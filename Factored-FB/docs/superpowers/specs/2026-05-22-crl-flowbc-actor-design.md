# CRL + FlowBC actor (JAX) for OGBench — Design

**Status:** approved (pending user review)
**Date:** 2026-05-22

## Goal

Give OGBench's **CRL** agent FB's **FlowBC actor** (a flow-matching velocity field
+ a one-shot noise-conditioned actor), in JAX/Flax, so we can run CRL with the DrQ
encoder **and** the FlowBC actor — matching FB's actor architecture and flow
mechanics, with the actor's Q-guidance driven by CRL's contrastive critic. Applies
to both modalities: **state** (`cube-single-play-v0`, no encoder) and **pixel**
(`visual-cube-single-play-v0`, DrQ front-end). Never edits vendored OGBench.

## Critical context (verified against the codebase)

- **FB FlowBC** (`agents/fb/flow_bc/{agent,model}.py`, `nn_models.py:269-433`) is an
  FQL-style scheme with two networks:
  - **`VectorField`** (`nn_models.py:416-433`): `net = [Linear(obs+act+1, h)→GELU] +
    [Linear(h,h)→GELU]×(L-1) + Linear(h, act)`; `forward(obs, action, t) =
    net(concat[obs, action, t])`. **Obs-conditioned only** (no goal). FB config
    (`configs/agent/fb_flowbc.yaml`): `hidden_dim=512, hidden_layers=4`.
  - **`NoiseConditionedActor`** (`nn_models.py:280-298`): `embed_z =
    simple_embedding(obs+z+act, h)`, `embed_s = simple_embedding(obs+act, h)`,
    `policy = [Linear(h,h)→ReLU]×hidden_layers + Linear(h, act)`; `forward(obs, z,
    noise) = tanh(policy(concat[embed_s(concat[obs,noise]),
    embed_z(concat[obs,z,noise])]))`. FB config: `hidden_dim=512, hidden_layers=2,
    embedding_layers=2`.
  - **`simple_embedding`** (`nn_models.py:199-206`, embedding_layers=2): `[Linear(in,
    h)→LayerNorm(h)→Tanh→Linear(h, h//2)→ReLU]`. Output dim `h//2` (=256 for h=512).
    So `policy` input = `concat[256, 256] = 512 = h`.
  - **FB actor update** (`agent.py:62-116`): `x_1=action`, `x_0~N(0,I)`, `t~U(0,1)`,
    `x_t=(1-t)x_0+t·x_1`, `vel=x_1-x_0`; `bc_flow_loss=MSE(vf(obs,x_t,t), vel)`;
    `actor_actions=actor(obs,z,noise)`; `Q` from forward map; `actor_loss=-Q.mean()`;
    if `bc_coeff>0`: `bc_loss=bc_coeff·MSE(actor_actions, compute_flow_actions(obs,
    noise))` (rollout no-grad), `actor_loss = actor_loss/|Q|.mean().detach() +
    bc_loss`; finally `actor_loss += bc_flow_loss`. `compute_flow_actions`
    (`agent.py:118-125`): `a=noise; for i in range(flow_steps): a += vf(obs, a,
    i/flow_steps)/flow_steps; clamp[-1,1]`. FB defaults: `flow_steps=10`,
    `bc_coeff=3.0` (manipulation), `lr_actor_vf=3e-4`. Eval (`act`) = one-shot
    `actor(obs, z, fresh_noise)`.
- **CRL** (`third_party/ogbench/impls/agents/crl.py`): contrastive bilinear critic
  (`GCBilinearValue`, `q1,q2`), Gaussian `GCActor` with `ddpgbc`/`awr` loss.
  `create` builds encoders from `encoder_modules[config['encoder']]`; `contrastive_loss`
  is the critic objective; `actor_loss` / `sample_actions` / `total_loss` are the
  actor surface we replace. `config['encoder']`, `frame_stack`, `p_aug` are placeholders.
- **Agent registry** (`agents/__init__.py`): `main.py` does `agent_class =
  agents[config['agent_name']]` (`main.py:77`). The `agents` dict is editable at
  runtime — the shim can register a new agent, exactly like DrQ registers into
  `encoder_modules`. `--agent` is a `config_flags` **config file** providing
  `get_config()` (`main.py:42`); the class is resolved separately by `agent_name`.
- **DrQ precedent:** `tools/wandb_mode_shim/sitecustomize.py` is first on the child
  PYTHONPATH (`run_gciql.py:101-103`); `ogbench_drq.py` holds the Flax encoder;
  parity bridges torch (`.venv`, in-proc) and jax (`/dev/shm/.venv-jax`, subprocess)
  via an `.npz`. We reuse this whole pattern.
- **Hard constraint:** never edit `third_party/ogbench` (pinned SHA).
- **Test harness:** `.venv/bin/python -m pytest`; JAX tests shell out to `.venv-jax`
  and skip when absent (`tests/jax_subprocess.py`).

## Deliberate deviations from FB (flagged for review)

1. **Q-guidance uses CRL's contrastive `min(q1,q2)`**, not FB's forward-map `F·z`
   (which is intrinsic to FB and has no CRL analog). This is the agreed integration.
2. **The one-shot actor's `z` is CRL's encoded goal, fed directly** — no FB `Norm`
   (`nn_models.py:395-398`) z-normalization, because CRL goals are not unit-norm task
   latents.
3. **Separate obs and goal encoders for the actor** (not CRL's single
   `GCEncoder(concat_encoder=...)`). This matches FB's structure (obs and z are
   separate inputs) and, at frame_stack=3, gives the actor a 9-channel obs encoding —
   *closer* to FB than CRL's 18-channel concat. The contrastive critic keeps its own
   `critic_state`/`critic_goal` encoders, unchanged.

## Components

### 1. Flow networks — `tools/wandb_mode_shim/ogbench_flow.py` (new)

Pure Flax modules + the rollout, importable and unit/parity-testable in isolation:

- `VectorField(nn.Module)`: `hidden_dim`, `hidden_layers`; `__call__(obs, action, t)`
  = `concat([obs, action, t])` → `[Dense(h)+GELU] + [Dense(h)+GELU]×(L-1) + Dense(act)`.
- `NoiseConditionedActor(nn.Module)`: `hidden_dim`, `hidden_layers`, `embedding_layers`;
  `_simple_embedding` = `[Dense(h)+LayerNorm+Tanh + Dense(h//2)+ReLU]` (for
  embedding_layers=2; general form mirrors `nn_models.py:199-206`); `__call__(obs, z,
  noise)` = `tanh(policy(concat[embed_s(concat[obs,noise]),
  embed_z(concat[obs,z,noise])]))`.
- `compute_flow_actions(vf, params, obs, noise, flow_steps)`: JAX Euler rollout
  matching `agent.py:118-125`, returns clamped actions. (Standalone fn so the parity
  test can exercise it directly.)

LayerNorm `epsilon=1e-5` (torch default); GELU uses the exact (erf) form to match
torch `nn.GELU()` default.

### 2. Agent — `tools/wandb_mode_shim/crl_flowbc.py` (new)

`class CRLFlowBCAgent(CRLAgent)` — inherits `contrastive_loss` (critic untouched),
overrides:

- **`create`**: builds the CRL `GCBilinearValue` critic (+ its `critic_state/goal`
  encoders) exactly as CRL does, plus `actor_vf` (`VectorField`) and `actor`
  (`NoiseConditionedActor`). When `encoder` is set, instantiates `actor_obs` and
  `actor_goal` encoders (`encoder_modules[...]`); for state, obs/goal pass through
  raw. ModuleDict keys: `critic`, `actor`, `actor_vf` (no `value`; FlowBC ≠ awr).
  Single `optax.adam(lr)` over all params (FB uses a separate `lr_actor_vf`, but here
  one optimizer with `lr=lr_actor_vf=3e-4` keeps things simple and matches FB's
  numeric lr; documented).
- **`actor_loss`**: replicates FB's update math (flow-matching L2 + Q-max + BC-distill,
  with the exact `-Q.mean()/|Q|.mean() + bc_coeff·MSE + bc_flow_loss` normalization),
  Q from `self.network.select('critic')(obs, actor_goals, actor_actions)` → `min(q1,q2)`.
  Randomness (`x_0`, `t`, `noise`) from the passed `rng`.
- **`total_loss`**: `contrastive_loss('critic') + actor_loss`. (No value/awr branch.)
- **`sample_actions`**: one-shot `actor(obs_enc, goal_enc, fresh_noise)`, clamped.
- **`get_config`**: CRL's config + `agent_name='crl_flowbc'`, `flow_steps=10`,
  `bc_coeff=3.0`, `lr_actor_vf=3e-4`, `actor_vf_hidden_dim=512`,
  `actor_vf_hidden_layers=4`, `actor_hidden_dim=512`, `actor_hidden_layers=2`,
  `actor_embedding_layers=2`. (`alpha`/`const_std`/`actor_loss` from CRL are unused by
  the FlowBC actor; left in the dict for compatibility.)

### 3. Injection — `sitecustomize.py` + `run_gciql.py` (modify)

- **`sitecustomize.py`**: new guarded block (independent of the DrQ block) — `import
  agents as _ogb_agents; from crl_flowbc import CRLFlowBCAgent;
  _ogb_agents.agents.setdefault('crl_flowbc', CRLFlowBCAgent)`. Wrapped in
  `try/except` so non-jax processes (the `.venv` test interpreter) stay inert.
- **`run_gciql.py`**: in `build_ogbench_argv` (or just before), resolve `agent_file`
  to an **absolute path when `REPO_ROOT/agent_file` exists**; otherwise pass the
  string through. This lets `agent_file: tools/wandb_mode_shim/crl_flowbc.py` work as
  `--agent` from the child's `cwd=impls`, while vendored `agents/*.py` (which do not
  exist under `REPO_ROOT`) pass through unchanged. The existing
  `OGBENCH_DRQ_FRONTEND=1`-on-`encoder=drq` gating is untouched.

### 4. Configs — `configs/gciql/` (new)

- `cube_single_state_crl_flowbc.yaml`: clone of `cube_single_state_crl.yaml` with
  `agent_file: tools/wandb_mode_shim/crl_flowbc.py`, `run_group:
  factored-fb-crl-flowbc`, `agent_overrides: {bc_coeff: 3.0}` (+ flow defaults from
  get_config). No encoder.
- `cube_single_visual_crl_flowbc_drq.yaml`: clone of `cube_single_visual_crl_drq.yaml`
  with `agent_file: tools/wandb_mode_shim/crl_flowbc.py`, `run_group:
  factored-fb-crl-flowbc-pixel-drq`, `agent_overrides: {bc_coeff: 3.0, batch_size:
  256, encoder: drq, frame_stack: 3, p_aug: 1.0}`.

### 5. Tests (`tests/`)

- **`test_ogbench_flow_nets.py`** — forward shape + determinism for `VectorField`,
  `NoiseConditionedActor`, and `compute_flow_actions` (jax subprocess).
- **`test_ogbench_flow_parity.py` — HARD GATE 1 (forward parity).** Weight-transplant:
  torch builds each net (in `.venv`), dumps weights + I/O; jax transplants and
  matches forward outputs **and** a 10-step `compute_flow_actions` rollout, `< 1e-5`.
- **`test_ogbench_flow_train_parity.py` — HARD GATE 2 (training-dynamics parity).**
  See the dedicated section below.
- **`test_crl_flowbc_injection.py`** — `agents['crl_flowbc']` registered; agent
  `create` builds (state and a fake-image case) without shape error (jax subprocess).
- **`test_crl_flowbc_configs.py`** — both configs compose and produce the expected
  `build_ogbench_argv` (`--agent=tools/wandb_mode_shim/crl_flowbc.py`,
  `--agent.bc_coeff=3.0`, run_group, env; pixel adds `encoder=drq` etc.).
- **`test_run_gciql_agent_resolve.py`** — `build_ogbench_argv` makes the repo-relative
  `crl_flowbc.py` absolute and leaves `agents/crl.py` untouched.

## HARD GATE 2 — supervised training-dynamics parity

**Requirement:** training the JAX FlowBC actor on the same values for some time must
produce the same outputs and the same training curves as FB's torch implementation.

**What is tested:** the **supervised** objectives shared with FB —
(a) the `VectorField` under the flow-matching L2 loss, and (b) the
`NoiseConditionedActor` under BC-distillation toward a **frozen** flow rollout. The
Q-guidance term is RL-specific to CRL (no FB counterpart) and is validated by the
smoke run, not here.

**How exactness is achieved (not just closeness):**

1. **Shared, pre-generated randomness.** Draw all stochastic inputs once in NumPy —
   the data batches `(obs, action)`, the flow `x_0`, the times `t`, and the actor
   `noise` for each of `N` steps — and feed the *identical* arrays to both frameworks.
   Neither framework's RNG is used in the loop.
2. **Identical initial weights.** Initialize in torch, transplant the same weights to
   the Flax modules (reuse the Gate-1 transplant mapping).
3. **Identical optimizer.** `Adam(lr=3e-4, β=(0.9,0.999), eps=1e-8)` — `torch.optim.Adam`
   vs `optax.adam` (same update formula `m̂/(√v̂+eps)`).
4. **float64 everywhere.** `jax.config.update('jax_enable_x64', True)` and torch
   `.double()`, so float32 rounding can't masquerade as a real divergence.

**Procedure (pytest orchestrates):** torch runs in-proc (`.venv`): transplant init
weights, run `N=100` Adam steps consuming the pre-generated arrays, record the
per-step loss and the final forward output on a held-out fixed input; save weights,
arrays, loss curve, and final output to `.npz`. The jax subprocess (`.venv-jax`)
loads the npz, transplants the **same** init weights, runs the **same** `N` steps on
the **same** arrays with optax Adam in x64, and asserts:

- per-step loss curve: `max |loss_jax[i] - loss_torch[i]| < 1e-6` for all `i`,
- final forward output: `max |out_jax - out_torch| < 1e-6`,
- final weights: `max |W_jax - W_torch| < 1e-6` per tensor.

Two such loops: one for the velocity field (flow-matching), one for the one-shot
actor (BC-distillation toward a fixed precomputed target). **Both are deployment
gates** — no CRL+FlowBC run launches until Gate 1 and Gate 2 pass (or SKIP only when
`.venv-jax` is absent). Fallback documented: if `optax`/`torch` Adam drift exceeds
`1e-6` purely from optimizer-impl differences, rerun the gate with plain SGD to
isolate network/gradient correctness, and investigate the Adam delta separately.

## Data flow

`run_gciql.py --config-name <crl_flowbc cfg>` → resolves `agent_file` to absolute →
child `main.py` (`.venv-jax`) with the shim on PYTHONPATH → `sitecustomize` registers
`crl_flowbc` (and, for pixel, the DrQ encoder + augment swap) → `main.py` loads
`get_config()` from the agent file, resolves `agents['crl_flowbc']`, calls
`CRLFlowBCAgent.create` → training writes checkpoints/CSVs to
`/dev/shm/gciql_outputs/factored-fb/<run_group>/sd<seed>_<ts>/`.

## Testing & verification order

1. Unit (shape/determinism) — Component 5 net tests.
2. **Gate 1** forward parity `< 1e-5`.
3. **Gate 2** training-dynamics parity `< 1e-6` (float64).
4. Injection + config + agent-resolve tests.
5. Smoke (manual, GPU + `.venv-jax`): 2k steps, `wandb=disabled` — **state first**,
   then pixel/DrQ; confirm the agent trains (flow + Q + distill losses finite and
   trending), eval runs, checkpoint written, throughput healthy.
6. Only after 1–5: wire these configs into the CRL sweep launch (update the existing
   CRL plan to point at the FlowBC configs).

## Risks / open notes

- **optax-vs-torch Adam exactness** at `<1e-6` is the main risk for Gate 2; float64 +
  the documented SGD fallback mitigate it.
- **GELU/LayerNorm form** must match torch defaults (erf-GELU, eps=1e-5) or Gate 1
  fails first — that localizes the issue early.
- **CRL builds extra encoders** (critic ×2 + actor obs/goal) for pixel — GPU memory
  checked by the smoke run, as with the CRL ddpgbc work.
- **`/dev/shm` is tmpfs** (reboot-wiped) — same caveat as all prior runs.

## Out of scope

- The sweep launch itself (covered by the CRL plan, retargeted after this lands).
- FlowBC for any agent other than CRL, or any env other than cube-single.
- Persisting `/dev/shm` outputs to durable storage.
- Editing vendored OGBench.
