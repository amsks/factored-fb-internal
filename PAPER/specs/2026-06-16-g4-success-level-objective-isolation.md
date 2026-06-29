# G4 — success-level objective isolation (z-conditioned actor on the IQL-value rep)

_2026-06-16. The last open arm of the GCIQL↔FB bridge program
(`2026-06-11-fb-psm-crl-gap-analysis-and-program.md`,
`2026-06-14-gciql-fb-morph-ladder.md`, master handoff `handoff-iwr-program.md`).
NB: this "G4" is the handoff's G4 (success-level objective isolation), **not** the
morph-ladder spec's G4 (zero-shot inference, cited)._

## Claim under test

Everything pointing at "the objective (IQL value-TD vs measure-matching) is FB's
binding ingredient" is so far **value-level** (cf-AUC):

- form exonerated — G3 (GCIQL, `value_arch=bilinear`, IQL, exact goal, flow actor)
  = **0.93** success;
- rank exonerated — d50 ≈ d128 ≈ d512 ≈ **0.90**;
- inference exonerated **at the value level** — `infer_ablation_gciql` shows the
  IQL bilinear value ranks goal-reaching states equally well under zero-shot
  `z = E[r·B]` (cf-AUC 0.92) as under the exact goal (0.92).

The gap: **G3's 0.93 uses an exact-goal actor; FB uses a zero-shot z-conditioned
actor.** The value *ranks* fine under inferred z, but a policy needs the actor to
*act* well under inferred z — never tested at the success level. G4 puts the
IQL-trained representation into FB's full zero-shot pipeline (bilinear + z-inference
+ z-conditioned actor) and measures **success**.

- **Success stays ~0.9** ⇒ the objective is the lever at the policy level too,
  holding FB's pipeline fixed; FB's failure is the measure objective, full stop.
  This is the reciprocal endpoint of the FB-IQL(0.20)↔G3(0.93) bridge.
- **Success collapses** ⇒ a success-level conditioning/policy-learning effect
  invisible to value-ranking; the story gains a nuance (objective + something about
  learning a z-conditioned policy).

Either way it is the decisive remaining experiment.

## The agent — one flag on `gciql_flow`

Add `actor_cond: 'goal' | 'z'` to the `gciql_flow` agent
(`tools/wandb_mode_shim/gciql_flow.py`).

- Default `'goal'` ⇒ **byte-identical** to the current `gciql_flow` agent
  (default-off contract).
- `'z'` requires `value_arch='bilinear'` (assert otherwise). Then:
  - **B := the value net's `psi`** (the action-free `GCBilinearValue`, the same map
    `infer_ablation_gciql` designates as B). One shared backward map, FB-faithful.
  - **Actor** conditions on `z = stop_grad(value.psi(goal))` instead of the raw
    28-dim goal observation. Implemented by feeding z as the `goals` argument to
    `GCNoiseActor` with `goal_encoder=None` (z is already a d-vector). The
    `stop_grad` keeps the representation **purely IQL-objective-trained** — the
    actor only *consumes* the rep, never reshapes it.
  - **Critic / value** unchanged from G3 (bilinear, IQL expectile V + double
    critic). The actor-loss Q-guidance term still calls
    `critic(obs, goal_obs, actor_actions)` (the critic computes its own psi
    internally); only the actor's *conditioning identity* becomes z.
  - **Flow vector field** (`actor_vf`) unchanged — it is state-only BC; the goal
    enters the policy only through the Q-guided one-shot actor, so only
    `GCNoiseActor` changes.

### Where z is computed in training

In `actor_loss`, the batch already carries `actor_goals` (raw goal obs). When
`actor_cond='z'`:

```
z = stop_gradient(value_psi(actor_goals))         # B := value net's psi, [B, d]
actor_actions = actor(obs, z, noise, params=grad_params)
q1, q2 = critic(obs, actor_goals, actor_actions)  # unchanged: critic computes psi internally
```

`value_psi(g)` = `network.select('value')(any_obs, g, info=True)[2]` (psi depends
only on the goal arg, as used in `infer_ablation_gciql`).

The one-shot↔flow distillation (`bc_coeff`) and `bc_flow_loss` are unchanged
(state-only). `sample_actions` gains the same branch: when `actor_cond='z'`, the
`goals` argument it receives is **already z** (the eval harness supplies it), so it
passes straight through to the actor with no psi call.

## Eval — standalone script `scripts/eval_zcond_gciql.py`

Follows the `phase_probe_gciql.py` / `infer_ablation_gciql.py` pattern: load the
agent via `phase_probe_gciql.load_agent(..., value_arch='bilinear',
value_latent_dim=d)`, roll the policy directly, never touch OGBench's eval loop.

Per task (`cube-single-play-singletask-task{1..5}-v0`), two modes on the **same
checkpoint**:

1. **exact-z**: `z = value.psi(goal_obs)` — the G3-analog *with* z-conditioning;
   isolates the actor-side z-bottleneck vs G3's raw-goal actor.
2. **inferred-z**: `z = E[r · value.psi(s')]` over relabeled transitions (FB's L0
   rule, `z = E[r·B]`) — the full FB zero-shot pipeline. Reuse the relabel/inference
   convention already in `infer_ablation_gciql` (faithful reward = cube `qpos[14:17]`
   dist < 0.04; obs layout is ambiguous so privileged qpos is required).

For each mode: feed z as `goals` to `agent.sample_actions`, roll episodes, count
**success** with the faithful cube-reward (same threshold/source as
`infer_ablation`). Report per-task and mean success for exact-z and inferred-z.

Output: `analysis/eval_zcond_gciql/s{seed}.json` with
`{seed, value_latent_dim, mean_success_exact, mean_success_inferred, per_task:{...}}`.

CPU/GPU: same env conventions as the other gciql probes —
`JAX_PLATFORMS=cpu MUJOCO_GL=egl /dev/shm/.venv-jax/bin/python` (CPU keeps it off
the training GPUs), pandas/pyarrow already installed in `/dev/shm/.venv-jax`.

## Runs

- 3 seeds × 500k steps, state `cube_single`, jax venv (GPU),
  `XLA_PYTHON_CLIENT_PREALLOCATE=false` to pack seeds per GPU, in **tmux**.
- wandb `entity=amsks`, group `gciql_g4_zcond`.
- **d = 512** primary — matches the G3 = 0.93 anchor, so the within-agent exact-z vs
  inferred-z dissociation is rank-controlled (rank already exonerated). **d = 50**
  optional FB-rank-parity confirmation.
- Launch via the same `run_gciql.py` path as the bridge runs (TORCH `.venv` has
  hydra; dispatches `main.py` to `/dev/shm/.venv-jax` via `storage=shm`). Set
  `save_interval ≤ train_steps` or no checkpoint is saved (handoff gotcha).
- Promote to 10×1M only if the result is clean and decision-relevant.

## The table G4 fills

| rep objective | actor conditioning | success |
|---|---|---|
| measure (plain FB) | zero-shot z | 0.42 |
| measure (FB-IQL critic) | zero-shot z | 0.20 |
| **IQL (G4)** | **exact z** | **?** |
| **IQL (G4)** | **zero-shot z** | **?** |
| IQL (G3) | exact goal (raw obs) | 0.93 |

## Tests / faithfulness

- **Parity guard**: `actor_cond='goal'` reproduces stock `gciql_flow` bit-for-bit
  (mirror `tests/test_gciql_flow_injection.py` if present).
- **z-cond actor**: forward pass with `actor_cond='z'` emits finite actions of the
  right shape; the z fed to the actor is d-dimensional.
- **stop-grad**: actor loss produces no gradient into the value net's psi params
  (the representation stays IQL-only).
- **Eval**: inferred-z vector has dim d; exact-z equals `value.psi(goal)`.

## Scope / caveats

- State-only; pixel out of scope.
- Eval uses privileged cube qpos for the faithful reward (dataset has none; obs
  `[19:21]` cube xy has unrecoverable per-axis scale/offset) — same constraint as
  `infer_ablation_gciql`.
- A clean read either direction is publishable: success ~0.9 confirms the objective
  thesis at the policy level; a collapse is a genuine new nuance.
- G4 says nothing new about the actor *architecture* (flow actor already exonerated
  at G1 = 72.3 / G3 = 0.93); it tests the *conditioning identity* (raw goal → z).
