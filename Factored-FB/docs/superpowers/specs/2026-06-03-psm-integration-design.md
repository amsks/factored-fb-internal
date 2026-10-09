# PSM Integration (state FlowBC + pixel DrQ+FlowBC) — Design Spec

**Date:** 2026-06-03
**Status:** Design — awaiting user review before implementation plan.
**Reference implementation:** `/home/adityamohan/git/baselines/PSM` (paper: *Proto Successor Measure: Representing the Behavior Space of an RL Agent*, Agarwal et al.; pure PyTorch, torchrl/Hydra/dmc-coupled).

## Goal

Integrate **PSM** into our codebase as a faithful baseline alongside FB, plugged into both experiment pipelines:
- **State:** PSM representation + **FlowBC** actor (mirrors `FBFlowBCAgent`).
- **Pixel:** PSM representation + **DrQ encoder + FlowBC** actor (reuses our existing DrQ front-end wiring).

PSM eval uses **reward-inference** (z from relabeled rewards), identical protocol to FB's `_infer_z`, for apples-to-apples comparison on the same OGBench cube-single tasks/metrics.

A **bit-exact equivalence test** against the reference implementation is the correctness oracle (see §9). This is non-negotiable: the base TD3-actor PSM must produce *exactly* the same outputs as reference given identical inputs.

## Decisions (from brainstorming)

1. **Fidelity:** Faithful PSM — reimplement PSM's defining proto-successor-measure structure (basis Φ + dual successor nets `sf_ψ`/`psm_ψ` + binary/Gaussian z + orthonormality), not a PSM-lite on FB rails and not a vendored wrap.
2. **Scope:** One combined spec covering **state + pixel**.
3. **Eval:** Reward-inference, matching FB's protocol.
4. **Architecture:** New `agents/psm/` package mirroring `agents/fb/` (+ `agents/rldp/`), reusing leaf components; FB code untouched.
5. **Equivalence:** Bit-exact (`torch.equal`, zero ULP) reference-equivalence test on the **TD3-actor** PSM.

## Background: PSM in FB terms (verified against reference)

| PSM | Role | FB analog |
|---|---|---|
| `phi` (Φ) | basis featurizer of the goal/next-state → outputs `z_dim` | backward map `B` |
| `sf_ψ` | conventional successor features for the **learned** policy; conditions on Gaussian `z ∈ R^z_dim`, outputs `z_dim` | forward map `F` |
| `psm_ψ` | **proto** successor net learned over a behavior *family*; conditions on **binary `z ∈ {0,1}^max_log_seed`** (default 16), outputs `z_dim` | (no FB analog) |
| `M = ψ(s,a,z)·φ(goal)ᵀ` | successor-measure matrix | `M = F·Bᵀ` |
| `Q = (sf_ψ·z).sum()` | control signal | `Q = (F·z).sum()` |

**Two distinct z dimensionalities (verified):** the SF/task z is `z_dim` (default 128, Gaussian-normalized, `sample_z`); the proto z is `max_log_seed`-dim (default **16**, binary, `sample_z_psm` via `int_to_binary_array`). `φ` and both ψ nets all *output* `z_dim` so `M = ψ·φᵀ` is well-formed; the two ψ nets differ only in their z-**conditioning input** width (16 vs 128). `psm_ψ = PsiMap(obs_dim, max_log_seed, action_dim, …)`, `sf_ψ = PsiMap(obs_dim, z_dim, action_dim, …)` (`psm.py:188–193`).

Per-step update (ref `agent/psm.py:695–773`, math at `:422–532`, actor at `:642–667`):
1. Sample **binary** `z ∈ {0,1}^z_dim` (`sample_z_psm`) → `update_psm`: target next-action from the **behavior sampler** (§4); `psm_loss = psm_diag + psm_offdiag + 1·ortho(φ)`; optimize **`psm_ψ` + `φ`**; soft-update `psm_ψ`,`φ` targets (τ).
2. Sample **Gaussian-normalized** `z` (`sample_z`), with `mix_ratio` rows replaced by `φ(goal_perm)` (analogous to FB `sample_mixed_z`) → `update_sf`: target next-action from the **learned policy**; `sf_loss = sf_diag + sf_offdiag` (**φ detached**, ortho coef 0); optimize **`sf_ψ`**; soft-update `sf_ψ` target.
3. `update_actor` driven by pessimistic `Q = (sf_ψ·z).sum()` — the seam where FlowBC plugs in (§5).

All ψ/φ nets carry a `num_parallel` ensemble; `get_targets_uncertainty` provides mean − pessimism·unc.

## File Structure

```
agents/psm/
  __init__.py
  model.py            # PSMModel: phi, sf_psi, psm_psi, encoders, normalizer, target nets, sample_z/sample_z_psm, reward_inference
  agent.py            # PSMAgent (TD3 actor): dual-ψ update, mixed-z, soft-updates, act, reward-inference, state_dict
  proto_sampler.py    # behavior-sampling actor for the proto-SM target (§4)
  nets.py             # PSM phi/psi/actor modules mirroring reference op-order (NOT FB's ForwardMap; see §9)
  flow_bc/
    __init__.py
    agent.py          # PSMFlowBCAgent: PSMAgent with TD3 actor -> FlowBC actor (Q from sf_psi)
configs/agent/psm.yaml          # mirrors fb.yaml
configs/agent/psm_flowbc.yaml   # mirrors fb_flowbc.yaml (inherits psm)
scripts/agents/psm/state.sh     # mirrors scripts/agents/fb/state.sh (+ pixel variant)
tests/test_psm_*.py
```

Reused unchanged: our normalizers, DrQ rgb-encoder/augmentator (via `build_pixel_cfgs(cfg)` → `rgb_encoder_cfg`/`augmentator_cfg` in the `shared` dict), `TruncatedNormal`, the FlowBC `NoiseConditionedActor` + `SimpleVectorField` (FlowBC variant only), `evals/ogbench.py`, the OGBench cube-single env + offline dataset. **FB classes are not modified.**

**Net-reuse exception (forced by §9):** PSM `φ`/`ψ`/TD3-actor live in `agents/psm/nets.py` and mirror the reference `models/psm_models.py` module structure and reduction order exactly, so reference weights transfer and forward is bit-identical. We deliberately do **not** reuse FB's `ForwardMap`/`BackwardMap` for these (their op order differs and would break bit-exactness).

## §5 — FlowBC actor (state + pixel)

`PSMFlowBCAgent(PSMAgent)` replaces the TD3 actor with the FlowBC actor exactly as `FBFlowBCAgent` does for FB:
- BC flow vector-field (`SimpleVectorField`, goal-unconditioned) trained by flow-matching on dataset actions.
- Q-driven actor head where **Q comes from `sf_ψ·z`** (mirroring FB's `forward_map·z`), pessimistic over the ensemble.
- `bc_coeff`, `flow_steps`, `lr_actor_vf` knobs mirror `fb_flowbc.yaml`.

**Pixel** is the same `PSMFlowBCAgent` plus the DrQ encoder/augmentator, wired through the existing `shared` pixel machinery — state vs pixel is purely a config switch (`configs/domain/visual_cube_single.yaml`'s `encoder: drq` + `augmentator`), identical to how FB does it. The DrQ host-bound pipeline caveat (`ogbench-drq-augment-host-bound`) applies as it does for FB.

## §4 — Behavior-sampling actor for the proto-SM target

PSM's `psm_ψ` target uses `SamplingSeedActor` (`psm.py:29–53`): a deterministic pseudo-random policy keyed by `(next_obs_hash, binary z)` so the proto basis spans a *family* of behaviors, not just the learned policy. The seed = `(⟨z, 2^i⟩ + next_obs_hash) % max_seed`, where `max_seed = 2^max_log_seed + 20000` (default `2^16 + 20000 ≈ 85k`, feasible to precompute), indexing a precomputed `seed→random_action` table. Note `z` here is the **16-dim binary** proto z, which is why the table is small.

Our offline `DictBuffer` has no `next_observation_hash`. **Plan:**
- `agents/psm/proto_sampler.py` reproduces the reference behavior exactly: same seed→action table construction and same `(hash(next_obs) + ⟨z,2^i⟩) % max_seed` indexing.
- Add a `next_observation_hash` field, precomputed once at dataset load, in the data loader **behind a flag** (default-off → byte-identical for FB and all other agents). Only PSM enables it.
- This sampler's equivalence to the reference table is asserted in the §9 test.

*(Considered alternative: hash on-the-fly from the obs tensor each step — no loader change, slightly slower. Rejected in favor of the precomputed-hash version for fidelity + throughput. Revisit if the loader change proves intrusive.)*

## §6 — Eval / inference (reward-inference, matches FB)

`PSMModel.reward_inference(next_obs, reward)` → `z = project(Σ r·φ(next_obs)/N)`, identical protocol to FB. `evals/ogbench.py` is unchanged (it already calls `agent.model.reward_inference` via `_infer_z`); PSM eval acts with the `sf_ψ`-driven policy (TD3 for the equivalence variant, FlowBC for experiments). No `eval_context` (that was goal-conditioning-specific). Same tasks, same metrics → directly comparable to FB.

## §7 — Config, registration, launcher

- `configs/agent/psm.yaml`: phi/sf/psm hidden dims + `num_parallel`, `z_dim`, `norm_z`, `lr_phi`/`lr_sf`/`lr_actor`, `target_tau`, pessimism penalties, `mix_ratio`, `ortho_coef`, proto-sampler knobs, `stddev_schedule`/`stddev_clip` (TD3 actor). Defaults to **PSM paper values** (e.g. `z_dim=128`, larger ψ hidden dims) but everything configurable.
- `configs/agent/psm_flowbc.yaml`: inherits `psm`, swaps actor → FlowBC (`NoiseConditionedActor` + `SimpleVectorField`), adds `bc_coeff`/`flow_steps`/`lr_actor_vf`.
- `make_agent` gains `psm` and `psm_flowbc` branches, reusing the existing `shared` dict (+ PSM-specific keys).
- `scripts/agents/psm/state.sh` (+ pixel variant) mirroring `fb/state.sh`: same conventions (wandb entity `amsks`, `/dev/shm`, SEEDS/GPUS/TRAIN_STEPS/RUN_GROUP, DRY_RUN).

## §8 — Hyperparameters & capacity

Default to PSM paper values, all configurable. For a fair FB head-to-head, a **capacity-matched** run (PSM `z_dim`/hidden matched to state FB) is exposed via config so it can be added as an ablation later — mirroring FB's `L_dim=256` ablation. No extra runs built now.

## §9 — Bit-exact reference-equivalence test (correctness oracle)

**Requirement:** Our base **`PSMAgent` (TD3 actor)** must produce **EXACTLY** the same outputs as the reference PSM given identical inputs — `torch.equal` / zero ULP, **not** `allclose`, not "matches to N decimals".

**Harness (`tests/test_psm_reference_equivalence.py`):**
- Runs on **CPU**, **float64**, with `torch.use_deterministic_algorithms(True)` and fixed seeds.
- Imports the reference network classes from `/home/adityamohan/git/baselines/PSM` (`models/psm_models.py`, `models/parallel_modules.py`, `models/modules.py`) **under our `.venv` torch** (eliminates torch-version kernel skew). If those modules can't import cleanly (torchrl/dmc deps), the test transcribes the exact reference forward/loss code verbatim with `psm.py:LINE` citations as the oracle.
- Constructs reference + our nets with the **same architecture/config**; copies the reference `state_dict` into ours (a key-mapping shim).
- Fixes one batch as explicit tensors: `obs`, `action`, `next_obs`, `goal`, the injected **z** (both binary and Gaussian variants — bypass random sampling), and the **next_action** (bypass the actor/behavior sampler so the comparison isolates the nets + loss + update).

**Assertions (`torch.equal` at every level):**
1. **Forward:** `φ(goal)`, `sf_ψ(s,a,z)`, `psm_ψ(s,a,z)`, `actor(s,z)` mean/sample, and the `M` matrices.
2. **Losses:** `psm_diag`, `psm_offdiag`, `ortho_diag`, `ortho_offdiag`, `psm_loss`; `sf_diag`, `sf_offdiag`, `sf_loss`; `actor_loss`, `Q`.
3. **Gradients:** every `.grad` tensor after `backward()` for `φ`, `sf_ψ`, `psm_ψ`, `actor`.
4. **Post-step parameters:** every parameter after the optimizer step (same optimizer, lr, betas, eps), and the target-net params after soft-update.

**Failure policy:** if any op-order difference makes `torch.equal` impossible, the test **fails loudly** and we fix the op order in `agents/psm/nets.py` — we do **not** relax to `allclose`. (PSM's math is matmuls + sum reductions + MLPs, all op-order-controllable, so bit-exactness is attainable.)

This test pins `PSMAgent` (TD3) to the reference. `PSMFlowBCAgent` is then defined as exactly `PSMAgent` with the actor swapped — a clean, separately-tested delta.

## Testing strategy (TDD, `.venv/bin/python -m pytest`)

Unit tests mirroring the FB suite:
- Nets build at correct dims; `state_dict`/`load_state_dict` round-trip.
- `sample_z` (Gaussian-normalized) and `sample_z_psm` (binary) shapes/properties.
- `proto_sampler` determinism + **equivalence to the reference table**.
- One `update` step runs with finite `psm_loss`/`sf_loss`/`actor_loss` — state and pixel.
- `reward_inference` returns correct-dim z.
- FlowBC `act()` output shape (state + pixel).
- 50-step e2e smoke for state + pixel (all variants finite).
- **§9 bit-exact reference-equivalence test.**
- 2k-step GPU sanity per variant before any campaign (exercises real eval + checkpoint save).

Known repo caveats honored: tests via `.venv/bin/python -m pytest` (not `uv run`); pre-existing pandas/script import failures are unrelated; commits omit the Claude co-author trailer.

## Out of scope (now)

- The 10-seed @1M PSM campaign launch (state + pixel) — separate decision after integration is validated, like the FB flow.
- The capacity-matched ablation runs.
- Online RL (PSM supports it; we target offline cube-single only).

## Open risks

- **`SamplingSeedActor` fidelity (resolved-but-verify):** the seed table is `2^max_log_seed + 20000 ≈ 85k` entries (binary z is 16-dim, not 128) — feasible to precompute and reproduce exactly. Reference action formula is `(torch.rand(action_dim) - 1)*2` per seed (note: yields `[-2, 0)`, likely a reference quirk but we reproduce it verbatim for bit-exactness, since the §9 test bypasses the actor by injecting `next_action`). **Action during planning:** confirm `max_log_seed`, the `manual_seed(i)`→action construction, and the `%max_seed` indexing reproduce byte-for-byte; reproduce the per-seed RNG draw order exactly.
- **Reference import under our torch:** torchrl/dmc deps may block importing `models/*`. Fallback (verbatim transcription of the forward/loss with `psm.py:LINE` citations) is specified in §9.
- **`next_observation_hash` reproduction:** the reference's commented loader (`psm.py:262`) hashes `str(next_obs_np[i]) % max_seed`. We must match its exact hashing for the proto-sampler equivalence; pin the precise hash function during planning.
- **Pixel throughput:** DrQ host-bound pipeline (see memory `ogbench-drq-augment-host-bound`).
