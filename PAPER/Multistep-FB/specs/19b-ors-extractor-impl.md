# Spec 19b — ORS shaping reward + GCIQL-flow extractor implementation contract

Audience: the implementing agent. Self-contained; science context in
`specs/19-ors-program.md`; upstream artifact = a frozen
`measure_density_behavior` checkpoint (spec 19a). TDD; default-off must be
BIT-identical.

## 0. What is being built

1. **A transcription duty (do this FIRST).** Read the ORS paper (ICLR
   2026, arXiv 2604.20627 — "Occupancy Reward Shaping: Improving Credit
   Assignment for Offline Goal-Conditioned RL"; check OpenReview
   EW8DskWQ1K and any released code). Transcribe and RECORD in a short
   note (`specs/19b-ors-transcription.md`) their exact conventions:
   (a) is r_W used ALONE as the reward or composed with the sparse
   reward; (b) their normalization/scaling of r_W; (c) the discount used
   inside the (1−γ) occupancy normalization vs the critic's γ; (d) any
   optimality-preservation condition their proof needs, so we do not
   silently violate it. Where this contract's defaults below conflict
   with the paper, THE PAPER WINS for the primary arm (bit-faithful
   transcription is the repo's pattern — PSM precedent); our variants
   become explicitly-labeled ablations.

2. **The shaping reward.** For goal g with hit-spec blocks/tolerances:

       d²(u, g)   = Σ_b ‖(u_b − g_b)/ε_b‖²          (per-block ε-scaled)
       r_W(s,a,g) = ors_reward_scale · (1−γ) · Σ_j w_j(s,a) · (−d²(u_j, g))

   where w(s,a) = softmax over a FIXED pool U_P of
   F_D(s,a)ᵀB_D(ũ_j)/τ from the frozen behavior-density towers, min-free
   (see §2.3 for the 2-head convention). The ε-scaling reuses the
   hit-spec machinery so cs/cd/scene are comparable and the metric
   matches the geometry the measure was trained in.

3. **The extractor.** Stock GCIQL-flow (IQL value/critic + flow actor,
   g-conditioned — the stack-divergence-compliant extraction path) in
   which the sparse relabelled reward is replaced by r_W inside the
   value/critic losses. NO hindsight columns, no phase switch, no measure
   loss — this is a plain reward-driven GCIQL run whose reward comes from
   a frozen network instead of `batch['rewards']`.

4. **A calibration probe (HARD GATE before any extractor launch).**

## 1. New file — `scripts/rescuing_fb/ors_calibration_probe.py` (P1.2 gate)

Eval-only, runs on a 19a checkpoint. The ground truth for a BEHAVIOR
occupancy is the DATASET itself (no rollouts). Sample N=2000 dataset
transitions (seeded, spread across trajectories, each with ≥500 same-
trajectory steps remaining) × the env's 5 task goals + 20 random dataset
goals; for each sampled index i and goal g compute r_W(s_i, a_i, g) from
the checkpoint, and as ground truth the empirical discounted future
distance Σ_{k≥1} γ^{k−1}(−d²(x_{i+k}, g)) within the
trajectory (truncate at horizon 500, renormalize by the truncated
discount mass) vs r_W(s_i, a_i, g). Report Spearman per env (pooled over
goals) + scatter JSON. GATE: Spearman ≥ 0.65. Also report Spearman on the
ACTION axis where the data allows (states with ≥2 near-duplicate (s,·)
pairs are rare — report coverage honestly; the state-axis Spearman is
the gate).

Tests: numpy reference on a synthetic trajectory where the truncated
discounted sum is hand-computable; determinism (bit-identical repeat —
the spec-18 lesson: use a repeat as the noise floor).

## 2. Extractor changes in `tools/wandb_mode_shim/gciql_flow.py`

### 2.1 `get_config()`

```python
config.ors_reward_ckpt = ''      # spec 19b: path to a measure_density_behavior run dir.
  # When set, the value/critic losses use r_W from the frozen (F_D, B_D) towers instead
  # of batch['rewards']. '' = off, bit-identical.
config.ors_pool_size = 4096      # queries drawn once at create from the train dataset
config.ors_pool_seed = 0         # recorded; pool is a Monte-Carlo device, not a partition
config.ors_reward_scale = 1.0    # OVERRIDDEN by the transcription note if the paper pins one
config.ors_temp = 0.0            # 0.0 = read tau from the frozen ckpt's config (default);
                                 # >0 overrides (ablation only)
```

### 2.2 Frozen-model machinery

Mirror the EXISTING `freeze_critic_ckpt` / `warm_start_ckpt` restore
machinery (read it first; reuse its loader): at `create()`, when
`ors_reward_ckpt` is set, load the 19a checkpoint's density_f/density_b
params into extra registered-but-never-trained modules (`ors_f`,
`ors_b`), assert the loaded config has `measure_density_behavior=True`
(consistency refusal, both directions, loud), and read γ/τ/hit_spec from
the LOADED config — refusing mismatches with the extractor's own
hit_spec/discount rather than silently mixing conventions.

Pool: drawn at create from the train dataset's next-state object block
coords with `ors_pool_seed` (the dataset is reachable at create the same
way `ex_observations` is — verify how main.py passes example batches and
plumb the pool rows alongside; if create genuinely cannot see the
dataset, generate the pool in the launcher into the run dir and load by
path, recording sha256 — 17b-style, documented). Precompute
`B_D(ũ)` once; store pool + embeddings as static non-pytree fields
(the 17b protos pattern — NEVER a pytree leaf).

### 2.3 r_W inside the value/critic losses

Where the stock GCIQL value/critic losses consume `batch['rewards']`
(grep for it; the relabelled reward is built at
`third_party/ogbench/impls/utils/datasets.py:244`), branch when ORS is
on:

```python
logits = einsum(ors_f([s, a]), b_emb_pool) / tau          # [2, B, P]
w = softmax(logits, -1); w = w.mean(0)                    # 2 heads: MEAN of the two
                                                          # softmaxes (r_W is a reward,
                                                          # not a value estimate -- no min;
                                                          # record this convention)
r_w = ors_reward_scale * (1 - gamma_frozen) * (w * (-d2_pool_to_goal)).sum(-1)
```

with `d2_pool_to_goal` the ε-scaled squared distances from every pool
point to that row's goal (`value_goals` for the value loss's reward,
matching wherever `batch['rewards']` was consumed — keep the SAME
goal key the stock loss pairs with the reward; do not mix). Whether r_W
REPLACES or is ADDED to the sparse reward: per the transcription note
(§0.1a). Stop-gradient everything ORS-side (frozen params make this
structural; assert it in the grad-hygiene test anyway).

`gc_negative` interplay: the stock reward is {−1,0}; r_W is a different
scale. The transcription note's normalization governs; if the paper
normalizes per-batch or by a dataset constant, implement exactly that and
log `ors_r_mean/std/min/max` info keys unconditionally when on.

### 2.4 Lattice

```python
if config.get('ors_reward_ckpt', ''):
    assert not config.get('measure_only', False), "the extractor is a plain GCIQL run"
    for bad in ('measure_density', 'measure_delta', 'measure_delta_partition',
                'measure_first_passage', 'warm_start_ckpt', 'freeze_critic_ckpt'):
        assert not config.get(bad, False)
    assert config.get('actor_kind', 'flow') == 'flow' and config.get('actor_cond', 'goal') == 'goal'
    assert config['encoder'] is None
    assert int(config.get('ors_pool_size', 0)) >= 256
```

## 3. Tests — `tests/test_gciql_ors_extractor.py`

1. `test_default_off_bit_identical` — `ors_reward_ckpt=''`: param tree +
   total_loss bitwise-identical to stock.
2. `test_rw_matches_numpy` — tiny frozen towers written by the test:
   r_W numpy reference (softmax-mean, (1−γ), scale, ε-scaled d²) matches
   the in-loss computation.
3. `test_reward_substitution` — with ORS on, the critic/value TD targets
   are built from r_W (assert by constructing a batch where
   batch['rewards'] is poisoned with NaN — the loss must be finite if and
   only if the substitution happened; if the transcription says ADD, use
   a sentinel value instead and assert the sum).
4. `test_frozen_consistency_refusals` — non-behavior ckpt refused;
   hit_spec/discount mismatch refused; pool_size < 256 refused.
5. `test_grad_hygiene` — grads of total_loss over `ors_f`/`ors_b` leaves
   are exactly zero (they are not in grad_params — assert structurally).
6. `test_linearity` — r_W is linear in the pool rewards:
   r_W(αr₁+βr₂) = αr_W(r₁)+βr_W(r₂) on random inputs.
7. `test_end_to_end_updates` — 30 updates with a synthetic frozen ckpt:
   finite, no NaN.

## 4. Configs + wave (launch per spec 19 §1 P1.3, only after the P1.2 gate)

`configs/gciql/{cube_single,cube_double,scene}_state_ors_d50.yaml` — stock
GCIQL-flow scaffolding (copy the plain gciql flow config the bridge
baseline uses; verify it is the same recipe class as the spec-17/18
configs minus measure knobs) + the `ors_*` knobs pointing at the P1.1
run dirs. run_groups `GCIQL-{Env}-Flow-ORS-d50`. Seeds: cs 0/1/2,
cd/scene 0/1/2 (the discriminators get full seeds; this is the program's
main bet). eval_interval per spec 19 §4.

Gates (spec 19 §1 anchors): cs within ±0.05 of the density arm (tie
expected — a LOSS flags the shaping signal); cd > .549 phase-1 anchor =
the headline; scene vs .927 hindsight = the honest comparison, and
> .48 phase-1 = the minimum interesting bar.

## 5. Out of scope

- Classical-FB z_g^dist shaping (P4, gated).
- Any change to the 19a training side.
- Reward composition variants beyond the transcribed convention + at most
  one labeled ablation (replace-vs-add) if the paper leaves it genuinely
  ambiguous.
