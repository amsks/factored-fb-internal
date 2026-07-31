# Spec: Is bilinear GCIQL implicitly learning a metric? (VIP / QRL / CRL comparison on OGBench cube)

**Date:** 2026-07-31
**Status:** Spec — to be implemented by a separate agent. No code in this document has been run.
**Branch context:** written against `feat/metaworld-runs` (dirty tree; `impls/critics/crl.py` exists but is untracked).

---

## 0. Research question

Bilinear GCIQL (`iql_bilinear`: `Q(s,a,g) = ⟨φ(s,a), ψ(g)⟩/√d`, `V(s,g) = ⟨φ(s), ψ(g)⟩/√d`) reaches ~90% on OGBench `cube-single-play-v0` (wandb group `GCIQL-State-Flow-Bilinear-d50`: 90.7 ± 2 @500k; d=512: 94 @1M; monolithic-MLP GCIQL: ~73). The question: **is the factored critic succeeding because its embedding space encodes a similarity/metric over states**, akin to what three prior lines *provably* learn:

1. **VIP** (Ma et al., ICLR 2023, arXiv:2210.00030) — hard-codes `V(s;g) = −‖φ(s) − φ(g)‖₂` (raw L2, **no** normalization, no temperature) and trains φ via the Fenchel dual of KL-regularized goal-conditioned offline RL with reward `1[s=g] − 1`. At optimum, embedding L2 distance ≈ discounted time-to-goal. Symmetric by construction.
2. **QRL** (Wang et al., ICML 2023, arXiv:2304.01203) — parametrizes `d_θ(s,g)` with an architecturally-guaranteed quasimetric head (IQE) and trains `max E[d(s,g)] s.t. E[relu(d(s,s') − 1)²] ≤ ε²` (Lagrangian). Converges to `d* = −V*` (optimal goal-reaching cost). Asymmetric, exact triangle inequality by construction.
3. **CRL** (Eysenbach et al., NeurIPS 2022, arXiv:2206.07568) — **explicitly bilinear** critic `f(s,a,g) = φ(s,a)ᵀψ(g)` trained with NCE against geometric-discounted future-state positives. At the binary-NCE optimum: `f*(s,a,g) = log p^π(s₊=g|s,a) − log p(g)` — log discounted-occupancy ratio, i.e. an on-policy value up to a per-goal constant.

Key theory bridge (Myers et al., ICML 2024, arXiv:2406.17098 — "Learning Temporal Distances"): any occupancy-style critic decomposes as `f(s,g) = c(g) − d(s,g)` where the candidate quasimetric is recovered by **self-distance subtraction**:

```
d_SD(s, g) := f(g, g) − f(s, g)
```

and this is a true quasimetric at the optimum of the *symmetrized* contrastive objective. This gives us a falsifiable, method-agnostic way to ask whether bilinear GCIQL's logits carry a metric.

Two competing hypotheses to discriminate (they are **not** the same claim):

- **H-sim (cosine-similarity hypothesis):** ψ (and φ) organize states so that the bilinear logit behaves like a symmetric similarity — approximately `‖φ‖‖ψ‖·cos(φ,ψ)` with concentrated norms → effectively cosine similarity, like a contrastive encoder.
- **H-qm (quasimetric-value hypothesis):** the logits encode a *directed* value geometry — `d_SD` is asymmetric where the environment is (e.g. non-prehensile cube states), approximately satisfies the triangle inequality within a trust region, and correlates with temporal distance — like QRL, not like cosine similarity.

Both could be partially true (symmetric similarity + small asymmetric correction). The probe battery below measures each property independently.

---

## 1. What each reference method learns (exact objectives to implement)

### 1.1 VIP — state-based variant (new critic `vip`)

Modality-agnostic dual objective (VIP Eq. 6). Per batch, sample sub-trajectories: initial obs `o₀`, adjacent transition `(o_k, o_{k+1})` with `t ≤ k < T`, goal `g = o_T` (final frame of the sub-trajectory). With `V(s,g) := −‖φ(s) − φ(g)‖₂` (plain L2 on the encoder output; **do not** L2-normalize φ; no temperature):

```
L(φ) = (1−γ)·E[ ‖φ(o₀) − φ(g)‖₂ ]
     + log E[ exp( ‖φ(o_k) − φ(g)‖₂ − r̃ − γ‖φ(o_{k+1}) − φ(g)‖₂ ) ]        (log-mean-exp over the batch)
```

with `r̃ = 1[o_k ≈ g] − 1` (use the same `gc_negative` sparse reward the repo already computes: `r = success − 1`; for VIP sampling, `r̃ = −1` unless `k+1 = T`... see §4.3), a small `eps=1e-6` inside the log for stability, `γ = 0.98` (VIP's default; also run γ=0.99 to match the RL agents — see run matrix). Reference implementation: `facebookresearch/vip`, `vip/trainer.py` (the loss is a literal transcription of the equation above) and `vip/models/model_vip.py` (`sim = -torch.linalg.norm(e1 - e2, dim=-1)`).

VIP needs no actor and no Q — pair it with `flowbc` (pure BC actor) so it slots into the existing `apply_update` loop unchanged. Its policy performance is irrelevant to this experiment; only φ matters.

**Encoder:** MLP `[512, 512, 512] → latent` with LayerNorm/gelu, matching `GCBilinearValue` towers, `latent ∈ {50, 512}` to match the GCIQL cells. This keeps capacity comparisons apples-to-apples.

### 1.2 QRL — offline, state-based (new critic `qrl`)

Components:
- **Encoder** `f: S → R^{k·l}` (same MLP trunk as above; output dim = `iqe_components k × dim_per_component l`, e.g. k=64, l=8 → 512, and a k=8×l=8=64 small variant to pair with d=50-ish capacity).
- **IQE head** (Wang & Isola, arXiv:2211.15120). Reshape `u, v ∈ R^{k·l}` to `(k, l)`. Per component i, the Lebesgue measure of a union of intervals:
  `d_i(u,v) = |⋃_j [u_ij, max(u_ij, v_ij)]|`  (asymmetric: interval empty when `v_ij ≤ u_ij`).
  Reduction: IQE-maxmean, `d = α·max_i d_i + (1−α)·mean_i d_i`, α = sigmoid(learnable), init 0.5.
  Implement in JAX; validate against `torchqmet.IQE` (github.com/quasimetric-learning/torch-quasimetric) on random tensors to 1e-5 — write this as a unit test with hardcoded fixtures generated once from torchqmet (do not add a torch dependency to the repo; commit the fixture arrays).
- **Objective** (offline QRL, github.com/quasimetric-learning/quasimetric-rl):
  ```
  L = −E_{(s,g)}[ softplus_damp(d(s, g)) ] + λ · ( E_{(s,s')}[ relu(d(s,s') − step_cost)² ] − ε² )
  ```
  step_cost = 1, ε = 0.25 (QRL repo default; the implementing agent should confirm against `quasimetric-rl` configs and note the value used), λ ≥ 0 a learnable Lagrange multiplier (parametrize `λ = softplus(λ_raw)`, dual ascent: λ_raw gets the *negated* constraint-violation gradient — in practice implement as in the QRL repo: a separate optimizer or sign-flipped gradient on λ_raw). `softplus_damp` is QRL's monotone concave transform on the global term (read the exact form from the repo; a plain mean also works as first cut, note which was used).
  `(s,g)` pairs: use the existing `GCDataset` `value_goals` sampling (random + trajectory goals); `(s,s')` from the transition batch.
- **No latent dynamics model, no actor extraction** — pair with `flowbc` as above. We only need `d_θ`.

### 1.3 CRL (existing untracked critic `crl`)

`impls/critics/crl.py` already implements sigmoid-BCE over the B×B logit matrix on `GCBilinearValue` (`value_exp=False`, no target nets). Before using it, the implementing agent must verify its goal sampling matches CRL's contract: **positives must be geometric future states from the same trajectory** (`value_p_curgoal = 0, value_p_trajgoal = 1, value_p_randomgoal = 0, value_geom_sample = true`, geometric parameter = 1−γ) — with B×B logits, the negatives come from the batch off-diagonal, so `value_p_randomgoal` must be 0. Fix `impls/configs/critic/crl.yaml` if it deviates. Latent dims {50, 512} to mirror GCIQL.

Note the semantic difference to keep in mind when interpreting results: CRL learns the **behavior policy's** occupancy; GCIQL (expectile 0.9) approximates an in-sample-max **optimal** value; QRL targets the optimal cost directly; VIP targets a KL-regularized optimal value. Their distances are *not* supposed to agree perfectly even in the limit — the QRL paper's MountainCar analysis shows CRL recovers on-policy, not optimal, geometry. The probes are designed to be robust to monotone reparametrizations (rank correlations) for exactly this reason.

---

## 2. Subject models and where their embeddings live

All heads expose `info=True` returning `(v, phi, psi)`:
- `GCBilinearValue` — `impls/utils/networks.py:376-377`
- `GCBilinearPsiValue` (used by `iql_bilinear`) — `networks.py:450-451`; `psi_of` is exposed as a Flax `method=` (`impls/critics/iql_bilinear.py:112,115`)

Subject: `iql_bilinear` trained checkpoints. Extract, for any batch of states/goals:
- `ψ_V(g)`, `φ_V(s)` from the **value** net (state-only tower — primary for geometry probes, since it needs no action),
- `ψ_Q(g)`, `φ_Q(s,a)` from the critic net (use dataset actions when needed),
- scalar `V(s,g)` and `Q(s,a,g)`.

Config recap of the subject (already the defaults in `impls/configs/critic/iql_bilinear.yaml`): expectile 0.9, γ=0.99, `psi_base: learned`, `project_z: false`, `double_q: true`, `iql_ortho_coef: 0.0`, hidden `[512,512,512]`, `value_latent_dim` set per run (50 / 512), reward `success − 1` (`gc_negative`), masks `1 − success`.

---

## 3. Distance constructions (per method, fixed notation)

For every trained model define one or more scalar maps `d̂(s, g)`; all probes run on each construction.

| id | method | construction |
|---|---|---|
| D1 | iql_bilinear | **steps-from-value** (TTGS, arXiv:2510.07257): `d̂(s,g) = log_γ(1 + (1−γ)·V(s,g))`, valid because with `r = −1`-until-goal, `V ≈ −(1−γ^T)/(1−γ)`. Clip V to `(−1/(1−γ) + δ, 0]` before the log. |
| D2 | iql_bilinear | **self-subtraction on raw logits** (Myers et al.): `d̂(s,g) = f(g,g) − f(s,g)` where `f(s,g) = ⟨φ_V(s), ψ_V(g)⟩/√d`. |
| D3 | iql_bilinear | **ψ-space cosine**: `d̂(s,g) = 1 − cos(ψ_V(s), ψ_V(g))` (embed *both* through ψ). Also the φ-space variant `1 − cos(φ_V(s), φ_V(g))`. |
| D4 | iql_bilinear | **ψ-space L2**: `‖ψ_V(s) − ψ_V(g)‖₂` (and φ-space variant). |
| D5 | crl | self-subtraction: `f(g,g) − f(s,g)` on its bilinear logits (state-only V tower if the critic has one; otherwise `φ(s, a_data)` with the dataset action, plus report sensitivity to action choice). |
| D6 | qrl | native: `d_θ(s, g)` (already a quasimetric). |
| D7 | vip | native: `‖φ(s) − φ(g)‖₂`. |
| D8 | gciql (monolithic) | steps-from-value D1 on its MLP `V(s,g)` — control: value quality without factorization. |
| D9 | random-init | D2–D4 on an **untrained** `iql_bilinear` net (same seed family) — null control. |
| D10 | raw obs | `‖s − g‖₂` in observation space, and `‖s_cube − g_cube‖₂` restricted to the cube-position coordinates — "no learning needed" control. |

D1 vs D2 comparison is itself informative: if they rank-agree strongly, the per-goal offset `c(g) = f(g,g)` is what turns logits into distances, exactly the Myers decomposition.

**Goal-input caveat:** in OGBench `GCDataset`, goals are full observations (future states), so `ψ(·)` accepts any state and D3/D4 are well-defined. At eval time, `env.reset(options=dict(task_id=..., render_goal=...))` provides `info['goal']` in the same observation space (`impls/utils/evaluation.py:37-80`).

---

## 4. Probe battery

All probes read from a fixed **probe set** built once and cached to disk (npz): from `cube-single-play-v0-val.npz` (exists at `~/.ogbench/data/`), sample
- `N_states = 10,000` states uniformly over the val set,
- `N_traj = 200` full validation trajectories,
- `N_pairs = 100,000` (s,g) pairs: 50% same-trajectory (with recorded time gap Δt), 50% cross-trajectory,
- `N_triplets = 1,000,000` (s, w, g) triples uniform over `N_states` (TTGS protocol),
- plus 50 evaluation rollouts of the trained iql_bilinear×flowbc policy (successful ones kept, with the episode's goal) for monotonicity curves.

Fixed RNG seed for the probe set; every model is probed on the *identical* arrays.

### P1. Temporal-distance correlation (does d̂ track dynamics?)
For same-trajectory pairs `(s_t, s_{t+Δ})`: Spearman ρ between `d̂(s_t, s_{t+Δ})` and Δ. Report per method/construction, plus the D10 baselines. Also report ρ *partialed on* D10 (rank-regress out raw-obs distance) to show learned structure beyond state-space proximity. Bin by Δ ∈ {1–10, 11–50, 51–200, >200} to expose horizon effects.

### P2. Goal-approach monotonicity (VIP-style curves)
Along each successful eval rollout with goal g: plot `d̂(s_t, g)` vs t; report Spearman ρ(d̂, −t) per episode (median over episodes) and the fraction of adjacent steps that are non-monotone ("bump rate"). This is VIP's headline probe (their `plot_reward_curves.py`).

### P3. Triangle-inequality violation (TTGS protocol, arXiv:2510.07257)
Over the 1M triples: violation rate `P[d̂(s,g) > d̂(s,w) + d̂(w,g)]` and mean positive violation magnitude `E[relu(d̂(s,g) − d̂(s,w) − d̂(w,g))]` (in the construction's own units; for D1/D8 that is steps). Report globally and restricted to a trust region (all three pairwise d̂ below the τ-quantile, τ = 0.5). Reference points from the literature: QRL ≈ 0.2–0.7% rate with ~1e-3-step magnitude (architectural); HIQL/GCIQL-style values 0.2–5.6% rate but ~580-step magnitudes when they fail.

### P4. Symmetry / asymmetry index
`σ = E[|d̂(s,g) − d̂(g,s)|] / E[d̂(s,g) + d̂(g,s)]` over cross-trajectory pairs (guard the denominator; exclude near-zero pairs). Discriminates H-sim (σ ≈ 0 by construction for D3/D4/D7/D10) from H-qm (σ > 0 where cube dynamics are irreversible, e.g. states where the block was dropped). Also report σ for D1/D2/D5/D6. Note this probe has no standard published baseline (open evaluation niche) — report raw numbers plus the D9 null.

### P5. Is the bilinear logit secretly cosine? (direct test of H-sim)
On `N_pairs`: (a) Pearson/Spearman between `⟨φ_V(s), ψ_V(g)⟩` and `cos(φ_V(s), ψ_V(g))`; (b) coefficient of variation of `‖φ_V(s)‖` and `‖ψ_V(g)‖` over the probe states (norm concentration ⇒ dot ≈ scaled cosine); (c) same for `ψ_Q, φ_Q`. If (a) > 0.95 and (b) < 0.1, the critic is effectively cosine similarity + a scalar scale — strong H-sim evidence.

### P6. Cross-method representational agreement (the "akin to VIP/QRL" question, quantified)
Two levels:
- **RSA (primary):** for each method, build the `N×N` distance matrix over a 2,000-state subsample using its native d̂ (D2 for iql_bilinear, D5 crl, D6 qrl, D7 vip, D10 controls). Report Spearman correlation between the upper triangles of every pair of matrices → a method×method agreement matrix. This is invariant to monotone reparametrization, so the on-policy/optimal semantic gap (§1.3) doesn't confound it.
- **CKA (secondary):** linear CKA between the *embedding matrices themselves* — `ψ_V` (iql_bilinear) vs `φ` (VIP) vs encoder `f` (QRL) vs `ψ` (CRL) on the same 2,000 states. CKA is not standard in this literature (retrieval/ordering probes are), so treat as supporting evidence only.
- Include D9 (random init) rows in both — every claimed agreement must clearly exceed the random-net row, and D10 rows to show agreement is not just "everyone learned raw state distance."

### P7. Effective rank & spectrum
Singular value spectrum and effective rank (`exp(entropy of normalized σᵢ²)`) of the `ψ_V` and `φ_V` matrices over probe states, per method. Cheap, and speaks to whether d=512 runs actually use the capacity (the d=5 cell collapsed to 9.6% success — spectrum of a d=512 run tells us the functional rank of the solution).

### P8 (tier 2, optional). Interpolation retrieval (Stable-CRL protocol)
Interpolate `(1−u)ψ(s₀) + u·ψ(g)` for u ∈ [0,1], nearest-neighbor retrieve from val states in ψ-space, and compute the permutation error `Σ_t |perm[t] − t|` against a real trajectory from s₀ to g. Only meaningful if P6 shows nontrivial structure; skip in the first pass if time-constrained.

---

## 5. Run matrix

Domain: `--domain cube-single` (the `env/domains/cube-single.yaml` variant with `eval_aggregate: true` — single folded success scalar, directly comparable to the historical `evaluation/overall_success`). Data: `cube-single-play-v0` (play data, on disk). Actor: `flowbc` everywhere (BC — identical policy-extraction across methods, and VIP/QRL don't need a critic-driven actor).

| # | critic | latent | seeds | steps | purpose |
|---|--------|--------|-------|-------|---------|
| 1 | iql_bilinear | 50 | 3 | 500k | subject (headline 90.7% cell) |
| 2 | iql_bilinear | 512 | 3 | 500k | subject, capacity check |
| 3 | gciql (monolithic) | — | 3 | 500k | value-quality control, no factorization |
| 4 | crl | 50 | 3 | 500k | contrastive reference |
| 5 | crl | 512 | 3 | 500k | contrastive reference |
| 6 | qrl | 64 (k8×l8) | 3 | 500k | quasimetric reference |
| 7 | qrl | 512 (k64×l8) | 3 | 500k | quasimetric reference |
| 8 | vip | 50 | 3 | 500k | metric-embedding reference (γ=0.98) |
| 9 | vip | 512 | 3 | 500k | metric-embedding reference |
| 10 | vip | 50 | 1 | 500k | γ=0.99 sensitivity run |
| D9 | iql_bilinear, untrained | 50, 512 | 3 | 0 | null control (checkpoint at step 0) |

30-ish GPU runs at 500k steps; on this cluster's cadence (existing Metaworld runs are 300k with `save_interval 50000`) this is ~1.5× a Metaworld campaign. `save_interval 100000` so probes can also be run *across training time* (probe-vs-step curves are cheap once the probe script exists and are highly informative — when does metric structure emerge relative to when success rises?). Launch via `scripts/vista_train.slurm` / a variant of `scripts/launch_matrix.sh` with `CRITICS="iql_bilinear gciql crl qrl vip"`, `DOMAINS="cube-single"`. Sanity gate: run #1 must reproduce ≥ ~85% success (historical 90.7 ± 2 was @500k with `value_latent_dim=50`; the known deviation — ψ not trained through the actor loss here — is documented in `iql_bilinear.py:31-36` and may cost a point or two). If it lands far below, stop and debug before burning the rest of the matrix.

Reproduction command for the subject:

```bash
python -m impls.main \
  --critic iql_bilinear --actor flowbc --domain cube-single \
  --seed 0 --train_steps 500000 \
  --eval_interval 100000 --eval_episodes 50 \
  --save_interval 100000 --save_dir exp/metric-probe/ \
  --override value_latent_dim=50
```

---

## 6. Implementation plan (isolated; file-by-file)

> **Implementation status (2026-07-31, first pass — frugal):** items 1, 2 (minus torchqmet
> fixture tests), 3, 4 (registry; crl.yaml verified — sampling already matches CRL's
> contract), 5, 6, 7 (aggregator as `eval/probe_report.py`) are DONE. Probes implemented:
> P1 (+ partial on raw-obs), P3, P4, P5, P7, self-distance, RSA via probe_report; P2
> (rollout monotonicity), P8, and CKA are deferred. Deviations, all flagged in code
> comments: QRL's global term uses the bounded transform d/(d+T) (config `qrl_transform_t`)
> instead of QRL's affine-softplus damping; QRL's λ lives inside the single `value` group
> (dual ascent via a sign-flipped gradient term) instead of a separate optimizer; VIP's
> sampler draws the anchor k uniformly then o₀ ~ U[traj_start, k] (marginally close to, not
> identical to, Algorithm 1's t-then-k order); no IQE parity fixtures yet (unit-test the JAX
> IQE against torchqmet before trusting QRL numbers); `metric_probe.py` defaults
> `probe_action_dim=5` (cube) since it builds the agent env-free. Smoke-tested via SLURM
> (gh-dev) — login nodes cannot run JAX (100-thread ulimit).

New code only — no edits to existing critics beyond registry lines and, if needed, the crl.yaml sampling fix (§1.3).

1. `impls/critics/vip.py` — `VIPCritic` implementing §1.1. `PROVIDES_V = True` (`V = −‖φ(s)−φ(g)‖`), no Q (`check_combo` should reject actors needing Q; flowbc is fine). Needs a custom batch shape (§6.1).
2. `impls/critics/qrl.py` — `QRLCritic` implementing §1.2, plus `impls/utils/quasimetrics.py` with the JAX IQE head + fixture-based parity test `tests/impls/test_iqe_static.py`.
3. `impls/configs/critic/vip.yaml`, `impls/configs/critic/qrl.yaml` — every knob listed (this repo's convention: bare-indexed keys, registered in `impls/utils/config_utils.py` `REQUIRED_KEYS` so unset knobs raise `ConfigError`).
4. Registry: add `"vip"`, `"qrl"` to `CRITICS` in `impls/agents/__init__.py:27-34`; track `impls/critics/crl.py` + its yaml (currently untracked — commit them as part of this work).
5. `eval/metric_probe.py` — the probe driver. CLI: `--run_dir <glob> --epoch <int> --probe_set <path> --out <json>`. Loads a checkpoint via `impls/utils/flax_utils.py:restore_agent`, builds/loads the cached probe set (§4), computes every applicable D×P cell, writes one JSON per (run, epoch) plus optional matplotlib PDFs (P2 curves, P6 heatmap, P7 spectra). Must run CPU-only-capable for cheap post-hoc sweeps.
6. `eval/probe_set.py` — builds and caches the probe arrays from the val npz (deterministic seed).
7. `scripts/metric_probe.slurm` + a small aggregator `eval/probe_report.py` that collects JSONs into the results tables (§7).
8. Tests: `tests/impls/test_vip_critic_static.py`, `test_qrl_critic_static.py` (loss values on tiny fixtures, shapes, λ-update sign), `test_iqe_static.py` (torchqmet parity fixtures), `tests/eval/test_metric_probe.py` (probe math on hand-computable toy inputs: a 1-D chain MDP where all probes have known answers — triangle violations = 0 for true distance, σ = 0 for symmetric constructions, ρ = 1 for P1).

### 6.1 VIP batch sampling
`GCDataset` (`impls/utils/datasets.py:82-137`) doesn't produce VIP's `(o₀, o_k, o_{k+1}, o_T)` tuples. Add a `VIPDataset` subclass (same file or `impls/utils/vip_dataset.py`): sample a trajectory, sample `t < T` uniformly, set `o₀ = obs[t]`, `g = obs[T]` (trajectory's final state — for play data this is the natural "goal"; also support geometric-future goal, flag `vip_goal: final|geom`, default `final` per the paper), sample `k ~ U[t, T−1]`, emit `(o₀, o_k, o_{k+1}, g, r̃)` with `r̃ = −1 + 1[k+1 == T]`. Select via the existing `dataset_class` config key.

### 6.2 Probe hooks
Everything needed already exists: `info=True` on the bilinear heads returns `(v, phi, psi)`; `psi_of` is a callable Flax method. For QRL/VIP, expose an equivalent `embed(obs)` method on the critic so `metric_probe.py` can treat all methods uniformly through a tiny adapter table `{critic_name: (embed_fns, distance_fns)}`.

### 6.3 Numerical guards
- D1: clip `1 + (1−γ)V` to `[1e−6, 1]` before `log_γ`.
- P4: exclude pairs with `d̂(s,g) + d̂(g,s) < 1e−3·median`.
- P3 on D3 (cosine): distances in `[0,2]`, violations still well-defined; report units per construction, never mix.
- All probes in float64 on CPU (checkpoint params are float32; cast up).

---

## 7. Deliverables and decision table

**Artifacts:** probe JSONs per (run, epoch); three summary tables (mean ± std over seeds, at final epoch):
- **T1 (metric axioms):** rows = D-constructions × methods, cols = P3 rate, P3 magnitude (global / trust-region), P4 σ, self-distance `E[d̂(s,s)]`.
- **T2 (dynamics alignment):** cols = P1 ρ (overall + partialed on D10 + per-Δ-bin), P2 median ρ, P2 bump rate.
- **T3 (cross-method):** the P6 RSA method×method matrix + CKA matrix; P5 cosine-proxy stats; P7 effective ranks.
Plus P-vs-training-step curves for runs #1–2 (probe every 100k checkpoint).

**Interpretation guide (pre-registered):**

| Outcome | Reading |
|---|---|
| D2/D3 P1-ρ ≈ VIP/QRL levels, ≫ D9/D10; P6 RSA(iql_bilinear, {vip,qrl}) high | Bilinear GCIQL *is* learning a temporal-metric-like embedding — the headline positive result. |
| P5 cosine-proxy > 0.95 with concentrated norms; P4 σ ≈ 0 | H-sim: it's effectively a cosine-similarity machine (contrastive-like), not a quasimetric. |
| P4 σ clearly > D9 null; P3 trust-region violations small | H-qm: directed value geometry, closer to QRL than to cosine similarity. |
| D1 tracks temporal distance but D3/D4 don't (ψ-geometry uninformative without φ) | The *value* is good but the *embedding geometry alone* carries no metric — factorization is a capacity/optimization trick, not implicit metric learning. This is the key discriminating cell: H-sim and H-qm both predict D3-or-D2 structure; "optimization trick" predicts only D1. |
| Nothing beats D10 (raw cube-position distance) | On cube, "metric" may be trivially readable from state geometry; move the question to a domain where raw-obs distance is misleading (maze/antmaze-style; note for follow-up, not this spec). |

**Known confounds to state in the writeup:** (a) on-policy (CRL) vs in-sample-optimal (GCIQL) vs optimal (QRL) semantics — mitigated by rank correlations; (b) cube's state space may make D10 strong — that's why it's a mandatory baseline row, not a footnote; (c) GCIQL's `d_SD` is only guaranteed quasimetric-like at the optimum of a *symmetrized contrastive* objective, which IQL is not — that's precisely what P3/P4 measure rather than assume; (d) play-data goal marginal `p(g)` differs from the eval task distribution — probes use val-set states, matching training's marginal.

---

## 8. References

**Companion theory note:** `docs/notes/2026-07-31-bilinear-gciql-metric-note.tex` — derives the identities that motivate probes P3–P7 (negated value is a quasimetric; bilinear GCIQL as low-rank factorization of the reachability kernel γ^d; CRL as the log-link/behavior-policy analog; Schoenberg/polarization bridge to VIP embeddings; triangle-violation as a value-suboptimality certificate).

- VIP: arXiv:2210.00030 · github.com/facebookresearch/vip (`vip/trainer.py`, `vip/models/model_vip.py`)
- QRL: arXiv:2304.01203 · github.com/quasimetric-learning/quasimetric-rl
- IQE: arXiv:2211.15120 · github.com/quasimetric-learning/torch-quasimetric ; quasimetric learnability (distortion/violation formalism): arXiv:2206.15478 ; MRN: arXiv:2208.08133
- CRL: arXiv:2206.07568 · google-research/contrastive_rl ; Stable CRL: arXiv:2306.03346 ; TD InfoNCE: arXiv:2310.20141
- Temporal distances / self-subtraction / CMD: arXiv:2406.17098 (Myers et al., ICML 2024); TMD follow-up: arXiv:2509.20478 ; interpolation planning: arXiv:2403.04082
- Triangle-violation protocol (TTGS): arXiv:2510.07257
- OGBench + reference impls (provenance of `GCBilinearValue`): arXiv:2410.20092 · github.com/seohongpark/ogbench
- In-repo anchors: `impls/critics/iql_bilinear.py` (subject), `impls/utils/networks.py:336-452` (bilinear heads, `info=True` hooks), `impls/utils/datasets.py:82-137` (`GCDataset`), `impls/utils/flax_utils.py:130-148` (checkpoint I/O), `impls/utils/evaluation.py:37-80` (eval protocol), `scripts/vista_train.slurm` (launch). Historical result provenance: `origin/gciql-fb-bridge:PAPER/fb-gciql-bridge/untried-design-space.md` and `configs/gciql/cube_single_state_gciql_flow_bilinear_d50.yaml` on that branch.
