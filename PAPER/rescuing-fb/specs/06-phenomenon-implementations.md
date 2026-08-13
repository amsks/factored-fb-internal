# Spec 06 — implementations for `phenomenon.tex` (rescuing-fb)

*2026-07-20. Pre-registered BEFORE implementation, per program discipline. Companion to
`../phenomenon.tex` §7–§8. Status of the claim boundary at writing: T10-a 0.823 / T10-b2
0.06–0.10 / T7 0.240–0.227 / T8 0.067 — all dead; strong negative claim licensed. Direction I
items 1–4 are the paper's method; Direction II items 5–7 are optional mechanism cells.*

Environment invariants (do not deviate): run in tmux; wandb entity `amsks`; jax venv
`/dev/shm/.venv-jax` (volatile — recreate if wiped), repo `.venv` for torch + tests;
`XLA_PYTHON_CLIENT_PREALLOCATE=false`; datasets under `/dev/shm/factored-fb/datasets` get
WIPED — restore from `~/bridge_ckpt_backup/datasets/`. After every campaign: add the group to
`scripts/results_map.py`, run `.venv/bin/python scripts/compile_results_csvs.py --refresh`.

---

## 1. GPI over goal candidates at inference (Direction I, priority 1 — CPU only)

**What.** New eval script `scripts/rescuing_fb/t6_gpi_eval.py` (fork of
`t6_generality_probe.py`): for each composite task `r = max_k r_{g_k}`, build candidate set
`Z = {z_r, ψ(g_1), …, ψ(g_K)}`; per env step, batch the K+1 policy calls `a_j = π(s, z_j)`
and one critic pass; execute `a_{j*}, j* = argmax_j min_heads Q(s, a_j, z_r)`.

- Checkpoints: RG-FB v0 @500k (`~/exp_rgfb/.../rgfb_v0_s*`) AND G4 d512 (both, so the read
  is not d50-specific).
- Protocol: T6's two-goal max-mixture cells (25 eps/cell, H=200, batched lockstep envs,
  cores ≤48). Report per-pair, plus the T6 anchors re-run through the GPI path as a
  no-regression gate (single-goal candidates ⇒ GPI must reproduce ~0.9).
- **Pre-registered reads:** two-goal max-mixture ≥0.60 = GPI fixes actor coverage (predict
  ~0.8+); 0.30–0.60 = partial (report per-pair split); ≤0.30 = actor coverage is NOT the
  mixture failure → re-open E1a ladder before any training change. Also log
  `argmax`-switch rate per episode (sanity: should be state-dependent, not constant).

## 2. RG-FB d512 + z-actor rerun (Direction I, priority 2)

Config fork of `configs/gciql/cube_single_state_rgfb_d50.yaml` → `..._rgfb_d512.yaml` with
`latent_dim: 512` only. 3 seeds, 500k (1M if the curve is still climbing at 500k), group
`RGFB-Cube-Single-State-d512-v0`. Eval via `scripts/rescuing_fb/eval_rgfb_zeroshot.py`
(match_data norm rule). **Reads:** zero-shot ≥0.88 = the d-gap explained the v0 shortfall;
0.83–0.88 = partial; ≤0.825 = d is not the gap → promote selection extraction (item 3).
Anchors: G4 exact-z d512 0.921; v0 zero-shot 0.825.

## 3. Selection-based extraction on RG-FB (Direction I, priority 3 — hours)

Fork the rerank harness (`scripts/` rerank protocol from the bilinear program) to consume
RG-FB checkpoints: N=32 proposals from the trained BC flow (goal-agnostic vf or the
z-conditioned vf — report both), score with the frozen min-head critic at the *inferred* z,
execute argmax. No training. **Reads:** ≥0.90 = deployment-mode RG-FB matches supervised
anchor (paper headline "zero-shot 0.90 via selection"); the N-sweep must not decline through
N=64 (off-support-peak check on the z-critic — if it declines, cap N and report). Optional
distill pass afterwards (IDQL-style) if the paper wants a parametric actor.

## 4. RG-FB v1 — task-prior scalar grounding (Direction I, priority 4, ~2 days impl)

New knobs on `tools/wandb_mode_shim/gciql_flow.py` (all default-off ⇒ v0/anchors bit-identical;
follow the `actor_vf_cond` pattern: param-tree-preserving, assert-lattice, tests first):

- `task_prior ∈ {'dirac' (default), 'mix'}`, `task_prior_probs = [p1, p2, p3]`
  (start 0.6/0.2/0.2).
- Mixture arm (p2): K∈{2,3} goals from the existing hindsight mix, hindsight-selected near
  batch next-states (reuse the relabel machinery — this is the sparse-in-batch mitigation);
  weights w ~ Dirichlet(1); `r_i = max_k 1[‖s'_i − g_k‖ ≤ ε]` with ε = env success radius on
  cube-xyz dims; `z = Σ w_k ψ(g_k)` renormalized by the match-data rule.
- Random-linear arm (p3): `z̃` uniform on sphere × mean‖ψ‖ (batch statistic, sg);
  `r_i = sg[ψ(s'_i)]·z̃/√d`.
- Loss: per-task-type normalization of the critic loss by `sg E|target|` (mirrors the actor's
  q normalization) so p3's dense rewards can't dominate; expectile V and Bellman Q **at the
  sampled z**, mask per the reward's own semantics (ball rewards: mask 0 inside the ball for
  mixture/Dirac arms; no mask for linear-reward arm — it has no absorbing set; document
  this asymmetry in the run notes).
- Actor: flow + distillation conditioned on sg[z] of the sampled task (all three arms).
- **Gradient discipline (CORRECTED 2026-07-20 — the earlier wording "no grad into ψ through
  z" was wrong and caused a bug):** the critic/value TD **must train ψ through z** for the
  Dirac and mixture arms (z = ψ(g) resp. Σw_kψ(g_k) are model outputs — this is v0's L_Q
  training both legs; with z fully sg'd, ψ never trains and the agent is the frozen-random-ψ
  row, 0.061). sg applies ONLY to: (i) the actor's copy of z (v0 parity), (ii) the p3 arm's
  reward features sg[ψ(s')] (else the reward is gameable), (iii) the mixture renorm SCALE
  factor `sg[mean_norm/‖Σwψ‖]` (keep direction gradients). The Dirac arm's reward
  convention is gc_negative (r∈{−1,0}); the mixture arm MUST match it (r = hit − 1,
  mask = 1 − hit), else near-identical z's get conflicting {0,1}-vs-{−1,0} targets. The
  actor's Dirac-arm task must come from `actor_goals` (traj-future only, v0's D8 split),
  not `value_goals`.
- Tests before launch (`tests/test_gciql_rgfb_v1.py`): (a) `task_prior='dirac'` bitwise ≡ v0
  update on a fixed batch; (b) mixture arm: z renorm correct, reward computed on relabelled
  batch, ψ RECEIVES gradient from the vc loss through z (arms 0/1) while the actor's z and
  p3's reward features are sg (verified via jax.grad); (c) p3 arm: loss scale within 2× of
  Dirac arm on a fixed batch; (d) assert-lattice refusals.
- Launch: 3 seeds, d512 if item 2 won else d50, group `RGFB-Cube-Single-State-<d>-v1-taskprior`.
- **Pre-registered reads (binding):** Dirac anchor ≥0.80 (non-regression vs v0 0.825);
  two-goal max-mixture (T6 protocol, *without* GPI) ≥0.60 success / 0.30–0.60 partial /
  ≤0.30 dead; region-goals ≥0.672 (non-regression). If GPI (item 1) already fixed mixtures,
  v1's p2 arm is optional — run v1 with p2=0 and claim the linear-reward slice only.

## 5. First-passage FB (Direction II, axis C — one masking line + 3 seeds)

In the jax measure cell (`gciql_flow.py` measure path — the same code T10 touched), new knob
`measure_first_passage=true`: `hit_ij = 1[‖s'_i − s⁺_j‖ ≤ ε]` on cube-xyz coords (ε=4cm =
0.4 obs units — REMEMBER decimeter scaling, verify against `cube_env.compute_observation`
like T1 did); `δ_ij = M_ij − hit_ij − γ(1−hit_ij)·sg[M̄_ij]`. **Diagonal discipline
(review finding 3): with `hit` supplied the diagonal is a SQUARED residual like every
other entry — `hit` is a representable 0/1 reward, so FB's linear diagonal attraction
(the Dirac-density device) must NOT be kept, else the diagonal pull is unbounded and the
fixed point stops being E[γ^T].** Everything else = the plain
measure-only cell (NOT T10's centering — this must be a single-axis move). 3 seeds, 500k,
group `GCIQL-State-Flow-Measure-FirstPassage-d50`.
**Reads:** ≥0.30 = semantics is a toxic axis (≫ the 0.08–0.10 measure ceiling); 0.10–0.30 =
partial; ≈0.10 = axis C exonerated, reframe the 82% misalignment number as correlate.

## 6. Expectile-measure FB (Direction II, axis B middle — ~1 day impl)

Knob `measure_expectile=true` on the same measure path: add `V_M(s,s⁺) = φ_V(s)·ψ̃(s⁺)/√d`
(new φ_V tower, ψ̃ shared); losses per column: expectile `ℓ²_τ(sg[M(s,a_data,s⁺)] − V_M)`
(τ=0.9) and backup `M(s,a,s⁺) ← r_δ + γ·sg[V_M(s',s⁺)]` with NO a′ anywhere. Composes with
item 5 (`measure_first_passage` supplies r_δ = hit and the (1−hit) mask). Scale question
answered pre-launch (per phenomenon §6 rule): with first-passage masking the fixed point is
bounded in [0,1] per column — run the composed cell by default; the unmasked variant needs
FB's sphere/ortho anchors (T10-b2 settings). 3 seeds, 500k,
group `GCIQL-State-Flow-Measure-Expectile[-FP]-d50`.
**Reads:** lifts ≫0.10 with T8 at 0.067 ⇒ "in-sample improvement is the missing bootstrap";
floors ⇒ axis B fully exonerated (both corners + middle dead).

## 7. Hindsight-weighted test points (Direction II, axis A — ~1 day impl)

Knob `measure_testpoint_dist='hindsight'`: off-diagonal test points s⁺ from the scalar
side's mix (0.5 geometric-future same-trajectory / 0.3 random / 0.2 current) instead of
batch next-states; **the diagonal (positive/attraction) term must come from
TRAJECTORY-FUTURE states ONLY — implemented as ψ(actor_goals), which are uniform-future on
the same trajectory (review finding 5: value_goals include 30% random states, which are
samples from no policy's future; attracting mass to them corrupts the object). With
first-passage composed, hit supplies the positives and no special diagonal is needed.
Unmasked expectile + hindsight is REFUSED by the assert lattice (no coherent reward).**
Print the caveat: the object is no longer the ρ-referenced measure (no importance
correction — intentional).
3 seeds, 500k, group `GCIQL-State-Flow-Measure-HindsightTP-d50`.
**Reads:** movement ≫0.10 validates the weighting half of the misalignment mechanism at the
loss level; null = weighting alone is not the toxin (the temporal-semantics half stands).

## 8. Stop-grad measure readout head (paper support — zero control risk by construction)

*Rationale (phenomenon §7.5 / Remark on the first-passage kernel): every measured cost of
"measure + scalar" came through shared parameters (E2 stop-grad localization, T10-a −8pts,
metamotivo aux-Q null). A measure head with NO gradient path into the control towers keeps
the occupancy extractables (density queries, OPE, γ-model-style reasoning) and produces the
honest fidelity number, at provably zero cost to control.*

**Phase 1 — post-hoc script (run this; no agent surgery).**
`scripts/rescuing_fb/measure_readout_head.py`, jax CPU-or-1-GPU:

- **Inputs**: any frozen RG-FB / bilinear-GCIQL run dir (arch knobs from the run's
  `flags.json`, restore pattern = `eval_rgfb_zeroshot.py`). Comparison arm: a plain-FB torch
  checkpoint (its native F·B *is* its measure readout — evaluate directly, no refit).
- **Feature dump (once)**: for the 1M-transition dataset, φ-features = concat of the twin
  towers `[φ_1(s,a); φ_2(s,a)]` (d=100 input), ψ-features = ψ(s') (goals are state-shaped,
  so ψ(s⁺) is well-defined). sg is enforced structurally — towers never enter the graph.
- **Head**: `M̂(s,a,s⁺) = φ_M(feat_φ)ᵀ ψ_M(feat_ψ)/√d_M`, φ_M/ψ_M = 2-layer 256 GELU MLPs,
  d_M=50.
- **Training signal** (flag `--estimator`):
  - `mc` (DEFAULT — cleanest grounding, no bootstrap): within-trajectory discounted
    visitation regression: positives (s_t,a_t,s_{t+k}) with weight γ^k against random
    test-point negatives (density-ratio/CRL-style form). No TD, no scale question.
  - `td_sarsa` (the FB-comparable estimator): batch measure TD with dataset-action
    bootstrap + the T10-b2 anchors (sphere-projected ψ_M, ortho λ=1, RAW base leg) — the
    anchors are REQUIRED here because there is no scalar term to pin scale (phenomenon §6
    rule).
- **Fidelity eval** (the deliverable): held-out episodes (last 100, excluded from head
  training); empirical discounted-occupancy targets on a (s,a)×s⁺ grid; report Spearman
  rank corr + calibration slope, **broken down by horizon bucket K≤5 / 5–20 / >20** (makes
  the misalignment bound visible in the calibration plot — the paper figure).
- **Extractable demos** (one panel each): (i) OPE of a random dense reward
  `r(s')=sg[ψ(s')]·z̃/√d` vs MC ground truth on held-out episodes; (ii) temporal distance
  `T̂ = log_γ(1+(1−γ)Q)` from the control head directly (first-passage identity — needs no
  measure head) vs realized hitting times on eval rollouts.
- **Pre-registered reads**: control untouched (structural — nothing is retrained; assert the
  checkpoint hash unchanged). Fidelity(RG-FB-feature head) ≥ fidelity(plain FB native F·B)
  ⇒ print "scalar grounding is measure-free but not measure-blind"; < ⇒ print the honest
  trade (the head is kept either way for the extractables). No success-rate read exists for
  this item by design.

**Phase 2 (optional, only if live logging wanted)**: co-training knob
`measure_readout='sg'` in `gciql_flow.py`, default-off, param-tree-preserving; test must
verify zero gradient into φ/ψ/vf via the jax.grad pattern from `tests/test_gciql_rgfb.py`.
Do NOT implement Phase 2 before Phase 1 shows the head is worth logging.

---

## PROGRAM RE-CENTER (2026-07-20, user decision — supersedes the ordering below)

**The success criterion for rescuing-fb is a SUCCESSOR MEASURE that works on cube. Nothing
else counts.** GPI (item 1, observed to help), RG-FB (items 2–4), and selection extraction
are demoted to *control anchors and paper baselines* — they are not the goal and no further
effort goes to them beyond letting the in-flight runs finish. Multi-goal GPI is explicitly
out of scope for this direction.

"Measure-native" (binding definition, so success can't be gamed): (i) the trained object is
a matrix M(s,a,s⁺) over a test-point field — not a per-task scalar regression; (ii) the
actor consumes only readouts of M; (iii) task entry stays zero-shot via the z/B readout.
First-passage variants QUALIFY (first-occupancy is a successor measure; Moskovitz et al.);
a different discount qualifies trivially. RG-FB does not.

**The measure ladder (run in this order):**
- **Rung 1 (RUNNING, tmux `s06gpu`)**: the four single/double-axis cells — FirstPassage /
  Expectile-FP / Expectile / HindsightTP, 3 seeds × 500k.
- **Rung 2 (RUNNING, tmux `s06corner`)**: the FULL-CORNER cell
  (`cube_single_state_measure_gciqlcorner_d50.yaml`) — all three axes at once
  (fp + expectile + hindsight). The scalar side needed all three together; this is the
  maximal measure candidate and the headline arm.
- **Rung 2W (RUNNING, tmux `s06cellw`) — CELL W, the theorem-backed win.** The existence
  theorem (2026-07-21): under the 0/−1+mask convention, `E[γ^T_hit] = 1+(1−γ)Q`, and
  substituting `M = 1+(1−γ)Q` into the first-passage measure residual gives **identically**
  `δ_M = (1−γ)·δ_Q^GCIQL` — the first-passage successor measure with hindsight-PAIRED
  columns and in-sample expectile backup IS bilinear GC-IQL (0.904) up to affine
  reparameterization. Verified numerically in
  `test_gciql_measure_axes.py::test_cellw_knobs_and_affine_identity`. Two knobs turn the
  corner cell into Cell W: `measure_offdiag_weight` (the corner's uniform B×B matrix makes
  the per-row task weighting ~99.6% UNPAIRED-random — axis A re-entering through the
  estimator; w=0 = diagonal/paired-only = the exact GC-IQL-equivalent; w≈0.3 = GC-IQL-like
  effective mix with genuine off-diagonal matrix supervision) and
  `measure_expectile_source='target_min'` (the anchor's double-DQN split). Arms:
  `cellw0` (w=0, guaranteed by the identity — the existence proof) and `cellw03` (w=0.3,
  the measure-native win candidate; launch when GPUs free). Configs
  `cube_single_state_measure_cellw{0,03}_d50.yaml`.
  **The claim this licenses regardless of every other arm:** *a successor measure that
  works on cube exists* — the first-passage measure at task-shaped weighting — and the
  paper's dial-walk (w: 0 → 1; hindsight → ρ columns; absorption on → off; γ split) charts
  exactly where measure-ness breaks between it and FB.
- **Occupancy completion (spec, run after a Cell W win): the renewal decomposition.**
  `M_occ(s,a,B) ≈ M_fp(s,a,B) · R(B)` — occupancy factorizes into the first-passage kernel
  times an (s,a)-INDEPENDENT recurrence factor R(B) = expected discounted revisit mass
  from the ball under the policy (exact under renewal at the ball; on play data R is large
  and action-free because μ re-grasps). Fit R per test point as a cheap readout on data →
  the agent then carries the FULL occupancy measure as a product, and the identity explains
  the entire failure record in one line: the joint estimator spends its capacity on R
  (which owns the mass) while control lives only in M_fp.
- **Rung 3 (code next if rungs 1–2 land ≤0.5): short-γ two-timescale measure.** A second
  measure head at γ_short≈0.8 (mean lookahead ~5 steps — inside the K≤5 gate window where
  the misalignment bound says the action information lives), actor ranks on a blend
  `Q = (1−β)·M_γ.99·z + β·M_γ.8·z`. Strictly occupancy-semantics (a bona fide successor
  measure at a different discount) — the lever for the strict-measure reading. Knobs:
  `measure_gamma_short`, `measure_short_blend`; composes with rung-1/2 knobs.
- **Rung 4 (with rung 3): mode-indexed forward map.** Condition φ on the observed contact
  channel obs[18] (two-headed φ switched by it). fbz precedent: indexing is how this
  architecture resolves conditional structure; fbz also says fitting alone is insufficient
  — so run only combined with rung 2 or 3 semantics, never alone.
- **Rung 5 (claim-grade, after any positive read): torch FB-FP port.** Port the winning
  semantics into `agents/fb/agent.py` proper (z-family training intact) so the claim is
  about FB, not a GCIQL graft.

**Success bars (pre-registered):** ≥0.85 = a successor measure works on cube (the headline);
0.5–0.85 = measure-native rescue real, iterate; 0.2–0.5 = partial (name the axis); ≤0.2 =
rung dead, escalate to the next rung. Anchors: measure ceiling 0.10, plain FB 0.42 @1M,
bilinear GC-IQL 0.904.

## Ordering & stopping rule (superseded by the re-center above; kept for the anchors)

1 → 2 → 3 → 4 (method chain; each gates the next's config), then 5 → 6 → 7 (mechanism cells,
only if the paper wants "which axis is toxic"). Item 8 Phase 1 is independent of the chain
(CPU-friendly, frozen checkpoints) — run it anytime; it is paper-support, not a gate. Stop
condition for the method chain: zero-shot ≥0.90 via any of 2/3/4 ⇒ freeze and write. Every
group lands in `results_map.py` + CSV refresh + a DECISIONS.md entry with the pre-registered
read applied verbatim.

## Do-not-build (carried from phenomenon §7.4, updated 2026-07-20 post-discussion)

- Centering/dueling M splits; double-leg scale norm; z-zeroed one-step; eval-time goal
  smoothing; unbounded ψ(s,g); task-norm reweighting; zHER; decouple-actor-z; whitened
  inference.
- **Measure-aux at any λ on SHARED parameters** (T10-a −8; E2 free-at-best). The sg readout
  head (item 8) is the only permitted form of a trained measure.
- **Metamotivo aux-Q summed with the measure** (`q_loss_coef>0`): already run —
  `fb_state_iql`, 10 seeds @1M = 0.468 ± 0.088 ≈ plain FB (0.42–0.48). Null; it moves no
  estimator axis (self-referential implicit reward B(E[BBᵀ])⁻¹z, learned-π bootstrap, no
  mask). Do not re-run under a new name.
- **Warm-start / curriculum measure ramps from ANY good checkpoint, including RG-FB**
  (anneal closure: sg-φ ramp 84.9 = neutralized-not-helpful; shared-legs ramp 0.9; iql-fade
  5.3; T10-b2 transient-then-decay). Curricula cross cliffs, not fixed points — the endpoint
  objective owns the endpoint.
- **Any conservatism/pessimism arm on measure cells** (T7: 0.42→0.240/0.227, dose-monotone
  harm).
