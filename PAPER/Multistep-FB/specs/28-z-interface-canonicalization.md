# Spec 28 — Canonicalizing the z interface: conditioning vs readout

Status: WRITTEN 2026-08-14, from an external feedback audit. House rules
as spec 27 (TDD, default-off bit-identical knobs, thin wrappers over
audited eval code + control gates against archived numbers, tests via
`.venv/bin/python -m pytest`, evals CPU jax, tmux, wandb amsks, no
co-author lines).

## 0. The verdict on the feedback (audited against the repo)

The feedback's central claim is correct and NEW: Density-FB's softmax
normalization destroys the linear identity `Q_r = Fᵀ E[rB]` that made
`z_r = E[r·B]` a valid reward→index inverse in standard FB, and the CE
loss pins only the product `FᵀB`, leaving B's Euclidean geometry gauge-
arbitrary. Our `z(g) = √d·normalize(B̄(g))` is a **primal** feature
vector, while the actor readout `r_z(u) = B̃(u)ᵀz/√d` needs a **dual**
coefficient `c = (Σ_B+λI)⁻¹m` — these coincide only under the
whitened/orthonormal geometry FB's ortho loss enforces and we (rightly)
dropped. This mechanism EXPLAINS spec 27's sharpest result (goal→z
retrieval works, reward→index assignment at chance .35/.47 vs .50)
instead of restating it.

Also correct: spec 24 changed conditioning AND actor reward readout
simultaneously (g+κ → z+Bᵀz), so the .736→.017 collapse is not yet
attributable to z-conditioning per se. The decisive missing experiment
is the 2×2 below — never specced, never run (audited).

Audit corrections to the feedback (bake into implementation):
- The g-path "exact kernel" is `_delta_phi(goals, u_raw)` — **query-set
  softmax-normalized**, rows sum to 1 over the M queries. Any new cell
  must use the same kg for parity, not raw κ.
- "No covariance correction anywhere" overlooks C2: `_density_rdecode_loss`
  already computes the ridge object `v* = (BᵀB+λI)⁻¹Bᵀr_c` — but only as
  a training shaping loss (solo arm flat: .108/ztv .009). The P0 probe
  already showed B is ridge-decodable (dense R² .68–.70 z-B vs .466 RBF
  control) ⇒ the geometry exists; the **interface** never uses it.
- ‖z‖=√d everywhere z enters F/actor. A ridge coefficient is NOT norm-
  preserving ⇒ on existing checkpoints it may be tested **readout-only**;
  conditioning on it is OOD and needs retraining.
- §7 (calibration ≠ extractability), §1.2 (dwell caution) restate our own
  DENSITY-FB-EXPLAINED §6.5 / spec 26 findings — no new action from them.

Binding constraints carried forward: general reward interface required
(goal-conditioned-only NOT acceptable); NO orthonormality loss on
training (convicted toxin: first-passage .967 no-ortho vs .225 with;
spec 25 refused it independently); no further tilt-strength sweeps (C1
deleted its own tilt at β=1.0).

Anchors: cs g-density .93 / BC .111; cd g .736 / BC .019 / z24 .017;
pmm g .315 / GCIQL .242 / BC .235. Stage-1 z25 (cs 500k): C1 .096/ztv
.054, C2 .108/.009, C3 .108/.602, full .108/.588 — ztv revived, success
never off the BC floor.

---

## P0 — Eval-only probes on existing checkpoints (no training; do first)

Checkpoints: z24 `~/exp_spec24/factored-fb/{CubeDouble,PointMassMaze-RND}
-…DensityZ-d50`; z25 stage-1 `~/exp_spec25_stage1/factored-fb/{C1Tilt,
C2Dec,C3Actor,Full}` (cs 500k); g anchors `~/exp_spec18/.../d18_cs_dens_s0`
+ the cd controls referenced by spec27 g4 JSONs. Reuse the pool/build
machinery of `eval_density_zeroshot.py`; P=16384 pools; every new probe
passes a control gate on the g-indexed anchor before any z cell is read
(the three-times-burned rule).

### P0a — Action-ranking probe (the decisive decomposition)

For ~64 fixed (s, g) pairs per checkpoint (states sampled from dataset
episodes that reach g, plus random states): sample K≈64 actions (dataset
actions at nearby states + flow-actor samples + uniform). Compute both
readouts under the SAME z-conditioned weights w_j(s,a,z(g)):
- q_g(s,a) = Σ_j w_j(s,a,z)·kg_j          (exact normalized kernel)
- q_z(s,a) = Σ_j w_j(s,a,z)·B̃(u_j)ᵀz/√d   (current latent reward)

Log per (s,g): Spearman_a(q_g, q_z); action-value span of each (max−min
and IQR); hit-region AUC of each against dataset "good" actions;
Δ_actor = max_k q_·(s,a_k) − q_·(s, π(s,z)); and actor sensitivity
E‖π(s,z)−π(s,z′)‖ over resampled z′ (the ztv metric measures the
MEASURE's sensitivity, never the actor's — this closes that gap).

Readout table:
- Spearman low + q_g span healthy ⇒ **readout indicted** (r_z wrong shape
  over the occupancy support).
- Spearman high + Δ_actor large ⇒ **actor extractor indicted**.
- q_g span ≈ 0 under z-conditioning ⇒ **conditioning indicted** — predicts
  2×2 cell (ii) fails; check before spending training compute on it.

### P0b — Three-encoding reward reconstruction

For each goal in the eval task set, three candidate indices:
- point: existing `_density_z_encode` output;
- moment: e2a's `analytic_zr` (exists), plus its UN-normalized variant;
- ridge: `(Σ_B+λI)⁻¹ m_g`, Σ_B and m_g = E[B·kg] from the P=16384 pool,
  λ swept {1e-3, 1e-2, 1e-1}·tr(Σ)/d (~20 new lines).

For each, reconstruct r̂(u)=B(u)ᵀz over the pool and score against kg:
hit/non-hit AUC, Spearman, top-k precision, hit-vs-hard-negative margin.
Gate G28-1: ridge ≫ point/moment on AUC+Spearman ⇒ the moment-vs-
coefficient confusion is real and repairable in the B basis (licenses
R-A). Ridge/point comparable and both bad ⇒ B's kernel geometry itself
is inadequate for reward shape (retrieval ≠ shape) ⇒ R-B.

### P0c — e2b rerun with ridge index (readout-only)

Rerun the spec 27 e2b assignment protocol (the coordinate-percentile
metric that discriminated — NOT policy success, dead-actor trap ×3) with
z_r^ridge in place of the moment. Question: does argmax percentile move
off chance (.35/.47) toward the g-control? A jump ⇒ reward→index is
recoverable by dual correction alone. Mandatory: bit-exact control gate
(archived e2b JSON reproduced) before reading the new cell.

---

## P1 — The 2×2: conditioning × actor-reward (retraining, gated on P0a)

Skip iff P0a returns "conditioning indicted" (q_g span ≈ 0 under z).
Otherwise: cube-single 500k s0 screen, spec-24-matched config
(z_rand_frac .5 unchanged — single-variable deltas only); promote
whichever cells discriminate to cube-double + seed s1.

| cell | measure/actor condition | actor reward | status |
|---|---|---|---|
| i   | g    | kg (exact)      | exists: cs .93 / cd .736 |
| ii  | z(g) | **kg (exact)**  | NEW — the decisive cell |
| iii | g    | **B̃ᵀz(g)/√d**   | NEW — readout validity alone |
| iv  | z(g) | B̃ᵀz(g)/√d      | exists: cs .111 / cd .017 |

Implementation (per cell ~30–50 lines + lattice + tests):
- Cell ii: knob `density_z_actor_reward ∈ {latent (default), kernel}`.
  `kernel` swaps rz24 in `_density_actor_q_z` for kg=`_delta_phi(goals,
  u_raw)`; F/actor conditioning stays z-only (g touches ONLY the frozen
  reward vector). Random-sphere rows have no coherent g: mask+renormalize
  the q term with the existing `z_hindsight` indicator, exactly as the BC
  term does (`_density_z_masked`, gciql_flow.py:1358–1371).
- Cell iii: symmetric knob on the g-path actor: reward = B̃(u)ᵀz(g)/√d
  with z(g) computed but NOT fed to F/actor.
- Default-off bit-identical: with knobs at default, training step outputs
  match current agent bit-exactly (standing test pattern).

Interpretation (pre-registered):
- ii succeeds (≫BC, ideally →cell-i level) ⇒ z is control-sufficient;
  failure = reward semantics of Bᵀz ⇒ repair R-A.
- ii fails while i succeeds ⇒ latent conditioning itself insufficient for
  control despite retrieval ⇒ repair R-B.
- iii fails ⇒ independent conviction of the Bᵀz readout (corroborates
  P0a/P0b regardless of ii).
- Optional tie-break cell ii′ (only if ii is ambiguous): ii with
  z_rand_frac=0 — removes the dilution amplifier.

---

## P2 — Repair fork (gated; pick ONE branch from P0/P1 evidence)

### R-A — Dual-corrected general reward interface (readout indicted)

Reward specified as samples {(u_j, r_j)} (FB's own interface — general
rewards preserved, hindsight goals give samples for free via kg):
- index: c_r = (Σ_B+λI)⁻¹ · (1/M)Σ_j B(u_j)r_j; condition F/actor on
  ẑ = √d·normalize(c_r) (norm restored for conditioning-distribution
  match — direction carries the dual correction);
- actor readout: integrate the ACTUAL reward samples, q = Σ_j w_j(s,a,ẑ)
  ·r(u_j) — Bᵀz never appears as a reward again. The latent is only a
  policy index; reward semantics stay external, which is exactly what
  made the g-path work.
- Retrain with this encoding in the task mixture (hindsight rows use
  c_{kg}); keep C3's hindsight-BC machinery. No ortho loss.

### R-B — Reward-embedding actor (conditioning indicted)

The standing spec24/27 fork, now licensed: FRE-style set encoder
{(u_j, r_j)} → e_r; π(a|s,e_r) trained on Density-FB's direct reward
integral. Task prior replaces 50% uniform spheres: hindsight goal
kernels + dense-distance + two-cube mixtures/conjunctions + behaviorally
grounded subtrajectory tasks.

### Parked (do NOT start without new evidence)
- Search/GPI over z (feedback Path C): requires a diverse π_z family;
  stage-1 shows all arms at the BC floor — nothing to search over yet.
- Linear value head Hᵀz (Path D): heaviest, re-adds FB algebra wholesale;
  only if both R-A and R-B fail.
- Ortho loss, tilt sweeps: convicted, see §0.

## Bookkeeping
- New groups → `scripts/results_map.py` + `compile_results_csvs.py
  --refresh` (standing debt already large — spec 27 Q5).
- DENSITY-FB-EXPLAINED.md: add the moment-vs-coefficient mechanism to
  §6.5 once P0b/P0c read out; density-fb.tex touches only after cells
  are confirmed (mid-run numbers never quoted).
