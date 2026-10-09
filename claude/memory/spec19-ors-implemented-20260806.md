---
name: spec19-ors-implemented-20260806
description: "2026-08-06/07 spec19 ORS IMPLEMENTED+COMMITTED (c5348cf, bf70d20, 8b33c52) and RUNNING: 19a behavior-density (cs done, cd training) + P1.2 gate PASSED on cs (Spearman .845) + cube-single ORS extractor launched 3 seeds. ⚠ the G4 'both measures fail calibration' null was an ARTIFACT (dropped half of a cancelling de-biasing pair) — corrected, all 6 ckpts PASS"
metadata: 
  node_type: memory
  type: project
  originSessionId: 576b53a0-d404-4a89-9380-6fece089b102
  modified: 2026-08-07T06:19:37.513Z
---

Bridge / measure track. Specs `PAPER/Multistep-FB/specs/19-ors-program.md` + 19a/19b
(+20 dwell, NOT started). Follows [[spec18-density-measure-20260806]].

**Program shape (spec 19 §1).** P0 debts → P1.1 behavior density → **P1.2 hard gate**
→ P1.3 extractors → P2/P3 dwell → P4 gated. The gate ordering is load-bearing: never
launch extractors on an uncalibrated shaping signal.

**19a `measure_density_behavior`** (`c5348cf`): spec-18 density with (i) NO goal
conditioning — F tower sees `[s,a]`, and (ii) SARSA bootstrap on the dataset's next
action, not the learned (goal-conditioned BC) actor. Actor is pure BC; `actor_q_center`
is REFUSED rather than left inert. GCDataset gained a gated `next_actions`.
- ⚠ **The forcing subtlety**: `create()` shadows its arg with `dict(config)`, so mutating
  only the local copy never reaches GCDataset — which in the real entrypoint holds a
  live ref to the CALLER's config and re-reads the flag PER SAMPLE (`datasets.py:225`;
  `main.py` :64 dataset → :78 create → :96 sample). Both objects are written now.
- Verified END-TO-END, not by unit test: a 300-step real run logs
  `density_sarsa_subst = 0.00195` = 2 rows / 1024 batch ≈ 1/traj_len. Without forcing
  the key is simply ABSENT. **This is the template for "prove the knob is load-bearing".**

**P1.2 gate PASSED on cube-single: Spearman 0.845** (n=50k, threshold .65); action-axis
.840 @ .1445 coverage. This SETTLED a concern worth remembering: the cs behavior density
looks diffuse (CE 6.53 vs log M 6.93, ess_median ~468/1024) — but **entropy of w is the
wrong instrument**. High entropy ≠ constant `r_W`; what matters is whether w varies
across (s,a), which only the gate measures. cd is much sharper (k_self .49 vs .057 — a
6-D conjunctive space separates points more at the same ε).
- Rigor note: I first gated at pool 16384 while the config trains at 4096 — i.e. would
  have validated a different signal than shipped. Re-ran at 4096: .8450 vs .8448 ⇒
  `r_W`'s ORDERING is pool-size-insensitive (unlike spec-18's eval readout, whose
  absolute Q needed P≥16384). **Always gate at the size you ship.**

**⚠ The G4 near-miss (the big lesson).** A new probe reported ~.13–.18 for BOTH density
and partition — "our measures order occupancy no better than chance" — which would have
redirected the whole program per spec 19 §3. Adversarial verification found an ARTIFACT:
the readout zeroed `r(x₀)` but kept its paired `−κ(x₀)`; spec 18 §1 defines those as a
de-biasing pair that CANCELS to `wᵀr/(1−γ)` = exactly the MC object, so dropping half
injected `−100·1[x₀∈box]`, anti-correlated with truth, damage growing with box size.
Corrected, all 6 PASS (tie-corrected .767–.790). **The decisive move was a CONTROL**:
feed the probe the MC estimate itself (must score ≈1.0) and noise (≈0) — that localized
the bug to the model side in minutes. Both controls are now permanent tests (`bf70d20`).
Honest per-multiplier read to quote alongside any pooled number: mult 1.0 → .41–.48,
2.0 → .59–.64, 4.0 → .83–.85 ⇒ coarse-scale ordering strong, FINE scale (native ε=0.4,
the ORS-relevant one) only moderate; pooling inflates because box size drives both vars.

**19b extractor** (`8b33c52`, built + tested, cs launched): plain g-conditioned GCIQL
whose TD reward is `r_W = scale·(1−γ)·Σ_j w_j·(−d²(u_j,g))` off frozen (F_D,B_D).
Transcription findings that OVERRIDE our spec defaults (paper wins, PSM precedent):
- `r_W` REPLACES `batch['rewards']` except success rows forced to 1.0 (`masks==0` is
  exactly the success indicator here: `datasets.py:262` sets `masks = 1 − successes`).
- Reward scale is PER TASK (Table 10: cd 0.25, scene 0.75) while their released code
  hardcodes one 0.75×. Configs: cd 4.0, scene 1.333 (both traced), cs 4.0 **INFERRED**
  (no Table 10 row) and labeled as such.
- ⚠ Their optimality-preservation Theorem 1 needs 4 assumptions (determinism, one-step
  reachability, strict potential monotonicity, layer-monotone behavior policy) that the
  paper never audits and cube contact dynamics violate — **do not claim it transfers**.
- Pool/embeddings are static nonpytree fields; pre-ORS restore explicitly round-tripped
  (the 17b failure mode: a new pytree leaf breaks `from_state_dict` for every existing
  checkpoint and only surfaces at collection).

**Launch gotcha:** hydra `+agent_overrides.ors_reward_ckpt=` FAILS when the key already
exists as a config placeholder — use `++`. Three launches exited 1 in <1s and my log
grep for error strings printed a reassuring "no errors". **Confirm launches by success
markers and exit status, never by absence of error strings.**

**RESULTS (cube-single, 500k, 3 seeds each).** P0.2 BC-only control = **0.111** — that
is what `bc_coeff=3.0` alone buys, and it makes the whole program's cs numbers
attributable for the first time:

| arm | cs | over cloning floor |
|---|---|---|
| BC-only (cloning alone) | .111 | — |
| delta (17a) | .207 | +.10 |
| **ORS (19b)** | **.739** | **+.63** |
| partition K256 (17b) | .884 | +.77 |
| density (18) | .929 | +.82 |
| fp-noortho (anchor) | .967 | +.86 |

⇒ The cs ORS result is a LOSS against the P1.3 gate (±.05 of density's .929) but is NOT
a failure: shaping clears cloning by +.63. Honest phrasing: "shaping works; reading the
measure directly works better on cs." ⚠ Confound specific to that arm: cs
`ors_reward_scale = 4.0` is **INFERRED** (no Table 10 row) while cd/scene are
transcribed — so the arm that lost is the one built on a guessed constant. Scale
ablation (1.0, 1.333, seed 0) launched to decide "wrong constant" vs "extraction was
never binding on cs".

**P1.2 gates: cs .845, cd .970** — both PASS. cd is better calibrated, consistent with
its sharper density (`k_self` .49 vs .057).

**State:** cs + cd behavior-density DONE; cs ORS ×3 DONE (.739); BC-only ×3 DONE (.111);
**cd ORS ×3 RUNNING** (the headline arm, verified scale 4.0, anchor .549); cs scale
ablation ×2 running. NOT launched: scene (behavior density + extractor), spec 20 dwell
(P2/P3), P4.

Related: [[spec18-density-measure-20260806]], [[spec17-delta-measure-20260806]],
[[primary-paper-dir-multistep-fb]], [[stack-divergence-audit-20260805]].
