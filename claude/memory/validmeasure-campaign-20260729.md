---
name: validmeasure-campaign-20260729
description: "valid-measure campaign CLOSED 2026-08-05 (§9.15c RESULT): wave-2 deconfound shows the MASS ANCHOR, not the nonneg hinge, causes BOTH the ordering loss (Spearman hw0.8 0.68->0.48 anchor-only) and the smooth0.8/ball1.6 transfer rescue; §9.15b findings #3 and #4 REFUTED; mcE dose-reduction and d128 both fail their PRIMARY gates; no arm passes all four -> validity<->ordering is structural to mass conservation"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2dafcc47-02b8-41af-948e-ac4c251a8661
  modified: 2026-08-05T01:28:45.147Z
---

**Bridge track.** The 12-hour "fully working successor measure" push
([[phase2-audit-20260728]], [[rewardz-launched-20260729]] → §9.11b was a WIN,
rz law adopted).

**D1 diagnostic (new, `scripts/rescuing_fb/negmass_diag.py`):** the ~50%
pointwise negativity is NOT cosmetic — negative entries carry **24–29% of
absolute mass** at ~36% of positive magnitude, and negativity is WORSE in
high-density quartiles (0.55–0.59 top vs 0.45 bottom). Mechanism call:
**truncated-basis (Gibbs) ringing** of the rank-50 bilinear fit to the sharp
boxcar hit kernel. {d50, boxcar, plain L2, M≥0} is over-constrained — pick 3.

**Arms (2 seeds each, phase-2 only, warm from `p1_s4{2,3}/final.pt`, all on the
adopted rzK50 stack; control = `p2_rzK50_s4{2,3}`):**
- `mcA` = nonneg hinge λ=1 + NEW `mass_coef=1` (per-row 1/N̂-weighted total-mass
  anchor toward 1/(1−γ); the anchor the hinge lacked when it inflated mass
  2.5–3.6×)
- `mcB` = mcA + `hit_kernel=quartic hit_eps=0.6` (C¹ kernel, ~0.77× ball mass)
- `mcC` = quartic alone (deconfound)

Chain per job: train 500k → L1fix → L2-B → L2-A → negmass. JSONs →
`results/spec10_torchfb/{L1fix,L2_B,L2_A,negmass}_mc{A,B,C}_s4{2,3}.json`.
**Gates G1–G4 + NULL pre-registered in PHASE2-LEDGER.md §9.15** (G1 neg_frac
≤0.10 AND neg-mass ≤0.05; G2 ratio_head 0.85–1.30 + ρ≥0.65; G3 success ≥0.92;
G4 transfer holds). NULL ⇒ signed bilinear itself is the obstruction → next is
structural nonneg factorization (softplus F,B; needs d≫50) or d128. Full debate
in `PAPER/rescuing-fb/specs/13-valid-measure-12h.md`.

**Smoke (1200 steps, not a result):** control byte-identical to archived rzK50
log; mcA at step 1000 had neg_frac 0.36→**0.015** with mass_ratio 0.09→0.60
rising toward 1.0 — sign and scale moving TOGETHER for the first time.
⚠ Remember the §9.11b lesson: the rzlaw smoke's negativity signal did NOT
survive to 500k.

**New knobs** (`agents/fb/measure/agent.py`): `mass_coef`, `hit_kernel` —
default-off byte-identical, 8 tests in `tests/test_fb_measure_mass.py`, wired
through `train.py` measure branch (whitelist-trap tested). For quartic arms
quote L1 **head** ratios only (exact-N̂ is ball-based, mismatched by design).

**WAVE-1 VERDICT (02:15, ledger §9.15b):** no arm passes all gates strictly, but
NOT the NULL. **mcA/mcB pass G1 decisively** (neg-mass share 0.27→**0.010–0.014**,
flat density gradient) with calibrated ratios (anchor pins mass_ratio=1.002 —
§9.9's sign/scale trade-off CLOSED) and G3/G4 pass (mcB both seeds).
**UNEXPECTED WIN: the penalties rescue smooth0.8** (0.24/0.43→up to 0.86) — the
last unsolved transfer family. **Residual defect = ONE number**: Spearman hw0.8
0.68→~0.50 on all penalty arms (hinge convicted by archived nn1.0 L1fix
0.58–0.66; anchor presumed innocent). **Quartic-alone = null** (neg 0.50–0.52 =
control) ⇒ Gibbs-ringing mechanism REFUTED as dominant; negativity is regression
slack only the hinge removes. hw0.4 1.4–1.9× on all arms (rank-limited → d128).

**WAVE 2 RUNNING (tmux `mcons2`, GPUs 0–3, §9.15c pre-registered):** mcD =
anchor-only (deconfound), mcE = nonneg **0.3** + mass 1 (dose reduction; primary:
G1 pass AND Spearman hw0.8 ≥0.65). **d128 chains armed (tmux `mcd128`/`mcd128p2`,
GPUs 5,6):** p1_d128 (ball) + p1_d128q (quartic) → p2 d128mcB / d128mcE, evals
chained; d128's pre-registered target = hw0.4 ratio ≤1.3. All land ~05:30 UTC.

**MULTI-ENV LAUNCHED (2026-07-30 ~04:05, tmux `mcmenv`, GPUs 4,7):** torch
two-phase on cube-double + scene s42 via NEW `hit_spec` knob (conjunctive
blocks, jax E7 mirror; tests in test_fb_measure_mass.py, 12 green). **Data
property CONFIRMED stronger there**: cube-double each cube stationary 72%,
revisit mass 32.2; scene buttons 99.6%/drawer+window 91%, revisit 37.0;
conjunctive mass 3.3e-4/3.7e-3. ⚠ TRAP FIXED: `configs/domain/{cube_double,
scene}.yaml` had literal `agent:` keys that silently reverted CLI
`agent=fb_measure_flowbc` to stock FB (recorded config carried measure knobs
anyway!) — regression-tested now. ⚠ OPEN: mass anchor's in-batch N̂ clamp (2/B)
dominates at conjunctive hit rates ~1/B ⇒ validity penalties OFF in multi-env
p2 until a trained-N̂-based anchor exists.

**Paper (`PAPER/FB-Manipulation-ICLR/main.tex`) rewritten twice tonight:**
(1) Eysenbach/Castro style — no abstract, contributions-only intro,
preliminaries→problem; standard GCRL vocabulary (FB loss framed as unsaturated
contrastive; method = GC TD with indicator reward); (2) data-first narrative —
§3 revisit-dominance (playdata.png figure + fresh stats: cube bit-identical 48%
of transitions, median run 44 steps), curriculum motivated by the two factors,
single main table (goal-reaching × held-out-reward returns) with OGBench-paper
baselines (GCIVL/GCIQL/CRL/HIQL quoted) + our-stack FB/CRL rows (FB scene =
0.047!). Backup: old_main_20260730_prerewrite.tex.

**WAVE-2 VERDICT — CAMPAIGN CLOSED (collected 2026-08-05, ledger §9.15c RESULT).**
The runs finished 07-30; the JSONs sat uncollected for six days while §9.16 took
the campaign. Read from `results/spec10_torchfb/{L1fix,negmass,L2_B}_*.json`:

- **`mcD` (anchor alone) REFUTES its own pre-registration in both halves.**
  Expected ρ≈0.68 + neg_frac≈0.5 (which would convict the hinge); got Spearman
  hw0.8 **0.483/0.468** and neg_frac **0.27**. The anchor alone does essentially
  ALL the ordering damage and already halves the negativity.
- **§9.15b #4 is WRONG** — ordering damage is the ANCHOR's, not the hinge's
  (anchor-only 0.48 is worse than hinge-only 0.58–0.66). **§9.15b #3's mechanism
  is WRONG** — mcD is not non-negative (0.27) yet rescues smooth0.8 equally
  (0.24/0.43→0.63/0.57) and ball1.6 better than any hinge arm (→0.62/0.85). The
  anchor is the active ingredient in the benefit AND the cost.
- **Consequence:** the validity↔ordering trade is structural to mass
  conservation (`E_g[M/N̂]=1/(1−γ)` is what a measure IS), not a hinge artifact.
  A weaker hinge cannot buy ordering back because the hinge never spent it.
- **`mcE` dose reduction FAILS PRIMARY** (neg_frac 0.148/0.157 vs ≤0.10;
  ρ 0.545/0.480 vs ≥0.65) ⇒ λ-interpolation branch CLOSED, **mcB adopted** as
  the validity arm with the ordering cost reported.
- **d128 = WEAK support for the rank story**: hw0.4 ratio 1.30 (mcE, 1 seed,
  exactly at the ≤1.3 bar) / 1.55 (mcB); does NOT lift Spearman (0.568/0.507).
  hw0.4 remains the one defect no arm has ever closed.
- **No arm passes all four gates.** Best validity = mcB; best transfer = mcD/mcE.
  Single-object analogue of the §9.16 multi-env dissociation ⇒ strengthens fork
  (B) (decoupled hindsight-grounded actor head): if validity costs ordering with
  ONE object, asking one object to be both measure and critic may be the wrong
  target. See [[p2-multienv-decay-mechanism-20260730]].

**Still owed:** multi-env s43 arms when GPUs free; results-CSV
refresh ([[results-csv-pipeline]]); committing torch measure agents + spec 12 +
rz law + mass/kernel/hit_spec code + domain-config fix.
