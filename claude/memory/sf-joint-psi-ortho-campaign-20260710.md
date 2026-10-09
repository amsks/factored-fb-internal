---
name: sf-joint-psi-ortho-campaign-20260710
description: "Joint-psi SF cells A+C: psi trained by the SF vector TD itself + FB ortho(1000) + sphere. Cell A is matched to C-fbz-projz-ortho1000 (0.231) in EVERY respect except the estimator (per-sample vs BxB). RUNNING tmux sfjoint, GPUs 3/5/6/7, launched 2026-07-10 ~01:05."
metadata: 
  node_type: memory
  type: project
  originSessionId: f75106fc-71c3-4a35-b220-0550320b953f
---

**Bridge track.** Spec `docs/superpowers/specs/2026-07-10-joint-sf-ortho-design.md`; plan
`docs/superpowers/plans/2026-07-10-joint-sf-ortho.md`; living doc = `PAPER/fb-gciql-bridge/
sf-design-space.md` §1a. (Both `docs/superpowers/` and `PAPER/` are gitignored.)

## The idea, and why it is nearly the FB ablation
ψ has never been trained by the SF loss — it was frozen, identity, or reward-regressed (E3).
`sf_joint_psi` un-stop-grads the base `b` so the TD residual reaches ψ, held non-degenerate by
FB's ortho. With `Δᵢ = φ(sᵢ,aᵢ) − γφ_tgt(s′ᵢ,a′ᵢ)`:

- joint-SF loss  = `‖Δ‖² − 2⟨Δ,ψ(s)⟩ + ‖ψ(s)‖²`
- FB measure     = `½ΔᵀCΔ/d − ⟨Δ,ψ(s′)⟩/√d` + `ortho = ½‖C−I‖²_F − d/2`,  `C := E_ρ[ψψᵀ]`

**FB's diagonal term IS the SF base term.** Ortho drives `C→I`, equating the quadratics;
`project_z` makes the norm terms constant. Three residuals survive: **(i) the estimator**
(FB pairs every `(sᵢ,aᵢ)` with every `ψ(s′ⱼ)`, B×B, residual in the metric `C`; SF uses only the
diagonal `i=i` in the identity metric), (ii) base index `ψ(s)` vs `ψ(s′)`, (iii) the absorbing
mask, which FB lacks (`measure_loss` has ZERO references to `batch['masks']`, and orthos
`ψ(next_observations)` — the ρ marginal, not the HER goals).

Cell A removes (ii)+(iii) by adopting FB's conventions wholesale ⇒ **matched to
`C-fbz-projz-ortho1000` = 0.231 ± 0.030 except the estimator.** The 2×2:

|  | reward-grounded ψ | ortho-grounded ψ |
|---|---|---|
| measure (B×B) | *empty* | **0.231** |
| SF (per-sample) | 0.000 / 0.036 (E3) | **cell A** |

A ≈ 0.23 ⇒ estimator irrelevant, SF story dissolves into the FB story. A ≈ 0.0 ⇒ the per-sample
vector TD is uniquely destructive.

## Code (committed 1a28995 → b6a7cc8, 31 tests green)
New knobs in `gciql_flow.py`: `sf_joint_psi`, `sf_base_index ∈ {current,next}`,
`sf_mask ∈ {goal,none}`, `sf_psi_ortho_coef` (on **ψ**; the pre-existing `sf_ortho_coef` is on
**φ** and at λ=1000 drove cells to 0.001–0.004 — never conflate). `project_z` threaded into
`GCFrozenBValue`/`GCLearnedBValue` (NOT Identity: its `_psi` returns the raw goal).
Five guards; guard 3 (`mask='goal' ⇒ fb_forward_z`) is **scoped to joint cells** — enforcing it
globally would break every legacy SF config. Tests `tests/test_gciql_sf_joint_psi.py` (9).

Ortho identity is asserted in its **exact finite-batch** form (the spec's `½‖C−I‖²_F − d/2` is the
B→∞ limit): `mean_{i≠j}⟨ψᵢ,ψⱼ⟩² = (B²‖C‖²_F − Σᵢ‖ψᵢ‖⁴)/(B²−B)`, `o_diag = −tr(C)`.

## RUNNING: tmux `sfjoint`, launched 2026-07-10 ~01:05
`scripts/launch_sf_joint.sh`, `GPUS="3 5 6 7"` (psi_base held 0/1/2/4), 6 runs = 2 cells × 3
seeds @500k, logs `/dev/shm/sfjoint_logs/`. 14 compute procs total on the box.
- A `GCIQL-State-Flow-SF-JointPsi-Ortho1k` — `b=ψ(s′)`, no mask, ortho1000, fbz + project_z
- C `GCIQL-State-Flow-SF-JointPsi-Ortho1k-Absorb` — `b=ψ(s)`, goal mask ⇒ `Q=γ^T` (hitting time)

**Diagnostics verified live @5-10k:** `psi_norm_mean = 7.0711` (=√50, project_z live),
`psi_ortho_diag = −50.0000` (=−d, constant ⇒ radial collapse impossible).

**DO NOT read `psi_ortho_off` on its own** (I got this wrong first pass). Asymptotically
`o_off = ½‖C‖²_F`, so at the WHITENED optimum `C=I` it equals `d/2 = 25` — the *same* value
random sphere-normed ψ gives, because random isotropic vectors already have `C ≈ I`. It cannot
distinguish the two. The discriminating quantity is the SUM:

    psi_ortho = o_off + o_diag = ½‖C‖²_F − tr(C) = ½‖C − I‖²_F − d/2   →  −d/2 = −25  iff  C → I
    ‖C − I‖²_F = 2·(psi_ortho + d/2)

**Measured @55–125k: psi_ortho ≈ −24.9, i.e. ‖C−I‖²_F ≈ 0.03–0.27 ⇒ ψ IS whitened**, so the
peaked-kernel argument (§4.2 of the spec) holds and the cells are healthy. `scripts/campaign_monitor.py`
reports `‖C−I‖²` directly.

`sf_loss` growing unbounded is BENIGN for a successor measure (PSM finding); not divergence.
The flow actor normalizes its Q-term by `sg[|q|.mean()]`.

## RESULT — DONE 2026-07-10, 3/3 seeds @500k, CSVs refreshed
- **Cell A = 0.120 ± 0.016** vs the matched B×B measure baseline **0.231 ± 0.030**.
  Non-overlapping CIs; per-seed Δ negative on all three (−0.116/−0.072/−0.144), everything else
  held identical. ⇒ **The estimator matters.** Same fixed point (ortho drives C→I), different
  test matrix: `½ΔᵀIΔ` (per-sample diagonal) vs `½ΔᵀCΔ/d` (B×B pairs). Mechanism = gradient
  density into ψ — the measure loss gives every `ψ(s′ⱼ)` gradient from all B forward vectors,
  while cell A's `ψ(s′ᵢ)` learns from its own sample only (~B× sparser).
- **Cell C = 0.125 ± 0.017 — a NULL.** Per-seed Δ vs A = +0.000/+0.000/+0.016. Absorption
  (`Q=γ^T`, first-hitting) vs visitation density makes NO difference on cube. Hypothesis dead.
- Ortho was healthy: `|ψ| = 7.071 = √d` exact, `‖C−I‖²_F ≈ 2.2–2.7` of `‖C‖²_F ≈ 52`. ψ WAS
  whitened; the shortfall is not an ortho failure.
- Cell A lands inside the OLD confounded SF band (0.08–0.16): joint-ψ + FB ortho did not lift the
  per-sample SF cell out of it. Both cells flat after 100k, like the whole measure family.

## NEXT
Nothing outstanding for this campaign. The estimator-axis 2×2 now has three of four corners; the
empty one is (measure estimator + reward-grounded ψ).

[[e3-sf-rewreg-campaign-20260710]] [[results-csv-pipeline]] [[always-run-in-tmux]]
[[wandb-entity-amsks]] [[commit-no-claude-coauthor]] [[jax-venv-location]]
