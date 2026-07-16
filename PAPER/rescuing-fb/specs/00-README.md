# Rescuing FB — program overview

*Started 2026-07-16. Goal: make FB-style zero-shot RL work on OGBench manipulation by keeping
the FB **API** (task space z, task encoder ψ/B, reward-moment inference, z-conditioned policy)
and swapping the **grounding** (policy-conditioned measure TD → hindsight scalar expectile TD).
Companion mechanism program: `PAPER/atlas/`. Bridge evidence base: `PAPER/fb-gciql-bridge/`.*

## Folder map

| File | What |
|---|---|
| `01-synthesis-deep-audit.md` | The 2026-07-16 deep audit + synthesis: code audits (jax + torch), mechanism story, literature juxtaposition, test queue T1–T9, RG-FB sketch, cross-OGBench plan. **Read first.** |
| `02-rgfb-spec.md` | RG-FB (Reward-Grounded Forward-Backward) — full method + implementation + eval spec, pre-registered success criteria. |
| `03-probes-spec.md` | Mechanism probes T1 (goal-smoothed rerank), T2 (DR3 co-adaptation), T3 (off-support peak × domain), with exact designs and pre-registered predictions. T4–T9 queue. |
| `../understanding-bilinear.tex/.pdf` | **LIVING DOC 1**: Socratic record of every experiment (Q / experiment with training details / findings), the bedrock open questions about bilinear Q-networks with queued experiments, and the bilinearity/representation-learning survey (VIP, BVN, CRL, QRL, linear MDPs, DR3, ...). |
| `../fb-variants-and-grounding.tex/.pdf` | **LIVING DOC 2**: implementation-level description of every FB variant, the annealing experiments and their three lessons, and the reward-grounding recipe + instances (RG-FB, grounded PSM proposal, centered measure T10). |
| `../meeting-notes-amy.*` | Meeting-prep condensations (superseded for research use by the living docs). |
| `../DECISIONS.md` | Running log (append-only) of what was run, changed, found. |

## Artifacts

- Probe outputs: `analysis/rescuing-fb/` (JSON + a short md per probe).
- RG-FB code: `tools/wandb_mode_shim/gciql_flow.py` (new knobs, default-off), tests in
  `tests/`, configs in `configs/gciql/`, zero-shot eval in `scripts/rescuing_fb/`.
- Training runs: `output_root=/home/adityamohan/exp_rgfb` (persistent), tmux session `rgfb`,
  wandb entity `amsks`.

## The claim being built

1. **Diagnosis** (done, atlas + bridge): the measure estimator — not the successor-measure
   interface, not the bilinear structure, not the actor — is what fails on manipulation
   (non-contractive, occupancy-weighted, bad fixed point on cube).
2. **Mechanism** (nearly done; T1–T3 close the gaps): bilinear GCIQL wins via cross-goal
   advantage pooling + DR3-protected stability + extraction match with the flow actor.
3. **Method** (this folder): RG-FB — a zero-shot agent grounded by scalar TD that matches the
   supervised SOTA on cube (target ≈0.95 zero-shot) and ports across OGBench.

## Status (2026-07-16)

- T1/T2/T3 probes: implementation + first runs dispatched.
- RG-FB v0: implementation dispatched (z-conditioned vf + zero-shot eval + 3-seed cube launch).
- Owed next: T4 (goal-conditioned vf on antmaze), T6 (zero-shot generality probe),
  param-matched control (T5), FB baselines T7/T8 for the paper.

## Relation to the bridge program (fb-gciql-bridge)

RG-FB **is a graft** — the terminal composition of the bridge's Direction A (FB structure onto
the IQL grounding), selected cell-by-cell from the measured lattice: bilinear form (D2, 0.904,
adopt) · φ z-conditioning (B1, 0.848, reject) · two-tower (0.86–0.87, unnecessary) · ortho1000
(0.072, reject) · z-cond actor (G4, 0.921, adopt) · moment inference (E1a, 0.955, adopt) ·
measure aux (E2 λ≤1, 0.86–0.92, optional) · z-cond BC vf (NEW, v0 tests it). Beyond the
composition, RG-FB changes the deployment contract: bridge cells were evaluated with exact
goals; RG-FB is evaluated zero-shot only. Dually, **T10 is the last new Direction-B cell**
(rescue the measure side). Paper skeleton: bridge = the search, atlas = the mechanism,
RG-FB = Direction A's constructive payoff, T10/T8 = Direction B's boundary.
After the campaign: add RGFB-Cube-Single-State-d50-v0 (and T10 groups) to
`scripts/results_map.py` + refresh `results/*.csv` (standing rule).
