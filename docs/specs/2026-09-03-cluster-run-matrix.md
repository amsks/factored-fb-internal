# Cluster run matrix — the queued waves (2026-09-03)

For execution on the cluster (this box's 8 GPUs are taken by the fb_tdj
parity arm + local work). Everything referenced here is committed on
`density-fb`; `git pull` gives you all configs and launchers. Written for a
fresh session: each wave names its exact config, budget, eval protocol, and
gate.

## Cluster ops (applies to every wave)

- Env vars: `EXORL_DATA_ROOT` and `EXORL_CACHE_ROOT` must point at the
  cluster's ExoRL data (RND buffers per domain) — jobs silently look in
  `$HOME` and die after allocation otherwise. OGBench waves need the
  datasets dir (auto-download to `~/.ogbench` works if nodes have egress).
- `XLA_PYTHON_CLIENT_PREALLOCATE=false` always;
  `XLA_FLAGS=--xla_gpu_enable_command_buffer=` (CUDA graphs OFF — lost ~25%
  of cells locally without it).
- Slurm bodies to adapt are in `scripts/*.slurm` (they targeted the old
  cluster layout); the local launchers `scripts/launch_exorl*.sh` encode
  the arm→config→override mapping and are the source of truth for flags.
- Checkpoint/resume landed (auto_resume, atomic saves) — safe under
  preemption.
- Eval: `scripts/exorl_paper_eval.py` (official protocols) and the tdj
  route (10k inference / 10 episodes) — per-wave below.
  `scripts/exorl_report.py` needs `--steps` passed EXPLICITLY.
- Never quote mid-run evals; final-step or last-k only.

## Wave A — ExoRL RLDP ortho-live, remaining domains  [READY, highest value]

The walker result (2026-08-31): `rldp_orthoa` (ortho LIVE at coef 1.0 —
overriding the official code's inert regularizer) scores 828/490/225/347 =
62–94% of published, vs 365/107/67/125 for the code-literal inert arm.
Cheetah/quadruped/pointmass were never run for this arm.

- Config: `impls/configs/critic/rldp_orthoa.yaml` (committed; the walker
  winner verbatim). Do NOT use `rldp_official.yaml` (that is the inert
  rung) and do NOT weaken the coef (1e-4 measured WORSE than inert).
- Matrix: {cheetah, quadruped, pointmass} × 4 seeds. Budget as walker ran:
  5M steps (two-phase per the yaml), discount 0.98 (0.99 pointmass — set
  via the domain override the launcher applies).
- Data: cheetah/pointmass standard RND; quadruped RND buffer must exist on
  the cluster (download from the denisyarats/exorl bucket if absent).
- Eval: official protocol — 50k re-simulated-reward inference samples,
  50 episodes, 4 tasks/domain (task sets in the 08-28 spec §2).
- Gate: compare against **td_jepa Table 1**, NOT RLDP's paper baselines
  (their baseline columns are under-reported — HILP 2.5× low; see
  docs/notes and the tdjepa memory). RLDP's own column is fine to quote.
- Cheap early screen: `scripts/rldp_action_sensitivity.py` sign test at
  2.5M — it called the walker ordering exactly, with no rollouts. Run it
  per arm at 2.5M and report; kill arms at/below chance (.50) only after
  confirming with us.

## Wave B — ExoRL HILP, first launch  [READY]

Staged since 08-31, never fired.

- Config: `impls/configs/critic/hilp.yaml` (official-HILP transcription;
  `hilp_tdj.yaml` is the td_jepa-matched variant — run BOTH if budget
  allows, they answer different questions: official replication vs
  same-protocol comparison).
- Matrix: {walker, cheetah, quadruped} × 4 seeds × 1M steps (official
  budget). Buffers: HILP's stated size is RND 5,000 episodes — subsample
  the standard buffer to its first 5,000 (the harness knob the 08-28 spec
  §2 defines); `hilp_tdj` uses the full-buffer tdj protocol instead.
- Eval: hilp arms — least-squares z over 10k transitions, √d-normalized,
  50 final episodes; hilp_tdj — the tdj route (10k inference, 10 eps).
- Gate: td_jepa Table 1 HILP column; HILP's own paper Table 3 is STILL
  UNTRUSTED in our notes (WebFetch transcription) — do not gate on it
  until re-extracted from the PDF.

## Wave C — density variants  [GATED on A/B]

Launch a method's density arm ONLY where its vanilla passes its gate on
that domain (user's standing rule).

- RLDP: `rldp_density_official.yaml` (compose with the orthoa settings —
  mirror whatever Wave A's passing arm ran, ortho included) and/or
  `rldp_density_tdj.yaml` for the tdj-protocol twin.
- HILP: the density-HILP critic (committed 732014e) on the hilp_tdj
  substrate — config per its yaml/launcher entry.
- Same domains, seeds, budgets, eval as the vanilla they mirror. Report
  format: official | our-vanilla | our-density per cell.

## Wave D — privileged-coordinate ablation, scene + cube-triple  [NEEDS ~half-day impl first]

Extends the 09-02 cube-single/double matrix (PAPER table_privcoord). NOT
launchable as-is: domain configs for the new envs do not exist yet.

Prerequisites (implement on the cluster checkout, commit, push back):
1. `impls/configs/domain/{cube_triple,scene}.yaml` following
   `cube_double.yaml`'s documented pattern. Derive obs layouts EMPIRICALLY
   from the envs (cube-double is 19 proprio + 2×9 block cols, block pos at
   19:22 / 28:31; triple should extend the pattern — VERIFY, don't assume).
   For scene, do NOT invent the hit-spec: transcribe it from the main
   worktree's launch scripts (`Factored-FB` branch gciql-fb-bridge,
   grep scripts/ for scene hit_spec, e.g. launch_table_fill_ogb_ms.sh); if
   absent, derive from the env's goal spec and document the derivation.
2. Hit-spec conjunction strings passed as the cube-double launcher does
   ("19:22:0.4,28:31:0.4" style); add a 3-block parse test if none exists.
3. Extend the 09-02 campaign launcher (`scripts/launch_cube_oc_density.sh`
   family — read it for the arm list) to the two domains.

Matrix: the SAME arm list as the 09-02 wave × {state, object-coords} ×
{scene-play-v0, cube-triple-play-v0} × 3 seeds × 500k — EXCLUDING all
FB-family arms (fb / fb_oc / fb_ref): FB is under eval-protocol repair
locally (docs/notes/2026-09-03-fb-vs-tdjepa.md — our FB cells were scored
through the goal-Dirac z while every reference number uses reward-inferred
z; the relabel eval is being built). FB rows join when that lands.

Eval: as the 09-02 wave (goal-conditioned, 20 episodes; eval.csv is one
row PER TASK — group by step, assert the full task count before
averaging).

## Priorities and budget

A > B > C (gated) > D. A+B ≈ 3×4×~2 + 3×4×2(arms)×~1 GPU-days; D adds
~28–42 half-day runs after its prerequisite lands. If the cluster queue is
tight, Wave A cheetah+quadruped and Wave B walker+cheetah are the minimum
decisive set.

Report everything back as per-domain tables against the gate columns; any
gate failure gets the ambiguity-flag/ablation-ladder treatment (08-28 spec
§4), never tuning toward the number.
