# Density-FB bridge-port + table-correction spec (for the implementation agent)

**Date:** 2026-08-16 · **Status:** Part A LANDED (gate job 915566, 0 failures) · Part B blocked on the A3 dev screen (job 915585)

## Status at a glance

| item | state |
|---|---|
| A1 lr parity | done — but the working key is **`lr_actor`**, not `actor.lr`; see the correction under A1 |
| A2 hard default | done — `measure_density_hard: true`; no published number moves |
| A3 ddpgbc port | done — `impls/actors/ddpgbc_density.py`, registered as `ACTORS["ddpgbc_density"]` |
| A4 instrumentation | done — `action_gap_l2`, `q_gap_data_minus_actor` in `flowbc.loss` info |
| Part A gate | **PASS** — `sbatch scripts/density_port_smoke.slurm`, job 915566: 6 new tests + 66-test regression subset + 200-step runs on both actor paths |
| A3 dev screen | **FAIL** — job 915585 finished flat 0.00–0.02 across all 500k (never trains; `density_ess_median` ~3 from the start, `goal_dist_final` ~0.28 throughout). ddpgbc as ported does not transfer to MW pick-place and cannot be the frozen actor. Open question: broken port vs genuine no — checkpoint videos not yet reviewed. |
| KT screens | **DONE (2026-08-17) — R1 CONFIRMED, COLLAPSE FIXED.** Three arms graduate on pr pick-place: **kt1-qc03 peak 0.980 / final 0.900** (best), kt1-qc01 0.740/0.720, combo-qc03-lr 0.920/0.760. Failures: bc10 & tau001 (raise peak, still decay — loop-slowing insufficient), g098/temp2/dim128/split (low peaks). Pattern = R1 exactly: only weakening the actor's ascent (q_coeff↓) repairs it. 2026-08-17 audit verdict (3 subagents): HPs at bridge parity (lr_actor only, screened), datasets clean (measure concentration is LEARNED: geometric ESS ~550 vs trained ~13), no port bugs (audit found only benign deltas + bon_probe eval inconsistency + missing mask/hit-spec disjointness assert). qc03 confirmation seeds 1/2: jobs 916996-97→917096-97. New audit arms: kt6-soft 917094, kt7-bc03 917095. Freeze candidate: `actor.q_coeff=0.3` (± lr_actor=3e-4 pending combo-vs-qc03 3-seed comparison). |
| KT screens (orig) | was: RUNNING (2026-08-17) — 9 flow-actor kill-test arms from the collapse spec §4, jobs 916951–916959: kt1-qc03/qc01, kt2-tau001/bc10, kt4-temp2/dim128, kt3-g098 (mechanism-side only), kt5-split, combo-qc03-lr (graduation candidate: `lr_actor=3e-4 actor.q_coeff=0.3`). Launcher extended; done-detection fixed for `\r\n` eval.csv lines. Read with `scripts/density_screen_report.sh`. |
| Part B | **LAUNCHED 2026-08-17T20:13Z — config FROZEN on soft** (`actor=flowbc`, `measure_density_hard=false`, all else stock). 27 jobs (`mw ms ogb`), SAVE_ROOT `$SCRATCH/factored-fb-density-table`. Freeze evidence: soft 3-seed finals 0.800/0.820/0.820 (all graduate, drops ≤0.10) vs qc03 0.900/0.560/0.820 (one decay); combo qc03+soft 0.860 single-seed reserved as a possible second-generation freeze via the edge-attribution track. On completion: replace every Density-FB cell in table.tex, collapse footnote e. |

Scripts added: `scripts/density_port_smoke.slurm` (Part A gate),
`scripts/launch_density_port_screens.sh` (dev screens),
`scripts/density_screen_report.sh` (reads the graduation gate off `.o` logs),
`scripts/launch_density_table.sh` (Part B; refuses to mix configs within one `SAVE_ROOT`).

The report script was validated against the existing runs before being trusted: it reproduces
the collapse spec's E1 exactly — pr pick-place peaks 0.50/0.26/0.52 and finals 0.18/0.08/0.14,
pr push graduates on all three seeds, prh pick-place peaks 0.72/0.84/0.82 with one seed
collapsing to 0.20.

**Date:** 2026-08-16 · **Original status:** SPEC — ready to implement
**Upstream:** branch `origin/gciql-fb-bridge` @ `59f5072` (same remote), reference file `tools/wandb_mode_shim/gciql_flow.py`. Our port base was `542c058`. Every bridge change since then is behind default-off knobs — nothing we ported was silently fixed; the gaps below are missing features and recipe verdicts.
**Companions:** `docs/specs/2026-08-14-density-fb-hpo.md` (uniform-config rule, dev/holdout split), `docs/specs/2026-08-16-density-fb-pickplace-collapse.md` (R1 bootstrap-loop hypothesis — the ddpgbc port below doubles as an R1 mitigation).

## Part A — Implementation work items

### A1 (P0, config only): lr parity
Bridge uses a single `lr=3e-4` for critic *and* actor. We split `lr_actor: 1.0e-4` (`impls/configs/actor/flowbc.yaml:9`). Add/verify an override path so density runs can set `actor.lr=3e-4`; do NOT change the flowbc default globally (GCIQL/biGCIQL cells were run at 1e-4 and must stay comparable). The density table config carries the override explicitly. (HPO spec H1; S0 screen already showed 0.133→0.367 on pr pick-place.)

> **Correction (implementation):** the key this tree reads is **`lr_actor`** (top level), not
> `actor.lr`. `actor.lr=3.0e-4` is valid dotlist, merges cleanly, and is **silently inert** —
> the run trains at 1.0e-4 and reports a number indistinguishable from a tuned one. The
> override path already existed; nothing was added. `env/config.py` now raises `ConfigError`
> on `actor.lr` / `critic.lr` (`_INERT_ALIASES`) naming the real key, so the spec's spelling
> fails loudly instead of quietly. Verified: `lr_actor=3.0e-4` → 0.0003, flowbc default and
> the gciql/iql_bilinear cells still 0.0001.

### A2 (P0, config only): hard variant is the default for density
Bridge verdict (`DENSITY-FB-EXPLAINED.md` @ `b811ea8`): "the hard variant is the one to ship" (cube-single 0.912 hard vs 0.725 soft). Flip `measure_density_hard: false` → `true` in `impls/configs/critic/density_fb.yaml:17`. All existing table cells already ran with the override on (tags `densityhard*`), so this changes no numbers — it removes a footgun for every future launch.

### A3 (P1, code): port the ddpgbc actor extraction (bridge spec22)
The only bridge mechanism that closes an antmaze-sized gap (hard-ddpgbc antmaze 0.920/0.868/0.844 vs hard-flow 0.772/0.580/0.656), and mechanistically relevant to the pick-place collapse (R1). Port from `gciql_flow.py` @ `59f5072`:
- `_density_actor_q` shared readout helper (~line 996);
- the `ddpgbc` gaussian DDPG+BC branch in `actor_loss` (~1306), including its hardcoded **centered** `q_scale` normalization;
- the matching bootstrap branch inside `density_loss` (~2518–2545): ddpgbc bootstraps `dist.mode()` at the target/frozen actor.
Land it as a new actor choice (e.g. `ACTOR=ddpgbc_density` or an actor-flag on flowbc's slot) wired through `impls/critics/density_fb.py`'s `q_fn`/`sample_bootstrap` seam (`density_fb.py:217-243`). Keep our flow path bit-identical (bridge knob discipline: new branch default-off). The `awr_flow` branch (~1272) is optional — port only if cheap while in the file.
- **Static tests:** extend `tests/impls/test_density_fb.py` — shapes, bootstrap uses mode not sample, flow path unchanged under default config.

### A4 (P2, code, mechanism-side): collapse instrumentation
In `flowbc.loss` info: log `Q(s,a_data) − Q(s,a_actor)` and `‖a_actor − a_data‖` (kill-test spec, "optional instrumentation"). One file, no behavior change.

### Explicitly OUT of scope
The z-track (bridge specs 24–29: z-indexed density, tilt/rdecode/zenc couplings, actor-reward decoupling, dual-corrected index), behavior-policy SARSA mode (19a), ORS shaping (19b). Separate research fork; not needed for `tab:main`.

## Part B — Correcting the table numbers on non-OGBench envs

Current Density-FB coverage in `PAPER/decoding-bilinear/table.tex` (tab:main): only the two **play-resample v3** cells are filled (pick-place 0.133, push 0.820). **Pending:** pnh-1m × 2, play-resample-hard v3 × 2, ManiSkill × 2 (`ms-pick-cube`, `ms-push-cube`), plus the OGBench OOD-goal split. Note: completed runs for prh and pnh-1m pick-place already exist in `$SCRATCH/factored-fb-tier1/` (prh 0.573, pnh-1m 0.460, seed triples in the collapse spec) under the current config — usable as-is ONLY if the frozen config ends up identical to what they ran with.

### Rules (all inherited, non-negotiable)
1. **Uniform config**: one density config wins or loses the whole table (HPO spec). Freeze the config FIRST (after A1/A3 screens on the dev cell clear the collapse-spec gates), then fill/rerun every cell with it. No per-cell tuning.
2. **Dev/holdout**: `mw-pick-place` play-resample v3 is the only DEV cell. Touch holdout cells (pnh-1m, prh, ManiSkill, mw-push) exactly once, with the frozen config.
3. 1M-only; γ=0.99 in every table cell; 3 seeds; final-step eval, 50 episodes; report seed triples next to any near-tie (eval reseeds the global RNG — CIs understate).
4. If the frozen config differs from the `densityhard*` runs (it will, at minimum via A1's `actor.lr`), the already-filled pr cells and the scratch prh/pnh-1m runs are **superseded — rerun them too**. Never mix configs within the Density-FB column.
5. Reporting honesty (R4 of the collapse spec): run with `EVAL_INTERVAL=50000`; if a cell still shows a peak-then-decay curve, the table cites the final number and the caption/footnote discloses the peak (e.g. "final of a decaying curve; peak 0.84@200k").

### Launch mechanics
- `scripts/vista_train.slurm`, `CRITIC=density_fb ACTOR=flowbc` (or ddpgbc variant if it wins the dev screens), frozen override string incl. `measure_density_hard=true measure_hit_spec=\"4:7:0.07\"` + repaired-protocol string (`obs/goal_mask_dims=[36:39] value_p_randomgoal=0 eval_goal_mode=synth`). `tr`-not-`xargs` quoting; per-TAG SAVE_DIR; DRYRUN smoke before each batch; no `--dependency` chains; read `.o` logs, not eval.csv (null bytes).
- **Metaworld cells** (pnh-1m, prh v3, pr v3 reruns): 500k steps, hit-spec `"4:7:0.07"`.
- **ManiSkill cells**: 300k steps (table convention), masked goal channel, corrected grasp-offset synth goal frame (footnote c protocol). Expect `ms-push-cube` to be a degenerate control (goal-blind BC saturates at 0.987) and masked `ms-pick-cube` to floor near 0.02–0.04 for everyone — these cells don't discriminate; fill them for completeness, don't tune on them.
- **Job count**: 6 pending non-OGBench cells × 3 seeds = 18, + 2 pr rerun cells × 3 = 6 → ≤ 24 jobs, inside the HPO envelope.

### Landmine found while implementing: the hit spec must carry literal quotes

`measure_hit_spec=4:7:0.07` passed unquoted on the dotlist is read by OmegaConf as a
**sexagesimal literal** and becomes `14820.07` (= 4·3600 + 7·60 + 0.07). Multi-block specs
(`4:6:0.07,6:7:0.14`) survive unquoted because the comma breaks the pattern — so this bites
only the single-block case, which is every Metaworld cell. The override string must carry
escaped quotes (`measure_hit_spec=\"4:7:0.07\"`, the idiom already in
`launch_measure_tier1.sh`), and the shell must not eat them: `tr`, not `xargs`.
`parse_hit_spec` now detects the sexagesimal result and raises naming the quoting fix rather
than reporting a bare parse error on `'14820.07'`.

### Table update procedure
For each finished cell: pull the three final-step numbers from `.o` logs, compute mean ± 1.96·s/√3, replace the `\emph{pending}` entry in `table.tex`, re-bold per the CI rule (bold only if runner-up mean lies outside leader's CI), update footnote d if the config changed (provenance: "this repo's port, frozen config X"), and update the caption's one-line reading if the pick-place verdict changes. Commit per completed dataset block, matching the existing commit style (`results: <block> complete — ...`).
