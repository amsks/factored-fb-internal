# 2026-09-15 — Transfer table completed with baselines, and the first ExoRL runs of our own pair

Two overnight jobs. Everything below is eval-only except the six new walker trainings.

## Job 1 — the reward-transfer table now has FB / HILP / RLDP

`PAPER/ICLR/content/experiments.tex::tab:reward-transfer` had only \name{}. It now carries
all four methods on both task routes, every cell re-scored from the SAME 1M checkpoints
under one protocol: 5 tasks per env, 10 episodes per task, 3 seeds.

- **Route A (goal conditioning)**: `scripts/reeval_checkpoint.py`, no inference flags, so the
  env's goal frame goes through each critic's own `map_eval_cond`. Driver
  `scripts/score_transfer_table.sh A`, 48 cells over 7 GPUs, ~2 h. Rows in
  `analysis/transfer_table/routeA.jsonl` (240).
- **Route B (reward inference)**: `scripts/relabel_rescore.py`, which RE-LABELS the reward per
  task from the next state's simulator qpos. Driver `scripts/score_transfer_routeB.sh`.
  Rows in `analysis/transfer_table/routeB.jsonl` (~240).
- Report: `scripts/transfer_table_report.py` (prints both routes + the latex rows).

| task | FB reward / goal | HILP reward / goal | RLDP reward / goal | Density-FB reward / goal |
|---|---|---|---|---|
| cube-single | 42.0±11.4 / 17.3±1.9 | 72.7±3.4 / **86.0**±5.9 | 36.7±2.5 / 10.7±5.7 | **84.7**±4.1 / 74.7±12.5 |
| cube-double | 4.7±0.9 / 3.3±2.5 | 8.0±0.0 / 33.3±0.9 | 2.0±1.6 / 3.3±2.5 | **34.0**±8.6 / **38.0**±8.5 |
| scene | 10.7±6.8 / 24.7±0.9 | 37.3±4.1 / 52.0±5.7 | 10.7±4.1 / 4.0±1.6 | **52.0**±1.6 / **59.3**±11.5 |
| puzzle-3x3 | 4.0±3.3 / **14.0**±2.8 | 0.7±0.9 / 2.0±1.6 | **9.3**±7.5 / 13.3±5.2 | 6.7±1.9 / 7.3±5.0 |

Reading: Density-FB is the only method that is strong on BOTH routes on the three tasks it
wins; FB and RLDP collapse when the task is a goal rather than a reward (42.0→17.3 and
36.7→10.7 on cube-single). HILP is the exception and BEATS us under goal conditioning on
cube-single (86.0 vs 74.7) — a cell that only exists because of the fix below.

### Two provenance problems found and fixed before scoring

1. **HILP's goal route was goal-blind in every previously published cell.** `map_eval_cond`
   was added to the HILP family on 2026-09-08; the tdjtab campaign predates it, so its
   in-run `eval.csv` rolled a zero task vector. Re-scored with the current code, HILP's
   cube-double goal cell moves 2.0 → 33.3 and cube-single 12.3 → 86.0.
2. **The RLDP cells were resolving to the ortho-0.01 runs** (`exp/…`), not the ortho-1 runs
   (`exp/ortho1`) the paper's tables quote. Fixed in the inventory; all RLDP cells now come
   from `exp/ortho1`.

### A bug in my first attempt, recorded so nobody repeats it

Route B run through `reeval_checkpoint --infer_cond_episodes 10000 --eval_on_inferred_cond`
scores near zero (cube-double Density-FB 2.7 instead of ~34). That path infers the task
vector from the DATASET's own reward column, which for OGBench play data is the
goal-reaching reward and is zeroed for unlabelled data — so it solves for a constant. The
reward must be re-labelled per task, which is what `relabel_rescore.py` does and what
impls/main.py's own relabel eval does. 47 such rows were discarded.

### Checkpoints

scene and puzzle baselines were not on this box; 24 runs (4 methods × 2 envs × 3 seeds,
config + params_1000000 + both eval csvs, 96 files) pulled from
`amsks/factored-fb-checkpoints` into `exp/hub/exp/`.

## Job 2 — why we rank low on ExoRL

**The answer is not tuning: there was no FB or Density-FB run on ExoRL at all.** Every
ExoRL number in this project is a retrofit (the density objective inside RLDP or HILP).
Six walker runs were trained overnight (500k, ddpgbc actor with the motivo overrides,
discount 0.98, 3 seeds; `scripts/launch_walker_dfb.sh`, save dir `exp/walker_dfb`) and
scored with the same driver as the rest of the block (`scripts/score_walker_dfb.sh` →
`exp/paper_eval_walker_dfb.jsonl`; report `scripts/walker_report.py`).

| critic | stand | walk | run | spin | mean over 4 tasks |
|---|---|---|---|---|---|
| Density-FB (density_fb_z) | 715 | 398 | 226 | 872 | **553 ± 12** |
| FB at ortho 1000 (fb_tdj) | 233 | 32 | 31 | 125 | 88 ± 82 |
| HILP (hilp_tdj) | 873 | 848 | 308 | 983 | 753 ± 18 |
| Density-HILP, regression index | 903 | 602 | 286 | 986 | 694 ± 41 |
| Density-HILP, feature index | 621 | 577 | 194 | 984 | 594 (n=1) |
| RLDP (rldp_tdj) | 802 | 596 | 294 | 644 | 584 ± 122 |
| Density-RLDP, feature index | 779 | 267 | 157 | 777 | 495 ± 5 |
| Density-RLDP, regression index | 614 | 244 | 139 | 739 | 434 ± 61 |

**The FB row is not usable and is my error**: `fb_tdj` is FB at ortho 1000, the value the
cube/td_jepa grid selected; ExoRL's FB standard is ortho 1. Three ortho-1 FB runs
(`--critic fb`) are training as of 06:10 and must be scored before any claim is made.

### Standing diagnostic for the retrofit losses

From the training logs, the predicted distribution on walker spreads over 28–48 candidates
(`density_ess`, entropy 3.0–4.4 nats) against 7–14 on cube-single (entropy 2.2–3.0), and in
both cases it matches its own target's entropy — so the target itself is diffuse at a
50-step horizon in locomotion. The value readout integrates over that diffuse distribution,
which smooths exactly the small action differences a dense locomotion reward turns on.
If the ortho-1 FB comparison shows Density-FB losing too, the first knob to sweep is
`measure_density_temp` (softmax temperature), not the task index — the index route was
already dissociated (regression 434/694 vs feature 495/594 on the two retrofits).

Also on record: `exp/walker_actorfix/qnorm` (actor Q-normalisation on Density-RLDP) is a
dead end — 12–24 return against the 203–384 baseline.

## Files

- `analysis/transfer_table/{routeA,routeB}.jsonl`, `scripts/{score_transfer_table.sh,score_transfer_routeB.sh,transfer_table_report.py}`
- `exp/walker_dfb/`, `exp/paper_eval_walker_dfb.jsonl`, `scripts/{launch_walker_dfb.sh,score_walker_dfb.sh,walker_report.py}`
- `exp/hub/exp/` (pulled checkpoints), `exp/paper_eval_walker_feat_full.jsonl` (objective-only retrofits, scoring)
- Paper: `tab:reward-transfer` rewritten with all four methods on both routes; surrounding text updated.
