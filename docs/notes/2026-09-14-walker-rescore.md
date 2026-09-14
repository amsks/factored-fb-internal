# Walker re-score after the reward-inference repair, and the 2026-09-14 walker block

## Job 1 — re-scoring the EXISTING 500k checkpoints with the current code

Question: how much of Density-RLDP's 362 (`docs/notes/2026-09-04-exorl-rldp-hilp-results-table.md`)
is explained by the 2026-09-11 reward-inference repair alone (`docs/notes/2026-09-11-rldp-hilp-ce-audit.md`:
the task vector is now the least-squares solve of the reward samples on the backward features,
not the reward-weighted feature mean). Nothing was retrained; the same `params_500000.pkl`
files under `exp/exorl-tdj/dmc-walker/{rldp_density_tdj,rldp_tdj}/` were scored again.
`rldp_tdj` is the control: its inference path was not touched by the repair, so it should
reproduce the old numbers up to eval noise.

Protocol (identical to the one that produced `exp/paper_eval_tdj_final.jsonl`): driver
`scripts/exorl_paper_eval.py --root exp/exorl-tdj --critic <c> --actor ddpgbc --domains dmc-walker
--seeds 0 1 2 --steps 500000 --episodes 10 --task-set tdjepa`, i.e. per (task, seed) one
`scripts/reeval_checkpoint.py` call: agent rebuilt from the run's own config.json, restore
params only, 10,000 reward-relabelled transitions for the task-vector inference (the td_jepa
count, `INFERENCE_SAMPLES`), `--eval_on_inferred_cond`, 10 episodes, tasks stand / walk / run /
spin, overrides discount=0.98, measure_hit_spec="0:24:15.9" (unused by these two critics),
actor.alpha=0 / q_normalize=false / q_noise_std=0.2 / boot_noise_std=0.2, rldp_official_nets=false
(legacy-arch pin). GPU 2, 2026-09-14 01:53–02:20Z. Output `exp/paper_eval_tdj_rescore_20260914.jsonl`
(24 rows, one per cell, no duplicates); logs `logs/walker/rescore_*.log`.
The old checkpoints load without any change: the repair touched only `infer_cond` (no new
parameter group in 'feature' mode), so no state-group mismatch arises.

Numbers are episode return, mean over 10 episodes per cell; "old" = `exp/paper_eval_tdj_final.jsonl`,
"new" = the re-score. Seed rows; last row mean ± population sd over 3 seeds (the convention of the
2026-09-04 table).

### rldp_density_tdj

| seed | stand old / new | walk old / new | run old / new | spin old / new | mean(4) old / new |
|---|---|---|---|---|---|
| s0 | 305 / 736 | 269 / 324 | 66 / 177 | 161 / 725 | 200 / 491 |
| s1 | 437 / 848 | 398 / 237 | 131 / 157 | 607 / 731 | 394 / 493 |
| s2 | 590 / 754 | 404 / 241 | 150 / 137 | 829 / 874 | 493 / 501 |
| **mean ± sd over seeds** | 444 ± 116 / 779 ± 49 | 357 ± 62 / 267 ± 40 | 116 ± 36 / 157 ± 17 | 532 ± 278 / 777 ± 69 | **362** / **495** (n=3) |

### rldp_tdj

| seed | stand old / new | walk old / new | run old / new | spin old / new | mean(4) old / new |
|---|---|---|---|---|---|
| s0 | 850 / 849 | 711 / 735 | 380 / 375 | 861 / 931 | 700 / 723 |
| s1 | 773 / 780 | 451 / 451 | 249 / 261 | 199 / 210 | 418 / 426 |
| s2 | 761 / 777 | 622 / 601 | 251 / 244 | 778 / 792 | 603 / 603 |
| **mean ± sd over seeds** | 795 ± 39 / 802 ± 34 | 595 ± 108 / 596 ± 116 | 293 ± 61 / 294 ± 58 | 613 ± 294 / 644 ± 312 | **574** / **584** (n=3) |

Reading: the control moves 574 → 584 (every cell within eval noise; the largest move is spin s0
861 → 931 on 10 episodes), so the eval path itself is unchanged. Density-RLDP moves 362 → 495:
stand 444 → 779 and spin 532 → 777 recover most of the gap to plain RLDP (802 / 644), run
116 → 157 improves slightly, walk 357 → 267 gets worse. So roughly 130 of the 212-point gap
(362 vs 574) is the reward-inference coefficient alone; the remaining ~90 (495 vs 584) is in the
trained networks and is what job 2 measures with the regression task index.

## Job 2 — new walker block (12 runs), launched 2026-09-14 01:58Z

Save dir `exp/walker_20260914/`, tmux session `walker` (one window per GPU 2–7, two runs queued
per window, seed 0 of all four arms first), logs `logs/walker/<critic>_s<seed>.log`, launcher
`logs/walker/run_one.sh`. Command per run (mirrors `scripts/launch_exorl_official.sh`'s td_jepa
branches): `impls.main --critic <c> --actor ddpgbc --domain dmc-walker --seed <s> --train_steps 500000
--eval_interval 100000 --save_interval 100000 --eval_episodes 10 --infer_cond_episodes 10000
--eval_on_inferred_cond` + overrides discount=0.98, the official-parity actor block
(motivo_archi, hidden_dims [1024], alpha 0, q/boot noise 0.2 clip 0.3, q_normalize false), and on the
two RLDP arms rldp_phase1_steps=2000000 (inert, two_phase false), measure_hit_spec="0:24:15.9",
ortho_coef=1. Final numbers are to be scored exactly as job 1 (`exorl_paper_eval --root
exp/walker_20260914 ... --task-set tdjepa`); the in-training eval rows (every 100k, task walk) are a
progress signal only — the 2026-08-31 arms had in-training eval at step 1 only.

Arms:
- (a) `rldp_tdj` + ortho_coef=1
- (b) `rldp_density_tdj_reg` = `rldp_density_tdj.yaml` + ortho_coef=1 + the regression task index
  (`impls/configs/critic/rldp_density_tdj_reg.yaml`; alias in `impls/agents/__init__.py`)
- (c) `hilp_tdj` as shipped
- (d) `hilp_density_tdj_reg` = `hilp_density_tdj.yaml` + the regression task index
  (`impls/configs/critic/hilp_density_tdj_reg.yaml`; alias)

Walker hit spec for the goal-reaching kernel: "0:24:15.9" — the Density-FB walker arm's
(`env/domains/dmc-walker.yaml`: full 24-dim observation block, radius 15.9 calibrated to
cube-single's random-pair hit rate). It is quoted in both yamls and in the override; the old
`rldp_density_tdj` runs had recorded it unquoted as the sexagesimal 1455.9 (harmless there, that
arm never read it).

Recorded config.json values, checked on all six running runs (seeds 0/1 of a, b; seed 0 of c, d):
- rldp_tdj: ortho_coef 1, discount 0.98, no task-index key (plain RLDP), measure_hit_spec 0:24:15.9
- rldp_density_tdj_reg: ortho_coef 1, rldp_density_task_index regression, windowed_goal_sampling true,
  measure_hit_spec 0:24:15.9, discount 0.98, temp 1.0, actor_q_center true, train_goal_ratio 0.5
- hilp_tdj: discount 0.98, hilp_discount 0.98, hilp_q_loss false
- hilp_density_tdj_reg: hilp_density_task_index regression, hilp_goal_sampling true,
  measure_hit_spec 0:24:15.9, discount 0.98, hilp_discount 0.98, temp 1.0, actor_q_center true
- all: actor motivo_archi true, alpha 0.0, lr_actor 3e-4 (the actor yaml's value, as in the
  2026-08-31 arms — the F1 note in the 2026-09-04 table still applies), batch 1024, z_dim 50,
  train_steps 500000 (command line; not a config.json key).

Discount note: the request said 0.99; the shipped `hilp_tdj.yaml` and the td_jepa DMC launcher
use 0.98 (the old hilp_density_tdj bug was 0.96 + 1M, fixed here to 0.98 + 500k). 0.98 was kept so
all four arms and td_jepa's own reference share one discount.

Smoke before launch: each of the four configs ran 300 steps (eval at steps 1 and 300, 1 episode):
all training losses finite, eval rows written. One code fix was needed for the two
`_reg` arms: in regression mode `infer_cond` received the dataset's rewards as [M, 1] and the
centered solve read that as M=1 (`impls/critics/{rldp,hilp}_density.py::infer_cond`, now flattens a
trailing singleton axis). 54 tests in test_rldp_density_levers / _inference and
test_hilp_density_levers / _contracts pass after the change.

Progress at 02:21Z:
- dmc-walker_hilp_density_tdj_reg_ddpgbc_seed_0: step 100000, eval rows 1
- dmc-walker_hilp_tdj_ddpgbc_seed_0: step 80000, eval rows 1
- dmc-walker_rldp_density_tdj_reg_ddpgbc_seed_0: step 30000, eval rows 1
- dmc-walker_rldp_density_tdj_reg_ddpgbc_seed_1: step 40000, eval rows 1
- dmc-walker_rldp_tdj_ddpgbc_seed_0: step 30000, eval rows 1
- dmc-walker_rldp_tdj_ddpgbc_seed_1: step 40000, eval rows 1
- queued after these: hilp_tdj s1 (GPU 2), hilp_density_tdj_reg s1 (3), rldp_tdj s2 (4),
  rldp_density_tdj_reg s2 (5), hilp_tdj s2 (6), hilp_density_tdj_reg s2 (7).
