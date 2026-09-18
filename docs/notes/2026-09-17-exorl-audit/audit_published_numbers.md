# Published zero-shot RL numbers on ExoRL (RND buffer, state observations) vs ours — audit 2026-09-17

Read-only audit. Sources were read from arXiv HTML (Touati 2209.14935, HILP 2402.15567, TD-JEPA 2510.00739;
text dumps in this scratchpad: `txt2_*.txt`), from the local RLDP PDF
(`/home/adityamohan/git/baselines/rldp/RLDP_full_paper.pdf`, text dump `txt_rldp.txt`), and from the
reference launchers under `/home/adityamohan/git/baselines/`. All published numbers below are episode
returns (max 1000 per episode) on the ExoRL RND exploration buffer with state observations unless flagged.

## A. Per-source extraction

### A1. Touati, Rapin, Ollivier, "Does Zero-Shot Reinforcement Learning Exist?", ICLR 2023 (arXiv 2209.14935)

Per-task table: **Appendix H.2, Table 2** ("Score of each method, split by task and replay buffer. Average over
ten random seeds, with ± estimated standard deviation on this average"). RND block, mean ± std over 10 seeds.
Columns: Rand, AEnc, ICM, Latent, Trans, Lap, LRA-P, CL, LRA-SR, FB, APS (APS trains only on its own buffer, so
APS on RND is "-").

| domain | task | FB | Lap (SF-Laplacian) | ICM (SF-ICM) | LRA-SR | Trans |
|---|---|---|---|---|---|---|
| walker | stand | 890 ± 15 | 836 ± 38 | 813 ± 10 | 828 ± 20 | 853 ± 18 |
| walker | walk | 760 ± 19 | 748 ± 75 | 518 ± 74 | 853 ± 15 | 607 ± 31 |
| walker | run | 388 ± 8 | 299 ± 20 | 251 ± 11 | 350 ± 14 | 250 ± 6 |
| walker | **flip** | 578 ± 10 | 569 ± 30 | 461 ± 9 | 454 ± 23 | 399 ± 19 |
| cheetah | run | 247 ± 9 | 50 ± 5 | 96 ± 8 | 138 ± 17 | 66 ± 4 |
| cheetah | run-backward | 185 ± 17 | 90 ± 7 | 160 ± 22 | 82 ± 8 | 143 ± 17 |
| cheetah | walk | 827 ± 41 | 330 ± 65 | 401 ± 40 | 446 ± 36 | 286 ± 30 |
| cheetah | walk-backward | 793 ± 66 | 499 ± 40 | 743 ± 60 | 352 ± 45 | 572 ± 100 |
| quadruped | jump | 651 ± 8 | 490 ± 48 | 176 ± 13 | 731 ± 9 | 806 ± 14 |
| quadruped | run | 429 ± 3 | 399 ± 26 | 98 ± 13 | 461 ± 6 | 478 ± 8 |
| quadruped | stand | 815 ± 2 | 720 ± 27 | 426 ± 45 | 944 ± 11 | 973 ± 2 |
| quadruped | walk | 528 ± 10 | 410 ± 30 | 90 ± 13 | 516 ± 30 | 552 ± 33 |
| maze | reach (20 goals, goal given) | 710 ± 8 | 707 ± 12 | 4 ± 1 | 532 ± 20 | 8 ± 4 |

Domain means (FB): walker 654.0, cheetah 513.0, quadruped 605.8. (Lap: 613.0 / 242.2 / 504.8.)
The APS- and Proto-buffer blocks of the same table exist (e.g. Proto walker walk FB 917 ± 7, APS cheetah run FB
267 ± 33) and are NOT RND numbers. The headline Figure 1 / Figure 2 scores average over all three buffers and are
normalized by offline TD3; they are not comparable to raw returns.

Protocol (Section 6 + Appendix G.5 Table 1, "Hyperparameters used in our experiments"):
- Buffer: ExoRL RND / APS / Proto; replay buffer size **5 × 10^6 transitions (10 × 10^6 for maze)** = 5,000 episodes of 1,000 steps.
- Observations: states. No pixel results.
- Gradient steps **10^6**; batch **1024**; discount **0.98 (0.99 maze)**; Adam lr **1e-4**; z dimension **50 (100 maze)**;
  z-sampling mixing ratio 0.5; target-network momentum **0.99** (i.e. tau 0.01); policy smoothing std 0.2, clip 0.3;
  orthonormality regularization weight 1; **reward labels for task inference 10^4**; seeds **10**.
- Eval episodes per task: **not stated in the paper**. The controllable_agent / url_benchmark code lineage
  (also used by HILP's zsrl fork, `hilp_zsrl/url_benchmark/train_offline.py:73,75`) defaults to 10 eval episodes
  during training and 50 for the final eval.
- Maze: "a known goal is presented and used to set z directly" — Touati's maze number is a goal-Dirac number, not a
  reward-inference number, and it is a 20-goal average, not the 4 reach_* tasks.
- Hyperparameters tuned by hand on walker + RND only, then shared across all methods and tasks (no per-method sweep).
- Walker task is **flip** ("walker_flip reward includes a component encouraging some minimal angular momentum", App. G.1).
- Architecture (App. G.2): F preprocesses (s,a) and (s,z) with 2×1024 to 512, then 2×1024 trunk; B is 3×256; actor same shape as F.

The compendium's "walker walk 760 ± 19 / cheetah run 247 ± 9" are exactly this table's RND-buffer FB cells (walker walk,
cheetah run). Same setting as everything else here, just single cells.

### A2. TD-JEPA — Bagatella, Pirotta, Touati, Lazaric, Tirinzoni, "TD-JEPA: Latent-predictive Representations for Zero-Shot Reinforcement Learning" (arXiv 2510.00739)

**Table 1** (Section 5.1). Two DMC blocks: the first is labelled DMC_RGB (pixels), the second, unlabelled block is
proprioception. Mean ± **standard error** over 5 seeds. Domain means only; no per-task table anywhere in the paper.

Proprioceptive block:

| domain | Laplacian | ICVF* | HILP | FB | RLDP | BYOL* | BYOL-γ* | TD-JEPA |
|---|---|---|---|---|---|---|---|---|
| DMC avg | 591.1 ± 10.7 | 619.3 ± 10.3 | 620.1 ± 8.4 | 648.2 ± 4.1 | 610.2 ± 13.5 | 618.6 ± 10.5 | 645.4 ± 10.5 | 661.2 ± 6.3 |
| walker | 769.7 ± 4.7 | 727.0 ± 16.2 | 796.4 ± 7.7 | 811.5 ± 5.9 | 723.9 ± 18.3 | 746.8 ± 11.0 | 786.1 ± 9.6 | 785.2 ± 6.7 |
| cheetah | 614.5 ± 18.9 | 606.3 ± 16.8 | 618.3 ± 5.8 | 672.7 ± 4.9 | 575.6 ± 44.9 | 622.8 ± 23.9 | 647.2 ± 9.0 | 688.7 ± 6.7 |
| quadruped | 635.0 ± 38.7 | 708.5 ± 14.2 | 694.8 ± 11.0 | 595.6 ± 9.1 | 665.0 ± 13.9 | 611.8 ± 28.1 | 683.1 ± 26.1 | 691.4 ± 5.0 |
| pointmass | 345.1 ± 22.4 | 435.5 ± 11.1 | 371.0 ± 37.1 | 513.0 ± 20.0 | 476.3 ± 39.4 | 493.0 ± 41.3 | 465.1 ± 17.6 | 479.3 ± 23.6 |

Pixel block (DMC_RGB), for the record: FB walker 324.4 ± 16.6, cheetah 622.4, quadruped 475.4, pointmass 402.8;
TD-JEPA 738.9 / 706.0 / 626.7 / 443.7. Do not mix with the proprio block.
Table 2 (encoder-depth ablation) gives FB proprio walker 788.7 / 811.5 / 812.4 for no / shallow / deep encoder.

Task sets (`metamotivo/envs/dmc_tasks/__init__.py:6-18`): walker **stand, walk, run, spin**; cheetah walk,
walk_backward, run, run_backward; quadruped stand, walk, run, jump; pointmass **8 tasks** (reach_top_left,
reach_top_right, reach_bottom_left, reach_bottom_right, reach_bottom_left_long, loop, square, fast_slow).
So the walker mean includes spin, not flip, and the pointmass mean is over 8 tasks, not our 4.

Protocol:
- Paper App. E.7: "All evaluation metrics are averaged over 10 episodes in OGBench, and **20 in DMC**, and computed
  over 10 random seeds for OGBench and **5 for DMC**." Launcher (`scripts/train/proprio/launch_fb_dmc.py:187-188`)
  says `num_episodes: 10`, `num_inference_samples: 10_000`. **Paper (20) and code (10) disagree on eval episodes.**
- Steps: paper **Table 4 says "Total gradient steps 2M"** for DMC; launcher `num_train_steps: 3_000_000`
  (`launch_fb_dmc.py:16`, same in rldp/td_jepa/td3 DMC launchers). **Paper (2M) and code (3M) disagree.**
- Buffer: `load_n_episodes: 5_000` from `<domain>/rnd/buffer` (`metamotivo/data_loading/dmc.py:15,19`) = 5,000 RND episodes.
- Task inference: 10,000 reward-labelled samples, z by linear regression / reward-weighted B (Sec. 3.3).
- Hyperparameters (Table 4, DMC proprio): discount 0.98, batch 1024, lr 1e-4, ortho coef 1.0 default, EMA 0.001
  (`f_target_tau = b_target_tau = 0.001` in launcher), z dim 50, z-mix 0.5, SF/F 3×1024, task encoder 2×256, actor 3×256,
  explicit state encoder of depth 0 (linear, 256-dim) in DMC.
- Selection (App. E.5): every method swept over 6 configs in DMC (`lr_b ∈ {1e-4, 1e-5}` × `ortho_coef ∈ {0.1, 1, 10}`
  in the launcher; Table 5 lists [0.01, 0.1, 1]) and **the best configuration per domain is reported**. Pointmass
  additionally uses discount 0.99 and actor lr 1e-6. Checkpoint = end of training (eval every 250k, see our
  replication note); the paper does not describe picking the best checkpoint over training.
- RLDP inside TD-JEPA is re-implemented in their framework (encoder trained in parallel, prediction horizon 5).

### A3. RLDP paper (`/home/adityamohan/git/baselines/rldp/RLDP_full_paper.pdf`, Table 1, PDF p.6-7)

Mean ± std over **4 seeds**, raw returns. Columns Laplace / FB / HILP / PSM / RLDP. Verified against the PDF text
dump; matches the transcription in `docs/notes/2026-08-28-rldp-paper-eval.md` §5.

| domain | task | Laplace | FB | HILP | PSM | RLDP |
|---|---|---|---|---|---|---|
| Walker | Stand | 243.70 ± 151.40 | 902.63 ± 38.94 | 607.07 ± 165.28 | 872.61 ± 38.81 | 877.69 ± 45.03 |
| Walker | Run | 63.65 ± 31.02 | 392.76 ± 31.29 | 107.84 ± 34.24 | 351.50 ± 19.46 | 324.85 ± 54.57 |
| Walker | Walk | 190.53 ± 168.45 | 877.10 ± 81.05 | 399.67 ± 39.31 | 891.44 ± 46.81 | 790.94 ± 67.55 |
| Walker | **Flip** | 48.73 ± 17.66 | 206.22 ± 162.27 | 277.95 ± 59.63 | 640.75 ± 31.88 | 491.64 ± 37.30 |
| Cheetah | Run | 96.32 ± 35.69 | 257.59 ± 58.51 | 68.22 ± 47.08 | 244.38 ± 80.00 | 236.31 ± 20.75 |
| Cheetah | Run Backward | 106.38 ± 29.40 | 307.07 ± 14.91 | 37.99 ± 25.16 | 296.44 ± 20.14 | 322.08 ± 39.28 |
| Cheetah | Walk | 409.15 ± 56.08 | 799.83 ± 67.51 | 318.30 ± 168.42 | 984.21 ± 0.49 | 895.31 ± 49.84 |
| Cheetah | Walk Backward | 654.29 ± 219.81 | 980.76 ± 2.32 | 349.61 ± 236.29 | 979.01 ± 7.73 | 984.76 ± 0.85 |
| Quadruped | Stand | 854.50 ± 41.47 | 740.05 ± 107.15 | 409.54 ± 97.59 | 842.86 ± 82.18 | 794.94 ± 43.25 |
| Quadruped | Run | 412.98 ± 54.03 | 386.67 ± 32.53 | 205.44 ± 47.89 | 431.77 ± 44.69 | 457.41 ± 74.70 |
| Quadruped | Walk | 494.56 ± 62.49 | 566.57 ± 53.22 | 218.54 ± 86.67 | 603.97 ± 73.67 | 465.40 ± 185.29 |
| Quadruped | Jump | 642.84 ± 114.15 | 581.28 ± 107.38 | 325.51 ± 93.06 | 596.37 ± 94.23 | 733.32 ± 55.30 |
| Pointmass | Top Left | 713.46 ± 58.90 | 897.83 ± 35.79 | 944.46 ± 12.94 | 831.43 ± 69.51 | 890.41 ± 60.79 |
| Pointmass | Top Right | 581.14 ± 214.79 | 274.95 ± 197.90 | 96.04 ± 166.34 | 730.27 ± 58.10 | 795.47 ± 21.10 |
| Pointmass | Bottom Left | 689.05 ± 37.08 | 517.23 ± 302.63 | 192.34 ± 177.48 | 451.38 ± 73.46 | 805.17 ± 20.44 |
| Pointmass | Bottom Right | 21.29 ± 42.54 | 19.37 ± 33.54 | 0.17 ± 0.29 | 43.29 ± 38.40 | 193.38 ± 167.63 |

Domain means: FB 594.7 / 586.3 / 568.6 / 427.3; HILP 348.1 / 193.5 / 289.8 / 308.3; RLDP 621.3 / 609.6 / 612.8 / 671.1;
PSM 689.1 / 626.0 / 618.7 / 514.1 (walker / cheetah / quadruped / pointmass).

Protocol (Sec. 5 p.6, 5.1.1 p.7, App. A.3.1 p.15, code `examples/rldp_train_dmc.py`):
- Data: "ExoRL suite ... exploratory datasets using RND"; code `load_n_episodes: 5_000`; states; episode length 1000.
- Steps: paper "train representations for **2 million** steps and train policy for additional **3 million** steps"
  (5M); code defaults `representation_steps=3_000_000`, `num_train_steps=6_000_000`. **Paper (5M) and code (6M) disagree.**
- Eval: paper "querying it for different task-rewards for **50 episodes**"; code default `num_eval_episodes: 10`. **Disagree.**
- Inference samples: paper silent; code `num_inference_samples: 50_000`.
- Seeds 4, mean ± std. Bold = within one std of the per-task best.
- Representation dim: "best for each method" (RLDP swept {64..1024}); Table 7 (p.22) redoes FB/PSM/RLDP at d=512
  where FB walker flip is 977.08 (vs 206.22 in Table 1) — the Table 1 FB flip cell is an outlier of their own sweep.
- Discount 0.98 (0.99 pointmass), lr 1e-4 (pointmass lr_b = lr_actor = 1e-6), horizon 5, ortho 1.
- Walker task is **flip** (Table 4, p.15: "stand walk run flip").

### A4. HILP — Park, Kreiman, Levine, "Foundation Policies with Hilbert Representations", ICML 2024 (arXiv 2402.15567)

**Appendix B, Table 3** "Full results on the state-based ExORL benchmark (4 seeds)", unnormalized return, mean ± std.
Datasets APS / APT / Proto / RND; environments Walker, Cheetah, Quadruped, **Jaco** (no point-mass maze). RND rows:

| env | task | GC-TD3 | Lap | FDM | FB | HILP |
|---|---|---|---|---|---|---|
| Walker | **Flip** | 298 ± 110 | 563 ± 40 | 416 ± 20 | 548 ± 94 | 563 ± 136 |
| Walker | Run | 166 ± 27 | 311 ± 29 | 295 ± 70 | 409 ± 15 | 401 ± 30 |
| Walker | Stand | 812 ± 133 | 844 ± 35 | 821 ± 84 | 866 ± 120 | 800 ± 61 |
| Walker | Walk | 703 ± 275 | 811 ± 77 | 476 ± 259 | 811 ± 52 | 855 ± 34 |
| Cheetah | Run | 65 ± 66 | 124 ± 32 | 107 ± 26 | 183 ± 83 | 262 ± 53 |
| Cheetah | Run Backward | 50 ± 35 | 113 ± 37 | 177 ± 62 | 153 ± 41 | 187 ± 55 |
| Cheetah | Walk | 281 ± 116 | 571 ± 111 | 494 ± 122 | 636 ± 291 | 823 ± 141 |
| Cheetah | Walk Backward | 146 ± 51 | 536 ± 92 | 652 ± 274 | 677 ± 85 | 843 ± 184 |
| Quadruped | Jump | 639 ± 106 | 508 ± 182 | 758 ± 98 | 642 ± 36 | 556 ± 101 |
| Quadruped | Run | 435 ± 38 | 346 ± 81 | 491 ± 7 | 436 ± 26 | 393 ± 42 |
| Quadruped | Stand | 910 ± 44 | 681 ± 191 | 971 ± 11 | 797 ± 72 | 810 ± 97 |
| Quadruped | Walk | 470 ± 21 | 441 ± 88 | 601 ± 82 | 642 ± 202 | 542 ± 32 |
| Jaco | Reach BL / BR / TL / TR | 1/1/0/0 | 18/7/5/8 | 53/38/36/44 | 18/29/45/22 | 19/18/8/5 (HILP-G 46/55/47/38) |

(CL, Rand, IDM, AE columns also exist; omitted.) Domain means on RND: FB 658.5 / 412.3 / 629.3; HILP 654.8 / 528.8 /
575.3; Lap 632.3 / 336.0 / 494.0. The headline Figure 5 is an IQM over 4 envs × 4 tasks × 4 datasets × 4 seeds
(includes APS/APT/Proto and Jaco), so it is not an RND-only number. Table 4 is the pixel-based version (64×64×3).

Protocol (Sec. 6.1, App. D.1, App. D.2 Table 5 "Hyperparameters for zero-shot RL"):
- Data: "first 5M transitions from each dataset" (= 5,000 episodes); states (plus a separate pixel table).
- Gradient steps **10^6** (5 × 10^5 pixel); batch **1024**; lr 5e-4 for φ, 1e-4 others; TD3 target smoothing **0.01**;
  discount **0.98**; latent dim **50**; **10,000 state samples for latent inference**; Hilbert discount 0.96 (walker) /
  0.98; expectile 0.5; Q-loss for HILP/SF on Quadruped and Jaco, vector loss otherwise.
- Seeds 4 (Table 3 caption). Eval episodes: not stated in the paper; the zsrl code defaults to 10 during training and
  50 at the final eval (`hilp_zsrl/url_benchmark/train_offline.py:73,75`).
- Tuned per environment on RND for HILP, FB, Lap, GC-TD3; hyperparameters then reused across datasets.
- Walker task is **Flip**.

### A5. Repo notes cross-checked
- `docs/2026-08-25-exorl-humenv-compendium.md:38` "Touati et al. report walker walk 760 ± 19 and cheetah run 247 ± 9" =
  Touati Table 2, RND block, FB column. Same setting as A1; correct.
- `docs/notes/2026-09-04-exorl-rldp-hilp-results-table.md` and `docs/notes/2026-08-31-tdjepa-replication-screen.md`:
  td_jepa's own RLDP/HILP at 500k on walker (RLDP 763/509/226/917 spin, HILP 864/873/301/981 spin) are the authors'
  code run by us, not published cells. The note already records that td_jepa uses spin and RLDP-paper uses flip, with
  different reward definitions (spin = tolerance(angmomentum ≥ 10) alone; flip = tolerance(≥ 5) × standing).

### A6. Cross-source discrepancies for the same method on the same buffer
- **FB walker flip**: Touati 578 ± 10, HILP-paper 548 ± 94, RLDP-paper 206 ± 162 (RLDP-paper's own Table 7 at d=512 gives 977).
  RLDP-paper's headline FB flip is 2.7× below the other two sources.
- **FB cheetah**: walk-backward 793 (Touati) / 677 (HILP-paper) / 981 (RLDP-paper); run 247 / 183 / 258. Domain mean spans
  412 (HILP-paper, 1M) → 513 (Touati, 1M) → 586 (RLDP-paper, 5M) → 673 (TD-JEPA, 3M, best-of-6 configs).
- **FB walker mean**: 654 (Touati, flip) / 658 (HILP-paper, flip) / 595 (RLDP-paper, flip) / 811.5 (TD-JEPA, spin). The
  spin task alone is worth ~+150 on the walker mean (spin scores ~900–980 for any decent policy; flip 500–600).
- **HILP**: HILP's own RND rows give walker 655 / cheetah 529 / quadruped 575; RLDP-paper reports HILP at 348 / 194 / 290
  (1.9–2.7× lower, 4 seeds, 5M steps); TD-JEPA reports 796 (spin) / 618 / 695. RLDP-paper's HILP column is an outlier.
- **RLDP**: own paper 621 (flip) / 610 / 613 / 671 (4 reach tasks) vs TD-JEPA's re-implementation 724 (spin) / 576 / 665 /
  476 (8 pointmass tasks). Only the cheetah and quadruped cells are on the same task set; they differ by ±50.
- Within-source conflicts: TD-JEPA paper says 2M steps and 20 DMC eval episodes, its launcher says 3M and 10; RLDP paper
  says 5M steps and 50 episodes, its code defaults say 6M and 10.

## B. Comparison table

Columns: published FB/HILP/RLDP/PSM per task (three sources with per-task numbers), TD-JEPA domain means (no per-task
table exists), our Density-FB (2M, 5 seeds), our FB at the reference configuration (`exp/walker_fbref`, 1M, 3 seeds).
Last column = our Density-FB 2M divided by the best published FB cell for that task (per-task: max over Touati,
HILP-paper, RLDP-paper; domain rows: max over those three and TD-JEPA). Walker row 4 is **spin for ours and TD-JEPA,
flip for the other three sources** — not the same task.

| task | Touati FB (RND, Tab.2) | HILP-paper FB (RND, Tab.3) | HILP-paper HILP (RND, Tab.3) | RLDP-paper FB (Tab.1) | RLDP-paper HILP (Tab.1) | RLDP-paper RLDP (Tab.1) | RLDP-paper PSM (Tab.1) | TD-JEPA FB (Tab.1) | TD-JEPA HILP | TD-JEPA RLDP | TD-JEPA TD-JEPA | ours Density-FB 2M/5s | ours FB-ref 1M/3s | ours DFB / best published FB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| walker stand | 890 | 866 | 800 | 903 | 607 | 878 | 873 | — | — | — | — | 715 | 920 | 0.79 (vs 903) |
| walker walk | 760 | 811 | 855 | 877 | 400 | 791 | 891 | — | — | — | — | 641 | 804 | 0.73 (vs 877) |
| walker run | 388 | 409 | 401 | 393 | 108 | 325 | 352 | — | — | — | — | 204 | 400 | 0.50 (vs 409) |
| walker flip/spin | 578 | 548 | 563 | 206 | 278 | 492 | 641 | — | — | — | — | 976 | 985 | n/a (spin vs flip) |
| **walker mean** | **654** | **658** | **655** | **595** | **348** | **621** | **689** | **811.5** | **796.4** | **723.9** | **785.2** | **634** | **777** | **0.78** (vs 812) |
| cheetah run | 247 | 183 | 262 | 258 | 68 | 236 | 244 | — | — | — | — | 60 | — | 0.23 (vs 258) |
| cheetah run_backward | 185 | 153 | 187 | 307 | 38 | 322 | 296 | — | — | — | — | 96 | — | 0.31 (vs 307) |
| cheetah walk | 827 | 636 | 823 | 800 | 318 | 895 | 984 | — | — | — | — | 303 | — | 0.37 (vs 827) |
| cheetah walk_backward | 793 | 677 | 843 | 981 | 350 | 985 | 979 | — | — | — | — | 469 | — | 0.48 (vs 981) |
| **cheetah mean** | **513** | **412** | **529** | **586** | **194** | **610** | **626** | **672.7** | **618.3** | **575.6** | **688.7** | **232** | — | **0.34** (vs 673) |
| quadruped jump | 651 | 642 | 556 | 581 | 326 | 733 | 596 | — | — | — | — | 609 | — | 0.94 (vs 651) |
| quadruped run | 429 | 436 | 393 | 387 | 205 | 457 | 432 | — | — | — | — | 404 | — | 0.93 (vs 436) |
| quadruped stand | 815 | 797 | 810 | 740 | 410 | 795 | 843 | — | — | — | — | 795 | — | 0.98 (vs 815) |
| quadruped walk | 528 | 642 | 542 | 567 | 219 | 465 | 604 | — | — | — | — | 467 | — | 0.73 (vs 642) |
| **quadruped mean** | **606** | **629** | **575** | **569** | **290** | **613** | **619** | **595.6** | **694.8** | **665.0** | **691.4** | **569** | — | **0.90** (vs 629) |
| maze top_left | — | — | — | 898 | 944 | 890 | 831 | — | — | — | — | 451 | — | 0.50 (vs 898) |
| maze top_right | — | — | — | 275 | 96 | 795 | 730 | — | — | — | — | 124 | — | 0.45 (vs 275) |
| maze bottom_left | — | — | — | 517 | 192 | 805 | 451 | — | — | — | — | 317 | — | 0.61 (vs 517) |
| maze bottom_right | — | — | — | 19 | 0 | 193 | 43 | — | — | — | — | 315 | — | 16.26 (vs 19) |
| **maze mean** | — | — | — | **427** | **308** | **671** | **514** | **513.0** | **371.0** | **476.3** | **479.3** | **302** | — | **0.59** (vs 513) |
Notes on the table:
- Maze "mean" ratio against a same-task-set FB: ours 302 vs RLDP-paper FB 427 on the same 4 reach tasks = **0.71**
  (the 0.59 in the table uses TD-JEPA's 8-task pointmass mean, which is not the same task set). Our maze
  bottom_right 315 is far above every published method except RLDP (193) — the published FB/HILP/PSM cells are 0–43.
- Walker mean, same task set as TD-JEPA (spin): ours Density-FB 634 vs TD-JEPA FB 811.5 (0.78); ours FB-ref 777 (0.96);
  ours Density-FB 1M/3 seeds 681 (0.84).
- Other in-house walker rows for reference (500k, 3 seeds, spin): our HILP re-implementation 879/884/338/972 = 768;
  our RLDP re-implementation 795/595/293/613 = 574; our FB at the default config 357/49/16/853 = 319.
- Our Density-FB 1M/3 seeds cheetah 62/84/262/380 = 197; the 2M/5-seed run is 232.
- Per-task, our FB-ref 1M is at published-FB level on walker: stand 920 (best published 903), walk 804 (Touati 760,
  HILP-paper 811, RLDP-paper 877), run 400 (best published 409). Density-FB 2M is 0.73–0.79 of the best published FB on
  walker stand/walk and 0.50 on run.
- Cheetah is where we are far below every source: 0.23–0.48 of the best published FB per task, 0.34–0.45 of the FB
  domain mean, and below even the weakest published FB (HILP-paper 412).
- Quadruped is at published FB level (0.90–1.00 of the best published domain mean; ties RLDP-paper FB 569 exactly).

## C. Protocol mismatches (ours: 2M gradient steps, 5 seeds, 10 eval episodes, 50k reward-labelled samples for inference,
5,000-episode RND buffer, state observations, walker spin, discount 0.98, maze tasks = the 4 reach_* tasks by reward inference)

vs Touati (ICLR 2023):
- steps 2M vs 1M (we train 2× longer); seeds 5 vs 10; inference samples 50k vs 10k; eval episodes 10 vs unstated
  (code lineage 10 / 50 final); buffer 5M transitions same (Touati maze uses 10M — ours unknown, check); walker
  spin vs flip; maze: Touati sets z from a known goal over 20 goals — not reward inference, not the 4 reach tasks;
  target momentum 0.99 (tau 0.01) vs whatever our config uses; single hand-tuned config shared across methods.

vs TD-JEPA (2025):
- steps 2M vs 3M in the launcher (paper Table 4 says 2M — cite the paper if we want parity, but the code says 3M);
  seeds 5 = 5; eval episodes 10 vs 20 in the paper (10 in the launcher); inference samples 50k vs 10k; buffer 5,000 RND
  episodes same; walker spin same; pointmass 4 tasks vs their 8 (their pointmass mean is not comparable to ours);
  their number is the best of 6 configurations per domain (lr_b × ortho), ours a single config; they pass states
  through an explicit 256-dim linear state encoder; EMA 0.001; batch 1024; standard error not std.

vs RLDP paper:
- steps 2M vs 5M (2M representation + 3M policy; code default 6M); seeds 5 vs 4; eval episodes 10 vs 50 in the paper
  (code default 10); inference samples 50k = 50k (code); buffer 5,000 RND episodes same; walker spin vs flip; maze
  4 reach tasks same; discount 0.98 / 0.99 pointmass same; representation dim chosen per method from a sweep;
  RLDP's ortho regularizer was found dead in the shipped code (memory note 2026-08-31), so their RLDP column is
  whatever the paper's runs actually did.

vs HILP paper:
- steps 2M vs 1M; seeds 5 vs 4; inference samples 50k vs 10k; eval episodes 10 vs unstated (code 10 / 50 final);
  buffer first 5M transitions same; walker spin vs flip; no point-mass maze (Jaco instead); batch 1024, tau 0.01,
  discount 0.98, latent 50; tuned per environment on RND; HILP/SF use the Q-loss variant on quadruped.

## D. Task naming, buffers, observations
- **Walker fourth task**: Touati = flip; HILP paper = Flip; RLDP paper = flip; TD-JEPA = **spin** (code
  `ALL_TASKS["walker"] = ["stand","walk","run","spin"]`; the paper text only says "e.g. walk, run, flip" loosely). Ours =
  spin. So our walker 4-task mean is comparable only to TD-JEPA's; against Touati / HILP / RLDP compare stand, walk, run only.
- **Buffers**: Touati Table 2 reports RND, APS and Proto blocks (headline figures average over all three; APS method
  only on the APS buffer); HILP Table 3 reports APS, APT, Proto, RND (headline Figure 5 aggregates all four) plus a
  pixel-based Table 4; RLDP = RND only; TD-JEPA = RND only (`dataset_expl_agent: "rnd"`) with a separate RGB block.
  All numbers used in Table B above are RND-buffer, state-observation cells.
- **Observations**: all cells in Table B are proprioceptive states. TD-JEPA's DMC_RGB block and HILP's Table 4 are
  pixels and were excluded; TD-JEPA's FB walker 324.4 is the pixel cell, not the state cell (811.5).
- **Maze**: Touati's maze = 20 goal-reaching tasks with the goal given (no reward inference); RLDP-paper and ours =
  4 reach_* tasks by reward inference; TD-JEPA = 8 pointmass tasks. Three different task sets.
