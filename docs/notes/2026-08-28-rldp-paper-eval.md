# RLDP official evaluation setup — extraction

Sources:
- Paper: `/home/adityamohan/git/baselines/rldp/RLDP_full_paper.pdf` — "Regularized Latent
  Dynamics Prediction is a Strong Baseline for Behavioral Foundation Models," anonymous,
  under review ICLR 2026, 25 pages. Extracted to plain text at
  `/tmp/claude-1001/.../scratchpad/rldp_paper_text.txt` via PyMuPDF (no `pdftoppm`/`pdftotext`
  installed in this environment, so the Read tool's native PDF path could not be used; text was
  extracted programmatically instead — page numbers below are PDF page numbers, i.e. the
  `===== PAGE N =====` markers, which are one greater than the printed "Under review..." page
  footer would suggest since page 1 has no footer number).
- Code: `/home/adityamohan/git/baselines/rldp/` (PyTorch). Key files: `README.md`,
  `examples/rldp_train_dmc.py` (the only end-to-end launch script in the repo — trains
  representations + zero-shot RL on ExoRL DMC domains), `rldp/rldp/agent.py` (`FBAgent`,
  training loop, losses), `rldp/rldp/model.py` (`FBModel`, architecture configs, reward/goal
  inference), `rldp/nn_models.py` (network building blocks), `rldp/buffers/buffers.py`,
  `rldp/misc/zbuffer.py`. There is **no D4RL or SMPL-humanoid launch script in this repo
  snapshot** — those setups are described only in the paper appendix, not runnable code here.

---

## 1. Benchmarks and domains

**ExoRL / DMC (main results, offline zero-shot RL, §5.1.1, Table 1; paper p.7 / PDF page 7):**
- Suite: ExoRL (Yarats et al. 2022) exploratory offline datasets on top of DeepMind Control
  Suite (Tassa et al. 2018) domains — **Walker, Cheetah, Quadruped, Pointmass**.
- Exploration buffer: **RND** (Burda et al. 2019) only. Paper Table 4 (PDF p.15) documents domain
  specs:
  | Domain | Type | Obs/Action dim | Tasks | Reward density (as labeled in Table 4) |
  |---|---|---|---|---|
  | Walker | Locomotion | 24/6 | stand, walk, run, flip | Dense |
  | Quadruped | Locomotion | 78/12 | jump, walk, run, stand | Dense |
  | Cheetah | Locomotion | 17/6 | walk, run, walk_backward, run_backward | Sparse |
  | Pointmass | Goal-reaching | 4/2 | reach_top_left, reach_top_right, reach_bottom_right, reach_bottom_left | Sparse |
  - Episode length 1000 for all DMC tasks (Table 4 caption, PDF p.15).
- Code (`rldp/examples/rldp_train_dmc.py:27-32`) — `ALL_TASKS` dict actually lists more Pointmass
  tasks than the paper's 4: `{"pointmass": ["reach_top_left","reach_top_right",
  "reach_bottom_right","reach_bottom_left","loop","square","fast_slow"]}`. Table 1 only reports the
  4 `reach_*` tasks, matching Table 4, so `loop`/`square`/`fast_slow` are defined but not in the
  reported eval set — **ambiguous whether they were used and dropped, or simply unused** (see §"could not find").
  Data path convention: `<dataset_root>/<domain_name>/<expl_agent>/buffer/*.npz`,
  `expl_agent="rnd"` by default (`load_data`, `rldp_train_dmc.py:92-120`;
  `TrainConfig.dataset_expl_agent: str = "rnd"`, line 137).
  Episodes loaded: `load_n_episodes: int = 5_000` (`TrainConfig`, line 139) — i.e. up to 5000 RND
  episode `.npz` files are read into a flat `DictBuffer`.

**SMPL 3D Humanoid (online zero-shot RL, §5.1.2, PDF p.7-8):**
- SMPL (Loper et al. 2023) full-body humanoid mimic environment. 358-dim proprioceptive obs,
  69-dim action space in [-1,1]. Episode length 300 (§A.3.2, PDF p.15).
- Representation-learning data: the **metamotivo 5M-transition dataset**
  (huggingface.co/facebook/metamotivo-M-1), which is "the final 5 million steps" of an FB-CPR
  agent trained online for 30M environment steps (§A.5.2, PDF p.20).
- Policy training uses **Conditional Policy Regularization (CPR)** from Tirinzoni et al. 2025
  (FB-CPR), which regularizes the policy toward expert human-demonstration trajectories to aid
  exploration in this high-dim online setting.
- Eval: full 45-task suite from Tirinzoni et al. 2025 (listed individually in Table 9, PDF p.24).

**D4RL / Gym-MuJoCo (low-coverage offline, §5.2, PDF p.8-9):**
- D4RL (Fu et al. 2020) OpenAI Gym MuJoCo domains: **halfcheetah, hopper, walker2d**, `medium`
  and `medium-expert` datasets (Table 5, PDF p.16): medium = 1e6 samples (early-stopped SAC
  rollout), medium-expert = 2e6 samples (50/50 mix of expert + suboptimal/random rollout).
  Appendix A.7.3 (Table 10, PDF p.23) additionally reports `medium-replay` variants as a
  supplementary low-coverage check.

## 2. Evaluation protocol

**Zero-shot task inference (reward → z):**
- Formula (Algorithm 1, §A.6, PDF p.21, and matching code `FBModel.reward_inference`,
  `rldp/rldp/model.py:128-136`): given N reward-labelled `(s, a, s')` transitions sampled from the
  offline dataset D, `z ← (1/N) Σ_i φ(s'_i) · r_task(s'_i)`, then projected onto the representation
  hypersphere (`project_z`, `model.py:117-120`: `z ← sqrt(d) · z/||z||`). Code implementation does
  **not** divide by N explicitly (`model.py:130-136` accumulates the raw sum `z += wr^T · B`) —
  harmless since the hyperspherical projection renormalizes direction only, but worth noting as a
  scale-vs-paper's-averaged-formula discrepancy.
- **Where the reward labels come from, concretely (DMC domains):** ExoRL's raw `.npz` files do
  **not** store task reward; `Workspace.reward_inference` (`rldp_train_dmc.py:315-334`) samples
  `num_inference_samples` transitions from the replay buffer, **re-simulates the environment
  physics** at each `(s', a)` via `env._physics.set_state`/`set_control` + `mujoco.mj_forward` /
  `mj_fwdPosition` / `mj_sensorVel` / `mj_subtreeVel`, then calls `env._task.get_reward(physics)`
  to get the ground-truth task reward for that transition. This is the "reward inference" sample
  set.
- **Number of reward-labelled samples:** `num_inference_samples: int = 50_000`
  (`TrainConfig`, `rldp_train_dmc.py:151`) — 50,000 transitions per z-inference call, one call per
  eval task per eval checkpoint.
- **D4RL / low-coverage setting:** reward inference instead uses "the corresponding reward
  function provided by each dataset" directly (§5.2, PDF p.9) — no physics replay needed since
  D4RL stores reward.
- **SMPL humanoid:** z-inference procedure not separately specified beyond the general Algorithm 1
  recipe; task specs come from the 45-task suite of Tirinzoni et al. 2025.

**Eval episodes:** Paper text states "querying it for different task-rewards for **50 episodes**"
(§5, PDF p.6, the general experimental-protocol paragraph that "For all datasets..."). **This
conflicts with the code default** `TrainConfig.num_eval_episodes: int = 10`
(`rldp_train_dmc.py:150`) — the shipped script's default runs 10 eval episodes/task, not 50. Not
resolved which one produced Table 1 (see "could not find").

**At what training step(s) numbers are reported:** `Workspace.eval` runs every
`eval_every_steps: int = 100_000` steps and also at `t=0` (`rldp_train_dmc.py:236-242`,
`eval()` at `t % eval_every_steps == 0`). The paper's headline Table 1 numbers are the
**final checkpoint** (after full training, i.e. representation_steps + policy steps completed —
"2 million [representation] steps and train policy for additional 3 million steps," §5.1.1, PDF
p.7) — not explicitly stated as "best over training" vs. "final only," but phrasing ("we train...
and train policy for additional...") implies **final-checkpoint reporting**, consistent with a
straight train-to-completion-then-eval protocol (no explicit best-checkpoint selection language
found anywhere in the paper).

**Online vs. offline:** DMC/ExoRL (§5.1.1) and D4RL (§5.2) are **fully offline** — representation
+ policy both trained from a fixed dataset, only the eval rollout is "online" execution.
SMPL humanoid (§5.1.2) uses an **offline representation phase** (2M steps on the metamotivo
dataset) followed by an **online policy phase** (20M environment steps, "no updates ... performed
on the representation network," §A.5.2, PDF p.20) using CPR/FB-CPR machinery — representation
learning stays offline throughout, only policy learning goes online here.

**Reward normalization:** D4RL uses D4RL's standard normalized-return convention (paper reports
"normalized returns," Table 2/10, standard 0=random/100=expert scaling implied but the exact
formula is not restated in-paper — it's the D4RL library default). SMPL humanoid: "normalized
scores with respect to fully-online trained representations and policy" i.e. normalized against
the FB-CPR oracle (§5.1.2, PDF p.7-8). DMC/ExoRL Table 1 numbers are **raw episode returns**, not
normalized.

## 3. Training config

**Shared architecture (F, B/encoder, actor) — from paper §A.4.1/A.5.1 (PDF p.16-17, 20) and code
(`rldp/rldp/model.py`, `rldp/nn_models.py`, `rldp/examples/rldp_train_dmc.py:37-89`):**

RLDP encoder φ/g (Figure 4, PDF p.16 = `EncoderMap` class, `nn_models.py:256-333`):
- φ (state net "zs"): feedforward MLP, 2 hidden layers × 256 units → d-dim embedding
  (`BaseMLP`, `l1`/`l2` hidden dim = `hidden_dim`; per-domain `hidden_dim` set via
  `agent_config.model.archi.b.hidden_dim = 256`, `rldp_train_dmc.py:63`).
- Action encoder "A": `nn.Linear(action_dim, 256)` (`za`, `nn_models.py:279`, `za_dim` default
  256, `BackwardArchiConfig.za_dim: int = 256`, `model.py:34`).
- g (state-action → next-latent, dynamics net): concatenate(φ(s), A(a)) → feedforward MLP 2
  hidden layers × 512 units → d-dim, hyperspherically normalized (`zsa = BaseMLP(zs_dim+za_dim,
  zsa_dim, hdim=512,...)`, `nn_models.py:280`), then linear layer `w` (`self.model =
  nn.Linear(zsa_dim, zs_dim)`, line 281) and final spherical re-normalization
  (`model_unroll`, lines 324-332: `sqrt(d) * normalize(...)`).
- Encoder kept **frozen after representation-training phase** ends (`update()` branches on
  `step < representation_steps`, `agent.py:172` vs. `else` FB/actor branch at line 199).

Forward map F(s,a,z) and actor π(s,z) (paper §A.5.1, PDF p.20, matching FB architecture of
Touati et al. 2023): two parallel embedding towers — `(s,z)` and `(s,a)` — each single
hidden layer of 1024 units → 512-dim, concatenated, then a further 1024-unit hidden layer, output
d-dim (F) or `action_dim`-dim with final Tanh (actor). Code: `ForwardArchiConfig(hidden_dim=1024,
hidden_layers=1, embedding_layers=2, num_parallel=2)`, `ActorArchiConfig(hidden_dim=1024,
hidden_layers=1, embedding_layers=2)` (`model.py:14-27`); F is an **ensemble of 2 parallel heads**
(`num_parallel=2` default) used for pessimism/uncertainty penalties.

**Per-domain overrides set in `create_agent()` (`rldp_train_dmc.py:37-89`):**
| Knob | Pointmass/Walker branch | Cheetah/Quadruped branch |
|---|---|---|
| `archi.z_dim` | 512 | 512 (both branches are identical in the current code — vestige of what was likely a per-domain sweep) |
| `lr_f` | 1e-4 | 1e-4 |
| `lr_b` | **1e-6** | 1e-4 |
| `lr_actor` | **1e-6** | 1e-4 |
| `discount` γ | **0.99** (pointmass only; not walker) | 0.98 |

Note: the discount branch is actually keyed only on `domain_name == "pointmass"` (0.99), all
others get 0.98 including Walker — the lr branch is keyed on `domain_name == "pointmass"` only too
(re-reading `rldp_train_dmc.py:69-76`: the `if domain_name == "pointmass":` branch sets
`lr_b=lr_actor=1e-6`; the `else` branch, covering walker/cheetah/quadruped, sets
`lr_b=lr_actor=1e-4`). Correcting the table above: only **Pointmass** gets the reduced
`lr_b=lr_actor=1e-6`; Walker/Cheetah/Quadruped all get `lr_f=lr_b=lr_actor=1e-4`. Only Pointmass
gets `discount=0.99`; Walker/Cheetah/Quadruped get `discount=0.98`.

Fixed across all DMC domains (`create_agent`, `rldp_train_dmc.py:56-87`):
- `archi.b.norm = True` (hyperspherical projection on encoder), `archi.norm_z = True`
- `archi.b.hidden_dim = 256`, `archi.b.hidden_layers = 2`
- `archi.f.hidden_dim = 1024`, `archi.f.hidden_layers = 1`
- `archi.actor.hidden_dim = 1024`, `archi.actor.hidden_layers = 1`
- `train.batch_size = 1024`
- `train.ortho_coef = 1` (**λ = 1**, matches paper's "we set the orthogonality regularization to
  1" for the main Table 1 results, §5.1.1 PDF p.7, and Fig. 3's finding that λ=1 performs best)
- `train.train_goal_ratio = 0.5` (mix of prior-sampled z vs. goal-encoded z for FB training)
- `train.fb_pessimism_penalty = 0`, `train.actor_pessimism_penalty = 0.5`

**Other `TrainConfig` (agent-level) defaults not overridden per-domain**
(`rldp/rldp/agent.py:16-36`):
- `fb_target_tau = 0.01` (target-network Polyak coefficient)
- `weight_decay = 0.0` (F/actor Adam); **encoder optimizer is AdamW with `weight_decay=1e-4`
  hardcoded** (`agent.py:66-71`, separate from the `TrainConfig.weight_decay` field)
- `clip_grad_norm = 0.0` (disabled)
- `stddev_clip = 0.3`, actor sampling std `Config.actor_std = 0.2` (`model.py:57`)
- `q_loss_coef = 0.0` (disabled — the optional USFA-style auxiliary Q-loss in `update_fb` is off
  by default)
- `update_z_every_step = 150`, `z_buffer_size = 10000` (used for online rollout z-mixing;
  not exercised in the offline `rldp_train_dmc.py` path since it has no online rollout loop)
- `representation_steps = 1_000_000` (agent-level default) — **overridden** by
  `TrainConfig.representation_steps: int = 3_000_000` at the `rldp_train_dmc.py` script level
  (`Workspace.__init__`, line 194) and by `enc_horizon` similarly (script default 5, line 145,
  overriding agent-level default of 5 anyway — same value).

**Script-level `TrainConfig` (the actual launch defaults, `rldp_train_dmc.py:131-169`):**
- `num_train_steps = 6_000_000` total
- `representation_steps = 3_000_000` (so representation phase = first 3M steps, FB+actor phase =
  remaining 3M steps by these code defaults)
- `encoder_hidden_dim = 512` → overrides `archi.z_dim` to 512 (Workspace.__init__ line 196) —
  consistent with the per-domain 512 already set in `create_agent`
- `encoder_norm = 1` (True) → overrides `archi.b.norm`
- `enc_horizon = 5` (encoding horizon H)
- `ortho_coef = 1.0` (redundant with `create_agent`'s own 1.0)
- `checkpoint_every_steps = 100_000`, `eval_every_steps = 100_000`, `log_every_updates = 10_000`

**Discrepancy vs. paper text:** Paper §5.1.1 (PDF p.7) states "We train representations for **2
million steps** and train policy for additional **3 million steps**" (5M total), whereas the
shipped script's defaults are `representation_steps=3_000_000` and `num_train_steps=6_000_000`
(so 3M repr + 3M policy = 6M total). The code defaults do not reproduce the paper's stated
step budget — the paper's numbers likely came from a different (undocumented) CLI invocation
overriding these two flags, not the file's literal defaults. **Flagging as unresolved** (see
below).

**D4RL / low-coverage config (§A.5.3, PDF p.20):**
- Representation pretraining: 1,000,000 gradient steps (§5.2, PDF p.8, "we first pretrain the
  representation learning methods on these datasets for 1 million gradient steps").
- Backward/encoder net: 2 hidden layers × 256 units → 512-dim embedding; RLDP encoding horizon
  **H = 1** here (not 5).
- Policy improvement uses a TD3+BC-style behavior-regularized loss (Eq. 14, PDF p.21):
  `L_P(π_z) = −λ ψ(s,a,z)ᵀz + ‖π_z(s) − a‖²`, `λ = α / (mean_i Q(s_i,a_i))`, **α = 2.5**
  (matching Fujimoto & Gu 2021's TD3+BC).

**SMPL humanoid config (§A.5.2, PDF p.20):**
- Representation pretraining: 2,000,000 gradient steps on the metamotivo 5M dataset; encoding
  horizon **H = 1**.
- Backward net B(s): 2-layer MLP, 256 hidden units → 256-dim embedding (matches the architecture
  of FB-CPR's own backward net, per the paper).
- Policy (FB-CPR/CPR machinery) trained online for 20,000,000 environment steps with the
  representation network **frozen** throughout.
- Oracle FB-CPR baseline: trained 30M environment steps, 5 seeds, results taken directly from
  Tirinzoni et al. 2025 (not re-run).

**Encoder architecture ablation defaults (§A.4, PDF p.16-19):** H swept over {1, 5, 10, 20} at
fixed λ=1 (Fig. 7); λ swept while H=5 fixed (Fig. 3, Fig. 5); paper settles on H=1 or H=5
depending on setting (as used above), not H=20 since higher H "results in slower encoder
training" (extra encoder forward passes per unroll step).

## 4. Seeds and statistics

- **DMC/ExoRL (Table 1, Table 3, Table 6, Table 7, Table 8) and SMPL humanoid (Table 9):**
  **4 seeds**, reported as **mean ± std** (paper states this explicitly, §5 PDF p.6: "All results
  are aggregated across 4 seeds"; table captions repeat "over 4 seeds").
  Boldface convention: "entries... within one standard deviation of the per-task best mean (i.e.,
  µ_i ≥ µ* − σ*)" (Table 1 caption, PDF p.7) — i.e. **bold = statistically-indistinguishable-from-
  best by a 1-σ overlap rule**, not a formal significance test.
- **D4RL low-coverage (Table 2, Table 10):** also mean ± std; boldface there instead means
  "statistically significant improvement" per a **one-sided/paired Wilcoxon signed-rank test**
  (Table 2 caption, PDF p.8) — a different (stricter, actual-hypothesis-test) criterion than
  Table 1's mean±σ-overlap rule. Number of seeds for D4RL is **not explicitly restated** in §5.2
  or §A.5.3 — presumably the same 4 seeds as elsewhere but not stated verbatim for this table
  (flagged below as unconfirmed).
- **Orthogonality-regularization ablation (Fig. 3):** compared via **one-sided Mann–Whitney U
  tests** on per-seed returns (not Wilcoxon) — a third distinct statistical-test convention used
  in the paper depending on the section.
- **SMPL FB-CPR oracle:** 5 seeds (only the oracle; the offline-learned methods incl. RLDP use 4
  seeds like elsewhere), per §A.5.2 (PDF p.20).
- No "best-of-N" seed selection language anywhere — all reported numbers are seed-averaged, not
  best-seed.

## 5. Official numbers — ExoRL-overlapping domains (Table 1, PDF p.6-7)

Raw episode returns, mean ± std over 4 seeds, offline zero-shot RL, representation dim = "best
for each method" (RLDP swept {64,128,256,512,1024} and reports the environment-best dim; FB/
HILP/PSM/Laplacian use "the representation dimensions previously identified as optimal for each
respective method" — **exact per-method dims not stated in text**, see gaps below).

| Domain | Task | Laplace | FB | HILP | PSM | **RLDP** |
|---|---|---|---|---|---|---|
| Walker | Stand | 243.70±151.40 | 902.63±38.94 | 607.07±165.28 | 872.61±38.81 | 877.69±45.03 |
| Walker | Run | 63.65±31.02 | 392.76±31.29 | 107.84±34.24 | 351.50±19.46 | 324.85±54.57 |
| Walker | Walk | 190.53±168.45 | 877.10±81.05 | 399.67±39.31 | 891.44±46.81 | 790.94±67.55 |
| Walker | Flip | 48.73±17.66 | 206.22±162.27 | 277.95±59.63 | **640.75±31.88** | 491.64±37.30 |
| Cheetah | Run | 96.32±35.69 | 257.59±58.51 | 68.22±47.08 | 244.38±80.00 | 236.31±20.75 |
| Cheetah | Run Backward | 106.38±29.40 | 307.07±14.91 | 37.99±25.16 | 296.44±20.14 | **322.08±39.28** |
| Cheetah | Walk | 409.15±56.08 | 799.83±67.51 | 318.30±168.42 | **984.21±0.49** | 895.31±49.84 |
| Cheetah | Walk Backward | 654.29±219.81 | 980.76±2.32 | 349.61±236.29 | 979.01±7.73 | **984.76±0.85** |
| Quadruped | Stand | **854.50±41.47** | 740.05±107.15 | 409.54±97.59 | 842.86±82.18 | 794.94±43.25 |
| Quadruped | Run | 412.98±54.03 | 386.67±32.53 | 205.44±47.89 | 431.77±44.69 | **457.41±74.70** |
| Quadruped | Walk | 494.56±62.49 | **566.57±53.22** | 218.54±86.67 | **603.97±73.67** | 465.40±185.29 |
| Quadruped | Jump | 642.84±114.15 | 581.28±107.38 | 325.51±93.06 | 596.37±94.23 | **733.32±55.30** |
| Pointmass | Top Left | 713.46±58.90 | **897.83±35.79** | **944.46±12.94** | 831.43±69.51 | 890.41±60.79 |
| Pointmass | Top Right | 581.14±214.79 | 274.95±197.90 | 96.04±166.34 | 730.27±58.10 | **795.47±21.10** |
| Pointmass | Bottom Left | 689.05±37.08 | 517.23±302.63 | 192.34±177.48 | 451.38±73.46 | **805.17±20.44** |
| Pointmass | Bottom Right | 21.29±42.54 | 19.37±33.54 | 0.17±0.29 | 43.29±38.40 | **193.38±167.63** |

(Bold = within 1σ of the per-task best mean, per the paper's own criterion; I've reproduced the
paper's exact boldface. RLDP is best or tied-best on 6 of 16 rows; PSM leads overall row-count.)

**Same-representation-dim (d=512) comparison, Table 7 (PDF p.22)** — FB/PSM/RLDP only (Random
Features control included), all at z_dim=512, same gradient-step budget: RLDP is competitive but
not table-1-best on several tasks (e.g. Walker Flip RLDP 492.94 vs FB 977.08 vs PSM 621.36) —
i.e. **FB's Table-1 numbers benefit from FB's own preferred (non-512) dimensionality**; Table 7
isolates the representation-dim confound. PSM and RLDP "generally achieve the strongest
performance" among learned methods per the paper's own read of Table 7.

## 6. Data preprocessing

- **Obs normalization:** `agent_config.model.norm_obs = False` by default in
  `create_agent()` (`rldp_train_dmc.py:53`) — i.e. **the DMC launch script trains without
  running-BatchNorm observation normalization** even though `FBModel` supports it
  (`nn.BatchNorm1d(obs_dim, affine=False, momentum=0.01)` when `norm_obs=True`, `model.py:71`).
  The `update()` loop still calls `self._model._obs_normalizer(obs)` unconditionally
  (`agent.py:182-185, 210-213`), which is a no-op `nn.Identity()` pass-through when `norm_obs`
  is off.
- **Action repeat / frame stack:** no mention anywhere in paper or code — DMC/ExoRL episodes are
  used at native `dm_control` step rate; no frame-stacking code path exists for the state-based
  DMC pipeline (frame-stack/pixel code paths exist in `EncoderMap`'s `use_pixel` branch but are
  unused by `rldp_train_dmc.py`, which is state-only).
- **Reward preprocessing:** none described beyond the D4RL-standard normalization noted in §2
  above and the humanoid oracle-normalization; DMC/ExoRL Table 1 numbers are raw returns.
- Data dtype: observations/actions cast to float32 on load (`load_data`,
  `rldp_train_dmc.py:106-108`).

## 7. Compute

**Not stated anywhere in the paper (main text or appendix) or in the code/README.** No GPU model,
wall-clock time, or per-run compute budget is given. Grepped the full extracted paper text for
"GPU," "hour," "compute," "A100/V100/H100," "wall-clock," "hardware" — zero hits.

---

## Facts I could NOT find / remain ambiguous

1. **Total training-step budget conflict (Table 1 numbers):** paper text says 2M representation +
   3M policy = 5M steps total; the shipped `rldp_train_dmc.py` script's *defaults* are
   `representation_steps=3_000_000`, `num_train_steps=6_000_000` (6M total). No launch command
   or config file in the repo shows the actual CLI overrides used to produce Table 1 — can't
   confirm which step budget generated the paper's headline numbers.
2. **num_eval_episodes conflict:** paper text says "50 episodes"; script default is
   `num_eval_episodes=10`. Not resolved which produced Table 1.
3. **Exact representation dimensions used per baseline in Table 1** (FB, HILP, PSM, Laplacian):
   paper says only "representation dimensions previously identified as optimal for each
   respective method" — no numeric values given, and no baseline configs are in this repo
   (repo only contains RLDP's own agent code, not FB/HILP/PSM/Laplacian implementations).
4. **RLDP's own winning representation dim per domain in Table 1** — paper says it swept
   {64,128,256,512,1024} and reports the best per-environment dim, but doesn't state which dim
   won for Walker vs. Cheetah vs. Quadruped vs. Pointmass.
5. **Whether Pointmass eval used 4 tasks (loop/square/fast_slow excluded) or all 7** —
   `ALL_TASKS["pointmass"]` in code has 7 entries but Table 1/Table 4 report only the 4 `reach_*`
   tasks; not explicit whether `eval_tasks` was overridden via CLI to restrict to 4, or whether
   loop/square/fast_slow were run and simply omitted from the paper table.
6. **No numeric hyperparameter table anywhere in the paper text itself** — batch size, learning
   rates, discount γ, target-network τ, gradient-clip, weight decay are **only** recoverable from
   the code defaults (§3 above); the paper appendix (§A.5) gives architecture widths/depths and a
   couple of loss-specific settings (H, λ, D4RL's α=2.5) but never states batch size, lr, γ, or τ
   in prose or a table.
7. **D4RL/low-coverage seed count** — not explicitly restated for §5.2/Table 2 (presumably 4,
   matching the rest of the paper, but not verbatim stated for this section).
8. **Compute/hardware/wall-clock** — entirely absent (see §7).
9. **No launch script or config in this repo for the D4RL or SMPL-humanoid experiments** — only
   `rldp_train_dmc.py` (ExoRL/DMC) exists; the exact CLI invocations, per-domain z_dim overrides,
   or eval-task lists used for Tables 2, 9, 10 are not present as runnable artifacts, only as
   appendix prose.
10. **Baseline (FB/PSM/HILP/Laplacian) implementations/configs** are not in this repo at all — the
    paper states "we rely on author's implementation to tune hyperparameters" (§5.1.1, PDF p.7)
    but those implementations/configs are external and not included here.
