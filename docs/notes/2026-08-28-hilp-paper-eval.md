# HILP zero-shot-RL evaluation setup (extraction)

Source paper: Park, Rybkin, Levine, "Foundation Policies with Hilbert
Representations," ICML 2024, arXiv:2402.15567 (fetched HTML v2:
https://arxiv.org/html/2402.15567v2 ; PDF: https://arxiv.org/pdf/2402.15567v2).
Source repo: https://github.com/seohongpark/HILP (default branch `master`,
browsed via GitHub API / raw.githubusercontent.com, not cloned).

**Extraction method / reliability note.** The paper was read via WebFetch
(HTML → small-model summarization), which is reliable for prose (objectives,
config text, README/code) but was **unreliable for large numeric tables** —
two independent extraction attempts at Table 3 returned different numbers for
the same cells (e.g. Walker-Flip-HILP came back as `573±37` and `563±136` in
two passes). The repo code (`sf.py`, `train_offline.py`, `dmc_benchmark.py`,
the `hilp_zsrl/README.md`) was fetched as raw files and is high-confidence —
it is verbatim source, not summarized-then-transcribed. Where a fact below
comes only from the unreliable table-extraction, it is marked **[LOW
CONFIDENCE — VERIFY AGAINST PDF]**.

---

## 1. The zero-shot RL experiment

- **Domains (repo: `hilp_zsrl/url_benchmark/dmc_benchmark.py`)**: `walker`,
  `cheetah`, `quadruped`, `jaco` (+ `point_mass_maze` exists in the dmc_benchmark
  task list but is **not** one of the four zero-shot-RL eval domains used in
  the paper's Table 3 / Figure 5 — paper text: "we consider four environments
  (Walker, Cheetah, Quadruped, and Jaco)").
- **Tasks per domain** (`dmc_benchmark.py`, verbatim):
  - `WALKER_TASKS = [walker_stand, walker_walk, walker_run, walker_flip]`
  - `CHEETAH_TASKS = [cheetah_walk, cheetah_walk_backward, cheetah_run, cheetah_run_backward]`
  - `QUADRUPED_TASKS = [quadruped_walk, quadruped_run, quadruped_stand, quadruped_jump]`
  - `JACO_TASKS = [jaco_reach_top_left, jaco_reach_top_right, jaco_reach_bottom_left, jaco_reach_bottom_right]`
  - Paper text confirms 4 test tasks/domain, e.g. "Flip, Run, Stand, and Walk
    for Walker."
- **Exploration/offline data buffer**: ExoRL (Yarats et al. 2022), built on
  the Unsupervised RL Benchmark (Laskin et al. 2021). Paper Table 3 caption
  and body text indicate **four** unsupervised exploration datasets per
  domain — **APS, APT, Proto, RND** — each contributing its own row. The
  released training commands in `hilp_zsrl/README.md` all use **RND** with
  `--num_episodes=5000` (Walker/Cheetah/Quadruped) or `20000` (Jaco); dataset
  conversion is via `hilp_zsrl/convert.py --env=... --task=... --method=rnd
  --num_episodes=... --use_pixels={0,1}`.
- **State or pixel**: both. State-based is the default (`obs_type=states`).
  Pixel variant: 64×64×3 observations, `frame_stack=3`, `agent.batch_size=512`,
  `num_grad_steps=500000` (vs 1,000,000 for state-based) — paper text: "we
  also employ their pixel-based variants with 64×64×3-sized observation
  spaces."

## 2. Evaluation protocol

- **Zero-shot task inference (paper, Eq. 8, quoted)**:
  `z* = argmin_{z∈Z} E_D[(r(s,a,s') − ⟨φ̃(s,a,s'),z⟩)²]`, closed form
  `z* = E_D[φ̃φ̃ᵀ]⁻¹ E_D[r(s,a,s')φ̃]`. Paper text: "we sample a small number
  of (s,a,s') tuples from the dataset D, compute the optimal latent z* ...
  using the above formula" — **exact sample count not stated in the prose**,
  but the code resolves it (see below).
  - **Code implementation** (`sf.py::infer_meta_from_obs_and_rewards`) does
    **not** literally build the (φφᵀ)⁻¹ closed form; it calls
    `torch.linalg.lstsq(phi, reward)` (batched least-squares over the
    accumulated φ features and rewards), then rescales:
    `z = sqrt(z_dim) * normalize(z)`. This is functionally the same
    least-squares regression, done via a numerically-stable solver instead of
    an explicit matrix inverse.
  - **Sample count, from code** (`train_offline.py::_init_eval_meta`): batches
    of size `cfg.batch_size` (1024 state-based / 512 pixel) are drawn
    repeatedly from the replay buffer and accumulated until reaching
    `agent.cfg.num_inference_steps`, whose **default is `10000`**
    (`SFAgentConfig.num_inference_steps: int = 10000`) — i.e. ~10 batches, ~10k
    (s,a,s') transitions, are pooled to solve for z* at eval time.
  - A second, goal-based inference path exists for goal-reaching tasks
    (`get_goal_meta`): `z = sqrt(z_dim) * normalize(φ(g) − φ(s))`, used when
    `goal_eval=True` (the HILP-G variant).
- **Eval episodes**: `num_eval_episodes: int = 10` used for **periodic**
  evaluation every `eval_every_steps=10000` steps during training; at the end
  of training the `finalize()` routine overrides this with
  `num_final_eval_episodes: int = 50` and re-evaluates — **the paper's
  reported numbers are the 50-episode final-checkpoint evaluation**, written
  to `test_rewards.json`.
- **Reporting step**: end of training — `num_grad_steps` default
  `1,000,000` (state-based) or `500,000` (pixel-based, set explicitly in the
  README pixel commands). No mid-training checkpoint averaging is described;
  it is a single final-checkpoint eval (matches user's memory note:
  "NEVER compare mid-run evals" — same discipline the source repo enforces
  structurally via `finalize()`).
- **Seeds / statistics**: Table 3's own caption states **"(4 seeds)"**
  unambiguously ("Full results on the state-based ExoRL benchmark (4 seeds).
  The table shows the unnormalized return averaged over four seeds in each
  setting"). Figure 5's caption: "results are aggregated over 4 environments,
  4 tasks, 4 datasets, and 4 seeds (i.e., 256 values in total)," using the
  **interquartile mean (IQM)** for the aggregate comparison, with "95%
  bootstrap confidence intervals in plots and standard deviations in tables."
  (Note: one earlier WebFetch pass mis-rendered these as "44"/"88" due to
  HTML/MathML digit-duplication artifacts — the Table 3 caption's plain-text
  "(4 seeds)" is the trustworthy reading; treat any "8 seeds" claim as an
  OCR artifact.)

## 3. Official ExoRL zero-shot RL numbers

**High confidence (paper prose, Figure 5)**: HILP achieves the best aggregate
zero-shot RL performance among the 9 (10, counting HILP-G) compared methods
on ExoRL, aggregated via IQM over 4 environments × 4 tasks × 4 datasets × 4
seeds (256 values), and achieves "the best or near-best scores in each
environment." Baselines compared: GC-TD3, CL (contrastive), Rand (random
features), IDM (inverse dynamics model), AE (autoencoder), Lap (Laplacian),
FDM (forward dynamics model), FB (forward-backward).

Table 3 (paper) structure, high confidence: rows are disaggregated **per
exploration dataset** (APS / APT / Proto / RND) × per task, columns are the 10
methods (`GC-TD3, CL, Rand, IDM, AE, Lap, FDM, FB, HILP (ours), HILP-G
(ours)`), reporting mean±std unnormalized return over 4 seeds. HILP-G numbers
are reported only for goal-reachable domains (e.g. Jaco), not for
locomotion-reward domains like Walker/Cheetah (those cells read `—` in what
was extracted).

**[LOW CONFIDENCE — VERIFY AGAINST PDF]** Two independent per-cell
transcriptions of Table 3 disagreed with each other (see reliability note
above), so the exact numeric cells are **not** reproduced here as facts. A
representative (unverified, possibly wrong) sample of what was extracted for
the RND-dataset row is:

| Domain | Task | HILP (unverified) | FB (unverified) |
|---|---|---|---|
| Walker | Flip | ~560–570 ± ~40–140 | ~330–550 ± ~90 |
| Walker | Walk | ~810–860 ± ~30–50 | ~810–840 ± ~50–100 |
| Cheetah | Run | ~260–370 ± ~50–70 | ~180–250 ± ~40–130 |
| Cheetah | Walk Backward | ~840–985 ± ~2–180 | ~680–980 ± ~3–85 |

Directionally these are consistent with the paper's claim (HILP ≥ FB on most
cells, both far above Rand/CL), but **do not cite specific numeric values from
this table without re-extracting from the actual PDF/HTML with a proper PDF
reader** — this environment had no `pdftotext`/`pdftoppm`/`pymupdf`/`pip`
available to render the PDF locally for exact transcription (see §5, "facts
not found").

## 4. Method structure (for a future port)

All of this section is **high confidence** — taken verbatim from
`hilp_zsrl/url_benchmark/agent/sf.py` (raw source fetched in full), which
implements HILP as a `feature_learner` option (`hilp`) inside the generic
`SFAgent` (successor-features) agent, alongside `fb_ddpg.py` (plain FB
baseline) and other feature learners (`lap`, `icm`, `transition`,
`contrastive`, `svd_sr`, `autoencoder`, `identity`, `random`).

### φ (Hilbert representation) objective

Two-head Hilbert value function trained as an IQL-style expectile regression
toward a **hindsight-relabeled goal-reaching sparse reward**:

```
# obs, next_obs from a transition; goals = future_obs (hindsight-relabeled state)
reward(s, s') = 1[ ||s - g|| < 1e-6 ] - 1          # sparse: 0 at goal, else -1
mask(s, s')   = 1 - 1[ ||s - g|| < 1e-6 ]          # 0 at goal (absorbing), else 1

v_i(s, g) = -sqrt(clamp(||phi_i(s) - phi_i(g)||^2, min=1e-6))     for i in {1,2}  (twin heads)

# target (double-Q style, min over twin target heads)
next_v   = min(next_v1_target, next_v2_target)
q        = reward + hilp_discount * mask * next_v
v_target = (v1_target(s,g) + v2_target(s,g)) / 2
adv      = q - v_target

q_i = reward + hilp_discount * mask * next_v_i_target      for i in {1,2}
L_HILP = sum_i  expectile_loss(adv, q_i - v_i(s,g), hilp_expectile)
       where expectile_loss(adv, diff, tau) = where(adv>=0, tau, 1-tau) * diff^2
```

- This is exactly an IQL-style expectile value regression (Kostrikov et al.
  2022 cited by the paper as the underlying offline-RL algorithm class) but
  with the **value function itself defined as a (negative) learned quasimetric
  distance** between twin feature embeddings `phi(s)` and `phi(g)` — i.e. φ is
  trained so that `-||phi(s)-phi(g)||` approximates the expectile-regressed
  optimal value of reaching g from s, which is the "Hilbert representation"
  (embeds the environment's temporal-distance structure isometrically-ish into
  a Hilbert/Euclidean space).
  - `phi1`/`phi2` (and their EMA targets `target_phi1`/`target_phi2`) are each
    an MLP `[obs_dim, hidden_dim, "ntanh", hidden_dim, "relu", feature_dim]`
    with `hidden_dim = phi_hidden_dim = 512` (default), `feature_dim = z_dim`
    (or `z_dim/2` if `feature_type='concat'`).
  - Target φ nets are EMA-updated every call with **hard-coded** `tau=0.005`
    (`utils.soft_update_params(self.phi1, self.target_phi1, 0.005)`), separate
    from the SF/actor's own `sf_target_tau=0.01`.
  - A running mean/std of `phi1(obs)` is tracked (`0.995`/`0.005` EMA) and the
    **mean only** is subtracted in the public `feature_net(obs)` used
    elsewhere (`phi = phi1(obs) - running_mean`); std is tracked but not
    applied as a divisor in the code shown.
  - Goals for the φ loss are `future_obs`, i.e. **hindsight-relabeled future
    states from the same trajectory** (see §6 for how future_obs is sampled —
    geometric/HER-style, with a `p_randomgoal` probability of substituting a
    fully random state instead).
  - `hilp_expectile` and `hilp_discount` are **per-domain hyperparameters**,
    not paper-wide constants (see the HP table in §4/§6): expectile is 0.5 for
    the standard HILP runs and 0.9 for the goal-conditioned `HILP-G` variant;
    discount is 0.96 for Walker and 0.98 for Cheetah/Quadruped/Jaco.

### z / skill parameterization and the successor-feature actor-critic on top

- `z` lives on a sphere of radius `sqrt(z_dim)`: `z = sqrt(z_dim) *
  normalize(gaussian_sample)` (`sample_z`), matching FB/SF conventions of
  z on `sqrt(d)·S^{d-1}`.
- On top of φ, a standard **successor-features (SF) actor-critic** is trained
  (this part is shared code with the other `feature_learner` choices, not
  HILP-specific):
  - `ForwardMap` (twin `F1, F2`, from `fb_modules.py`) predicts SF-style
    features `F(s,z,a) ≈ E[Σ γ^t φ(s_t)]`.
  - Critic target: `target_phi = phi(next_obs).detach()` (feature at the next
    state, using whichever `feature_type` — `state`, `diff`, or `concat` of
    both), `next_F = F_target(next_obs, z, next_action)` (double-Q min via
    `Q_i = <F_i, z>`), `target_F = target_phi + discount * next_F`.
  - SF loss: either an MSE on `Q = <F,z>` vs `target_Q = <target_F,z>` when
    `q_loss=True`, or a full-vector MSE `F` vs `target_F` when `q_loss=False`
    (`q_loss` is itself a **per-domain flag** — see §6).
  - Actor: `Actor`/`DiagGaussianActor` (from `fb_modules.py`) conditioned on
    `(obs, z)`; actor loss `-Q.mean()` (deterministic/DDPG-style,
    `boltzmann=False` default) or `(temp*log_prob - Q).mean()` if
    `boltzmann=True`; `Q = min(<F1,z>, <F2,z>)`.
  - **z-mixing during training** (`mix_ratio=0.5` default): with probability
    `mix_ratio` per-sample, `z` for that sample in the batch is replaced by a
    "relabeled" z computed as `normalize(phi(future_obs[perm]) @ pinv(cov(phi)))
    * sqrt(z_dim)` — i.e. some fraction of training batches use
    hindsight-style z's derived from the same batch's φ-covariance rather than
    pure random z's, analogous to FB's `future_ratio`/relabeling trick.
- **Zero-shot z inference at eval** (repeated from §2 for completeness):
  `infer_meta_from_obs_and_rewards` does `z = lstsq(phi, reward)` then
  `z = sqrt(z_dim)*normalize(z)`; `feature_type` controls whether `phi` is
  `phi(s)` (`'state'`), `phi(s')-phi(s)` (`'diff'`), or `concat(phi(s),
  phi(s'))` (`'concat'`, with `z_dim` split in half between the two phi heads
  in that case).

### Network sizes / all training HPs (defaults, `SFAgentConfig` in `sf.py`)

| field | default |
|---|---|
| `z_dim` | 50 |
| `hidden_dim` (actor/forward-map) | 1024 |
| `phi_hidden_dim` (φ nets) | 512 |
| `feature_dim` | 512 |
| `batch_size` | 1024 (state) / 512 (pixel, per README) |
| `lr` | 1e-4 |
| `lr_coef` | 5 (φ optimizer uses `lr_coef * lr` = 5e-4) |
| `sf_target_tau` | 0.01 |
| `hilp_discount` | 0.98 (config default; **overridden per-domain**, see §6) |
| `hilp_expectile` | 0.5 (config default; **0.9 for HILP-G**) |
| `mix_ratio` | 0.5 |
| `q_loss` | True (config default; **per-domain override**, see §6) |
| `update_z_every_step` | 300 |
| `num_inference_steps` | 10000 |
| `stddev_schedule` | "0.2" (fixed exploration std for the DDPG-style actor) |
| `stddev_clip` | 0.3 |
| `nstep` | 1 |
| `feature_type` | 'state' (default; `'diff'` for HILP-G) |

Paper text corroborates `z_dim = D = 50` for zero-shot RL ("we use D=50 for
zero-shot RL").

Top-level `train_offline.py::Config` (shared across agents):

| field | default |
|---|---|
| `discount` | 0.98 |
| `future` | 0.99 (geometric-sampling parameter for HER-style future_obs) |
| `p_randomgoal` | 0 (config default; **0.375 in all HILP training commands**) |
| `num_grad_steps` | 1,000,000 (state) / 500,000 (pixel, README) |
| `num_eval_episodes` | 10 (periodic) |
| `num_final_eval_episodes` | 50 (used for the reported numbers, via `finalize()`) |
| `eval_every_steps` | 10,000 |
| `replay_buffer_episodes` | 5,000 (Walker/Cheetah/Quadruped); 20,000 (Jaco) |
| `frame_stack` | 3 (pixels only) |
| `action_repeat` | 1 |
| `image_wh` | 64 |
| `checkpoint_every` | 100,000 |

**Per-domain HP overrides** (verbatim from `hilp_zsrl/README.md` example
commands — these are the paper's actual reproducing commands, not just
defaults):

| Domain | `hilp_expectile` | `hilp_discount` | `q_loss` | `p_randomgoal` | `replay_buffer_episodes` |
|---|---|---|---|---|---|
| Walker | 0.5 | 0.96 | False | 0.375 | 5000 |
| Cheetah | 0.5 | 0.98 | False | 0.375 | 5000 |
| Quadruped | 0.5 | 0.98 | True | 0.375 | 5000 |
| Jaco (HILP) | 0.5 | 0.98 | True | 0.375 | 20000 |
| Jaco (HILP-G) | 0.9 | 0.98 | True (feature_type=diff, goal_eval=True) | 0.375 | 20000 |

Pixel variants add: `obs_type=pixels agent.batch_size=512 num_grad_steps=500000`,
otherwise identical per-domain HPs to the state-based commands.

## 5. Repo layout

- Root: `README.md` points to two independent implementations:
  `hilp_gcrl/` (goal-conditioned RL experiments, not examined here) and
  `hilp_zsrl/` (zero-shot RL / ExoRL experiments — this is the relevant one).
- `hilp_zsrl/` is a fork of the FB/"Does Zero-Shot RL Exist?" codebase
  (`facebookresearch/controllable_agent`) — README states this explicitly.
  - `hilp_zsrl/README.md` — the exact reproducing commands (quoted in full
    above / in §4,§6).
  - `hilp_zsrl/convert.py` — converts raw ExoRL datasets into the repo's
    replay-buffer format (`--env --task --method --num_episodes --use_pixels`).
  - `hilp_zsrl/requirements.txt`, Python 3.8, conda env `hilp_zsrl`.
  - `hilp_zsrl/url_benchmark/train_offline.py` — the main entrypoint (Hydra
    config, `Config` dataclass, training loop, `_init_eval_meta`, eval loop,
    `finalize()` for the 50-episode final numbers).
  - `hilp_zsrl/url_benchmark/base_config.yaml` — Hydra defaults
    (`agent: ddpg` default group, submitit/SLURM launcher config — **not**
    where HILP's per-domain HPs live; those are CLI overrides per the README).
  - `hilp_zsrl/url_benchmark/agent/sf.py` — **the file that actually contains
    HILP**: `SFAgentConfig`, the `HILP(FeatureLearner)` class (φ objective),
    the `FeatureLearner` zoo (`lap`, `icm`, `transition`, `contrastive`,
    `svd_sr`/`svd_srv2`, `svd_p`, `autoencoder`, `identity`, `random`), and
    `SFAgent` (the shared successor-feature actor-critic + z-inference logic).
  - `hilp_zsrl/url_benchmark/agent/fb_ddpg.py` + `fb_modules.py` — the **FB
    baseline** agent (`z_dim=50`, `hidden_dim=1024`, `backward_hidden_dim=512`,
    `ortho_coef=1.0`, `fb_target_tau=0.01`) and shared network modules
    (`Actor`, `DiagGaussianActor`, `ForwardMap`, `BackwardMap`, `mlp`,
    `OnlineCov`) reused by `sf.py`.
  - `hilp_zsrl/url_benchmark/agent/ddpg.py` — vanilla DDPG + `make_aug_encoder`
    (pixel-obs augmentation/CNN encoder, referenced by `sf.py` for
    `obs_type='pixels'`; not independently examined here).
  - `hilp_zsrl/url_benchmark/agent/gctd3.py` — the `GC-TD3` baseline used in
    Table 3.
  - `hilp_zsrl/url_benchmark/in_memory_replay_buffer.py` — the HER-style
    replay buffer: geometric future-state sampling
    (`future_idx = step_idx + Geometric(p=1-future)`, clipped to episode
    length), `p_randomgoal`-probability random-goal substitution, frame-stack
    concatenation for pixels, simple discount scaling (`discount *
    stored_discount`), **no observation normalization/clipping**.
  - `hilp_zsrl/url_benchmark/custom_dmc_tasks/` — custom DMC XMLs/task defs
    for `cheetah`, `hopper`, `jaco`, `point_mass_maze`, `quadruped`, `walker`
    (extends stock `dm_control` suite with these variants + reward functions).
  - `hilp_zsrl/url_benchmark/dmc_benchmark.py` — domain/task name lists
    (quoted in full in §1) and `PRIMAL_TASKS` (a default single training task
    per domain used elsewhere in the pipeline, e.g. unsupervised
    pretraining/expl-agent bookkeeping — not the zero-shot eval tasks).
  - `hilp_zsrl/url_benchmark/dmc.py`, `utils.py`, `my_utils.py`, `logger.py`,
    `video.py` — environment wrappers, misc helpers, logging/video, not
    examined in detail.
- **Config/flag defaults that differ from the paper's stated defaults**: the
  `SFAgentConfig` dataclass ships with `hilp_discount=0.98` and `q_loss=True`
  as its own defaults, but the actual paper commands override these
  per-domain (Walker uses `hilp_discount=0.96`, `q_loss=False`; Cheetah uses
  `hilp_discount=0.98`, `q_loss=False`) — i.e. **do not trust the dataclass
  defaults alone**; use the README's per-domain CLI invocations.

## 6. Data preprocessing / buffer matching

- **Obs normalization**: none applied in `in_memory_replay_buffer.py` — no
  running-mean/std normalization or clipping of states or pixels found in the
  buffer code. (The HILP φ-net itself keeps a running-mean buffer that is
  subtracted from `phi1(obs)` output — see §4 — but that's a representation
  normalization inside the model, not a preprocessing step on raw
  observations.)
- **Action repeat**: `action_repeat=1` (config default; no override seen in
  the README example commands) — i.e. standard DMC control-timestep, no
  repeat.
- **Frame stack**: `frame_stack=3`, pixels only (`obs_type=pixels`); ignored
  for state-based runs.
- **Pixel size**: 64×64×3 (`image_wh=64`), converted via `convert.py
  --use_pixels=1` producing a `replay_pixel64.pt` buffer file (vs
  `replay.pt` for state).
- **Goal/future relabeling for the φ objective and z-mixing**: `future_obs`
  is drawn via `future_idx = step_idx + Geometric(p = 1 - future)` with
  `future=0.99` — i.e. a HER-style geometric horizon (mean look-ahead ≈
  `1/(1-future) = 100` steps), clipped to the episode boundary; then with
  probability `p_randomgoal=0.375` (all HILP commands set this explicitly,
  overriding the `Config` default of 0) that sampled future_obs is instead
  replaced by a **uniformly random observation from the buffer** (not
  necessarily from the same episode/trajectory).
- **Buffer construction to match**: download ExoRL per
  `https://github.com/denisyarats/exorl`, then run
  `hilp_zsrl/convert.py --save_path=... --env={walker,cheetah,quadruped,jaco}
  --task={run,run,run,reach_top_left} --method=rnd
  --num_episodes={5000,5000,5000,20000} --use_pixels={0,1}` per domain (task
  arg to `convert.py` selects which single ExoRL sub-buffer/task-labeled
  trajectory set to convert — cross-reference against `PRIMAL_TASKS` in
  `dmc_benchmark.py` if replicating other domains' default task string).

---

## Facts I could NOT find / confirm

1. **Exact per-cell numeric values of Table 3 / Table 4** — two independent
   WebFetch-based extraction passes disagreed with each other; I had no
   working PDF-text tool in this environment (`pdftotext`, `pdftoppm`,
   `pymupdf`/`fitz`, and even `pip`/`pip3` itself were all unavailable, so I
   could not render or parse `hilp.pdf` locally — I downloaded it to
   `/tmp/.../scratchpad/hilp.pdf` but could not extract text from it). **To
   get trustworthy numbers, re-run this with a PDF-capable environment** (or
   ask a human to paste the table) rather than trusting the numbers sketched
   in §3.
2. **Table 4's exact contents/caption** — never successfully retrieved
   (WebFetch reported the HTML was truncated before reaching it); presumed to
   be the pixel-based ExoRL results table (Table 3 is explicitly captioned
   "state-based"), but this is inference, not confirmed text.
3. **Whether the paper's stated seed count is 4 or 8** — Table 3's caption
   unambiguously says "(4 seeds)," but one earlier extraction pass rendered
   the main-text opening as "8 random seeds." Given the demonstrated
   MathML/HTML digit-duplication artifacts in this pipeline (e.g. "44
   environments, 44 tasks..." for what Figure 5's caption correctly gives as
   "4 environments, 4 tasks..."), I believe 4 is correct and 8 is an
   artifact, but did not get a second independent confirmation of the
   main-text sentence.
4. **A dedicated "Appendix D hyperparameter table"** — the paper apparently
   has an Appendix D ("Implementation Details"/"Experimental Details") but I
   could not get WebFetch to surface a clean, tabular HP listing from it
   (responses said the relevant subsection was truncated/not included in what
   it read). All HP values in this doc instead come from the **code and
   README**, which is arguably more reliable for a port anyway, but a
   paper-stated per-domain HP table (if one exists, distinct from the README
   commands) was not independently confirmed.
5. **Pixel encoder architecture details** (CNN layer sizes, DrQ-style
   augmentation specifics) — `make_aug_encoder` is referenced from
   `agent/ddpg.py` and used by `sf.py` for `obs_type='pixels'`, but that file
   was not fetched/examined in this pass.
6. **HILP-G's exact non-Jaco applicability** — the README only shows a
   goal-conditioned (`HILP-G`) command for Jaco; whether HILP-G was also run
   on Walker/Cheetah/Quadruped in the paper (Table 3 showed `—` for those
   cells in the unreliable extraction) was not confirmed from paper prose.
