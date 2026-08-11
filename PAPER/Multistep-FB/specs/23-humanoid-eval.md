# Spec 23 — a non-OGBench domain for the density measure: Meta Motivo humanoid assessed, ExoRL adopted, humanoid deferred

Status: WRITTEN 2026-08-11. Parents: spec 18 COLLECTED (density measure on
jax GCIQL-flow, state, offline: cube-single .929 / cube-double .736 /
scene .628) + user request 2026-08-11 ("the method needs a non-OGBench
domain; candidate = the humanoid evaluation of Meta's MetaMotivo line,
arXiv 2504.11054"). This spec is the EVALUATION of that idea plus the
resulting eval plan. Spec only — no implementation contract is issued yet;
P0 below gates whether one gets written.

Naming: the file keeps the requested name `23-humanoid-eval` although the
verdict routes the primary eval to ExoRL (DMC). "Humanoid" below always
means the SMPL humanoid of HumEnv/Meta Motivo, never DMC `humanoid`.

Every external factual claim in this spec is registered with its source
URL in §8. Structural claims about our own stack are per spec 18/19.

## 0. One paragraph

arXiv 2504.11054 is "Zero-Shot Whole-Body Humanoid Control via Behavioral
Foundation Models" (FB-CPR / Meta Motivo, ICLR 2025): an FB agent trained
ONLINE (30M env steps of unsupervised interaction) in HumEnv, an SMPL
humanoid with 358-dim observations and 69-dim actions, regularized toward
AMASS motion-capture trajectories, and evaluated zero-shot on three task
families (45 rewards, 50 goal poses, motion tracking). It is a strong
brand-name target but a poor round-1 fit for us: the published protocol
has no offline training set (the nearest offline object is the released
replay buffer of their own trained agent), the observation/action scale is
~9x/14x our cube stack, and — decisively — our density head is built on a
low-dimensional goal-coordinate kernel (d_c = 3–6 on cube/scene) while
every humanoid task family lives on the full ~70–358-dim pose, so all
three families would need protocol modifications before we could report a
number a reviewer can compare. Verdict: INVALID as the primary
non-OGBench domain; the nearest valid alternative is ExoRL on DMC
(walker / cheetah / quadruped / point_mass_maze) — genuinely offline,
unsupervised-exploration data (the closest public analogue of unlabeled
play), FB's ORIGINAL eval domain with published zero-shot numbers (FB =
85% of offline TD3 with a good buffer), MIT-licensed, and
dimensionally comfortable for the d=50 bilinear stack. The humanoid
survives as a DEFERRED stretch arm (goal-reaching family only, trained
offline on the released buffer, FB-CPR checkpoint quoted as an online
ORACLE, modified metric declared as such), gated on the ExoRL wave.

## 1. What arXiv 2504.11054 actually is (verified, sources in §8)

- Paper: "Zero-Shot Whole-Body Humanoid Control via Behavioral Foundation
  Models", Tirinzoni, Touati, Farebrother, Guzek, Kanervisto, Xu,
  Lazaric, Pirotta. ICLR 2025. Method = FB-CPR (forward-backward
  representations with conditional-policy regularization); released model
  = Meta Motivo.
- Environment: HumEnv — SMPL-skeleton humanoid (24 rigid bodies, 23
  actuated joints) in MuJoCo, 450 Hz physics / 30 Hz decisions. State =
  proprioception: body pose 70 + body rotations 144 + linear/angular
  velocities 144 = 358 dims. Action = 69-dim PD-controller targets.
- Eval protocol, three families:
  1. Reward optimization: 45 reward functions, average return over
     300-step episodes. Zero-shot inference: z_r = 1/n Σ_i r(s_i)B(s_i)
     over reward-labelled sampled states.
  2. Goal reaching: 50 manually selected stable poses; success rate +
     proximity. Inference: z_g = B(s_goal).
  3. Motion tracking: held-out mocap motions; success rate + EMD.
     Inference: per-step z along the demonstration (ERFB-style).
- Training data: AMASS motion capture, 8,902 motions (~29 h) train / 990
  (~3 h) test — OBSERVATION-ONLY (no actions), used for the
  discriminator/regularization and mocap-based resets; plus 30M steps of
  ONLINE unsupervised environment interaction. There is NO fixed offline
  transition dataset in the published protocol.
- Headline: FB-CPR reaches 61% of task-specific TD3 on rewards, ~83%
  tracking success on the test set, strong goal-reaching (e.g. +118% over
  ASE); judged more human-like than task-specific TD3 in 83% of human
  comparisons. Baselines: TD3, Goal-GAIL, PHC (task-specific); Goal-TD3,
  CALM, ASE (multi-task/unsupervised); MPPI, Diffuser, H-GAP (planning).
- Released artifacts: 6 checkpoints on HuggingFace (metamotivo-S-1..5 at
  24.5M params, metamotivo-M-1 at 288M); per-model data buffers
  (`buffer.hdf5` = full training replay buffer, and
  `buffer_inference_500000.hdf5`; fields include qpos/qvel among others);
  inference API (`reward_inference`, `reward_wr_inference`,
  `goal_inference`, `tracking_inference`, `act(obs, z)`); full training
  scripts (`examples/fbcpr_train_humenv.py`, and notably
  `examples/fb_train_dmc.py` — their own plain-FB DMC pipeline). HumEnv
  ships benchmark code (`humenv.bench` Reward/Goal/Tracking evaluations)
  that runs against ANY agent exposing `act(obs, z)` plus the relevant
  inference method via wrappers. Both repos are CC BY-NC 4.0; mocap-based
  features require a separate (registration-gated) AMASS/SMPL license.

## 2. Validity assessment (honest)

Question: is the Meta Motivo humanoid eval a valid target for OUR setting
— state-based, trained OFFLINE from a fixed unlabeled dataset, jax
GCIQL-flow stack with the spec-18 density head (goal-conditioned
successor density over a LOW-DIM goal-coordinate block, ε-kernel +
softmax over ρ_x queries, d=50 bilinear towers, flow-BC actor)?

**(a) Their training is not offline.** FB-CPR is an online agent (30M
interaction steps); AMASS is observation-only and therefore cannot train
our F(s,a,g), critic bootstrap, or flow-BC actor (all need actions). The
only action-labelled data in the ecosystem is the RELEASED TRAINING
REPLAY BUFFER of their trained models (`buffer.hdf5`, downloadable per
checkpoint). An offline variant "train on the released buffer" is
technically runnable and is the modification we would adopt — but it is
epistemically weaker than it looks: the buffer is the replay stream of an
increasingly competent agent under its own exploration scheme, i.e.
agent-replay data, not play data; results on it answer "can our method
distill FB-CPR's own data offline", not "does our method work on a new
domain's unlabeled data". Defensible as a stretch arm, not as the
headline non-OGBench result. (Also CC BY-NC 4.0 — fine for a paper,
still a footnote.)

**(b) Scale.** 358-dim obs / 69-dim actions vs cube's ~40 / 5. Two
distinct risks. (i) Stack feasibility: unknown but plausibly surmountable
— FB-CPR itself uses a bigger latent (their S model is 24.5M params vs
our d=50 towers); a capacity sweep would be on the critical path. (ii)
The REAL blocker: our density head is a kernel density over a
low-dimensional goal-coordinate block (hit_spec blocks, d_c = 3 on
cube-single, 6 on cube-double/scene). Humanoid goals are full body poses
(~70-dim pose, 358-dim state); an ε-kernel + batch-softmax density over
even a 20-dim block is a nearest-neighbour lookup at any realistic batch
size (curse of dimensionality; ESS → 1 by construction, the spec-18
degeneration signature). Any humanoid arm must therefore REDEFINE the
goal coordinate to a reduced block (e.g. root height + root planar
velocity + torso orientation) — which changes the success metric away
from the published 50-pose benchmark.

**(c) Can we run their eval against an arbitrary policy?** Yes —
`humenv.bench` evaluates any agent exposing `act(obs, z)` (wrappers
provided), and the reward functions/goal poses/tracking motions ship with
the benchmark. This part is NOT a blocker. But family coverage is: the
tracking family is out of scope for our method (needs per-step latent
inference along a demonstration; we have no analogue), and the reward
family's 45 rewards are functions of large parts of the 358-dim state,
while our zero-shot reward readout evaluates r at pool points that are
BLOCK coordinates — only rewards expressible on the chosen reduced block
are readable. Realistically we could run: goal family (modified metric),
and a locomotion-flavoured subset of the reward family. Two of three
families modified or dropped.

**(d) Fair baseline.** The released FB-CPR checkpoints are an ORACLE
reference only (online-trained, 30M interactive steps + AMASS
regularization) — never a baseline for an offline method. A fair baseline
set on the humanoid would be: our stock GCIQL-flow and the repo's torch
plain FB, both trained on the same released buffer. All such numbers
would be new (no published offline-humanoid anchors exist in this line) —
so every humanoid claim would rest entirely on our own controls, on a
modified metric. That is a weak reviewer-facing position.

**Verdict: INVALID as the primary non-OGBench domain** (three
independent blockers: no offline play data in-protocol; goal-coordinate
kernel vs full-pose tasks; no published offline anchors — every family
needs modification). **Valid only as a deferred, explicitly-modified
stretch arm** (§6). The nearest valid alternative is ExoRL, adopted in
§4.

## 3. Why ExoRL is the right substitute (explicit weighing)

- **Actually offline, actually unlabeled.** ExoRL datasets are
  transitions collected by unsupervised exploration algorithms (RND,
  ProtoRL, APS, ICM, DIAYN, SMM, Disagreement, APT, Random) on DMC
  domains — 10M steps per (domain, algorithm) — downloadable by script,
  MIT-licensed, reward-relabelable for multiple downstream tasks per
  domain. This is the closest public analogue of unlabeled play data
  outside OGBench.
- **It is FB's original eval domain.** "Does Zero-Shot RL Exist?"
  (Touati, Rapin, Ollivier; arXiv 2209.14935, ICLR 2023) evaluated FB and
  10 SF variants exactly here: 13 tasks over maze (20 goals), walker
  (stand/walk/run/flip), cheetah (walk/run/walk-backward/run-backward),
  quadruped (stand/walk/run/jump), on RND/APS/Proto buffers, scored as %
  of offline TD3 trained on the same buffer; headline FB = 85% of
  supervised performance with a good buffer. Published anchors exist for
  BOTH the method family and the topline. Beating or matching FB here is
  a claim reviewers can verify; the humanoid offers no such anchor.
- **Meta's own repo ships `fb_train_dmc.py`** — DMC is the shared ground
  the metamotivo line itself maintains for plain FB. We stay inside the
  lineage the user pointed at while keeping the offline protocol honest.
- **Dimensionally comfortable**: walker/cheetah obs are SMALLER than
  cube's; quadruped (~78-dim obs, 12-dim actions) is the built-in scale
  probe, two steps up from cube instead of nine.
- Cost: DMC locomotion is less novel than a humanoid and is not
  manipulation. Accepted — the purpose of this wave is "the method is not
  an OGBench artifact", not "the method controls a humanoid". The
  humanoid remains available as the deferred arm once the cheap, anchored
  claim is banked.

## 4. Adopted eval: ExoRL (DMC), density measure vs FB on FB's home turf

### 4.1 Data acquisition

- `git clone https://github.com/denisyarats/exorl`; `./download.sh
  <domain> rnd` for domains: `point_mass_maze`, `walker`, `cheetah`,
  `quadruped`. RND is the primary buffer (the best-behaved buffer in the
  published FB study); `proto` is the robustness buffer, downloaded only
  if the P1 gate passes.
- Storage: datasets land outside the repo per convention (`datasets*/`
  untracked); record byte sizes and episode counts in the run configs.
  Datasets are episodic .npz (physics states + obs + actions); rewards
  are relabelled per task by the ExoRL tooling.
- Budget knob: published FB protocol trains on these buffers as released;
  we record `dataset_size` explicitly (full 10M primary; a 5M subsample
  arm only if memory forces it — declared, not silent).

### 4.2 Conversion + adapters (implementation contract to be written as
spec 23a AFTER P0 passes; scope preview only)

- ExoRL episodes → the repo's GCDataset format (observations, actions,
  terminals; goal relabelling machinery unchanged). No reward stored —
  rewards enter only at eval (zero-shot) or for the offline-TD3-style
  controls.
- A thin dm_control eval adapter (episode length 1000, standard DMC
  conventions) parallel to the OGBench eval loop. `humenv`/metamotivo
  code is NOT a dependency of this wave.
- Obs-layout verification step is MANDATORY at implementation time: the
  block indices in §4.3 are stated from standard DMC layouts and must be
  checked against the actual arrays before any hit_spec is frozen (the
  fb-state-run-name lesson: verify from recorded data, not from names).

### 4.3 Goal coordinate (hit_spec) per domain

Same object as on cube: a small block of state coordinates + per-block ε,
`start:end:eps` format.

- `point_mass_maze` (P0): block = (x, y) position, expected dims 0:2 of
  the 4-dim obs; ε matched to the published goal-reaching tolerance of
  the 20-goal protocol. This is structurally IDENTICAL to cube-single
  (Dirac goals in a 2–3-dim coordinate) — the cleanest transfer of the
  existing recipe, hence P0.
- `walker`: block = (torso height, horizontal velocity) — the two
  coordinates the stand/walk/run rewards are functions of (flip adds
  angular velocity; include it in a 3-dim block if the layout check
  confirms a clean index). ε from coordinate scales of the dataset
  marginal (record the quantile rule used).
- `cheetah`: block = (horizontal velocity,) 1-dim (+ optional torso
  height); all four tasks are speed-band rewards.
- `quadruped`: block = (torso height, horizontal speed) analogues; the
  78-dim obs makes the layout check non-trivial — this is deliberately
  the LAST domain (scale probe).

### 4.4 What "goal success" and "zero-shot reward" mean here

- **Goal family (maze only):** success = final/any-time proximity of the
  block coords to the goal within the published tolerance, 20 goals,
  matching the 2209.14935 maze protocol; report mean success. Our stack
  runs it natively (g-conditioned actor; goal = block coords), like cube.
- **Reward family (walker/cheetah/quadruped):** zero-shot = NO
  reward-labelled training, NO fine-tuning. At eval, reward r is exposed;
  the density readout produces Q̂_r(s, a, g) via the spec-18 PRIMARY
  readout (pool from the dataset block marginal), and the task policy is
  the RETRIEVAL DECODE already used in the spec-14/18 harnesses: score a
  candidate-goal pool by the readout, condition the actor on the argmax
  goal (re-decoded at episode start; per-step re-decode is a secondary
  arm). Report: raw DMC return (0–1000 scale) AND % of the published
  offline-TD3-on-same-buffer topline — the 2209.14935 metric — quoting
  the published topline rather than rerunning it (rerun only if a
  reviewer-grade discrepancy appears).
- The comparison to published FB is at the METRIC level (% of topline on
  the same domain/task/buffer), with the protocol difference (our decode
  is goal-retrieval, theirs is z_r inference) stated in one honest
  sentence wherever the number appears.

### 4.5 Baseline set

1. **Stock GCIQL-flow** (measure off), same data: goal family natively;
   for the reward family it gets the SAME retrieval decode over its
   Q(s,a,g) — the "is the density head doing anything" control — plus
   the cheap argmax-goal control (condition on the dataset state with
   highest reward under r's block restriction).
2. **Plain torch FB** (repo stack — this codebase is a port of the
   metamotivo/td_jepa FB, i.e. DMC is its native habitat), same data,
   standard z_r = E[r·B] inference. The direct method-family baseline.
3. **Published numbers** (quoted, not rerun): FB / best-SF / offline TD3
   from arXiv 2209.14935 on the matching (domain, task, RND) cells.
4. **BC-only control** (actor q-term zeroed) — the spec-19 P0.2 lesson:
   no number is attributable to the measure without it.

## 5. Phased plan + pre-registered gates

**P0 — smallest feasibility probe (write spec 23a only if this passes):**
point_mass_maze, RND buffer, seed 0: density arm + stock GCIQL control +
BC-only control, existing d50 recipe, 500k steps. Manual layout check +
dataset-conversion round-trip test are part of P0.
- Gate G0a (plumbing): training healthy — CE finite, median ESS ≥ 4 at
  end (spec-18 statistic, exactly), no NaN.
- Gate G0b (signal): density goal success ≥ 0.8 × stock GCIQL success,
  and BOTH beat BC-only by ≥ 0.1 absolute. Maze+RND is the easiest cell
  in the published study; failing G0b here kills the ExoRL route and the
  finding is reported, not patched.

**P1 — the headline cell:** walker, RND, 4 tasks, density (3 seeds) vs
torch FB (3 seeds) vs controls.
- Gate G1: density ≥ published FB's %-of-topline on ≥ 2 of 4 walker
  tasks, and ≥ our own torch-FB rerun on ≥ 2 of 4. (Two gates because
  published FB and our port may not coincide; both are reported.)
- Health gate G2: same ESS/entropy statistics as spec 18 §7 G2.

**P2 — breadth:** cheetah (4 tasks), then quadruped (4 tasks, the scale
probe). Same gate structure; quadruped has NO pass/fail gate in round 1
(pre-registered as exploratory — first time the recipe leaves d_c ≤ 6
comfort on a 78-dim obs).

**P3 — robustness (only after G1):** repeat walker on the `proto` buffer
(the published study's buffer-sensitivity axis).

**P4 — humanoid stretch arm:** §6; gated on G1 AND an explicit user
go-ahead (it is expensive and CC BY-NC-flavoured).

Pre-registered interpretations:
- G0b passes, G1 fails vs published FB but passes vs our torch FB ⇒ the
  port, not the method, is the gap — report both, investigate the port
  before any method claim.
- Density ≥ FB on walker but degrades on quadruped ⇒ the block-kernel
  recipe has a dimensionality ceiling — report as a RESULT with the ESS
  signature, do not patch in-wave (multi-scale ε is a pre-listed round-2
  item, spec 18 §9).
- All gates pass ⇒ the paper claim "not an OGBench artifact; competitive
  on FB's original benchmark, offline, zero-shot" is licensed.

Post-collection: every run_group into `scripts/results_map.py` +
`.venv/bin/python scripts/compile_results_csvs.py --refresh` (standing
rule).

## 6. The deferred humanoid arm (exact modifications that would make it
defensible)

If P4 is opened, the arm is valid ONLY with all of the following
declared:
1. **Data**: train offline on the released `buffer.hdf5` of ONE named
   checkpoint (e.g. metamotivo-S-1) — agent-replay data, stated as such;
   AMASS is not usable for training our stack (observation-only). Verify
   at implementation that the buffer's fields yield (obs, action,
   next_obs) transitions; qpos/qvel are confirmed present, the
   transition-completeness of the schema is NOT yet verified — that check
   is the first task of the arm.
2. **Families**: goal-reaching only, plus at most a locomotion subset of
   the 45 rewards whose reward is a function of the reduced block.
   Tracking is out of scope, stated up front.
3. **Goal coordinate**: a reduced block (root height + planar velocity +
   torso orientation; ≤ 8 dims), with success = ε-proximity in the
   REDUCED coordinates — a MODIFIED metric, never presented as the
   published 50-pose benchmark. Run `humenv.bench` machinery where
   possible for episode rollout, but score on the declared metric.
4. **References**: FB-CPR checkpoint evaluated under the SAME modified
   metric as an online ORACLE (labelled oracle, never baseline); stock
   GCIQL-flow + torch FB on the same buffer as the actual baselines.
5. **Capacity sweep pre-registered**: d ∈ {50, 128, 256} — the 358/69-dim
   spaces void the d=50 default; a d-sweep failure is reported as a
   scale result, not silently tuned away.

## 7. Anti-scope (round 1)

No humanoid training before P4 opens; no tracking family anywhere; no
online fine-tuning; no pixel path; no new measure variants inside this
wave (the wave evaluates the SHIPPED spec-18 recipe on new domains); no
reruns of published toplines without a demonstrated discrepancy; no
multi-scale ε / learned temperature (spec 18 anti-scope carries over);
no DMC `humanoid` domain (absent from both the ExoRL task list and the
published FB study — it would reintroduce the no-anchor problem).

## 8. Claim register (every step-1 claim with its source)

From https://arxiv.org/abs/2504.11054 :
- Identity: "Zero-Shot Whole-Body Humanoid Control via Behavioral
  Foundation Models", Tirinzoni et al., ICLR 2025; method FB-CPR; model
  Meta Motivo; trains on observation-only AMASS + unsupervised online
  interaction; evaluated on motion tracking, goal reaching, reward
  optimization.

From https://arxiv.org/html/2504.11054 :
- HumEnv state 358 = pose 70 + rotations 144 + velocities 144; action
  69-dim PD targets; MuJoCo 450 Hz physics / 30 Hz decisions; SMPL
  skeleton, 24 bodies / 23 actuated joints.
- Eval: 45 rewards, 300-step episodes, average return; 50 stable goal
  poses, success + proximity; tracking success rate + EMD on held-out
  motions.
- Inference: z_r = 1/n Σ r(s_i)B(s_i); z_g = B(s); tracking via ERFB
  along the demonstration.
- Data: AMASS 8,902 train motions (~29 h) / 990 test (~3 h); 30M online
  interaction steps; no fixed offline transition dataset in-protocol.
- Numbers: reward = 61% of task-specific TD3 (177% of Diffuser); tracking
  ~83% success on test motions; goal reaching outperforms ASE by 118%;
  human eval: more human-like than TD3 in 83% of cases. Baselines: TD3,
  Goal-GAIL, PHC, Goal-TD3, CALM, ASE, MPPI, Diffuser, H-GAP.

From https://github.com/facebookresearch/metamotivo (+ raw README):
- 6 checkpoints on HuggingFace: metamotivo-S-1..5 (24.5M params),
  metamotivo-M-1 (288M).
- Per-model buffers: `data/buffer.hdf5` (full training buffer) and
  `data/buffer_inference_500000.hdf5`, via `hf_hub_download`; fields
  include qpos and qvel among others.
- API: `reward_inference(next_obs, reward)`, `reward_wr_inference`,
  `goal_inference(next_obs)`, `tracking_inference(next_obs)`, then
  `act(obs, z, mean=True)`.
- `humenv.bench` RewardEvaluation/GoalEvaluation/TrackingEvaluation run
  against agents exposing `act(obs, z)` (wrappers provided).
- Training scripts: `examples/fbcpr_train_humenv.py`,
  `examples/fb_train_dmc.py`. License CC BY-NC 4.0.

From https://github.com/facebookresearch/humenv :
- SMPL humanoid, MuJoCo backend, tuned friction/actuation; 9 configurable
  reward classes; benchmark code for reward / goal-reaching / tracking;
  mocap-based initialization requires separately licensed AMASS/SMPL
  data. License CC BY-NC 4.0.

From https://github.com/denisyarats/exorl :
- 6 DMC domains (cartpole, cheetah, jaco, point_mass_maze, quadruped,
  walker); 9 collection algorithms (APS, APT/ICM, DIAYN, Disagreement,
  ICM, ProtoRL, Random, RND, SMM); 10M steps per collection; download via
  `./download.sh <domain> <algo>` from S3; multiple downstream tasks per
  domain via reward relabelling. MIT license (DeepMind parts Apache 2.0).

From https://arxiv.org/abs/2209.14935 (+
https://openreview.net/pdf?id=4jqhCamGsO ):
- "Does Zero-Shot Reinforcement Learning Exist?" — FB "reaching 85% of
  supervised RL performance with a good replay buffer, in a zero-shot
  manner"; 13 tasks on maze (20 goals) / walker (stand, walk, run, flip)
  / cheetah (walk, run, walk backward, run backward) / quadruped (stand,
  walk, run, jump); buffers RND, APS, Proto from ExoRL; metric = % of
  offline TD3 trained on the same buffer; FB the most consistent method.

## 9. Open questions carried into spec 23a

- Exact obs layouts + block indices for the four DMC domains (mandatory
  verification step, §4.2).
- Whether the retrieval decode needs per-step re-decoding on the speed-
  band tasks (episode-start decode is the round-1 default).
- ε selection rule for velocity coordinates (quantile of the dataset
  marginal vs task tolerance — pin ONE rule before launch).
- Whether the torch-FB DMC baseline reuses `fb_train_dmc.py` HPs or the
  2209.14935 HPs (pin before P1; record in config either way).
- Humanoid buffer schema completeness (obs/action/next_obs) — first task
  of P4, if opened.
