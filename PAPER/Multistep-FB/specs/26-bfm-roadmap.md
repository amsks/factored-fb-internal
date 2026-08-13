# Spec 26 — Roadmap: self-normalized occupancy BFM for manipulation → bottleneck domains

Status: WRITTEN 2026-08-13. Parent: `density-new.tex` (the program note; read
it first — thesis, claims map, gates). Goal as stated by the user: show we
can learn a behavior foundation model for manipulation with this method,
then generalize to a bottleneck-state domain (robot air hockey; humanoid
stays a gated stretch per spec 23). This file is the master ordering plus
implementation contracts for the immediate phases; later phases get full
specs only when their gates pass (spec-19 program pattern).

## 0. Priority order

- **R0 (this week, parallelizable):** R0a fork-and-dwell synthetic (P1
  gate = cheapest falsification of the thesis); R0b BFM interface +
  tracking eval on EXISTING checkpoints (no training; makes the "BFM"
  claim concrete); R0c air-hockey offline-data audit (research-only;
  gates the flagship early instead of late).
- **R1:** OGBench multiplicity stress (the causal experiment; needs R0a
  pass).
- **R2:** non-FB generality (joint critic; needs R1 machinery).
- **R3:** air hockey port (needs R0c pass + R1 result).
- Parallel, already specced elsewhere: spec 25 index-coupling screens
  (running), spec 21/22 leftovers, CSV-refresh debt.

## 1. R0a — fork-and-dwell synthetic (contract)

New standalone module `scripts/synthetic/fork_dwell.py` + study driver
`scripts/synthetic/run_fork_dwell_study.py`. NO dependency on the jax
agent stack (numpy/jax-cpu, tabular-scale networks; minutes per cell).

MDP (pin exactly): states = {decision D, left region L1..Lm, right region
R1..Rm, neutral N}; at D, action left → L1, right → R1, a third action →
N; inside a region, advance one step with prob 1−1/T_dwell, stay with
prob 1/T_dwell is WRONG — pin instead: geometric dwell, stay in region
cell with prob p_stay = 1 − 1/T_dwell, exit to D otherwise; episode
horizon H=200, γ=0.99 default. Dataset: uniform behavior policy episodes;
duplication knob replicates stationary transitions (s'=s region cells)
×{1,2,5,10,20}; T_dwell ∈ {5,20,44,100}.

Arms (all sharing one query-pool + Bellman-target implementation):
standard-FB (scalar FᵀB, squared+linear loss, tiny ortho arm optional),
density-CE (this program), density-MSE (normalized target, squared-error
fit — the CE-vs-normalization deconfound), joint-MLP unnormalized, joint-
MLP normalized. Ground truth computed exactly (matrix inversion) for M
and ν; log per cell: critic scale, total predicted mass vs truth,
entropy, action gap at D (Q(left)−Q(right) under a left-favoring test
reward), greedy-policy success, fixed-point error.

Tests: exact-ν against closed form on a 3-state case; duplication leaves
the true MDP unchanged (ground truth invariant); each arm's loss
finite/decreasing on a smoke run.

GATE P1 (pre-registered): normalized arms' action gap and success are
materially flatter in duplication factor than unnormalized arms'. If NOT:
stop, revise the thesis (density-new.tex says so); report either way.
Secondary read: CE vs MSE separates "normalization" from "fitting loss".

## 2. R0b — BFM interface + tracking eval (contract; eval-only, existing ckpts)

Two deliverables that make the foundation-model claim concrete without
new training:

**(i) `scripts/rescuing_fb/bfm_act.py`** — one entry point
`act(obs, prompt, ckpt)` with prompt ∈ {goal observation; reward
function; demo trajectory}. Routing: goal → direct conditioning (native);
reward → the v2 zero-shot machinery's search-based index selection
(import, don't reimplement); demo → sliding-window waypoint conditioning:
extract the demo's object-coordinate path via the ckpt's hit-spec,
condition the actor on the configuration Δ steps ahead of the current
nearest-progress point (Δ knob, default 25; progress = argmin object
distance along the demo, monotone non-decreasing). Thin: every branch
calls existing audited code.

**(ii) `scripts/rescuing_fb/eval_tracking.py`** — the tracking eval
OGBench lacks: hold out N=50 play trajectories (episode-level split by
index, seeded, never seen as demos before), run the sliding-waypoint
policy from the demo's own start state (teleport, as the G4 probe does),
metrics = mean time-aligned object-path error, waypoint hit-rate
(fraction of demo waypoints reached within ε), and completion (final
configuration within ε of the demo's final). Controls: BC-only ckpt
(floor) and a random-goal control (conditioning on shuffled waypoints —
tests that tracking is real, not drift). 3 density ckpts (soft + hard) on
cube-single first; cube-double if cs is alive.

GATE R0b: density beats BOTH controls on waypoint hit-rate with the split
documented. Pass ⇒ the three-prompt BFM claim (goal/reward/demo) is
demonstrated on manipulation play and goes into the paper's framing; fail
⇒ report; the demo prompt drops to future work (goal+reward prompts
already collected).

Tests: waypoint-progress monotonicity on a synthetic path; shuffled-
control wiring; determinism repeat.

## 3. R0c — air-hockey data audit (research-only; no code)

WebSearch/WebFetch: what offline datasets exist for the Robot Air Hockey
testbed (arXiv 2405.03113 repo + the air-hockey-challenge stack) and the
interaction-resampling line (arXiv 2606.11525)? For each candidate: is it
downloadable, state-based, with actions, play-like coverage of contacts,
and license-clean? ⚠ prior finding (spec 15): the TommyShen play datasets
were NOT public; verify freshly, do not assume. If nothing suitable
exists: cost out generating our own with the testbed's simulation (their
repo ships environments) — a scripted/noisy-play collection policy is
acceptable for the flagship, but say so in the note. Deliverable: a
half-page verdict appended to specs/26 + go/no-go for R3. Also: the
"first to do X" literature check for the flagship claim.

## 4. R1 — OGBench multiplicity stress (contract; gated on P1)

Knob `measure_dup_passive` (float ≥ 1, default 1.0 = bit-identical) on
the TRANSITION SAMPLING of the measure/critic losses ONLY: passive rows
(‖Δ block-coords‖ < δ, δ = 0.1·min block ε; precomputed mask at dataset
load) are drawn with probability multiplied by the factor,
renormalized. THE DECONFOUND (density-new.tex §5, non-negotiable): the
goal-relabeling distributions (value_goals, actor_goals) are computed
from the UNDUPLICATED sampler — a knob-on run must show bit-identical
goal-sampling statistics (test). A secondary labeled arm
`measure_dup_passive_goals_too` runs the confounded version once for the
record.
Arms: standard-FB torch stack (the repo's plain FB, same duplication
applied to its sampler — one matching knob there) and density-hard jax
arm; factors {1,5,20} round 1 (3 points, 3 seeds cs; 1 seed screens
cd/antmaze); metrics per density-new.tex §5 (scale, mass, entropy, action
gap via the existing G4 machinery, goal success, one zero-shot family).
Gates: the P2 prediction (FB degrades with factor on manipulation;
normalized degrades less; antmaze separates least), quoted as slopes with
seeds, not single cells.

## 5. R2 — non-FB generality (stub; full spec on R1 collection)

`measure_density_joint` knob: c = MLP([s,a,g,ũ]) replacing FᵀB scoring in
density_loss + actor path (readouts lose precomputed B embeddings —
accept the eval cost, document). Unnormalized-joint control = same MLP
under the scalar squared/linear loss. Rerun the R1 sweep's cs column.

## 6. R3 — air hockey (stub; gated on R0c + R1)

Port checklist when opened: env adapter + dataset loader; hit_spec =
puck xy (+velocity block decision); contact-phase stratifier (pre/
contact/post, reusing the R1 passive mask machinery); the standing recipe
end-to-end on one buffer; then the R1 multiplicity sweep on post-contact
frames; then the reward-map demo (terminal reward families over puck
position through the existing zero-shot harness). Flagship storyboard per
density-new.tex §10.

## 7. Anti-scope

No humanoid work (spec 23 verdict stands; revisit only after R3); no PSM
extension (density-new.tex §7.2 reasons); no DICE composition round 1; no
new index mechanisms here (spec 25 owns that); no real-robot work before
the simulation flagship exists.

## R0c verdict (2026-08-13)

**Recommendation: GO-WITH-SELF-GENERATED-DATA** for the literal air-hockey
flagship (R3). No public offline air-hockey play dataset exists, but both
testbeds ship simulators, and — reversing the spec-15 finding — the
TommyShen play datasets ARE now public and give an immediate downloadable
dynamic-contact play corpus for a warm-up leg.

### 1. Dataset audit

**(a) Robot Air Hockey testbed (Chuck et al., arXiv 2405.03113;
https://arxiv.org/abs/2405.03113).** Code: `github.com/CalCharles/air-hockey-rl`
(linked from https://rlairhockey.github.io/). Repo = Box2D paddle+puck sim
(juggle task) + real-UR5 stack, gym-style env package `airhockey/`, TD3
training, pretrained sim policies in `latest_models/canonical/`; actively
developed (pushed 2026-08-13). The paper reports 400 human teleop
trajectories (350 mouse + 50 shadow, 8 participants) and 1M sim BC
timesteps, but **neither is released** — no dataset link in repo or project
page; "code requests contact calebc@cs.utexas.edu". ⚠ Repo has **NO
license file** (GitHub API license=null) — usable only after emailing the
author. Verdict: simulator yes, offline data no.

**(b) air-hockey-challenge ecosystem (NeurIPS'23 challenge; retrospective
arXiv 2411.05718; https://arxiv.org/abs/2411.05718).** Code:
`github.com/AirHockeyChallenge/air_hockey_challenge` — **MIT**, MuJoCo,
Kuka iiwa-14, maintained (qualifying-2025 branch; docs
https://air-hockey-challenges-docs.readthedocs.io/). Data: participants
could "download the dataset obtained from the modified simulator" once per
day during the challenge, but **no permanent public dataset release** is
documented in the retrospective, repo, or docs; real-robot data explicitly
limited by hardware availability. Verdict: best-licensed simulator; no
offline data.

**(c) Interaction-resampling line (Shen et al., IWR, arXiv 2606.11525;
https://arxiv.org/abs/2606.11525; page https://iwr-arxiv.github.io/).**
The method itself is **online** CRL (SGCRL/CRTR/PPO/SAC baselines), first
real-world goal-conditioned air-hockey agent (25%→60%); the paper/page
release no air-hockey data. BUT the spec-15 finding is now stale: as of
~2026-07 the HF profile https://huggingface.co/TommyShen hosts
**`TommyShen/factored-fb-play`** (updated ~18d ago, **MIT**, ~1.35M
transitions, 9 tasks): per-episode `.npz` with
observation/action/next_observation/reward/terminal/timeout/success (state-
based, with actions), two subsets — `play-noise` (expert+action-noise
rollouts) and `play-resample` (**OGBench-style diverse-goal chaining** =
genuinely play-like coverage). Tasks: Box2D dynamic ball-pushing
(`b2d-center-ctrl`, `b2d-center-wall`, `b2d-goal-ctrl`, `-target`,
`-target-vel`, `-target-hard`, `-maze` — the IWR Box2D variants: agent
ball strikes a free ball into goal regions, contact-rich 2D dynamics, but
**not** the paddle+puck air-hockey testbed) + ManiSkill pick/push-cube +
Metaworld pick-place/push. Sibling `TommyShen/Factor-FB-Offline` (MIT,
1.57 GB, 14 tasks) is TD-MPC2 **expert** demos — wrong coverage profile
for us. Verdict: downloadable, state+actions, play-like, license-clean —
the only ready-made candidate; usable as the dynamic-contact warm-up
domain for R3, not as literal air hockey.

### 2. Self-generation cost-out

Both testbeds ship gym-style envs, so scripted noisy-play collection is
cheap. Preferred: **air_hockey_challenge (MIT, MuJoCo)** — roll out the
shipped baseline hit/defend/prepare agents with action noise + random
mixing + goal resampling (OGBench-play-style chaining), log state+action
at sim rate; MuJoCo iiwa runs ~1k steps/s/core ⇒ 1–5M transitions in
CPU-hours. Fallback: Chuck Box2D sim (faster, ships pretrained TD3
policies as scripted actors) pending a license email. Needed pieces:
play wrapper (goal chaining + noise schedule), `.npz` episode writer
matching our unified loader, `hit_spec` = puck xy(+vel block), contact-
phase labels from puck-paddle distance (feeds the R1 passive-mask reuse).
Per density-new.tex §10 a scripted/noisy-play corpus is acceptable for
the flagship **if disclosed in the note** — disclose it.

### 3. Flagship novelty check

No prior work found combining (i) reward-free/unsupervised offline
training, (ii) zero-shot user-specified terminal reward maps, and (iii)
air hockey or comparable contact-rich dynamic manipulation. Nearest
neighbors: **shen2026 IWR** (arXiv 2606.11525 — air hockey, but online,
goal-conditioned only, no reward prompts); **PSM** (Agarwal et al., RLJ
2025, arXiv 2411.19418 — zero-shot arbitrary rewards, no dynamic contact
domain); **Soft FB** (Bagatella et al., arXiv 2602.06769 — general
utilities zero-shot, standard benchmarks); **BFM-Zero** (arXiv 2511.04131
— promptable unsupervised-RL BFM, humanoid not manipulation); **FRE**
(arXiv 2402.17135 — functional reward encodings, zero-shot from offline
data, no contact-rich domain); challenge-adjacent air-hockey work is
planning/model-based/online (e.g. arXiv 2407.03705, 2406.00518). The
combination claim in density-new.tex §10 — plus the dwell/multiplicity-
robustness axis, which NONE of the above address — remains distinct.

### Go/no-go for R3

GO, contingent on self-generated data: adopt air_hockey_challenge MuJoCo
as the R3 env (MIT), self-generate noisy-play; optionally warm up the R3
port on `TommyShen/factored-fb-play` Box2D (download today, zero cost)
before the air-hockey buffer exists; email calebc@cs.utexas.edu in
parallel for the teleop data + Box2D license (400 human trajectories
would upgrade the flagship's data story if granted).
