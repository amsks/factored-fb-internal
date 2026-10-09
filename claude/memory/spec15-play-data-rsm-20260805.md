---
name: spec15-play-data-rsm-20260805
description: "2026-08-05 spec15 WRITTEN: play-data survey + dwell empirics + RSM (renewal successor measure) proposal — occupancy = arrival x dwell fits r>=0.99 on 7 datasets (OGBench single/double/triple/quad/scene + ManiSkill/Metaworld expert); dwell GROWS with clutter (22->140 steps); NO published work reports dwell/revisit stats of play data (open territory); TommyShen play-noise/play-resample NOT actually on HF yet (repos = TD-MPC2 expert demos + BC teacher ckpts; tl-play-data code private/unreachable)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 1ba5ea79-da0c-4583-a89e-0a11cfad5238
  modified: 2026-08-05T23:38:52.349Z
---

Bridge track, follows [[stack-divergence-audit-20260805]]. User asked to (1)
validate the play-data setting with references, (2) test whether the
dwell/revisit phenomenon transfers, (3) brainstorm a less hand-crafted
algorithm than reaching-TD + measure two-phase.

**Docs written (both untracked in PAPER/):**
- `PAPER/rescuing-fb/specs/15-play-data-and-unification.md` — the spec:
  semi-Markov/renewal reading of play data, RSM algorithm, empirics, OGBench
  collection mechanics, TommyShen premise correction, E3/E4/E5 plan.
- `PAPER/rescuing-fb/play-data-survey.md` — annotated lit survey (Lynch
  play-LMP 2019 -> CALVIN/TACO -> OGBench -> PlayWorld 2026; the standard
  justifications with quotes; "playworld" = Yin et al. 2026 arXiv:2603.09030,
  Lynch's sim is "Playground/3D Playroom").

**Key empirics (agent-computed, script scratchpad dwell/dwell_stats.py —
COPY IT SOMEWHERE DURABLE before the scratchpad dies):** 50 eps-ball centers
per dataset, eps=4cm, gamma=.99. Mass model P(arrive)xE[gamma^t_first]x
(1-gamma^L)/(1-gamma) fits cross-region mass at log-Pearson .991-.9995 on
ALL of: cube-single/double/triple/quadruple-play, scene-play, ManiSkill
pick-cube (expert), Metaworld bin-picking (expert). Mean sojourn 22/60/120/
140 steps single->quad while P(arrive) drops .38->.10; longest run carries
86-99.9% of in-region time; revisit prob .2-.43 (secondary). Run lengths
BIMODAL (grazing passes + rests) => a single dwell constant is wrong, a
regression D(g) is needed. OGBench collector mechanics (verified in
third_party/ogbench): play = open-loop plan oracle + smoothed plan noise,
immediate subtask resampling, ~11 pick-places/1001-step ep; dwell grows with
object count because each cube is re-picked w.p. 1/#top-blocks.

**RSM proposal (spec15 §3):** H(s,a,g)=phi(s,a)^T psi(g) first-passage TD
(= our working phase 1 / jax hindsight arm, goal-cond NOT z-cond) +
supervised dwell head D(g) regressed on placement events (no TD, no
sparsity problem) + readout z_r = E[psi(g) D(g) r(g)], Q = phi^T z_r; task
vector exists ONLY at inference; actor trained on H alone. Kills phase 2,
mass anchor, nonneg hinge, actor freeze. Pre-registered failure mode:
action-dependent dwell rewards ("hold in air"). Next: E4 (eval-only RSM
readout on existing cube-single ckpts), E3 (dwell action-independence
regression), E5 (RSM on jax hindsight cd/sc vs spec14 anchors 0.872/0.45).

**TommyShen RESOLVED (08-05 evening):** user fetched `origin/tl-play-data`
(box git creds dead — user ran `! git fetch` interactively). Collectors
read; spec §6b-RESOLVED has the scored checklist. play = TD-MPC2
stochastic prior + eps/Gaussian noise, one goal/ep (the noise CONTROL);
play-resample = continuous goal-chaining (reach→hold 8→resample valid
goal, no reset, goal-switch markers recorded) = OGBench structure with a
reactive LEARNED controller instead of the scripted plan oracle (⇒ cousin
of OGBench-noisy on the controller axis). Data itself STILL unreachable:
doc's HF id `TommyShen/factored-fb-play` 404s; files on Tommy's box at
/raid/tongle/rl/fb-data/collected_play{,_resample}/. Pre-registered: dwell
factorization holds there but with SMALL dwell (hold≈8) ⇒ dose-response
test of the FB pathology. Local ~/git/datasets/Factor-FB-Offline{,-tdmpc2}
= just the July expert-demo/checkpoint snapshots, no play files.

**E2 CLOSED 2026-08-05 late: independent transfer MEASURED.** Tommy's v2
Metaworld play-resample (scripted Sawyer oracle, goal re-draw on
success+dwell/timer, local ~/git/datasets/factoredfb-metaworld-play, 2
regimes x {push,pick-place}): arrival×dwell fit log-Pearson .91-.99,
ratio 1.0-1.24, runs 27-55 steps mean. NEW: revisit channel BIGGER than
OGBench (.45-.64 vs .2-.4; first sojourn 74-80% of mass vs 86-95%) ⇒
define D(g) as per-episode in-region mass given arrival (fold revisits),
not first-sojourn-only. Battery scripts/rescuing_fb/dwell_mw_play.py,
results/dwell_mw_play.json. ⚠ obs[36:39] = the collector's goal INSIDE
the observation — mask via goal_mask_dims=[36,37,38] for any critic
training on this data or BC reads the answer off its input. Object xyz =
achieved_goal (= obs[4:7]); success radius .05 push / .07 pick-place.

**External review integrated 2026-08-05 (user-supplied assessment, mostly
accepted):** (1) resampling does NOT distinguish OGBench play vs noisy —
BOTH resample; axes = controller class + noise structure (spec §6b
corrected); (2) cross-dataset step counts incomparable across control
frequencies (40/20/10 Hz) — report seconds or γ(Δt)=exp(-Δt/τ) before any
cross-family paper table; (3) new experiments E6 (CALVIN scene_obs dwell
battery = external validity) and E7 (CAUSAL: first-hit-truncate/dedup/
reweight cube-single, retrain FB — turns the dwell story into an
intervention); (4) terminology = "persistent occupancy/post-arrival dwell"
not "revisitation"; (5) ε=4cm = OGBench's own success radius (answer to
the hand-tuned-ε objection); (6) PLATO reference added (verify citation).
⚠ The review quotes draft statistics (48% stationarity, median run 44,
25-31 post-arrival visits, regressor rank corr .60 vs .06) that are NOT
from this session's audit — different statistic definitions (whole-cube
stationarity vs our per-ε-ball runs); reconcile before quoting either set.

Related: [[p2-multienv-decay-mechanism-20260730]] (the phase-2 record RSM
explains), [[validmeasure-campaign-20260729]] (mass-anchor ordering loss =
TD-on-product pathology), [[firstpassage-measure-works-20260722]].
