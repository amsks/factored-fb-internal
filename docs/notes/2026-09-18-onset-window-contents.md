# What the +-5 "motion onset" windows contain (cube-single-play-v0)

Read-only data audit, 2026-09-18. Question: the rebalancing sampler
(`impls/utils/datasets.py:105-172`, `enable_persist_oversample(pool="onset", window=5)`)
upweights rows within +-5 rows of a cube motion onset. Before using it: are those rows the
securing / carrying decisions, or transitions after the cube is already moving? Do failed
grasp attempts enter the windows? Script: session scratchpad `onset_contents.py`
(numpy only, calls the real `enable_persist_oversample` so the window mask is sampler-exact);
raw numbers in `onset_contents.json` beside it.

## Indices and thresholds

Observation layout, `ogbench/manipspace/envs/cube_env.py:766-794` (state branch, one cube):
`obs[0:6]` arm joint pos, `obs[6:12]` arm joint vel, `obs[12:15]` effector xyz
(`(pos - [0.425,0,0]) * 10`), `obs[15:17]` cos/sin yaw, `obs[17]` gripper opening x 3,
`obs[18]` gripper contact (`||cfrc_ext(right_pad)||/50`, `manipspace_env.py:438-440`),
`obs[19:22]` cube xyz (same shift and x10). `obs[17]/3` counts DRIVER travel: it rises as the
Robotiq closes (`scripts/probes/cube_bottleneck/restore.py:27-32`), so "closed" = `obs[17]/3 > 0.5`
(the probe's `tau_grip`, `phases.py:41`). `actions[:,4]` is the gripper delta,
`target_opening = opening + a_gripper` (`manipspace_env.py:371-385`), so `a[4] > 0` commands
closing. Cube rests at z = 0.020 m (`banks.py:TABLE_Z`), "lifted" here = z > 0.04 m (2 cm above
rest). The sampler threshold 0.005 is in the x10 observation scale, i.e. **0.5 mm of cube
travel per step** -- a finger touch registers as motion.

Compact loader: 1000 episodes x 1001 rows; the last two rows of each episode carry
`terminals=1` and the last row `valids=0`, so the sampler's pool is rows 0..998 of every
episode, and it masks motion on the terminal row 999 (`datasets.py:137-138`). Window shifts
are in pool positions (`datasets.py:143-149`), so 0.7% of window rows come from the episode
preceding the onset's episode.

Other thresholds: "near" = ee-cube distance < 3 cm; "closing" = closed OR `a[4] > 0.2`;
"securing-like" row = near AND closing.

## 1. What an onset is (13,252 onsets in 1,000,000 valid rows; 1.33% of rows; windows cover 13.37%)

Segment = moving rows from the onset until the cube is at rest (< 0.5 mm/step) again.

| class | rule (real metres) | n | frac | median segment |
|---|---|---|---|---|
| LIFT | z rises > 2 cm above rest height in the segment | 10,974 | 0.828 | 43 steps |
| PUSH | xy travel > 2 cm, |dz| < 1 cm | 58 | 0.004 | 6 |
| DROP (onset from a still hold in the air) | rest z > 4 cm, falls > 2 cm | 0 | 0.000 | -- |
| BLIP | segment < 3 steps or total travel < 1 cm | 2,171 | 0.164 | 2 |
| OTHER | none of the above | 49 | 0.004 | 5 |

At the onset row itself (all classes): ee-cube distance median 0.38 cm, ee within 3 cm in
99.1%, closing command in 89.6%, gripper already closed in only 27.8%, contact channel > 0.5 in
16.2%, cube lifted 0%. So the onset row is the moment the fingers are closing on the cube --
the securing decision -- not a row where the object is already moving freely. PUSH onsets are
the exception (ee 6.6 cm away, never closing on the cube) and are 0.4% of onsets; the two
PUSH examples in the figure show the fingers closing fully (closedness -> 1.0) beside the
cube, i.e. a missed grasp that bumped it.

BLIPs (16%): 66.8% are followed by a LIFT onset within 20 steps (a stutter of the same
grasp: 13.1% of LIFT onsets are preceded by a BLIP), 25.2% are abandoned (ee retreats > 3 cm,
no lift within 30 steps) -- these ARE failed grasp attempts, caught because the touch moved
the cube by > 0.5 mm.

## 2. Window composition (133,706 sampler-exact rows)

| quantity | fraction of window rows |
|---|---|
| pre-onset (t <= onset) | 0.553 |
| post-onset | 0.447 |
| cube within 1 cm of its rest position | 0.851 |
| post-onset AND cube still within 1 cm | 0.303 |
| post-onset AND cube displaced > 1 cm | 0.144 |
| cube lifted (> 2 cm) | 0.065 |
| cube moving (>= 0.5 mm/step) | 0.534 |
| ee within 3 cm of cube | 0.971 |
| gripper closed (closedness > 0.5) | 0.467 |
| closing command (a[4] > 0.2) | 0.748 |
| securing-like (near AND closing) | 0.766 |
| row belongs to two or more windows | 0.089 |

By class (per window, not deduplicated): LIFT windows 0.85 still / 0.15 post-and-displaced /
0.07 lifted / 0.80 securing-like; BLIP windows 0.92 still / 0.71 securing-like; PUSH windows
0.39 post-and-displaced / 0.01 securing-like.

## 3. Failed grasp attempts

Attempt = run of >= 2 consecutive rows with ee within 3 cm, gripper closed or closing
command, cube not lifted, and no macroscopic cube motion (total travel < 1 cm) over the run
plus the next 10 steps. (The strict version "no 0.5 mm motion at all" finds 278: at that
threshold almost every closing near the cube nudges it.)

| quantity | value |
|---|---|
| runs (>= 2 rows) near+closing | 25,032 |
| ... cube moved during the run (grasp/push worked) | 22,144 |
| ... no motion during, motion within 10 steps after (slow closure) | 2,288 |
| ... attempts (no motion > 1 cm within run + 10) | 600 |
| ... of which abandoned (ee retreats > 3 cm, no LIFT within 30) | 377 |
| attempt rows | 2,075 = 0.21% of valid rows |
| attempt length median / p90 | 3 / 6 rows |
| attempt rows inside an onset window | 0.619 |
| abandoned-attempt rows inside an onset window | 0.733 |
| attempts touching a window | 0.52 |

Failed attempts are NOT excluded: about 60-70% of their rows already sit in the windows,
via the BLIP onsets. The remaining ~30-40% are closings that never moved the cube by 0.5 mm.

## 4. Carrying rows and mid-transport drops

| quantity | value |
|---|---|
| carrying rows (lifted AND moving) | 365,153 = 36.5% of valid rows (99.7% with gripper closed) |
| carrying rows inside an onset window | 0.024 |
| controlled lowerings (hand descends with the cube, grip loosens < 0.1; the old loose "drop" rule) | 1,334 events |
| real drops (lifted, z falls > 2 cm within 10 steps, AND cube separates from hand: ee-cube distance grows > 1 cm or cube falls >= 1.5 cm more than the hand) | 12 events, 45 rows = 0.0045% of rows |
| ... starting from a still hold vs inside a moving segment | 0 / 12 |
| drop rows / events inside or touching an onset window | 0.000 / 0.000 |

Even the 12 "drops" are marginal (hand descends nearly as far as the cube, 1-2 cm of
separation): the scripted play policy essentially never loses the cube mid-carry. The
mid-transport failure the manipulation analysis attributes to FB
(`docs/notes/2026-09-04-density-fb-central-perspective.md`) is a policy failure at
evaluation, not a transition type present in the data; no sampler can upweight it.

## 5. Proposed "decision" pool

Rule, on `(observations, actions, valids)` with real-metre geometry `cube = obs[:,19:22]/10`,
`eff = obs[:,12:15]/10`, `closed = obs[:,17]/3 > 0.5`, `closing = actions[:,4] > 0.2`,
`near = ||eff - cube|| < 0.03`, `lifted = cube_z > 0.04`, `mv[t] = ||cube[t+1]-cube[t]|| >= 0.0005 m`
(same episode; `valids`/terminals as the sampler), within-episode index arithmetic:

1. onset windows: `onset[t] = ~mv[t-1] & mv[t]`; rows `|r - t| <= 5` (as today);
2. attempt windows: for each maximal run `[s, e)` of `near & (closed | closing) & ~lifted & valids`
   with `e - s >= 2` and `max_{s <= u < e+10} ||cube[u] - cube[s]|| < 0.01`, rows `|r - s| <= 5`;
3. drop windows: `drop[t] = lifted[t] & (min_{k<=10} z[t+k] - z[t] < -0.02) &
   ((dist[t+k*] - dist[t] > 0.01) | (dz_cube[t..t+k*] - dz_ee[t..t+k*] < -0.015))`, `k*` the
   argmin; rows `|r - s| <= 5` around the first row `s` of each maximal run;
4. decision = union of 1-3, restricted to valid rows.

Shares of valid rows: onset windows 0.1337, attempt windows 0.0050, drop windows 0.0001;
**union 0.1368** (136,808 rows; 41% of attempt-window rows were already onset-window rows).
It can be targeted with the same `transient_mass` knob. Composition of the union: 0.756
securing-like, 0.064 carrying. It moves the pool by 2.3% relative to today's, because the
data hold almost no post-acquisition failures to add.

## 6. Figures

`PAPER/ICLR/figures/appendix/onset_windows_examples.png`: six columns (two LIFT onsets, two
PUSH onsets, one abandoned attempt, the highest real "drop"), rows = cube z [cm], cube xy
speed [mm/step], ee-cube distance [cm], gripper (closing command `a[4]` in red, closedness
`obs[17]/3` in black); the +-5 window is shaded. LIFT columns: the hand arrives, the closing
command steps up inside the window, the cube leaves the table 3-5 steps after the onset row
and the 40-step carry lies entirely outside the window. PUSH columns: fingers close to 1.0
(nothing between them) and the cube skids 1-2 cm. Attempt column: closing at the cube, cube
stays put. Drop column: a fast lowering from 33 cm with the cube slipping the last few cm;
the window sits on the carry, and there is no release command.
`onset_windows_composition.png`: onset classes, window composition, and the coverage of
failed-attempt / carrying / drop rows by the windows.

## Verdict

1. The windows weight the securing decision, not the carry: 97% of window rows have the hand
   at the cube, 77% have it closing, 85% have the cube still within 1 cm of rest, 6% have it
   lifted, and only 14% are post-onset rows with the cube displaced.
2. Carrying (36.5% of the data) is covered at 2.4%; the sampler leaves the transport phase
   at its natural weight, so it does not address FB's post-acquisition failure.
3. Failed grasp attempts are mostly inside the windows already (62-73% of their rows), because
   at 0.5 mm any finger touch counts as an onset (the BLIP class, 16% of onsets, a quarter
   of them abandoned grasps).
4. Genuine mid-transport drops do not exist in the play data (12 marginal slips in 1M rows);
   that failure mode is FB's own at evaluation and cannot be upweighted from this dataset.
5. "Motion onset" here is operationally "fingers closing on the cube"; the proposed decision
   pool (onset U attempt U drop windows) is 13.7% of rows vs 13.4% today -- a near no-op,
   so the sampler can be run as is, described as upweighting grasp-closure rows.
