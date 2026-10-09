# Figure plan — effective controllability (2026-09-04)

Framing: **effective controllability** of a state factor = fraction of dataset
transitions on which it changes. Arm ≈ 1; cube .52 (single), .28 (each, double),
drawer/window .09 (scene); walker ≈ 1 for every factor.
Phenomenon: FB's successor estimate has fidelity proportional to a factor's effective
controllability; when the reward depends on a weakly controlled factor, zero-shot control
fails. Normalized successor estimation (Density-FB) restores fidelity.

Checkpoints. **Primary = seam jax, matched harness**: `fb_ref` (3 seeds) vs
`density_fb_state` (3 seeds), cube-single 500k; cube-double `density_fb_state` 0-2,
`fb_ref` 0. Loader/restore/banks: `scripts/probes/cube_bottleneck/`.
**Replication = rlbrew torch 1M HF checkpoints** (FB, PSM, RLDP, GCIQL, state + pixel):
FB-side figures repeated in a second implementation; Density-FB has no torch twin, so
the torch panels show the failure only. Do not mix 1M torch and 500k jax in one panel.
Lesson from the probe: every measure is conditioned on its own policy; any "truth" from
rollouts must be rolled under the arm's own actor and labelled so.

---

## Fig 1 — HERO: where will the cube be? (predicted occupancy heatmaps)

Rows: one pre-grasp contact state (P2) and one held state (P3), from the probe banks.
Columns: **truth | FB | Density-FB**. Cell: top-down heatmap over the table (cube xy),
colour = discounted occupancy of the cube position, goal marked.
- Predicted: evaluate the successor estimate on the 1024-row dataset pool
  (`measure_probe.fb_readout` / `density_readout`), bin pool rows by cube xy, sum mass
  per bin (FB: clip negatives, normalise; Density-FB: softmax weights as they are).
- Truth: 20 rollouts of the arm's OWN actor from the same state, H = 100, discounted
  histogram of cube xy. Labelled "under its own policy".
- Inset per cell: mass split into **cube moved / cube unchanged** (Δcube > 3 cm).
Expected: truth and Density-FB put mass along the path to the goal; FB puts it at the
cube's current position. Second row of insets (appendix): same split for the arm
(Δeffector) — FB spreads mass over arm motion with the cube fixed.
Data: none new beyond ~6k rollout steps per state.

## Fig 2 — fidelity along a trajectory (the rlbrew "hazy" figure, made quantitative)

Panel A: a held-start rollout. x = step. Lines: true cube x-position; cube position
**decoded from the critic's backward embedding** by a linear probe fit on the dataset
(`B(s)` for FB, `B` tower for Density-FB); same for effector position. FB: effector
decode tracks, cube decode flat-lines during transport; Density-FB: both track.
Panel B: per-phase R² of that probe, cube vs effector, per method, 3 seeds — the
representation-level fidelity number. Torch HF checkpoints extend Panel B to PSM, RLDP,
GCIQL (FB-side replication, state and pixel).
Existing: fb_cube_failure Finding 3 (R²(B→d) .39 state / .76 pixel) is this figure's
ancestor; reuse `evals/representation_profile.py` for the torch side.

## Fig 3 — the aggregate: success vs effective controllability, all methods

x = effective controllability of the task factor (walker 1.0, cube-single .52,
cube-double .28, scene .09). y = success (walker: return, own axis). One line per
method: FB, RLDP, PSM, HILP and their density variants at **state input**.
Existing cells: privcoord table (cube-single/double), tab:main (scene: Density-FB .64,
FB 0), walker tdj (rldp 574 / rldp_density 362 / hilp 768 / hilp_density 730,
Density-FB 756 / FB 760). **Missing**: RLDP/PSM/HILP + density on scene (Wave D),
PSM on cube-double, walker for PSM. Companion table with seeds per cell.
Also the **clean dial**: walker passive-object, controllability set to 1.0/.5/.3/.1 by
construction, FB vs Density-FB vs first-passage, 3 seeds — one curve without the task
confound (36 runs, spec to write).

## Fig 4 — bar plots: value fidelity per factor (the mechanism number)

At the 64 held-bank states and the 5 task goals, per method and seed:
- **cube-goal sensitivity**: std of Q(s, a_data, goal) when the goal's cube position is
  moved over a grid on the table, arm pose of the goal fixed.
- **arm-goal sensitivity**: std of Q when the goal's arm pose is varied (draw from
  dataset states with the same cube position), cube position fixed.
Bars: two factors × methods, cube-single and cube-double. Expected: FB high on arm, low
on cube; Density-FB the reverse (note: by construction its goal reward is on the object,
say so in the caption; the FB bar is the finding). No rollouts.
Second bar group: **held-start success** H0 vs H1 per method (from Test 1) — "giving FB
the grasp does not help", the control-bottleneck refutation in one bar pair.

## Fig 5 — illustrative schematic (no data)

Left: play trajectory, arm path dense, cube path a few short segments; below it the
per-transition "factor changed" strips: arm ~all steps, cube ~half. Middle: the
regression target FB sees for the cube factor — a tall bar at "unchanged" and a thin
tail — vs the normalized distribution Density-FB fits, same shape but on the simplex,
with the tail visibly resolved. Right: consequence — value as a function of the cube
goal: flat (FB) vs peaked-and-decaying (Density-FB).

## Fig 6 / tables — performance

- Goal-conditioned: `table_main_1m.tex` (1M, 3 seeds; cube-single/double/scene/antmaze,
  Density-FB soft/hard × flow/DDPG+BC vs FB, BC, GCIQL, HIQL).
- Zero-shot reward: `experiments.tex` §reward-transfer + the centered dual index (.82
  cube-single); state the interface (reward samples, M ≥ 256).
- Privileged coordinates: `table_privcoord.tex` as is.
- ExoRL parity: walker same-protocol (ours vs td_jepa) + official-RLDP (0.27× → 0.76×
  with live ortho) — appendix; the headline is "neutral where controllability ≈ 1".

## Order of work
1. Fig 4 and Fig 1 predicted panels: eval-only on seam checkpoints, one day.
2. Fig 1 truth rollouts + Fig 2 probes: two days incl. torch replication.
3. Fig 3 missing cells: scene retrofits (Wave D impl), then the walker dial (36 runs).
