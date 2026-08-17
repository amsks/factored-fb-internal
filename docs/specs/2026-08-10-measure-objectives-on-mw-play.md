# TD-FB and density-FB: porting the successor-measure agents into impls/ and running them on MW play data

**Date:** 2026-08-10 (rev 2 — direction changed: port the AGENTS into our stack, not our
data into theirs) · **Status:** SPEC — ready to execute
**Source:** `origin/gciql-fb-bridge` @ `542c058` (fetched). Both objectives live in
`tools/wandb_mode_shim/gciql_flow.py` (4147 lines) on that branch; its HEAD commit is the
"Successor-measure objectives — code guide". An inspection worktree exists at the session
scratchpad (`bridge-wt/`); keep one checked out for side-by-side reading.

## 1. Goal

Bring the two successor-measure objectives into `feat/metaworld-runs` as **two separate,
isolated agents**:

- **`density_fb`** — the density successor measure (spec 18): one-stage, from scratch,
  two-tower softmax density `f(s,a,g,u) = F(s,a,g)ᵀB(u)/τ`. Cube-single 0.929.
- **`td_fb`** — the two-phase curriculum: phase 1 first-passage (absorbing TD), phase 2
  ball-visitation with hindsight columns, warm-started from phase 1. Scene 0.927,
  cube-double 0.53, cube-single ~0.93.

Then measure both against our existing BC / monolithic GCIQL / bilinear GCIQL arms on:

- `play-resample` (v3, 2000×500 = 1M transitions/task),
- `play-noise` (low noise, 1000×100 = 100k),
- `play-noise-hard` (high noise, 1000×100 = 100k),

at `$SCRATCH/fb-data/<regime>/data/metaworld/{mw-push,mw-pick-place}.npz` (all verified
present). **Size caveat for every table:** the noise regimes are 10× smaller than
play-resample; cross-REGIME comparisons are size-confounded (see the E1 cube ladder in
`2026-08-10-deconfound-size-coverage.md`); within-regime ARM comparisons are clean.

## 2. Where the agents go, and why the interface already fits

In our stack an "agent" is a critic×actor pair composed by `impls/agents/__init__.py::Agent`
(`Agent.create(critic_name, actor_name, ...)`). The isolation the user wants — separate
agents whose changes don't touch shared code — maps to:

- `impls/critics/density_fb.py` → `DensityFBCritic`, registered as `"density_fb"`,
- `impls/critics/td_fb.py` → `TDFBCritic`, registered as `"td_fb"`,
- one registry line each in `impls/agents/__init__.py` `CRITICS`, one yaml each under
  `impls/configs/critic/`, and NOTHING else shared is modified. Each file self-contains its
  nn modules (copy the tower definitions from `gciql_flow.py` rather than reusing
  `impls/utils/networks.py` — LayerNorm/init details are part of what made these numbers,
  and self-containment is the isolation guarantee).

The critic protocol (`impls/critics/base.py`) already provides everything these objectives
need — this was verified against the code, not assumed:

- **Bootstrap actions from the learned actor**: `Agent.update` draws
  `boot = actor.sample_bootstrap(net, batch, cond, rng)` and passes it into every
  `critic.stage_loss(stage, params, target, batch, cond, boot)`. Density's TD target
  ("target towers' distribution at the bootstrap action") consumes `boot` directly.
- **Arbitrary groups/stages/targets**: `GROUPS`/`STAGES`/`TARGETS` (group→tau) — the two
  towers (or φ/ψ) are groups with their own target copies; `_LR_KEY` in
  `impls/agents/__init__.py` needs entries for the new group names.
- **Actor coupling through `q_fn`**: `FlowBCActor.loss` (bc_coeff=3.0 default — the SAME
  value the bridge measure recipes use) pulls `Q = critic.q_fn(...)`, normalizes by
  `sg[|Q|.mean()]`, adds the distill term. The measure M(s,a,g) is served as `q_fn`.
- **Hindsight columns**: `batch['value_goals']` from our GCDataset = the trajectory-future
  goals the two-phase `measure_testpoint_dist=hindsight` branch needs.
- **Pure-BC control**: `bcflow` (= flowbc with q_coeff 0) composes with any critic, so the
  BC-only floor cell comes for free.

**Known gap:** the bridge recipes set `actor_q_center: true` (subtract the occupancy's large
shared baseline before the greedy term). Our actor has no centering knob. Implement it
INSIDE the new critics' `q_fn` (subtract the batch mean of Q when the knob is on) — it
composes with the actor's `sg[|Q|.mean()]` normalization; whether that composition
reproduces their numbers is exactly what the cube anchor (P5) gates.

## 3. Implementation plan

### P1 — `impls/critics/density_fb.py`

Port these symbols from `gciql_flow.py` near-verbatim (grep the symbol, line numbers
drift): `_parse_hit_spec` (~35), `_delta_phi` (~1403), `_density_queries` (~1430),
`_raw_block_cols` (~1442), `density_loss` (~1975, the whole objective in ~40 lines), the
density tower modules from `create` (~3147), and the density branch of `actor_loss`
(~1089) — the last one becomes `q_fn` (it evaluates the measure with FROZEN tower params;
keep the stop-gradient structure).

Structure: `GROUPS = ("density_f", "density_b")` (+ targets for both, tau from config —
mirror the bridge's soft-update of both towers, `update` ~2587); single stage; `PROVIDES_V
= False`; `PROVIDES_COND = {"value_goal", "goal"}` (goal-conditioned, like gciql — the
unified adapter then feeds the eval goal frame). Config yaml `impls/configs/critic/
density_fb.yaml` with the bridge's knob NAMES kept verbatim for greppability:
`measure_hit_spec` (default `"19:22:0.4"`, cube), `measure_density_dim: 50`,
`measure_density_temp: 1.0`, `actor_q_center: true`, plus `lr` mappings.

Contract details that are load-bearing (from the guide, encode in tests):
- rewards/kernel counted at the NEXT state (`x_{t+1}`);
- `_delta_phi` eps-scales INTERNALLY: kernel term gets RAW block cols, B tower gets
  pre-scaled queries — porting must preserve which callsite feeds which;
- softmax over the QUERY axis, not the batch axis (the wrong one also trains and also
  looks plausible);
- queries resampled every update from the batch's own next-state object coords;
- NO ortho, NO mass anchor, NO scale norm on this path (each separately measured harmful;
  ortho took cube-single 0.967→0.225).
- log the train metrics under the bridge's exact names — `density_ce`,
  `density_ess_median` (MEDIAN), `density_ess_h0/h1`, `density_k_self` — so the guide's
  healthy ranges (ess 87–91 on cube, k_self ≈0.057 at eps 0.4) transfer as calibration.

Out of scope: `measure_density_behavior`/ORS (needs `next_actions` plumbing and has no
consumer here), `measure_density_hard`, EpsGainTable/NhatHead (knob-gated extras).

### P2 — `impls/critics/td_fb.py`

Port: `_parse_hit_spec` (import from density_fb or a tiny shared `impls/critics/
_measure_common.py` — acceptable shared surface: parser + hit matrix only),
`_hit_matrix` (~1384), `_measure_terms` (~1488 — its docstring is the best single
explanation; port the docstring too), `measure_loss` (~1574, keep the hindsight branch,
`tpd = ...` ~1674), the bilinear φ/ψ tower modules, `warm_start` restore semantics
(~4128).

One module, both phases — the phases differ ONLY in loss flags, so the param tree is
identical and a phase-2 warm start is a straight params restore:

| knob | phase 1 (first-passage) | phase 2 (ballvisit-hs) |
|---|---|---|
| `measure_first_passage` | true | **true** (gates the hit indicator; phase 2 drops only the absorption — setting it false would drop `hit` entirely and degenerate to a plain FB TD residual) |
| `measure_fp_absorbing` | true | **false** |
| `measure_stopgrad_psi` | false | **true** (ONLY valid with warm start — FrozenPsi trap) |
| `measure_testpoint_dist` | hindsight | hindsight |
| `measure_ortho_coef` | 0.0 (locked) | 0.0 (locked) |
| `actor_q_center` | false | true |

Assert in `build()`: `stopgrad_psi and not warm_start` is a hard error (the bridge's
assert-lattice equivalent — a from-scratch run with frozen ψ floors the cell for a reason
unrelated to any hypothesis). `measure_testpoint_dist=batch` may exist as a knob but the
configs never use it (known dead on conjunctive envs; ledger §9.16 — don't spend runs).

### P3 — Warm-start support in `impls/main.py`

New flags `--warm_start_path` / `--warm_start_epoch`, distinct from the existing
`restore_path` (which is full-state RESUME: params + opt state + rng). Warm start restores
**params and target params only** for all groups (critic AND actor, matching the bridge's
whole-agent warm start), with FRESH optimizer state and rng. Implement in
`impls/utils/flax_utils.py` beside `restore_agent`; it must not add any pytree field to
Agent (the bridge's restore-breakage gotcha).

### P4 — Tests (the executable contract — port, don't invent)

Adapt the assertions of the bridge's `tests/test_gciql_measure_density.py` (CE vs numpy
reference, simplex rows, eps-scaling, grad hygiene, default-off bit-identity) to
`density_fb`, and write the td_fb equivalents from `_measure_terms`/`measure_loss`
semantics (absorbing vs non-absorbing fixed points on a hand-built 2-state chain, hit
matrix reproduces the MW success rule on a crafted batch, hindsight columns come from
`value_goals`). Plus: registry/combo test (density_fb×flowbc, td_fb×flowbc, both×bcflow
build and run one update on dummy data), and a bit-identity test that existing critics are
untouched. Optional but valuable: a CPU parity harness that imports the bridge worktree's
`gciql_flow.py` and asserts `density_loss`/`measure_loss` agree with our port on a fixed
synthetic batch + params — if the cross-import is painful, the ported numpy-reference
tests are the fallback contract.

### P5 — Cube anchors (BLOCKING gate before any MW run)

Via the existing OGBench adapter (`env/ogbench.py`), 500k steps:

| run | expect | gate |
|---|---|---|
| density_fb × flowbc, cube-single, 3 seeds | ≈0.929 | ≥0.85 mean |
| td_fb phase 1, cube-single, 1 seed | ≈0.967 | ≥0.90 |
| td_fb phase 2 (warm from that phase 1), cube-single, 1 seed | ~0.93 | ≥0.85 |

If an anchor misses, the delta is in the port — first suspects: the `actor_q_center`
placement (P2 known gap), tower init/LayerNorm details, softmax axis, eps double-scaling.
Diff against the bridge file side-by-side; do not "tune" the MW configs to compensate.

### P6 — MW configs and protocol

Prerequisite: **E0 of the deconfound spec** (synth eval goals + distance logging) must be
in place — measure agents are goal-conditioned and inherit the fixed-frame eval bug
otherwise.

Every MW run in this program uses the E3 protocol: `obs_mask_dims=[36,37,38]
goal_mask_dims=[36,37,38] p_randomgoal=0 eval_goal_mode=synth`, 50 eval episodes, 500k
steps (match the record numbers' budget; our historical 300k baselines are NOT comparable
cells). Hit-spec **is** the env success rule, in raw meters, no scaling:
`measure_hit_spec="4:7:0.05"` (mw-push) / `"4:7:0.07"` (mw-pick-place) — the same
coincidence cube has (eps 0.4 obs-units = 4 cm = its success radius). Masking dims 36:39
never touches the hit block 4:7. Note in the run log that `p_randomgoal=0` also shapes
`value_goals` (all hindsight columns grounded — the bridge's hindsight intent); if measure
arms look goal-coverage-starved, a `p_randomgoal=0.3` ablation cell is the pre-registered
follow-up, not a silent recipe change.

## 4. Experiment matrix

Arms (all × flowbc unless noted): `bcflow` BC floor, `gciql`, `iql_bilinear` d50,
`density_fb`, `td_fb`-p1, `td_fb`-p2 (warm from same-seed p1).

- **Tier 0 — anchors (5 runs):** P5 above. Blocking.
- **Tier 1 — core (54 runs):** the 3 measure arms (density, p1, p2) × {mw-push,
  mw-pick-place} × {play-resample, play-noise-hard} × 3 seeds, PLUS baselines
  {bcflow, gciql, biql} × 2 tasks × play-noise-hard × 3 seeds (18). Baselines on
  play-resample are **E3 of the deconfound spec** (same protocol, same 500k if
  coordinated — check what E3 launched before duplicating; if E3 ran 300k, rerun at 500k
  here and say so in the table).
- **Tier 2 — low noise (36 runs):** all 6 arms × 2 tasks × play-noise × 3 seeds. With
  obs masking the old BC-saturation argument may not hold, so it's measurable — but it's
  the least discriminating regime and 100k-sized; run after Tier 1 reads out.
- Phase-2 chaining: verify the same-seed phase-1 checkpoint exists, then submit — never
  SLURM `--dependency` (purged-job failures).

### What to record

- The bridge metric names (P1) per run; `density_ess_median ≥ 4` stays a gate;
  `density_k_self` will differ from cube's 0.057 (eps- and space-dependent — record,
  compare across MW tasks, don't gate on the cube value).
- Success at 500k mean±sd, per-episode final distances (E0 logging) for matched-radius
  re-scoring (E5 of the deconfound spec).
- td_fb cells: ALWAYS quote phase-1 and phase-2 together — the p2−p1 delta is the
  curriculum's value and a negative delta is the known collapse signature.
- "Success is blind to measure validity" (guide §10): the G4 Spearman calibration probe is
  cube-specific machinery; port it only if cheap, otherwise mark calibration N/A and keep
  measure-validity claims out of the writeup.

### Pre-registered predictions

1. **play-resample is the measure agents' best shot** (1M, goal-diverse, dwell ~0.09). If
   density_fb ≥ biql-d50 there (as on cube, 0.929 vs ~0.92), occupancy parameterization
   generalizes to MW play and the cube story gains a second environment.
2. **Discriminability compression:** MW's P(relabeled goal already satisfied at 1–2×H) is
   0.249/0.155 vs cube 0.013 — hit events are dense, measure targets carry more of the
   action-independent dwell constant. Expect compressed arm separation on the noise
   regimes (dwell 0.21–0.25) and density_ess unlike cube's 87–91.
3. **td_fb p2 ≥ p1 on play-resample** (hindsight columns are exactly what goal-diverse
   play provides); on high-dwell noise data the delta may shrink toward zero.
4. If ALL measure arms sit at the BC floor while biql holds its known play-noise-hard
   guidance (+0.27 push), the measure objectives too are cube-geometry-conditional —
   sharpening the boundary story, not contradicting it.

## 5. Gotchas (their guide + our campaign history, merged)

From the bridge guide: any new knob must be provably load-bearing (they once shipped a
loss multiplied by a default-0.0 coefficient — trained nothing, logged plausibly); never
add a pytree data field to the agent (breaks restore for all old checkpoints, surfaces at
collection time — `nonpytree_field`, which our Agent already uses for critic/actor);
instrument the exact statistic a gate names (median, not mean); ortho is toxic on this
family; never add pessimism to a measure cell (0.42→0.24); FrozenPsi only with warm start.
Their `XLA_FLAGS=--xla_gpu_enable_command_buffer=` stream-capture crash was their
launcher's; our stack hasn't needed it — if `CUDA_ERROR_STREAM_CAPTURE_INVALIDATED`
appears at logging time, that flag is the known fix.

From ours: login nodes can't run JAX — SLURM only (`scripts/vista_train.slurm`); per-TAG
SAVE_DIR subdirs (run dirs don't encode overrides — the dirclash lesson); job `.o` logs
are ground truth over eval.csv; `make_config` overrides are a dotlist LIST; don't touch
`output_scale` (init crash, `networks.py:462`, still unfixed); v1 datasets
(`play-resample-v1`, `play-resample-hard-v1`) are archived defectives — never point runs
at them; the eval-RNG restore already in `env/unified.py` is why new error bars are
honest — don't compare seed-sd against pre-fix runs.

## 6. Order and cost

1. P1–P4 (agents + tests): the bulk of the work, ~1–2 days; density_fb first (one stage,
   self-contained), td_fb second (reuses parser/hit matrix).
2. P5 anchors: 5 cube runs (~2–3 h each) — gate.
3. P6 + Tier 1: 54 runs. Tier 2 after readout: 36 runs.

≈ 95 training runs total. Tier 1 answers "do the successor measures transfer to MW play
data"; the arms line up in ONE table with our RQ1 baselines for the first time.
