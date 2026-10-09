# spec18 §7 bookkeeping: register spec-17 run groups in the living results CSVs

## What was added

`scripts/results_map.py` (appended before the closing `}` of `MAP`, after the
existing spec07 entries — did not touch anything else in the file, which was
already dirty from another track's work):

```python
'GCIQL-CubeSingle-Flow-Measure-Delta-d50': dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='measure-delta',
                                         graft="spec17a (variant A): signed event target hit(x')-hit(x) on the region successor measure "
                                               "(temporally-centered, non-absorbing; deletes the dwell-plateau level channel from supervision)"),
'GCIQL-CubeSingle-Flow-Measure-DeltaPartition-d50': dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='measure-delta-partition',
                                         graft='spec17b (variant B): one-stage K=256 simplex successor measure over fixed hit-spec prototypes '
                                               '(delta_head CE-TD to sg[(1-gamma)phi(x)+gamma*p_bar]; actor q = p.phi(goal), frozen head)',
                                         notes='d17_cs_part_s1 crashed twice with a CUDA fault (dirs preserved as '
                                               'd17_cs_part_s1.crashed_70k_cuda_1043 / .crashed_230k_cuda_1138) and is re-running from scratch '
                                               'as of 2026-08-06; those two dead dirs are NOT completed runs -- see compile_audit.txt partial handling'),
```

Both routed to CSV2 (`bridge_grafts.csv`), env=`cube` (matching every other
cube-single measure entry in the map), agent=`GCIQL`, actor=`fql_flow`
(`actor_kind: "flow"` confirmed in `flags.json`) — same convention as every
neighboring spec06/07/08 measure-variant row.

## Refresh run

`.venv/bin/python scripts/compile_results_csvs.py --refresh` (foreground,
~90s, exit code 1 — the 1 is PRE-EXISTING and unrelated: 36 groups from other
in-flight campaigns (ZX/Dense/DelayedActor/FrozenActor/Occ-Phase2/etc., plus
3 smoke groups) are unmapped in `results_map.py` as of this session's start;
none of them are spec-17 groups and I did not touch them per the "don't
revert/reformat unrelated entries" instruction).

```
wandb: [wandb.Api()] Loaded credentials for https://api.wandb.ai from /home/adityamohan/.netrc.
[compile] /home/adityamohan/git/Factored-FB/results/summary_agents.csv: 98 rows
[compile] /home/adityamohan/git/Factored-FB/results/bridge_grafts.csv: 175 rows
[compile] audit -> /home/adityamohan/git/Factored-FB/results/compile_audit.txt  !! 36 UNMAPPED
```

## Compiled spec-17 rows (`results/bridge_grafts.csv`)

```
cube,GCIQL,fql_flow,measure-delta,...,0.2067,0.0634,3,"0.152 0.264 0.204",...,GCIQL-CubeSingle-Flow-Measure-Delta-d50,
cube,GCIQL,fql_flow,measure-delta-partition,...,0.884,0.0,2,"0.884 0.884",...,partial_seeds="d17_cs_part_s1(died@70000) d17_cs_part_s1(died@230000) d17_cs_part_s1(died@45000)",GCIQL-CubeSingle-Flow-Measure-DeltaPartition-d50,...
```

- **Delta (17a)**: mean 0.2067, n=3, per-seed `0.152 0.264 0.204` — **exact
  match** to the ground truth (s0=0.152, s1=0.264, s2=0.204).
- **DeltaPartition (17b)**: mean 0.884, n=2, per-seed `0.884 0.884` — **exact
  match** to the ground truth (s0=0.884, s2=0.884), and n_seeds correctly
  drops to 2 (s1 excluded).

No discrepancies against the ground truth.

## Crashed/partial dir handling — verified, not a bug here

The compiler doesn't glob local run directories at all; it queries the wandb
API by `run_group` and, per run, only accepts a value if the run's logged
step (summary or nearest history point) falls in `[490k, 510k]`
(`STEP_LO`/`STEP_HI`). Three separate wandb run objects were all logged under
the *name* `d17_cs_part_s1` (the two crashed attempts + the current restart's
progress at fetch time), dying at steps 45k/70k/230k — all far below
`STEP_LO`. All three landed in `partial_seeds` (`died@70000`, `died@230000`,
`died@45000`) and were correctly **excluded** from `vals`/the mean — the
machinery did not fold the crashed 230k run in with the two completed 500k
seeds.

**Caveat worth flagging (not a bug found here, but a latent risk in the
shared script, left as-is per instructions):** the exclusion works because
these crashes happened well outside the `[490k,510k]` window. A run that
crashed or was killed *just under* 490k (or a resumed run whose wandb id
changes but whose last logged step lands inside the window before the
current attempt catches up) would be silently accepted as a full "seed" with
no distinguishing marker beyond the `@<step>` suffix in `per_seed` — there is
no crash/finished-state check, only a step-window check. That's an existing
property of `compile_results_csvs.py` (STEP_LO/STEP_HI gate is the only
safeguard), not something introduced or fixed by this task; I did not modify
the compiler.

## STATUS

Done. `scripts/results_map.py` updated (2 new entries, additive only);
`results/bridge_grafts.csv` and `results/compile_audit.txt` regenerated via
`--refresh`. Both spec-17 rows compiled and match the given ground truth
exactly (delta 0.152/0.264/0.204; part 0.884/[s1 excluded]/0.884). No changes
made to `results/summary_agents.csv` (not applicable) or any other
`results_map.py` entries. Files touched: `scripts/results_map.py`,
`results/bridge_grafts.csv`, `results/compile_audit.txt`. No git actions
taken.

## Concerns

- The pre-existing 36 UNMAPPED groups (other tracks' campaigns) still make
  `compile_results_csvs.py` exit 1 — not spec-17's concern, left alone, but
  worth someone else's attention before this is treated as a clean CI-style
  gate.
- `d17_cs_part_s1` is still not finished; re-run this refresh once it
  completes so the DeltaPartition row picks up 3 seeds instead of 2.
