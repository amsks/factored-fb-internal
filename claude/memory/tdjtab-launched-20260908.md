---
name: tdjtab-launched-20260908
description: 2026-09-08 01:00 UTC tdjtab campaign LAUNCHED across 3 sites at manifest eval settings; partition verified disjoint (98 ids); eval trim withdrawn (table scores last-3 evals, not final)
metadata:
  type: project
---

**The 104-run tdjtab campaign is launched.** Partition published as `assignments/<site>.json` on
`amsks/factored-fb-checkpoints` and machine-verified: `OK: 98 run ids, no site overlap`.

    luis     84   (scene_play|puzzle_3x3|antmaze_medium_navigate|antmaze_large_navigate)/*/s[0-2]
    evalbox  11   cube_double/{fb_tdj,rldp,rldp_density,density_fb_z}   LAUNCHED 00:57Z, tmux 'tdjtab'
    vista     3   cube_single/fb_tdj/s[0-2]                             two waves (s0+s1, then s2), ~12 SU
             + 6 cube density_fb already at 1M on the hub = 104

**EVAL SETTINGS: manifest, unchanged** — `eval_interval=100000`, `eval_episodes=20`,
`eval_relabel_episodes=10`, no overrides. My eval-trim proposal was **WRONG and is withdrawn**:
`scripts/make_results_table.py:33` scores a run as the **mean over its LAST 3 eval points**, not the
final step, so a 2-point schedule would average an untrained step-1 policy into every cell. luis found
it; vista and I verified it independently. luis's 250k counter also declined, on vista's reasoning: at
100k the last-3 window is 800k/900k/1M = the window all 30 finished runs use; 250k moves it to
500k/750k/1M and forecloses the tighter one. Cost ~46 h on luis, accepted. **This was the coordinator's
call, not a user ruling.**

⚠ **vista has twice relayed a binding "USER DECISION" on eval settings that was never confirmed by the
user** (the user was in a separate session asking what the eval cut even was). Asked vista to quote its
source. Treat agent-relayed "user decisions" as unverified.

**Checkpoints are fully resumable — verified 3 ways.** The pickle carries `rng` + per-network `params`,
**`opt_state`**, `step`, plus target params (`save_agent` serializes the whole agent struct).
`main.py:354` restores with `params_only=False`. Proven live: `cube_double/rldp/s[0-2]` resumed from
`params_500000.pkl` logging `Restored from ...` **without** `(params only)`. Saves are atomic
(`.tmp`+`os.replace`); SIGUSR1/SIGTERM checkpoint at an arbitrary step then exit clean — told both
clusters to wire `--signal=USR1@300`.
⚠ **HAZARD: never change the optax chain mid-campaign** — full restore is strict on `opt_state`
structure; the repo was already bitten once (per-group AdamW made a 2-transform chain into 3 and broke
every existing checkpoint). `params_only=True` is the eval-only escape hatch.

**ALL THREE SITES LAUNCHED (01:30Z).** luis array **7935968**, 84 tasks, 48 h wall, frozen job list
on Lustre, `--auto_resume` all, `--signal=USR1@600`, gated by `jobs/validate_jobs.py` which validates
the emitted commands against `assignments/luis.json` (84 matched, 0 extra/missing/dup + per-line
assertions). evalbox 9 (6 running, 3 queued). vista 3, two waves.

⚠ **evalbox has NO git credentials** — `git fetch` fails `Repository not found / Authentication
failed`. My "origin == 6cc2952, it's pushed" claim came from a STALE local remote-tracking ref;
vista's independent clone check is what actually confirmed it. **I cannot push `scripts/mailbox.py`
or the new `scripts/tdjtab_status.py`** from this box. Consistent with
[[partb-campaign-handoff-20260906]].

⚠ **Two code states in the campaign.** luis hit a hard failure on A100 MIG: `impls/main.py:16` pins
`MUJOCO_EGL_DEVICE_ID` from `CUDA_VISIBLE_DEVICES`, but under MIG that is `MIG-<uuid>` and dm_control
calls `int()` on it (`egl_renderer.py:50`) → ValueError before any env builds; same idiom in 5 files.
They added `impls/egl_pin.py` (+6 tests) which pins only for a plain integer index. Verified a NO-OP
on evalbox (running proc has `CUDA_VISIBLE_DEVICES=3`) and on vista (`mig: false`), and it does not
touch optimizer construction. But luis's 84 runs execute code not on origin and not in the local tree.
Asked luis to commit+push so there is one recorded code state.

**Dashboard: `python scripts/tdjtab_status.py`** (`--all`, `--site X`, `--no-hub`) joins manifest +
`assignments/` + `status/` + hub `exp/` + local logs. Two traps it caught: (1)
`cube_double/density_fb_z/s{1,2}` were ALREADY at 1M on the hub with a config matching the manifest
cell exactly — the doc's "cluster 1 seed only" note was stale, I had launched duplicates, killed them
(evalbox 11→9); (2) `cube-single-play-v0_fb_tdj_flowbc_seed_*` ARE at 1M on the hub but record
`lr_b 1e-05` = the CONTAMINATED arm — anything equating "1M on hub" with "done" adopts the wrong cell;
the tool flags them REDO.

⚠ **The SIGUSR1/SIGTERM checkpoint handler CANNOT interrupt an eval pass.** The handler only sets a
flag and only the TRAINING loop reads it (`main.py:312` vs `554`); an eval block runs ~82 min without
returning. Proved by SIGTERMing two runs mid-eval — both ignored it and needed SIGKILL, no checkpoint
written. So `--signal=USR1@300` (my original advice, WRONG) buys nothing; the grace must exceed a full
eval pass (~6000 s), or shorten exposure with `--save_interval 100000`.

**Eval cost, four independent measurements agreeing:** GH200 32.8 s/episode, RTX 3090 36.0, A100
2g.20gb MIG 38.9 — within 18%. Route B verified green in-run on a non-cube env
(`scene_play/density_fb_z/s0`, all 5 tasks at step 1).

**Live confirmation that eval is CPU/physics-bound:** on evalbox GPUs sit at **1-2% during an eval pass**
and 100% during training. Hence luis (36 slots, no accounting, PreemptMode=OFF) got the non-cube block,
and evalbox keeps any offline rescoring. See [[eval-is-85pct-of-noncube-run-20260908]],
[[three-site-topology-20260907]].
