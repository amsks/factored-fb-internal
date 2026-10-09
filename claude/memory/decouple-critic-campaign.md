---
name: decouple-critic-campaign
description: "DONE (2026-06-29): decouple-actor-z × {IQL, zHER} cube runs — both WORSE than plain FB; decoupling actor-z does not rescue the flow actor in any cell"
metadata: 
  node_type: memory
  type: project
  originSessionId: 7cf3b146-6071-4cb2-bea7-dd5fc8d584d6
---

RUNNING (launched 2026-06-26, tmux `decpl2`, GPUs 0-5): completes the
decouple-actor-z matrix that [[handoff-iwr-program]] / phase2 left half-done.
Script `scripts/launch_decouple_critic.sh` (cube-single, flow actor `fb_flowbc`,
1M×3 seeds each, decouple_actor_z=true train_goal_mode=traj train_goal_ratio=0.5):

- `fb_cube_decouple_iql` — decouple + `critic_mode=iql` (the real gap; never run before)
- `fb_cube_decouple_zher_traj` — decouple + `critic_mode=measure`, traj/0.5 (clean
  one-variable comparison vs `fb_state_zher`=0.32; phase2's `fb_cube_decouple` only
  ran measure+**perm**/0.5 = 0.39, so it was a 2-variable change, not isolated)

**Why:** phase2 showed decoupling actor-z does NOT help the flow actor (decouple
0.39 ≈ coupled zher 0.32 / iql 0.39, all below fb_state_plain 0.48), but the
decouple×IQL cell was missing and the decouple×zHER cell wasn't a clean ablation.

**How to apply:** ~5h/run (~55 FPS). Resume = pull final `evaluation/overall_success`
from wandb amsks/factored-fb for the two groups; compare vs fb_state_zher (0.32) /
fb_state_iql (0.39) / fb_state_plain (0.48). Outputs in volatile /dev/shm/decpl2_out.
See [[tmux-sessions-registry]].

**RESULTS @1M (DONE 2026-06-29, EXIT=0, success/overall from decpl2_logs):**
- **decouple-IQL** (`fb_cube_decouple_iql`): 0.36/0.22/0.04 → mean **~0.21** (s2 collapsed).
  vs fb_state_iql 0.39, plain 0.48 → WORSE. (Identical numbers to phase2 `iqllb`
  = IQL+learned-B+ratio1 — the decouple knob added nothing on top of IQL+learned-B.)
- **decouple-zHER** (`fb_cube_decouple_zher_traj`): 0.24/0.16/0.14 → mean **~0.18**.
  vs fb_state_zher 0.32, plain 0.48 → WORSE. (Earlier phase2 `decpl`=measure+perm/0.5
  = 0.44/0.50/0.24 mean 0.39 was the best decouple cell, still < plain 0.48.)
- **VERDICT:** decoupling actor-z does NOT help the flow actor in the IQL cell or the
  clean traj-zHER cell — completes the matrix, confirms phase2's negative. Nothing inside
  the FB framework (decouple/IQL/zHER/IWR) recovers controllability; binding constraint
  stays end-to-end coupling. **Backed up** to ~/bridge_ckpt_backup/decpl2_out (6 final.pt).

⚠️ **GOTCHA — under `critic_mode=iql`, `decouple_actor_z` AND `train_goal_ratio` are
both INERT.** decouple-IQL (decouple=true, ratio=0.5) is **byte-identical** across all 21
eval checkpoints to phase2 `iqllb` (decouple=false, ratio=1.0, fixed_b=none) — same seed →
same RNG → same run. So decouple-IQL is NOT an independent data point; it collapses onto the
learned-B/ratio=1 IQL control (21±40). decouple only does anything in **measure mode**
(perm 39±34 / traj 18±13, both < plain FB 43). Don't double-count diql as a separate row.

**WRITTEN to paper (2026-06-29):** `PAPER/fb-gciql-bridge/fb-gciql-bridge.tex` §3.7 — added
2 measure-mode decouple rows + promoted the z=g block (controls iqllb 21, her_idnorm 20) from
"in-flight" to final; PDF rebuilt (/tmp/tectonic). AntMaze §4.1 left as-is (existing stat tighter).
