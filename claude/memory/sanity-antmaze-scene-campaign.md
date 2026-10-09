---
name: sanity-antmaze-scene-campaign
description: "RUNNING (launched 2026-06-25 ~06:24 UTC): 4 sanity/control run-groups (500k x3 seeds) in tmux sanity500 — disentangles why GCIQL-bilinear-d50 scored only 24% on antmaze + extends manipulation comparison to scene."
metadata: 
  node_type: memory
  type: project
  originSessionId: 592e4344-752f-4754-87f1-5a90dc556133
---

**Why:** the antmaze GCIQL-bilinear-d50 smoke scored only **~24%** (vs OGBench
stock GCIQL ~71 @1M) while FB-flowbc-d50 got **~59%** — so the "FB beats GCIQL on
navigation" dissociation was NOT apples-to-apples (GCIQL side ran a cube-tuned
bridge config: bilinear-d50 value + flow actor + bc_coeff=3.0). Config audit vs
`third_party/ogbench/impls/agents/gciql.py` + official
`impls/hyperparameters.sh`: every numeric IQL hyperparam (lr3e-4, disc0.99,
expectile0.9, tau0.005, alpha0.3, batch1024) MATCHES; the 47pt gap is the 3
deliberate bridge mods. Official antmaze-medium GCIQL = pure defaults + alpha=0.3.

**RUNNING — tmux `sanity500`, all 500k × 3 seeds (0,1,2), launched via
`scripts/launch_sanity_antmaze_scene.sh`. Logs `/dev/shm/sanity_logs/<tag>.log`
(end marker `EXIT=<code>`).** GCIQL packed 2/GPU (XLA preallocate=false):
- **G1 `g1_s*`** stock GCIQL+DDPGBC on antmaze-medium-navigate (cfg
  `cube_single_state`, agent `agents/gciql.py`, `agent_overrides.alpha=0.3`) —
  GPUs 0/1/2. run_group `GCIQL-AntMaze-Medium-Stock-DDPGBC`. **Must reproduce ~71**
  → proves harness/data clean.
- **G2 `g2_s*`** GCIQL MLP-value + Flow actor on antmaze (cfg
  `cube_single_state_gciql_flow`, value_arch=mlp default, bc_coeff=3.0) — GPUs
  0/1/2. run_group `GCIQL-AntMaze-Medium-Flow-MLP`. Isolates bilinear-vs-MLP with
  flow held fixed (only delta vs the 24% run is value_arch bilinear→mlp).
- **G3 `g3_s*`** GCIQL Bilinear-d50 + Flow on scene-play (cfg
  `..._bilinear_d50`) — GPUs 3/3/4. run_group `GCIQL-Scene-Flow-Bilinear-d50`.
- **G4 `g4_s*`** FB FlowBC on scene-play (`train.py domain=scene`, torch .venv) —
  GPUs 5/6/7. wandb_group `fb_scene_500k`. (scene.yaml: bc3.0 ortho1.0 lr_b1e-4)

**Where results land @500k:** GCIQL → `/dev/shm/gciql_outputs/factored-fb/<run_group>/sd*/eval.csv`
col `evaluation/overall_success`. FB → torch outputs + log metric
`eval/reward/eval/success` (= `success/overall`); wandb amsks/factored-fb.

**Done check:** `tmux ls | grep sanity500`; `grep -L EXIT=0 /dev/shm/sanity_logs/*.log`.
ETA ~2.5–5h (flow-actor eval is the slow part; eval_on_cpu=1).

**UPDATE 2026-06-25:** scene done — GCIQL-bilinear-d50+flow ~0.29 ≫ FB-flowbc ~0.05 (manip
thesis holds, opposite of antmaze where FB>GCIQL). **g1/g2 antmaze STALLED**: eval_on_cpu=1
gaussian/flow rollout was glacial (~76 s/it), only step-1 logged after 9.7h → KILLED &
RELAUNCHED in tmux `phase2` (`scripts/launch_phase2_controls.sh`). FIX: g1 stock GCIQL uses
**eval_on_cpu=0 (GPU eval) → ~1.14 s/it (~67x faster)** — the cpu-eval rule is FLOW-actor-only;
gaussian GPU eval is fine and is what OGBench uses for the ~71 number. g2 flow keeps cpu eval
but 10 eps + eval@500k only. Phase2 also runs the FB z=g controls — see [[zg-identity-fb-experiment]].

**RESOLVED 2026-06-29 — the antmaze killer is the FLOW ACTOR, not the bilinear value.**
Phase2 antmaze-medium controls @500k×3 (gciql_outputs eval.csv overall_success):
- **G1 Stock-DDPGBC** (gaussian/ddpgbc actor, MLP value): 0.6/0.86/0.7 → **0.72** ✓
  reproduces OGBench ~71 → harness/data CLEAN.
- **Bilinear-d50 + DDPGBC** actor: 0.9/0.94/0.8 → **0.88** → the bilinear value is FINE
  on antmaze (even beats stock MLP value with the same ddpgbc actor).
- **Bilinear-d50 + FLOW** actor (the original 24% config): 0.34/0.16/0.16 → **0.22** ✓
  reproduces the ~24%.
- **MLP value + FLOW** actor (G2): 0.6/0.1/0.3 → **0.33** (high variance).
⇒ Holding the value fixed and swapping ddpgbc→flow actor crashes antmaze 0.88→0.22;
the value form is exonerated. The cube-tuned **flow actor (bc_coeff=3.0)** is the antmaze
killer. So the "FB>GCIQL on navigation" dissociation was an artifact of GCIQL's flow actor,
NOT a value/capacity story. Use G1 Stock (0.72) as the paper's antmaze GCIQL anchor.

Conventions: [[wandb-entity-amsks]], [[jax-gpu-packing-preallocate]],
[[always-run-in-tmux]], [[gciql-flow-agent]], [[antmaze-gciql-vs-fb-smoke]],
[[handoff-bridge-10seed-1M-campaign]].
