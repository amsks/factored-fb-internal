# FB (torch) config matrix — deltas vs the base cell

Unlike GCIQL (one YAML per cell), FB variants are **hydra override sets in launch
scripts**. This file collects every FB cell we ran, as deltas against the common base.

**Base = `configs/agent/fb_flowbc.yaml`** (inherits `configs/agent/fb.yaml`) on
`domain=cube_single`, with the campaign-common overrides used by every launcher:

```
agent=fb_flowbc domain=cube_single ortho_coef=1000 lr_b=1e-4 bc_coeff=3.0
```

i.e. plain FB + FlowBC actor, learned B (`fixed_b=none`), measure critic
(`critic_mode=measure`), uniform buffer goals (`train_goal_mode=perm`,
`train_goal_ratio=0.5`), z_dim=50. **This base = `fb_state_plain`/`fb_flowbc`,
the 0.48 @1M anchor.** (wandb gotcha: group `fb_state_L256` is ALSO plain FB —
see memory `fb-state-run-name-map`.)

Key knobs live in `configs/agent/fb.yaml` (lines 33–46) and are implemented in
`agents/fb/agent.py` / `agents/fb/model.py`.

---

## 1. Core ablations (`scripts/launch_bridge_10seed_1M.sh`, 10 seeds × 1M)

| wandb group | delta vs base | what it asks |
|---|---|---|
| `fb_state_zher` | `train_goal_mode=traj` | zHER: z from own-trajectory future instead of uniform buffer goals |
| `fb_state_iql` | `train_goal_mode=traj q_loss_coef=1.0` | aux Q-MSE loss on top of the measure critic |
| `fb_state_iqlcritic_fix` | `train_goal_mode=traj critic_mode=iql` | REPLACE measure loss with goal-conditioned expectile (IQL) critic on F·z |

## 2. z=g identity-B (`scripts/launch_zg_cube.sh`, 3 seeds × 1M)

Extra common overrides: `fixed_b=identity train_goal_mode=traj train_goal_ratio=1.0`

| wandb group | delta vs base | result |
|---|---|---|
| `fb_cube_zg_iql` | + `critic_mode=iql` | **0.44** — learned B is FB's bottleneck |
| `fb_cube_zg_her` | + `critic_mode=measure` | ~0.19 |

## 3. Phase-2 confound controls (`scripts/launch_phase2_controls.sh`, 3 seeds × 1M)

Extra common overrides: `train_goal_mode=traj train_goal_ratio=1.0`

| wandb group | delta vs base | what it controls |
|---|---|---|
| `fb_cube_her_learnedB_r1` | + `fixed_b=none critic_mode=measure` | is identity B the problem? (vs zg_her) |
| `fb_cube_her_idnorm_r1` | + `fixed_b=identity_norm critic_mode=measure` | does bounding ‖z‖ fix the measure blowup? |
| `fb_cube_iql_learnedB_r1` | + `fixed_b=none critic_mode=iql` | single-var control for zg_iql 0.44 → **0.21** |

## 4. Decouple-actor-z (`scripts/launch_decouple_critic.sh`, 3 seeds × 1M)

Extra common overrides: `decouple_actor_z=true train_goal_mode=traj train_goal_ratio=0.5`

| wandb group | delta vs base | result |
|---|---|---|
| `fb_cube_decouple_iql` | + `critic_mode=iql` | ~0.21 |
| `fb_cube_decouple_zher_traj` | + `critic_mode=measure` | ~0.18 — decoupling never rescues the flow actor |

## 5. FB→GCIQL morph knob ladder (`scripts/launch_bridge_knobs.sh`, 3 seeds × 500k)

Ladder-specific base (NOT the campaign base): `fixed_b=none critic_mode=iql
train_goal_mode=traj ortho_coef=0 lr_f=3e-4 lr_b=3e-4 lr_actor=3e-4 bc_coeff=3.0`
and `GMIX = critic_goal_mix=[0.5,0.2,0.3] actor_goal_mix=[1.0,0.0,0.0]` (GCIQL's mixes).

| wandb group | delta vs ladder base |
|---|---|
| `fb_bridge_a0_cubexyz` | `iql_reward=cube_xyz critic_goal_mix=[1,0,0]` (privileged reward, traj goals only) |
| `fb_bridge_r0_base` | `iql_reward=goal_match` + GMIX (GCIQL's reward + goal sampling) |
| `fb_bridge_r1_decond` | r0 + `decondition_F=true` (φ(s,a) task-agnostic) |
| `fb_bridge_r2_dnorm` | r0 + `d_norm=true` (F·z / √d scaling) |
| `fb_bridge_r3_allon` | r0 + `decondition_F=true d_norm=true` (full bilinear-GCIQL form) |

Full run→success map: memory `fb-state-run-name-map` + `results/` CSVs
(`summary_agents.csv`, `bridge_grafts.csv`).

## 6. Goal-conditioned FB branch cells (pulled runs, `scripts/pull_fb_runs.sh`)

| wandb group | delta vs base | knob |
|---|---|---|
| `fb_state_v1` | `goal_cond=true` | V1: explicit goal obs into forward map + actor |
| `fb_state_v2` | `fixed_b=cube_xyz` | V2: privileged USFA backward map B=cube xyz (z_dim→3) |

## 7. SG-ISFB (torch) — `configs/agent/isfb_flowbc.yaml` (a real YAML, 3 seeds × 500k)

Delta vs base: `agent=isfb_flowbc`, `ortho_coef=1.0`, `train_goal_mode=traj
train_goal_ratio=1.0`, `iql_expectile=0.9`, `isfb_goal_mask=true`,
`alpha_n=1.0 alpha_m=1.0 awr_beta=1.0 awr_wmax=100`, and the cell axis
`measure_insample ∈ {coord, scalar_gate_hard, scalar_gate_awr}`
(configs also mirrored as `configs/gciql/cube_single_state_isfb_*.yaml` for the JAX shim).
Result: NULL — coord 0.33 / awr 0.27 / hard 0.08, all < plain 0.48.

## 8. Knobs that exist but were folded into other cells

`bilinear_v` (FB-IQL bilinear V head), `onestep`, the IWR/coverage reweight family
(`reweight_alpha/reweight_mode/iwr_*`), `weight_z` — all default-off in `fb.yaml`;
IWR/coverage was its own earlier (pre-bridge) campaign.

---

*Companion file: `config-matrix-gciql.md` (auto-generated per-YAML deltas for the JAX side).*
*Anchors: plain FB 0.48 · zg_iql 0.44 · FB-IQL 0.21–0.24 · bilinear GCIQL 0.90 @1M.*
