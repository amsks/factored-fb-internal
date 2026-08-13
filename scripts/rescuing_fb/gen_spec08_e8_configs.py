"""Generate the spec-08 E8 config matrix: the two-phase curriculum on ALL OGBench
state-based manipulation environments.

E8 extends E7 (cube-double + scene) to the full 9-env manipulation suite. The recipe is
frozen at the cube-single winner (0.967, `cube_single_state_measure_firstpassage_noortho_d50`)
and the ONLY thing that varies per environment is the goal region `measure_hit_spec`,
derived from each env's own success criterion and verified against privileged state by
`verify_hit_specs.py`.

Three cells per environment:
  phase1  first-passage (absorbing) measure TD, no orthonormality  -> the reaching value
  phase2  ball-visitation (non-absorbing) TD on the frozen phase-1 psi -> the measure
  anchor  GC-IQL bilinear d=50 with the same flow actor            -> the bar to clear

Existing configs are never overwritten; the script reports what it skipped.

Run:  .venv/bin/python scripts/rescuing_fb/gen_spec08_e8_configs.py [--write]
"""

import os
import sys

CFG_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'gciql')

PROPRIO, CUBE_BLOCK, BUTTON_BLOCK = 19, 9, 4


def cube_spec(n):
    return ','.join(f'{PROPRIO + CUBE_BLOCK * i}:{PROPRIO + CUBE_BLOCK * i + 3}:0.4' for i in range(n))


def button_spec(n):
    return ','.join(f'{PROPRIO + BUTTON_BLOCK * i}:{PROPRIO + BUTTON_BLOCK * i + 2}:0.5' for i in range(n))


# key -> (env_name, run_group tag, hit_spec, region description)
ENVS = {
    'cube_single': ('cube-single-play-v0', 'State', cube_spec(1),
                    '1 cube within 4cm'),
    'cube_double': ('cube-double-play-v0', 'CubeDouble', cube_spec(2),
                    'both cubes within 4cm'),
    'cube_triple': ('cube-triple-play-v0', 'CubeTriple', cube_spec(3),
                    'all 3 cubes within 4cm'),
    'cube_quadruple': ('cube-quadruple-play-v0', 'CubeQuad', cube_spec(4),
                       'all 4 cubes within 4cm'),
    'scene': ('scene-play-v0', 'Scene',
              '19:22:0.4,28:30:0.5,32:34:0.5,36:37:0.72,38:39:0.6',
              'cube + 2 buttons + drawer + window'),
    'puzzle_3x3': ('puzzle-3x3-play-v0', 'Puzzle3x3', button_spec(9),
                   'all 9 button states match'),
    'puzzle_4x4': ('puzzle-4x4-play-v0', 'Puzzle4x4', button_spec(16),
                   'all 16 button states match'),
    'puzzle_4x5': ('puzzle-4x5-play-v0', 'Puzzle4x5', button_spec(20),
                   'all 20 button states match'),
    'puzzle_4x6': ('puzzle-4x6-play-v0', 'Puzzle4x6', button_spec(24),
                   'all 24 button states match'),
}

# measured by measure_region_density.py at B=1024 (off-diagonal positives per row)
DENSITY = {
    'cube_single': 13.90, 'cube_double': 0.46, 'cube_triple': 0.03,
    'cube_quadruple': 0.01, 'scene': 2.85, 'puzzle_3x3': 2.04,
    'puzzle_4x4': 0.06, 'puzzle_4x5': 0.01, 'puzzle_4x6': 0.01,
}

HEAD = """defaults:
  - storage: shm
  - _self_
variant: state
env_name: {env}
agent_file: tools/wandb_mode_shim/gciql_flow.py
seed: 0
train_steps: 500000
log_interval: 5000
eval_interval: 50000
save_interval: 100000
eval_episodes: 50
eval_on_cpu: 1
run_group: {group}
restore_path: null
restore_epoch: null
exp_name: null
agent_overrides:
"""

BASE_OVERRIDES = """  bc_coeff: 3.0
  value_arch: bilinear
  value_latent_dim: 50
"""

MEASURE_OVERRIDES = """  measure_only: true
  measure_coef: 1.0
  measure_legs: critic
  measure_ortho_coef: 0.0
  measure_scale_norm: true
  measure_first_passage: true
  measure_hit_spec: "{spec}"
"""

PHASE2_EXTRA = """  measure_fp_absorbing: false
  measure_stopgrad_psi: true
  actor_q_center: true
"""


def phase1_doc(key, env, region, spec, dens):
    return f"""# Spec 08 E8 -- PHASE 1 of the two-phase curriculum on {env}.
# Port of the cube-single winner (cube_single_state_measure_firstpassage_noortho_d50,
# 0.967) with ONE change: the goal region. Everything else -- bilinear d=50, twin-min,
# no orthonormality, scale-normed measure, flow BC + distilled greedy actor, 500k -- is
# held fixed so the environment is the only moving part.
#
# Region mirrors the env's own success criterion ({region}):
#   {spec}
# Verified against privileged state by scripts/rescuing_fb/verify_hit_specs.py.
#
# Density pre-flight (measure_region_density.py, B=1024): {dens} off-diagonal positives
# per row, vs cube-single's 13.90. The diagonal always fires, so this is the sparsity of
# the ONLY signal that teaches many-to-one region structure. Read a floor here against
# the matching {key}_state_gciql_flow_bilinear_d50 anchor, which sees the identical
# region sparsity through its own reward.
"""


def phase2_doc(env, region, dens):
    return f"""# Spec 08 E8 -- PHASE 2 (ball-visitation) of the two-phase curriculum on {env}.
# The exact phase-2 recipe that produced 0.952/0.928 on cube-single:
#   warm-start every module from the matching phase-1 checkpoint (the launcher supplies
#     +agent_overrides.warm_start_ckpt=<run_dir> +agent_overrides.warm_start_epoch=500000)
#   measure_fp_absorbing=false   drop the (1-hit) absorption => M = E[sum gamma^t 1[region]]
#                                (genuine occupancy, revisits counted)
#   measure_stopgrad_psi=true    freeze the backward map psi carried over from phase 1
#   actor_q_center=true          centered greedy normalizer (M carries a large shared
#                                recurrence baseline that otherwise dilutes the signal)
# with the SAME goal region as phase 1 ({region}).
#
# Pre-registered read: critic scale plateaus near the Monte-Carlo visitation mass of the
# region with no inflation (cube-single: 24 vs MC truth 24.7), and native success holds at
# the phase-1 level. Inflation here = the estimator hygiene does not survive this region.
# Region density {dens} positives/row at B=1024 -- a sparse region has little revisit mass,
# so expect a SMALLER plateau than cube-single's, not a larger one.
"""


def anchor_doc(env, dens):
    return f"""# Spec 08 E8 -- ANCHOR for {env}: GC-IQL with a bilinear d=50 critic and the same
# flow actor class as the measure cells. This is the bar the two-phase curriculum must
# clear, and it is env-specific: the note's 0.90 bar is cube-single-only, and scene's
# turned out to be 0.292 (stock-MLP 0.516, FB 0.047).
#
# It is also what makes a FLOOR interpretable. The anchor is trained on the same data with
# the same actor and the same goal-reaching event, so if both anchor and measure cells
# floor on this env, the env is hard for goal-conditioned TD generally rather than for the
# measure objective specifically. Region density is {dens} positives/row at B=1024.
#
# 500k steps to match the campaign (the cube-single bilinear-d50 anchor config is 1M; the
# launcher overrides train_steps for that env so every anchor in the table is at 500k).
"""


def build():
    out = {}
    for key, (env, tag, spec, region) in ENVS.items():
        dens = DENSITY[key]
        out[f'{key}_state_measure_firstpassage_noortho_d50.yaml'] = (
            phase1_doc(key, env, region, spec, dens)
            + HEAD.format(env=env, group=f'GCIQL-{tag}-Flow-Measure-FirstPassage-NoOrtho-d50')
            + BASE_OVERRIDES + MEASURE_OVERRIDES.format(spec=spec) + 'wandb_mode: online\n')
        out[f'{key}_state_measure_ballvisit_d50.yaml'] = (
            phase2_doc(env, region, dens)
            + HEAD.format(env=env, group=f'GCIQL-{tag}-Flow-Measure-BallVisit-FrozenPsi-d50')
            + BASE_OVERRIDES + MEASURE_OVERRIDES.format(spec=spec) + PHASE2_EXTRA
            + 'wandb_mode: online\n')
        out[f'{key}_state_gciql_flow_bilinear_d50.yaml'] = (
            anchor_doc(env, dens)
            + HEAD.format(env=env, group=f'GCIQL-{tag}-Flow-Bilinear-d50')
            + BASE_OVERRIDES + 'wandb_mode: online\n')
    return out


def main():
    write = '--write' in sys.argv
    cfgs = build()
    wrote = skipped = 0
    for name in sorted(cfgs):
        path = os.path.normpath(os.path.join(CFG_DIR, name))
        if os.path.exists(path):
            print(f'  SKIP (exists)  {name}')
            skipped += 1
            continue
        print(f'  {"WRITE" if write else "would write"}    {name}')
        if write:
            with open(path, 'w') as f:
                f.write(cfgs[name])
        wrote += 1
    print(f'\n{wrote} new, {skipped} existing (left untouched)')
    if not write:
        print('re-run with --write to create them')
    return 0


if __name__ == '__main__':
    sys.exit(main())
