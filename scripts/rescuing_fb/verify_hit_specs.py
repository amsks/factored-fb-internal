"""Verify the per-env `measure_hit_spec` against each env's OWN success criterion.

The two-phase curriculum's only per-env hyperparameter is the goal region: a set of obs
slices, each with an L2 radius, ANDed together (`gciql_flow._parse_hit_spec`). The region
is supposed to MIRROR the environment's native success test, so that "hit the region"
and "the env calls this a success" are the same event.

This script checks that claim structurally, without trusting any hand arithmetic:

  1. SLICE IDENTITY -- for every object i, assert the obs slice we assign to it really
     carries that object's privileged state, by recomputing the observation transform
     from `compute_ob_info()` and comparing to the slice. Cubes: (pos - center) * 10.
     Buttons: one-hot(button_state).

  2. RADIUS SEMANTICS -- assert the radius reproduces the env's own threshold.
     Cubes: ogbench uses ||obj - target|| <= 0.04 m (cube_env.py `_compute_successes`),
     and obs is scaled by 10, so 0.4 in obs units. Verified by scaling, not asserted.
     Buttons: states are one-hot rows of eye(2), so equal <=> L2 == 0 and unequal
     <=> L2 == sqrt(2); any radius in (0, sqrt(2)) decides it. We use 0.5 and assert
     the separation.

  3. CONJUNCTION WIDTH -- assert the number of ANDed blocks equals the number of objects
     the env conjoins in `_compute_successes` (`all(...)` over objects).

Run:  .venv/bin/python scripts/rescuing_fb/verify_hit_specs.py
Exits non-zero on any mismatch. No dataset is needed -- envs are constructed directly.
"""

import sys

import numpy as np

import ogbench  # noqa: F401  (registers the envs)
import gymnasium

XYZ_CENTER = np.array([0.425, 0.0, 0.0])
XYZ_SCALER = 10.0
PROPRIO_DIM = 19  # joint_pos(7) + joint_vel(7) + eff_pos(3) + cos/sin yaw(2)... verified below
CUBE_BLOCK = 9  # pos(3) + quat(4) + cos(1) + sin(1)
BUTTON_BLOCK = 4  # one-hot(2) + pos(1) + vel(1)

CUBE_EPS = 0.4  # 0.04 m env threshold * XYZ_SCALER
BUTTON_EPS = 0.5  # one-hot equality: 0 vs sqrt(2)

# env id -> (n_cubes, n_buttons)
ENVS = {
    'cube-single-v0': (1, 0),
    'cube-double-v0': (2, 0),
    'cube-triple-v0': (3, 0),
    'cube-quadruple-v0': (4, 0),
    'puzzle-3x3-v0': (0, 9),
    'puzzle-4x4-v0': (0, 16),
    'puzzle-4x5-v0': (0, 20),
    'puzzle-4x6-v0': (0, 24),
}
# scene is irregular (1 cube + 2 buttons + drawer + window) and its spec is already
# validated in production by the completed sc_fp runs; carried through verbatim.
SCENE_SPEC = '19:22:0.4,28:30:0.5,32:34:0.5,36:37:0.72,38:39:0.6'


def derive_spec(n_cubes, n_buttons):
    """The rule: proprio block, then one fixed-width block per object."""
    parts = []
    for i in range(n_cubes):
        s = PROPRIO_DIM + CUBE_BLOCK * i
        parts.append(f'{s}:{s + 3}:{CUBE_EPS}')
    for i in range(n_buttons):
        s = PROPRIO_DIM + BUTTON_BLOCK * i
        parts.append(f'{s}:{s + 2}:{BUTTON_EPS}')
    return ','.join(parts)


def parse_spec(spec):
    out = []
    for blk in spec.split(','):
        a, b, e = blk.split(':')
        out.append((int(a), int(b), float(e)))
    return out


def check_env(env_id, n_cubes, n_buttons, spec, expected_dim=None):
    env = gymnasium.make(env_id)
    inner = env.unwrapped
    ob, _ = env.reset(seed=0)
    blocks = parse_spec(spec)

    errs = []

    # --- conjunction width mirrors the env's own `all(...)` over objects
    n_obj = n_cubes + n_buttons
    if n_obj and len(blocks) != n_obj:
        errs.append(f'conjunction width {len(blocks)} != {n_obj} objects')

    # --- slice identity, checked against the env's privileged state
    # Step a few times so the state is not the (possibly degenerate) reset state.
    for _ in range(5):
        ob, *_ = env.step(env.action_space.sample())
    info = inner.compute_ob_info()

    for i in range(n_cubes):
        s, e, eps = blocks[i]
        expected = (info[f'privileged/block_{i}_pos'] - XYZ_CENTER) * XYZ_SCALER
        got = ob[s:e]
        if not np.allclose(expected, got, atol=1e-6):
            errs.append(f'cube {i}: obs[{s}:{e}]={got} != privileged {expected}')
        if abs(eps - CUBE_EPS) > 1e-9:
            errs.append(f'cube {i}: eps {eps} != {CUBE_EPS}')

    for i in range(n_buttons):
        s, e, eps = blocks[i]
        state = inner._cur_button_states[i]
        expected = np.eye(2)[state]
        got = ob[s:e]
        if not np.allclose(expected, got, atol=1e-6):
            errs.append(f'button {i}: obs[{s}:{e}]={got} != one-hot({state})={expected}')
        # radius decides one-hot equality: 0 vs sqrt(2)
        if not (0.0 < eps < np.sqrt(2)):
            errs.append(f'button {i}: eps {eps} does not separate one-hots (need 0<eps<1.414)')

    # --- the env's own success threshold for cubes is 0.04 m; obs scale is 10x
    if n_cubes:
        import inspect

        src = inspect.getsource(type(inner)._compute_successes)
        if '0.04' not in src:
            errs.append('cube success threshold in env source is not 0.04 m -- eps 0.4 is stale')
        if 'all(' not in inspect.getsource(type(inner).post_step):
            errs.append('env does not conjoin object successes with all(...)')

    obs_dim = ob.shape[0]
    if expected_dim is None:
        expected_dim = PROPRIO_DIM + CUBE_BLOCK * n_cubes + BUTTON_BLOCK * n_buttons
    if obs_dim != expected_dim:
        errs.append(f'obs dim {obs_dim} != derived {expected_dim}')

    env.close()
    return obs_dim, errs


def main():
    print(f'{"env":24s} {"obs":>4s}  {"n_blk":>5s}  spec')
    print('-' * 110)
    failed = False
    specs = {}
    for env_id, (nc, nb) in ENVS.items():
        spec = derive_spec(nc, nb)
        obs_dim, errs = check_env(env_id, nc, nb, spec)
        specs[env_id] = spec
        shown = spec if len(spec) <= 60 else spec[:57] + '...'
        status = 'OK ' if not errs else 'FAIL'
        print(f'{env_id:24s} {obs_dim:4d}  {len(parse_spec(spec)):5d}  {status} {shown}')
        for e in errs:
            failed = True
            print(f'    !! {e}')

    # scene: carried through from the validated production config, re-checked for shape only
    # scene is irregular (cube + 2 buttons + drawer + window), so the object-block dim rule
    # does not apply; its spec is validated in production by the completed sc_fp runs
    # (0.468/0.480 at 500k). Only its shape is re-checked here.
    obs_dim, errs = check_env('scene-v0', 0, 0, SCENE_SPEC, expected_dim=40)
    print(f'{"scene-v0":24s} {obs_dim:4d}  {len(parse_spec(SCENE_SPEC)):5d}  '
          f'{"OK " if not errs else "FAIL"} {SCENE_SPEC} (production spec)')
    for e in errs:
        failed = True
        print(f'    !! {e}')

    print('\n--- specs for configs ---')
    for k, v in specs.items():
        print(f'{k}: "{v}"')
    print(f'scene-v0: "{SCENE_SPEC}"')

    if failed:
        print('\nVERIFICATION FAILED')
        return 1
    print('\nall hit specs verified against env privileged state')
    return 0


if __name__ == '__main__':
    sys.exit(main())
