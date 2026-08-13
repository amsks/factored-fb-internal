"""Pre-flight: how sparse is each env's goal region at the training batch size?

The two-phase curriculum trains on an n x n matrix of hindsight pairs: rows are next
observations s'_i, columns are relabeled goals g_j = s'_j, and the target is the region
indicator h_ij = AND over `measure_hit_spec` blocks of ||s'_i[a:b] - g_j[a:b]|| <= eps.

The diagonal always fires (h_ii = 1 by construction), so the learning signal off the
diagonal is what varies across environments -- and it varies enormously, because the
conjunctive regions of multi-object envs are far sparser than cube-single's single ball.
Wave 1 measured cube-single at 14.0 off-diagonal positives per row vs cube-double's 0.51
(27x sparser), which is why that wave shipped sparsity controls.

This script reports the same statistic for every env in the campaign, so a floor can be
read against a known density rather than being confounded with it.

Run:  .venv/bin/python scripts/rescuing_fb/measure_region_density.py
"""

import sys

import numpy as np

DATA_DIR = '/dev/shm/ogbench_data/data'
BATCH = 1024
N_BATCHES = 20
SEED = 0

# env dataset name -> measure_hit_spec (verified by verify_hit_specs.py)
SPECS = {
    'cube-single-play-v0': '19:22:0.4',
    'cube-double-play-v0': '19:22:0.4,28:31:0.4',
    'cube-triple-play-v0': '19:22:0.4,28:31:0.4,37:40:0.4',
    'cube-quadruple-play-v0': '19:22:0.4,28:31:0.4,37:40:0.4,46:49:0.4',
    'scene-play-v0': '19:22:0.4,28:30:0.5,32:34:0.5,36:37:0.72,38:39:0.6',
    'puzzle-3x3-play-v0': ','.join(f'{19 + 4 * i}:{21 + 4 * i}:0.5' for i in range(9)),
    'puzzle-4x4-play-v0': ','.join(f'{19 + 4 * i}:{21 + 4 * i}:0.5' for i in range(16)),
    'puzzle-4x5-play-v0': ','.join(f'{19 + 4 * i}:{21 + 4 * i}:0.5' for i in range(20)),
    'puzzle-4x6-play-v0': ','.join(f'{19 + 4 * i}:{21 + 4 * i}:0.5' for i in range(24)),
}


def parse(spec):
    return [(int(a), int(b), float(e)) for a, b, e in (p.split(':') for p in spec.split(','))]


def hit_matrix(rows, cols, blocks):
    """[R,C] AND over blocks -- the numpy mirror of GCIQLFlowAgent._hit_matrix."""
    hit = np.ones((rows.shape[0], cols.shape[0]), dtype=bool)
    for a, b, e in blocks:
        d = np.linalg.norm(rows[:, None, a:b] - cols[None, :, a:b], axis=-1)
        hit &= d <= e
    return hit


def main():
    rng = np.random.default_rng(SEED)
    print(f'batch={BATCH}, {N_BATCHES} batches, region = env-faithful conjunction\n')
    print(f'{"env":26s} {"obs":>4s} {"blocks":>6s} {"offdiag +/row":>14s} {"pair rate":>11s}  '
          f'{"vs cube-single":>14s}')
    print('-' * 92)

    baseline = None
    rows_out = []
    for name, spec in SPECS.items():
        path = f'{DATA_DIR}/{name}.npz'
        try:
            with np.load(path) as f:
                obs = f['observations']
        except FileNotFoundError:
            print(f'{name:26s} {"":>4s} {"":>6s} {"MISSING DATASET":>14s}')
            continue

        blocks = parse(spec)
        per_row = []
        for _ in range(N_BATCHES):
            idx = rng.choice(obs.shape[0], size=BATCH, replace=False)
            s = obs[idx].astype(np.float64)
            h = hit_matrix(s, s, blocks)
            np.fill_diagonal(h, False)  # the diagonal fires by construction; exclude it
            per_row.append(h.sum() / BATCH)
        m = float(np.mean(per_row))
        rate = m / (BATCH - 1)
        if baseline is None:
            baseline = m
        ratio = f'{baseline / m:.1f}x sparser' if m > 0 else 'inf (no offdiag)'
        print(f'{name:26s} {obs.shape[1]:4d} {len(blocks):6d} {m:14.2f} {rate:11.2%}  {ratio:>14s}')
        rows_out.append((name, m))
        del obs

    print('\nreference (wave 1, same statistic): cube-single 14.0 | cube-double 0.51 | scene 3.07')
    print('wave-1 finding: scene\'s SPARSER conjunction (3.07) beat its denser cube-only control')
    print('(61.9) by 2x, so low density is not by itself a predictor of failure.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
