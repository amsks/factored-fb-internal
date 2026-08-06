#!/usr/bin/env python
"""scripts/rescuing_fb/make_delta_prototypes.py -- spec 17b section 1.1.

Draws K fixed prototypes of the measure_hit_spec object-configuration blocks
from an OGBench dataset npz, for the `measure_delta_partition` (variant B)
one-stage simplex successor measure
(PAPER/Multistep-FB/specs/17b-delta-partition-impl.md). phi_k(x) is a
softmax-RBF over these K prototypes (GCIQLFlowAgent._delta_phi); this script
only draws the prototypes -- no training happens here.

Reimplements the `_parse_hit_spec` grammar from
`tools/wandb_mode_shim/gciql_flow.py` rather than importing it: gciql_flow
transitively imports jax/flax (only present in the jax venv), while this
script is numpy-only and must run in the plain repo `.venv` (the launcher
calls it before training starts, spec 17 section 6). Keep the two grammars in
lockstep by hand if either changes -- same 'start:stop:eps' comma-separated
block spec.

CLI:
    make_delta_prototypes.py --dataset <npz> --hit_spec '19:22:0.4' \\
        --k 256 --seed 0 --out results/delta_protos/cube_single_K256_s0.npy

Prototypes are stored in RAW obs units (no epsilon scaling) -- eps scaling
happens at use, inside GCIQLFlowAgent._delta_phi. Protos should be written
somewhere durable (results/delta_protos/, NOT /dev/shm, which gets wiped --
the checkpoint's config records this path and restore needs the file
present).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _parse_hit_spec(spec: str):
    """Same grammar as tools/wandb_mode_shim/gciql_flow.py:_parse_hit_spec --
    comma-separated 'start:stop:eps' observation blocks ANDed together. A
    STRING (not a list): ml_collections rejects list-valued agent overrides
    on the Hydra command line (spec06 launch gotcha) -- this script is not
    itself an agent override, but the grammar is kept identical on purpose so
    the same measure_hit_spec string works in both places."""
    blocks = []
    for part in str(spec).split(','):
        part = part.strip()
        if not part:
            continue
        fields = part.split(':')
        assert len(fields) == 3, f"hit-spec block must be 'start:stop:eps', got {part!r}"
        a, b, e = int(fields[0]), int(fields[1]), float(fields[2])
        assert 0 <= a < b, f"hit-spec block {part!r}: need 0 <= start < stop"
        assert e >= 0.0, f"hit-spec block {part!r}: radius must be >= 0"
        blocks.append((a, b, e))
    assert blocks, f"hit_spec parsed to no blocks: {spec!r}"
    return tuple(blocks)


def extract_block_columns(observations: np.ndarray, blocks) -> np.ndarray:
    """Concatenate the hit-spec block columns of `observations` in block
    order -> [N, d_c], d_c = sum(stop-start). Prototypes and phi() both key
    off this same column layout (raw obs units, _parse_hit_spec order)."""
    cols = [observations[:, a:b] for (a, b, _e) in blocks]
    return np.concatenate(cols, axis=1)


def draw_prototypes(block_cols: np.ndarray, k: int, seed: int) -> np.ndarray:
    """K DISTINCT row indices drawn uniformly (without replacement) by the
    seeded rng -- coverage over the dataset's visited object configurations,
    not a clustering fit. Every returned row is literally one of the
    dataset's rows (restricted to the block columns)."""
    n = block_cols.shape[0]
    assert k >= 2, f"k must be >= 2, got {k}"
    assert n >= k, f"dataset has only {n} rows, need >= k={k} to draw distinct prototypes"
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=k, replace=False)
    return block_cols[idx].astype(np.float32)


def sanity_stats(protos: np.ndarray) -> dict:
    """K, d_c, per-dim min/max, and the mean nearest-neighbor distance between
    prototypes (coverage sanity -- collapsed/duplicate draws show up as a
    near-zero mean NN distance)."""
    k, d_c = protos.shape
    diffs = protos[:, None, :] - protos[None, :, :]          # [K,K,d_c]
    dist = np.sqrt(np.sum(diffs ** 2, axis=-1))               # [K,K]
    np.fill_diagonal(dist, np.inf)
    nn_dist = dist.min(axis=1)                                # [K]
    return {
        'k': k, 'd_c': d_c,
        'dim_min': protos.min(axis=0), 'dim_max': protos.max(axis=0),
        'mean_nn_dist': float(nn_dist.mean()),
    }


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', required=True,
                     help="OGBench dataset .npz path (needs the 'observations' key)")
    ap.add_argument('--hit_spec', required=True,
                     help="measure_hit_spec grammar, e.g. '19:22:0.4' or a comma-joined conjunction")
    ap.add_argument('--k', type=int, default=256, help="number of prototypes to draw")
    ap.add_argument('--seed', type=int, default=0, help="numpy rng seed for the draw")
    ap.add_argument('--out', required=True, help="output .npy path ([K, d_c] float32)")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    blocks = _parse_hit_spec(args.hit_spec)
    with np.load(args.dataset) as data:
        observations = np.asarray(data['observations'])
    block_cols = extract_block_columns(observations, blocks)
    protos = draw_prototypes(block_cols, args.k, args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, protos)

    stats = sanity_stats(protos)
    print(f"[make_delta_prototypes] hit_spec={args.hit_spec!r} blocks={blocks}")
    print(f"[make_delta_prototypes] dataset={args.dataset} N={observations.shape[0]} "
          f"K={stats['k']} d_c={stats['d_c']} seed={args.seed}")
    print(f"[make_delta_prototypes] per-dim min: {np.array2string(stats['dim_min'], precision=4)}")
    print(f"[make_delta_prototypes] per-dim max: {np.array2string(stats['dim_max'], precision=4)}")
    print(f"[make_delta_prototypes] mean nearest-neighbor distance: {stats['mean_nn_dist']:.6f}")
    print(f"[make_delta_prototypes] wrote {out_path} shape={protos.shape} dtype={protos.dtype}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
