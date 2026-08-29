"""agents/dfb/hit_spec.py — hit-spec blocks and the eps-kernel, torch port.

Faithful ports of the jax reference (tools/wandb_mode_shim/gciql_flow.py):
    _parse_hit_spec       -> parse_hit_spec
    _raw_block_cols       -> raw_block_cols
    _density_queries      -> scaled_block_cols
    _delta_phi            -> delta_phi

CONVENTION PIN (the spec-18 §4.2 double-scaling trap): `delta_phi` takes RAW
(unscaled) rows for BOTH arguments and applies the per-block eps scaling
internally; `scaled_block_cols` is the eps-scaled twin used ONLY as the B-tower
input. Never feed scaled rows to `delta_phi`.
"""

from __future__ import annotations

import torch


def parse_hit_spec(spec) -> tuple:
    """Parse 'start:stop:eps[,start:stop:eps...]' -> tuple of (start, stop, eps)
    observation blocks. String form (not a list) matches the jax launch convention.

        cube-single  '19:22:0.4'
        cube-double  '19:22:0.4,28:31:0.4'
        scene        '19:22:0.4,28:30:0.5,32:34:0.5,36:37:0.72,38:39:0.6'
    """
    blocks = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        fields = part.split(":")
        assert len(fields) == 3, f"hit-spec block must be 'start:stop:eps', got {part!r}"
        a, b, e = int(fields[0]), int(fields[1]), float(fields[2])
        assert 0 <= a < b, f"hit-spec block {part!r}: need 0 <= start < stop"
        assert e >= 0.0, f"hit-spec block {part!r}: radius must be >= 0"
        blocks.append((a, b, e))
    assert blocks, f"measure_hit_spec parsed to no blocks: {spec!r}"
    return tuple(blocks)


def raw_block_cols(obs_rows: torch.Tensor, blocks) -> torch.Tensor:
    """[B, d_c] RAW (unscaled) concatenated block columns of obs_rows — the
    `protos` argument to `delta_phi`, which scales internally."""
    return torch.cat([obs_rows[:, a:b] for (a, b, e) in blocks], dim=-1)


def scaled_block_cols(obs_rows: torch.Tensor, blocks) -> torch.Tensor:
    """[B, d_c] eps-scaled block coords — the resampled query support AND the
    B-tower input rows (jax `_density_queries`)."""
    return torch.cat([obs_rows[:, a:b] / e for (a, b, e) in blocks], dim=-1)


def delta_phi(obs_rows: torch.Tensor, protos: torch.Tensor, blocks) -> torch.Tensor:
    """Query-set-normalized eps-kernel rows (jax `_delta_phi`): softmax over K of
    -0.5 * sum_blocks ||(x_blk - proto_blk)/eps_blk||^2.

    obs_rows [B, obs_dim] FULL observation rows (sliced here per block);
    protos   [K, d_c] RAW block columns in `blocks` concatenation order.
    Returns [B, K] rows on the simplex (rows sum to 1 over the K protos: the
    base measure is kernel x query-pool density — spec 30 Rev 1 statement).
    """
    parts_x, parts_p, off = [], [], 0
    for (a, b, e) in blocks:
        w = b - a
        parts_x.append(obs_rows[:, a:b] / e)
        parts_p.append(protos[:, off:off + w] / e)
        off += w
    xc = torch.cat(parts_x, dim=-1)                  # [B, d_c]
    pc = torch.cat(parts_p, dim=-1)                  # [K, d_c]
    diffs = xc[:, None, :] - pc[None, :, :]          # [B, K, d_c]
    logits = -0.5 * (diffs ** 2).sum(dim=-1)         # [B, K]
    return torch.softmax(logits, dim=-1)
