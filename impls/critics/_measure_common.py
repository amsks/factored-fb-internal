"""Shared surface for the successor-measure critics (density_fb, td_fb): the hit-spec
parser and the [R,C] goal-region indicator both objectives build their targets from.
Ported near-verbatim from tools/wandb_mode_shim/gciql_flow.py (bridge worktree,
origin/gciql-fb-bridge @ 542c058). Deliberately minimal — per spec
docs/specs/2026-08-10-measure-objectives-on-mw-play.md P2, the acceptable shared surface
between the two critics is the parser + hit matrix ONLY; every other symbol (the density
kernel `_delta_phi`/`_density_queries`/`_raw_block_cols`, `_measure_terms`) lives in its
own critic file so the two stay isolated.

`scaled_hit_cols` is a later addition OUTSIDE that pair: it serves the rldp/psm critic
families' `backward_input: "hit_spec"` mode (object-coordinate B/phi input), and like the
parser it is generic hit-spec machinery, not part of either measure objective.
"""
import jax.numpy as jnp


def _looks_sexagesimal(part):
    """Is this token a bare number, i.e. what OmegaConf turns 'a:b:eps' into?"""
    try:
        float(part)
    except (TypeError, ValueError):
        return False
    return True


def parse_hit_spec(spec):
    """Parse `measure_hit_spec` -> tuple of (start, stop, eps) observation blocks.

    The goal-region hit indicator h(s', g) is a CONJUNCTION over one or more blocks, each
    with its own radius (cube-single: one block; cube-double/scene: several) —

        cube-single  '19:22:0.4'
        cube-double  '19:22:0.4,28:31:0.4'

    A STRING (not a list): dotlist overrides on our launch path can't carry a list value
    either, so the string-grammar convention carries over unchanged from the bridge."""
    blocks = []
    for part in str(spec).split(','):
        part = part.strip()
        if not part:
            continue
        fields = part.split(':')
        # A single unquoted block on the dotlist path is a REPEAT landmine: OmegaConf reads
        # `measure_hit_spec=4:7:0.07` as a sexagesimal literal and hands us 14820.07
        # (= 4*3600 + 7*60 + 0.07). Multi-block specs survive unquoted because the comma
        # breaks the pattern, so this only ever bites the common single-block case. Name the
        # cause rather than let the arithmetic show through as a bare parse error.
        if len(fields) == 1 and _looks_sexagesimal(part):
            raise AssertionError(
                f"hit-spec {part!r} looks like OmegaConf's sexagesimal parse of an UNQUOTED "
                f"'a:b:eps' override. Pass it with escaped quotes so the value stays a "
                f"string: --override 'measure_hit_spec=\"4:7:0.07\"' (launchers: "
                f'measure_hit_spec=\\"...\\" inside the OVERRIDE string, `tr` not `xargs`).')
        assert len(fields) == 3, f"hit-spec block must be 'start:stop:eps', got {part!r}"
        a, b, e = int(fields[0]), int(fields[1]), float(fields[2])
        assert 0 <= a < b, f"hit-spec block {part!r}: need 0 <= start < stop"
        assert e >= 0.0, f"hit-spec block {part!r}: radius must be >= 0"
        blocks.append((a, b, e))
    assert blocks, f"measure_hit_spec parsed to no blocks: {spec!r}"
    return tuple(blocks)


def scaled_hit_cols(x, blocks):
    """[..., d_c] eps-scaled hit-spec block columns of full observation rows — exactly the
    transformation density_fb.py::_density_queries applies to build the B tower's input
    (slice each `blocks` block, divide it by its eps, concatenate), generalized to any
    leading shape so a single eval goal row ([D]) works as well as a batch ([N, D]).

    This is the `backward_input: "hit_spec"` input map for the rldp/psm critic families:
    their B / phi towers consume this in place of the raw observation. Callers validate
    obs width at build time (the blocks must fit inside the observation); at trace time a
    too-short row would silently wrap/clip under jax slicing, so keep the build check."""
    return jnp.concatenate([x[..., a:b] / e for (a, b, e) in blocks], axis=-1)


def hit_matrix(row_obs, col_obs, blocks, dtype):
    """[R,C] goal-region indicator: AND over blocks of ||row[a:b] - col[a:b]|| <= eps.

    `blocks` comes from parse_hit_spec. With the default single cube-single block this is
    exactly `(norm(nobs[:,19:22] - col[:,19:22]) <= 0.4)` — the env's own success rule."""
    hit = None
    for (a, b, e) in blocks:
        dist = jnp.linalg.norm(row_obs[:, None, a:b] - col_obs[None, :, a:b], axis=-1)
        h = (dist <= e).astype(dtype)
        hit = h if hit is None else hit * h
    return hit
