"""Hit-spec helpers shared by the critics: the parser for `goal_spec` (which
observation columns define the task and the radius that counts as reaching it) and the
eps-scaled column map used as a backward-network input.
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
    """Parse `goal_spec` -> tuple of (start, stop, eps) observation blocks.

    A task is reached when every block is within its radius (cube-single: one block;
    cube-double/scene: several):

        cube-single  '19:22:0.4'
        cube-double  '19:22:0.4,28:31:0.4'

    A string rather than a list, because dotlist overrides cannot carry a list value."""
    blocks = []
    for part in str(spec).split(','):
        part = part.strip()
        if not part:
            continue
        fields = part.split(':')
        # OmegaConf reads an unquoted single-block override `goal_spec=4:7:0.07` as a
        # sexagesimal literal (14820.07). Multi-block specs survive because the comma breaks
        # the pattern. Name the cause rather than fail with a bare parse error.
        if len(fields) == 1 and _looks_sexagesimal(part):
            raise AssertionError(
                f"hit-spec {part!r} looks like OmegaConf's sexagesimal parse of an UNQUOTED "
                f"'a:b:eps' override. Pass it with escaped quotes so the value stays a "
                f"string: --override 'goal_spec=\"4:7:0.07\"' (launchers: "
                f'goal_spec=\\"...\\" inside the OVERRIDE string, `tr` not `xargs`).')
        assert len(fields) == 3, f"hit-spec block must be 'start:stop:eps', got {part!r}"
        a, b, e = int(fields[0]), int(fields[1]), float(fields[2])
        assert 0 <= a < b, f"hit-spec block {part!r}: need 0 <= start < stop"
        assert e >= 0.0, f"hit-spec block {part!r}: radius must be >= 0"
        blocks.append((a, b, e))
    assert blocks, f"goal_spec parsed to no blocks: {spec!r}"
    return tuple(blocks)


def scaled_hit_cols(x, blocks):
    """[..., d_c] eps-scaled hit-spec columns of observation rows (slice each block, divide
    by its eps, concatenate), for any leading shape. The backward-network input under
    `backward_input: hit_spec`. Callers check at build time that the blocks fit inside the
    observation; jax slicing would silently clip a too-short row."""
    return jnp.concatenate([x[..., a:b] / e for (a, b, e) in blocks], axis=-1)

