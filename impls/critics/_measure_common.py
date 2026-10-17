"""Hit-spec parsing: which observation columns define the task and the success radius."""

import jax.numpy as jnp


def _looks_sexagesimal(part):
    """Whether the token parses as a number."""
    try:
        float(part)
    except (TypeError, ValueError):
        return False
    return True


def parse_hit_spec(spec):
    """Parse 'start:stop:eps,...' into a tuple of (start, stop, eps) blocks.

    Example: '19:22:0.4,28:31:0.4'.
    """
    blocks = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        fields = part.split(":")
        # OmegaConf parses an unquoted 'a:b:eps' as a sexagesimal number
        if len(fields) == 1 and _looks_sexagesimal(part):
            raise AssertionError(
                f"hit-spec {part!r} was parsed as a number; quote it: "
                f"--override 'goal_spec=\"4:7:0.07\"'."
            )
        assert len(fields) == 3, (
            f"hit-spec block must be 'start:stop:eps', got {part!r}"
        )
        a, b, e = int(fields[0]), int(fields[1]), float(fields[2])
        assert 0 <= a < b, f"hit-spec block {part!r}: need 0 <= start < stop"
        assert e >= 0.0, f"hit-spec block {part!r}: radius must be >= 0"
        blocks.append((a, b, e))
    assert blocks, f"goal_spec parsed to no blocks: {spec!r}"
    return tuple(blocks)


def scaled_hit_cols(x, blocks):
    """Eps-scaled hit-spec columns [..., d_c]; callers check the blocks fit the observation."""
    return jnp.concatenate([x[..., a:b] / e for (a, b, e) in blocks], axis=-1)
