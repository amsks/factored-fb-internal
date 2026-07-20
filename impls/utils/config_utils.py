"""Config composition for the impls/ tree.

Composes a run config from three sibling groups — critic/, actor/, domain/ — under
impls/configs/. Replaces PR1's hardcoded main.py:get_config(). Returns a plain dict so
downstream code (Agent.create, the critics/actors) never sees a DictConfig.
"""
import os

from omegaconf import OmegaConf

CONFIG_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs")

# Keys each critic needs at the top level. Checked up front so a missing key is a named
# error here rather than a bare KeyError inside Agent.create's optimizer construction.
REQUIRED_KEYS = {
    "fb": ("lr_f", "lr_b", "lr_actor", "z_dim", "batch_size", "discount"),
    "gciql": ("lr", "lr_actor", "batch_size", "discount", "expectile", "tau",
              "value_hidden_dims", "layer_norm", "gc_negative"),
}


class ConfigError(ValueError):
    """Raised when a composed config is missing a key the seam requires."""


def _load_group(group, name):
    path = os.path.join(CONFIG_ROOT, group, f"{name}.yaml")
    if not os.path.exists(path):
        raise ConfigError(f"no {group} config named {name!r} (looked for {path})")
    return OmegaConf.load(path)


def load_config(critic, actor, domain, overrides=None):
    """Compose critic + actor + domain configs into a plain nested dict."""
    cfg = OmegaConf.merge(
        _load_group("domain", domain),
        _load_group("critic", critic),
        _load_group("actor", actor),
    )
    if overrides:
        set_overrides = [o for o in overrides if not o.startswith("~")]
        del_overrides = [o[1:] for o in overrides if o.startswith("~")]
        if set_overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(set_overrides))
        for key in del_overrides:
            # OmegaConf has no dotlist "delete" verb, so walk to the parent container
            # and pop the leaf ourselves. This has to happen after the `set_overrides`
            # merge so that e.g. `~lr` always wins regardless of override ordering.
            *parents, leaf = key.split(".")
            node = cfg
            for p in parents:
                node = node[p]
            del node[leaf]
    out = OmegaConf.to_container(cfg, resolve=True)

    for key in REQUIRED_KEYS.get(critic, ()):
        if key not in out or out[key] is None:
            raise ConfigError(
                f"critic {critic!r} requires config key {key!r}, which the composed config "
                f"(domain={domain}, critic={critic}, actor={actor}) does not define"
            )
    return out
