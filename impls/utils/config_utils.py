"""Config composition: a run config from three groups -- critic/, actor/, domain/ -- under
impls/configs/. Returns a plain dict so the critics and actors never see a DictConfig.
env/config.py::make_config is the entry point main.py uses (it also reads env/domains/).
"""
import os

from omegaconf import OmegaConf

CONFIG_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs")

# Keys each critic needs, checked up front so a missing key is a named error here rather
# than a bare KeyError inside make_dataset() or Agent.create. Dotted keys reach into nested
# group dicts. Keyed by the --critic name as given (aliases have their own yamls).
_FB_KEYS = (
    "lr_f", "lr_b", "lr_actor", "z_dim", "batch_size", "discount",
    "L_dim", "num_parallel", "ortho_coef",
    "train_goal_ratio", "norm_z", "f_target_tau", "b_target_tau",
    "forward.hidden_dim", "forward.hidden_layers", "forward.embedding_layers",
    "backward.hidden_dim", "backward.hidden_layers", "backward.norm",
    "left_encoder.hidden_dim", "left_encoder.hidden_layers", "left_encoder.norm",
)
REQUIRED_KEYS = {
    "fb": _FB_KEYS,
    # rldp builds the same forward/backward/left_encoder modules plus the predictor, and
    # needs `horizon` + dataset_class so make_dataset() emits the h-step window.
    "rldp": _FB_KEYS + (
        "horizon",
        "predictor.hidden_dim", "predictor.hidden_layers", "predictor.embedding_layers",
        "dataset_class",
    ),
}


def _get_dotted(d, dotted_key):
    """Walk a dotted key (e.g. "forward.hidden_dim") through nested dicts. Returns
    (value, present) so a present-but-None value can be distinguished from a missing one."""
    node = d
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return None, False
        node = node[part]
    return node, True


class ConfigError(ValueError):
    """Raised when a composed config is missing a required key."""


def _load_group(group, name):
    """Load <group>/<name>.yaml. A top-level `base: <other>` key loads <group>/<other>.yaml
    first and merges this file over it."""
    path = os.path.join(CONFIG_ROOT, group, f"{name}.yaml")
    if not os.path.exists(path):
        raise ConfigError(f"no {group} config named {name!r} (looked for {path})")
    cfg = OmegaConf.load(path)
    if "base" in cfg:
        base = cfg.pop("base")
        cfg = OmegaConf.merge(_load_group(group, base), cfg)
    return cfg


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
        value, present = _get_dotted(out, key)
        if not present or value is None:
            raise ConfigError(
                f"critic {critic!r} requires config key {key!r}, which the composed config "
                f"(domain={domain}, critic={critic}, actor={actor}) does not define"
            )
    return out
