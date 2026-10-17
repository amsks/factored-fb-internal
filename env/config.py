"""Run config composition with domains from env/domains/ (fallback impls/configs/domain/)."""
import os

from omegaconf import OmegaConf

from impls.utils.config_utils import (
    upgrade_config,
    _load_group, _get_dotted, REQUIRED_KEYS, ConfigError,
)

DOMAINS_ROOT = os.path.join(os.path.dirname(__file__), "domains")

# Override keys that are read nowhere, mapped to the key that is read
_INERT_ALIASES = {"actor.lr": "lr_actor", "critic.lr": "lr"}


def _load_domain(name):
    path = os.path.join(DOMAINS_ROOT, f"{name}.yaml")
    if os.path.exists(path):
        return OmegaConf.load(path)
    return _load_group("domain", name)


def make_config(critic, actor, domain, overrides=None):
    """Compose critic, actor and domain configs into a plain dict."""
    cfg = OmegaConf.merge(
        _load_domain(domain),
        _load_group("critic", critic),
        _load_group("actor", actor),
    )
    if overrides:
        set_overrides = [o for o in overrides if not o.startswith("~")]
        del_overrides = [o[1:] for o in overrides if o.startswith("~")]
        if set_overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(set_overrides))
        for key in del_overrides:
            *parents, leaf = key.split(".")
            node = cfg
            for p in parents:
                node = node[p]
            del node[leaf]
    out = upgrade_config(OmegaConf.to_container(cfg, resolve=True))

    # Reject overrides of keys that are never read
    for alias, real in _INERT_ALIASES.items():
        value, present = _get_dotted(out, alias)
        if present:
            raise ConfigError(
                f"override {alias}={value!r} does nothing in this tree: the key that is "
                f"actually read is {real!r} (see impls/agents/__init__.py::_LR_KEY). Pass "
                f"{real}={value!r} instead — the run would otherwise train at the default "
                f"and report a number nobody could tell apart from a tuned one."
            )

    for key in REQUIRED_KEYS.get(critic, ()):
        value, present = _get_dotted(out, key)
        if not present or value is None:
            raise ConfigError(
                f"critic {critic!r} requires config key {key!r}, which the composed config "
                f"(domain={domain}, critic={critic}, actor={actor}) does not define"
            )
    return out
