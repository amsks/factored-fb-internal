"""env/config.py — compose a run config with the DOMAIN group owned by env/.

impls/utils/config_utils.load_config reads all three groups (critic/actor/domain) from
impls/configs/. To keep every environment config inside env/ (and impls/ untouched), this
composer loads the domain from env/domains/ and reuses config_utils' critic/actor loaders +
override/required-key logic read-only. Domains not found under env/domains/ fall back to
impls/configs/domain/ (so the collaborator's OGBench domains, e.g. cube_single, still work).
"""
import os

from omegaconf import OmegaConf

from impls.utils.config_utils import (
    _load_group, _get_dotted, REQUIRED_KEYS, ConfigError,
)

DOMAINS_ROOT = os.path.join(os.path.dirname(__file__), "domains")


def _load_domain(name):
    path = os.path.join(DOMAINS_ROOT, f"{name}.yaml")
    if os.path.exists(path):
        return OmegaConf.load(path)              # env-owned domain
    return _load_group("domain", name)           # fall back to impls/configs/domain (unchanged)


def make_config(critic, actor, domain, overrides=None):
    """Compose critic + actor (from impls/configs) + domain (from env/domains) -> plain dict.
    Same semantics as impls.config_utils.load_config, only the domain source differs."""
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
    out = OmegaConf.to_container(cfg, resolve=True)

    for key in REQUIRED_KEYS.get(critic, ()):
        value, present = _get_dotted(out, key)
        if not present or value is None:
            raise ConfigError(
                f"critic {critic!r} requires config key {key!r}, which the composed config "
                f"(domain={domain}, critic={critic}, actor={actor}) does not define"
            )
    return out
