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

# Override spellings that are valid dotlist but name nothing this tree reads. Upstream's
# bridge config is flat with a single `lr`, so specs transcribed from it reach for `actor.lr`
# / `critic.lr`; both merge cleanly and change nothing. Mapped to the key that is read.
_INERT_ALIASES = {"actor.lr": "lr_actor", "critic.lr": "lr"}


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

    # Overrides that name a key this tree does not read are SILENT: OmegaConf happily creates
    # `actor.lr` next to the real `lr_actor` and the run trains at the default, producing a
    # wrong table cell with no error anywhere. The bridge specs are written against upstream's
    # flat single-`lr` config, so its spellings are exactly the ones that land here inert.
    # Fail loudly on the known aliases rather than let a silently-mistuned run reach a table.
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
