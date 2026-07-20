"""Config composition for the impls/ tree.

Composes a run config from three sibling groups — critic/, actor/, domain/ — under
impls/configs/. Replaces PR1's hardcoded main.py:get_config(). Returns a plain dict so
downstream code (Agent.create, the critics/actors) never sees a DictConfig.
"""
import os

from omegaconf import OmegaConf

CONFIG_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs")

# Keys each critic needs — checked up front so a missing key is a named error here rather
# than a bare KeyError inside Agent.create's optimizer construction, or (for gciql's
# GCDataset keys) inside make_dataset(), which runs *before* Agent.create. Dotted keys
# (e.g. "forward.hidden_dim") reach into nested group dicts.
REQUIRED_KEYS = {
    # fb.py:build() reads z_dim/L_dim/num_parallel plus the forward/backward/left_encoder
    # sub-dicts (fb.py:130-154); the FBCritic dataclass reads the remaining scalars.
    "fb": (
        "lr_f", "lr_b", "lr_actor", "z_dim", "batch_size", "discount",
        "L_dim", "num_parallel", "ortho_coef", "fb_pessimism_penalty",
        "train_goal_ratio", "norm_z", "f_target_tau", "b_target_tau",
        "forward.hidden_dim", "forward.hidden_layers", "forward.embedding_layers",
        "backward.hidden_dim", "backward.hidden_layers", "backward.norm",
        "left_encoder.hidden_dim", "left_encoder.hidden_layers", "left_encoder.norm",
    ),
    # gciql.py:build() reads lr/batch_size/discount/expectile/tau/value_hidden_dims/
    # layer_norm; datasets.py:GCDataset (built *before* Agent.create, datasets.py:96-116)
    # reads the goal-sampling and dataset_class keys.
    "gciql": (
        "lr", "lr_actor", "batch_size", "discount", "expectile", "tau",
        "value_hidden_dims", "layer_norm", "gc_negative",
        "value_p_curgoal", "value_p_trajgoal", "value_p_randomgoal", "value_geom_sample",
        "actor_p_curgoal", "actor_p_trajgoal", "actor_p_randomgoal", "actor_geom_sample",
        "dataset_class",
    ),
    # isfb.py:build() bare-indexes every FB key (it builds the same forward/backward/
    # left_encoder modules) PLUS the ISFB terms; stage_loss reads batch['masks'], which only
    # GCDataset produces, so the GCDataset keys are required too (they are consumed by
    # make_dataset(), which runs *before* Agent.create).
    "isfb": (
        "lr_f", "lr_b", "lr_actor", "z_dim", "batch_size", "discount",
        "L_dim", "num_parallel", "ortho_coef", "fb_pessimism_penalty",
        "train_goal_ratio", "norm_z", "f_target_tau", "b_target_tau",
        "forward.hidden_dim", "forward.hidden_layers", "forward.embedding_layers",
        "backward.hidden_dim", "backward.hidden_layers", "backward.norm",
        "left_encoder.hidden_dim", "left_encoder.hidden_layers", "left_encoder.norm",
        "measure_insample", "iql_expectile", "alpha_n", "alpha_m", "alpha_v",
        "awr_beta", "awr_wmax", "awr_normalize_adv", "goal_mask",
        "gc_negative",
        "value_p_curgoal", "value_p_trajgoal", "value_p_randomgoal", "value_geom_sample",
        "actor_p_curgoal", "actor_p_trajgoal", "actor_p_randomgoal", "actor_geom_sample",
        "dataset_class",
    ),
    # iql_bilinear.py:build() bare-indexes every gciql key (same expectile-V + twin-Q
    # objective) PLUS the bilinear-head and actor-conditioning knobs that _validate() and
    # the head constructor read. Every one of these is indexed without .get(), on purpose:
    # a knob that silently defaults is a knob a sweep can silently fail to set.
    "iql_bilinear": (
        "lr", "lr_actor", "batch_size", "discount", "expectile", "tau",
        "value_hidden_dims", "layer_norm", "gc_negative",
        "value_latent_dim", "psi_base", "project_z", "double_q", "iql_ortho_coef",
        "actor_cond", "actor_z_from_critic", "actor_z_infer",
        "value_p_curgoal", "value_p_trajgoal", "value_p_randomgoal", "value_geom_sample",
        "actor_p_curgoal", "actor_p_trajgoal", "actor_p_randomgoal", "actor_geom_sample",
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
        value, present = _get_dotted(out, key)
        if not present or value is None:
            raise ConfigError(
                f"critic {critic!r} requires config key {key!r}, which the composed config "
                f"(domain={domain}, critic={critic}, actor={actor}) does not define"
            )
    return out
