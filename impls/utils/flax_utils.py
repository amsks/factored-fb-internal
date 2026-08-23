"""Flax train-state primitives for the factored impls/ tree.

Ported from third_party/ogbench/impls/utils/flax_utils.py (pinned 1d41409), minus the
unused `apply_loss_fn`, plus `soft_update` (Polyak, from PSMFlows agents/psm.py:_soft).
"""
import functools
import glob
import os
import pickle
from typing import Any, Dict, Mapping, Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

nonpytree_field = functools.partial(flax.struct.field, pytree_node=False)


class ModuleDict(nn.Module):
    """A dictionary of modules.

    Allows sharing parameters between modules and provides a convenient way to access them.
    """

    modules: Dict[str, nn.Module]

    @nn.compact
    def __call__(self, *args, name=None, **kwargs):
        """Forward pass.

        For initialization, call with `name=None` and provide the arguments for each module in `kwargs`.
        Otherwise, call with `name=<module_name>` and provide the arguments for that module.
        """
        if name is None:
            if kwargs.keys() != self.modules.keys():
                raise ValueError(
                    f'When `name` is not specified, kwargs must contain the arguments for each module. '
                    f'Got kwargs keys {kwargs.keys()} but module keys {self.modules.keys()}'
                )
            out = {}
            for key, value in kwargs.items():
                if isinstance(value, Mapping):
                    out[key] = self.modules[key](**value)
                elif isinstance(value, Sequence):
                    out[key] = self.modules[key](*value)
                else:
                    out[key] = self.modules[key](value)
            return out

        return self.modules[name](*args, **kwargs)


class TrainState(flax.struct.PyTreeNode):
    """Custom train state for models.

    Attributes:
        step: Counter incremented by 1 after each `apply_gradients` call.
        apply_fn: Apply function of the model.
        model_def: Model definition.
        params: Parameters of the model.
        tx: optax optimizer.
        opt_state: Optimizer state.
    """

    step: int
    apply_fn: Any = nonpytree_field()
    model_def: Any = nonpytree_field()
    params: Any
    tx: Any = nonpytree_field()
    opt_state: Any

    @classmethod
    def create(cls, model_def, params, tx=None, **kwargs):
        """Create a new train state."""
        if tx is not None:
            opt_state = tx.init(params)
        else:
            opt_state = None

        return cls(
            step=1,
            apply_fn=model_def.apply,
            model_def=model_def,
            params=params,
            tx=tx,
            opt_state=opt_state,
            **kwargs,
        )

    def __call__(self, *args, params=None, method=None, **kwargs):
        """Forward pass.

        When `params` is None the stored params are used and gradients are stopped; pass the
        traced params to flow gradients.
        """
        if params is None:
            params = self.params
        variables = {'params': params}
        if method is not None:
            method_name = getattr(self.model_def, method)
        else:
            method_name = None

        return self.apply_fn(variables, *args, method=method_name, **kwargs)

    def select(self, name):
        """Helper to select a module from a `ModuleDict`."""
        return functools.partial(self, name=name)

    def apply_gradients(self, grads, **kwargs):
        """Apply the gradients and return the updated state."""
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)

        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        )


def soft_update(online_params, target_params, tau):
    """Polyak: target <- tau*online + (1-tau)*target. Matches PSMFlows agents/psm.py:_soft."""
    return jax.tree_util.tree_map(lambda o, t: o * tau + t * (1 - tau), online_params, target_params)


def save_agent(agent, save_dir, epoch):
    """Save the agent to a pickle under save_dir/params_{epoch}.pkl."""
    save_dict = dict(agent=flax.serialization.to_state_dict(agent))
    save_path = os.path.join(save_dir, f'params_{epoch}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(save_dict, f)
    print(f'Saved to {save_path}')


def resolve_run_dir(path_glob):
    """Resolve a --restore_path / --warm_start_path glob to THE single run directory it
    matches. Factored out so callers that need something else from the same checkpoint
    directory (env/online.py's replay-buffer state) resolve it exactly as restore_agent
    does, instead of re-globbing and possibly disagreeing."""
    candidates = glob.glob(path_glob)
    assert len(candidates) == 1, f'Found {len(candidates)} candidates: {candidates}'
    return candidates[0]


def restore_agent(agent, restore_path, restore_epoch):
    """Restore the agent from restore_path (glob) / params_{restore_epoch}.pkl."""
    restore_path = resolve_run_dir(restore_path) + f'/params_{restore_epoch}.pkl'
    with open(restore_path, 'rb') as f:
        load_dict = pickle.load(f)
    agent = flax.serialization.from_state_dict(agent, load_dict['agent'])
    print(f'Restored from {restore_path}')
    return agent


def _splice_matching(fresh, checkpointed):
    """Replace `fresh` (a pytree of params) by `checkpointed` iff their leaf counts and
    shapes match exactly; otherwise leave `fresh` untouched. Returns (params, copied:bool).
    Mirrors the bridge's warm_start_ckpt splice (tools/wandb_mode_shim/gciql_flow.py
    ~4128): a group present in the checkpoint whose architecture doesn't match the fresh
    agent's (e.g. a differently-sized module) stays at its fresh init rather than raising,
    since a warm start across a deliberately-changed architecture is a valid (if unusual)
    call the caller made on purpose."""
    fresh_leaves, treedef = jax.tree_util.tree_flatten(fresh)
    try:
        ck_leaves, _ = jax.tree_util.tree_flatten(checkpointed)
    except Exception:
        return fresh, False
    if len(fresh_leaves) != len(ck_leaves):
        return fresh, False
    if not all(jnp.shape(a) == jnp.shape(b) for a, b in zip(fresh_leaves, ck_leaves)):
        return fresh, False
    return jax.tree_util.tree_unflatten(treedef, [jnp.asarray(x) for x in ck_leaves]), True


def warm_start_agent(agent, warm_start_path, warm_start_epoch):
    """Splice a TRAINED checkpoint's params into a freshly-created `agent` — params AND
    target params ONLY, for every group whose shape matches, with FRESH optimizer state
    and rng (unlike `restore_agent`, which is a full-state RESUME: params + opt state +
    rng). This is td_fb's phase-2 warm start (docs/specs/2026-08-10-measure-objectives-on-
    mw-play.md P3): the whole agent (critic AND actor groups) is spliced, matching the
    bridge's own warm_start_ckpt convention, which restores every module whose param tree
    shape matches the checkpoint rather than only the critic's.

    Does NOT add a field to Agent's pytree (the bridge's restore-breakage gotcha) — it
    only rewrites the params living inside the EXISTING `net`/`target` fields, so
    `restore_agent` on old checkpoints is unaffected.

    `warm_start_path` is a glob matching exactly one run dir (as `restore_path` is);
    `warm_start_epoch` selects params_{epoch}.pkl inside it."""
    candidates = glob.glob(warm_start_path)
    assert len(candidates) == 1, f'Found {len(candidates)} candidates: {candidates}'
    ckpt_path = candidates[0] + f'/params_{warm_start_epoch}.pkl'
    with open(ckpt_path, 'rb') as f:
        load_dict = pickle.load(f)
    ck = load_dict['agent']
    ck_net = ck.get('net', {}) or {}
    ck_target = ck.get('target', {}) or {}

    new_net = dict(agent.net)
    copied, fresh = [], []
    for g in agent.net:
        ck_group = ck_net.get(g)
        ck_params = ck_group.get('params') if isinstance(ck_group, dict) else None
        if ck_params is not None:
            spliced, ok = _splice_matching(agent.net[g].params, ck_params)
            if ok:
                new_net[g] = agent.net[g].replace(params=spliced)
                copied.append(g)
                continue
        fresh.append(g)

    new_target = dict(agent.target)
    for g in agent.target:
        if g in ck_target:
            spliced, ok = _splice_matching(agent.target[g], ck_target[g])
            if ok:
                new_target[g] = spliced

    print(f'[warm_start] spliced {copied} from {ckpt_path}; fresh-init {fresh}; '
          f'fresh optimizer state and rng', flush=True)
    return agent.replace(net=new_net, target=new_target)
