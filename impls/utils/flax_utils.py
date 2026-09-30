"""Flax train-state primitives for the factored impls/ tree.

Ported from OGBench upstream impls/utils/flax_utils.py (pinned 1d41409), minus the
unused `apply_loss_fn`, plus `soft_update` (Polyak averaging).
"""
import functools
import glob
import os
import pickle
from typing import Any, Dict, Mapping, Sequence

import flax
import flax.linen as nn
import jax
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
    """Polyak: target <- tau*online + (1-tau)*target. """
    return jax.tree_util.tree_map(lambda o, t: o * tau + t * (1 - tau), online_params, target_params)


def save_agent(agent, save_dir, epoch):
    """Save the agent to a pickle under save_dir/params_{epoch}.pkl.

    Written atomically (tmp file + rename): checkpoints are also written on SIGUSR1/SIGTERM
    right before a walltime kill, and --auto_resume picks the latest N, so a half-written
    file must never exist under the final name."""
    save_dict = dict(agent=flax.serialization.to_state_dict(agent))
    save_path = os.path.join(save_dir, f'params_{epoch}.pkl')
    tmp_path = save_path + '.tmp'
    with open(tmp_path, 'wb') as f:
        pickle.dump(save_dict, f)
    os.replace(tmp_path, save_path)
    print(f'Saved to {save_path}')


def resolve_run_dir(path_glob):
    """Resolve a run-directory glob to the single directory it matches."""
    candidates = glob.glob(path_glob)
    assert len(candidates) == 1, f'Found {len(candidates)} candidates: {candidates}'
    return candidates[0]


def _keep_fresh_opt_state(fresh_sd, loaded):
    """Return `loaded` with every 'opt_state' subtree replaced by the fresh agent's.

    Deliberately NOT a shape check: the point is to ignore the optimizer entirely, whatever
    it looks like on either side."""
    if not isinstance(loaded, dict) or not isinstance(fresh_sd, dict):
        return loaded
    out = {}
    for k, v in loaded.items():
        if k not in fresh_sd:
            out[k] = v
        elif k == 'opt_state':
            out[k] = fresh_sd[k]
        else:
            out[k] = _keep_fresh_opt_state(fresh_sd[k], v)
    return out


def restore_agent(agent, restore_path, restore_epoch, params_only=False):
    """Restore the agent from restore_path (glob) / params_{restore_epoch}.pkl.

    params_only=True keeps the freshly-built optimizer state and restores everything else,
    which is what evaluation wants: an eval never steps the optimizer, so it should not
    depend on the optimizer chain matching the one the checkpoint was trained with.

    Params stay strict under this flag: a shape or key mismatch in the weights raises."""
    restore_path = resolve_run_dir(restore_path) + f'/params_{restore_epoch}.pkl'
    with open(restore_path, 'rb') as f:
        load_dict = pickle.load(f)
    loaded = load_dict['agent']
    if params_only:
        loaded = _keep_fresh_opt_state(flax.serialization.to_state_dict(agent), loaded)
    agent = flax.serialization.from_state_dict(agent, loaded)
    print(f'Restored from {restore_path}' + (' (params only)' if params_only else ''))
    return agent
