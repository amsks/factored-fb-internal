#!/usr/bin/env python
"""prepare_anneal_ckpt.py — write a finetune-ready checkpoint for annealed SM finetuning.

Two facts about `restore_agent` make this necessary (spec section 3):

1. It deserializes the WHOLE TrainState, so `network.step` comes back as the anchor's value
   (1_000_001). A lambda schedule keyed on `network.step` would therefore start fully ramped and
   the run would silently be a constant-lambda cold start.
2. `optax.adam(learning_rate=<callable>)` ends in `ScaleByScheduleState`, while a constant LR ends
   in `EmptyState`. So an anchor saved with a constant LR CANNOT restore into a scheduled optimizer.

Rather than fight either, we write a new checkpoint: the anchor's params, a FRESH `opt_state` taken
from the finetune agent's own (scheduled) `tx`, and `step = 0`.

Fresh Adam moments are deliberate. Restored moments are tiny at convergence, so the first SM
gradient would be amplified by 1/sqrt(v) -- a concrete mechanism by which one large early step
destroys the value geometry and manufactures a spurious "the measure objective is incompatible"
result. Fresh moments + the LR warmup remove that failure mode.

Run under the jax venv, from third_party/ogbench/impls:

    cd third_party/ogbench/impls && /dev/shm/.venv-jax/bin/python \
        ../../../scripts/prepare_anneal_ckpt.py \
        --src ~/bridge_ckpt_backup/.../GCIQL-State-Flow-Bilinear-d50/sd000_20260615_011444 \
        --epoch 1000000 --out /dev/shm/anneal_ckpt/sd000 --lr-warmup 10000
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import flax
import jax
import jax.numpy as jnp

# Resolve imports independently of the cwd. `python scripts/prepare_anneal_ckpt.py` puts the
# SCRIPT's directory on sys.path, not the cwd, so `agents.gciql` (which lives under
# third_party/ogbench/impls) would otherwise resolve to an empty namespace package. We add both
# the shim (for gciql_flow) and impls (for agents.* / utils.*) explicitly. The repo root is
# deliberately NOT added: it contains a torch-side `agents/` package that would shadow impls'.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, 'third_party', 'ogbench', 'impls'))
sys.path.insert(0, os.path.join(_REPO, 'tools', 'wandb_mode_shim'))


def splice_checkpoint(finetune_agent, src_pkl: str, out_dir: str) -> str:
    """Write `<out_dir>/params_0.pkl`: the source checkpoint's params inside `finetune_agent`'s
    state (fresh opt_state, step=0). Returns the written path.

    `finetune_agent` must already be constructed with the finetune config, so its `opt_state`
    matches the scheduled `tx`. We copy ONLY `network.params` across; everything else stays as the
    finetune agent built it."""
    with open(src_pkl, 'rb') as f:
        src = pickle.load(f)['agent']

    state = flax.serialization.to_state_dict(finetune_agent)
    src_params = src['network']['params']

    # The param trees must match exactly, or the architectures differ (e.g. fb_forward_z changes
    # phi's input dim). Fail loudly here rather than mid-run.
    tgt_params = state['network']['params']
    s_struct = jax.tree_util.tree_structure(src_params)
    t_struct = jax.tree_util.tree_structure(tgt_params)
    if s_struct != t_struct:
        raise SystemExit(
            f'param tree mismatch: the finetune config does not match the checkpoint architecture\n'
            f'  source: {s_struct}\n  target: {t_struct}')
    for a, b in zip(jax.tree_util.tree_leaves(src_params), jax.tree_util.tree_leaves(tgt_params)):
        if a.shape != b.shape:
            raise SystemExit(f'param shape mismatch: {a.shape} vs {b.shape}')

    state['network']['params'] = src_params
    state['network']['step'] = 0          # <- lam(0)=0 and the LR warmup both start here
    # state['network']['opt_state'] is left as the finetune agent's FRESH init (deliberate)

    spliced = flax.serialization.from_state_dict(finetune_agent, state)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'params_0.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(dict(agent=flax.serialization.to_state_dict(spliced)), f)
    return out_path


def _dims_from_params(params):
    """Infer (obs_dim, action_dim) from the bilinear param tree, so the script needs no env.

    value.phi  Dense_0 kernel: (obs_dim, hidden)                  [ensemble=False]
    critic.phi Dense_0 kernel: (2, obs_dim + action_dim, hidden)  [ensemble=True -> leading head axis]
    """
    obs_dim = params['modules_value']['phi']['Dense_0']['kernel'].shape[0]
    critic_in = params['modules_critic']['phi']['Dense_0']['kernel'].shape[1]
    return obs_dim, critic_in - obs_dim


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', required=True, help='anchor run directory containing params_<epoch>.pkl')
    ap.add_argument('--epoch', type=int, default=1_000_000)
    ap.add_argument('--out', required=True, help='output directory; receives params_0.pkl')
    ap.add_argument('--lr-warmup', type=int, default=10_000)
    ap.add_argument('--latent-dim', type=int, default=50)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    from gciql_flow import GCIQLFlowAgent, get_config

    src_pkl = os.path.join(args.src, f'params_{args.epoch}.pkl')
    if not os.path.exists(src_pkl):
        raise SystemExit(f'no such checkpoint: {src_pkl}')
    with open(src_pkl, 'rb') as f:
        src_params = pickle.load(f)['agent']['network']['params']
    obs_dim, act_dim = _dims_from_params(src_params)
    print(f'  inferred obs_dim={obs_dim} action_dim={act_dim} from the checkpoint')

    cfg = dict(get_config())
    cfg['encoder'] = None
    cfg['frame_stack'] = None
    cfg['value_arch'] = 'bilinear'
    cfg['value_latent_dim'] = args.latent_dim
    cfg['anneal_lr_warmup'] = args.lr_warmup      # <- makes opt_state match the finetune tx
    ex_obs = jnp.zeros((2, obs_dim))
    ex_act = jnp.zeros((2, act_dim))
    agent = GCIQLFlowAgent.create(args.seed, ex_obs, ex_act, cfg)

    out = splice_checkpoint(agent, src_pkl, args.out)
    print(f'  wrote {out}  (step=0, fresh opt_state, lr_warmup={args.lr_warmup})')


if __name__ == '__main__':
    main()
