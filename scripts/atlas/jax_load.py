"""Shared jax-runtime setup and checkpoint loading for the atlas.

MUST run under /dev/shm/.venv-jax/bin/python (jax 0.10.0); the repo .venv has no jax.
Handles both agent classes: gciql_flow (bilinear/mlp flow + ddpgbc cells, simplesf) and
stock gciql (the mlp_ddpgbc actor-isolation control). See analysis/atlas/DECISIONS.md.
"""
import json
import os
import sys

R = '/home/adityamohan/git/Factored-FB'


def setup_jax_path():
    for p in (R, f'{R}/scripts', f'{R}/third_party/ogbench/impls', f'{R}/tools/wandb_mode_shim'):
        if p not in sys.path:
            sys.path.insert(0, p)


def _read_agent_cfg(ckpt_dir):
    """flags.json may sit in ckpt_dir or its parent; return the 'agent' sub-dict."""
    for cand in (os.path.join(ckpt_dir, 'flags.json'),
                 os.path.join(ckpt_dir, '..', 'flags.json')):
        if os.path.exists(cand):
            return json.load(open(cand)).get('agent', {})
    return {}


# agent_name -> (module, class); mirrors scripts/oodgrid_run_jax.py:62 JAX_AGENTS.
_AGENTS = {'gciql_flow': ('gciql_flow', 'GCIQLFlowAgent'),
           'gciql': ('agents.gciql', 'GCIQLAgent')}


def load_agent(ckpt_dir, step):
    """Load a jax agent from a checkpoint dir. Dispatches on agent_name in flags.json.
    Returns the restored agent (has .network.select(...), .sample_actions(...))."""
    setup_jax_path()
    import importlib

    import jax.numpy as jnp
    from utils.flax_utils import restore_agent
    ag = _read_agent_cfg(ckpt_dir)
    name = ag.get('agent_name', 'gciql_flow')
    if name not in _AGENTS:
        raise ValueError(f'unhandled agent_name={name!r}')
    modname, clsname = _AGENTS[name]
    mod = importlib.import_module(modname)
    cls = getattr(mod, clsname)
    get_config = mod.get_config
    cfg = dict(get_config())
    cfg.update({k: v for k, v in ag.items() if k in cfg})
    cfg['encoder'] = None
    if 'frame_stack' in cfg:
        cfg['frame_stack'] = None
    ex_obs = jnp.zeros((1, 28), jnp.float32)
    ex_act = jnp.zeros((1, 5), jnp.float32)
    agent = cls.create(0, ex_obs, ex_act, cfg)
    return restore_agent(agent, ckpt_dir, step)


# Registry: critic name -> (checkpoint dir, step). Verified on persistent disk.
_B = os.path.expanduser('~/bridge_ckpt_backup/gciql_outputs/factored-fb')
CRITICS = {
    'bilinear_flow':   (f'{_B}/GCIQL-State-Flow-Bilinear-d50/sd000_20260615_011444', 500000),
    'bilinear_ddpgbc': (f'{_B}/GCIQL-State-Flow-Bilinear-d50-ddpgbc/sd000_20260617_221451', 500000),
    'mlp_flow':        (f'{_B}/GCIQL-State-Flow/sd000_20260609_234409', 1000000),
    'mlp_ddpgbc':      (f'{_B}/factored-fb-gciql/sd001_20260518_201036', 500000),
    'simplesf':        (f'{_B}/GCIQL-State-Flow-SimpleSF-Arch/cube_single_state_gciql_flow_simplesf_arch_s0', 500000),
}
