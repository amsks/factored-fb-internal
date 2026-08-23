"""Generate the bit-identity baseline for the seam boot contract (workstream 0).

Run this ONCE, BEFORE any actor/critic edit, and commit the .npz it writes. Every
later task in workstream 0 must keep tests/impls/test_boot_contract.py green
against it.

    JAX_PLATFORMS=cpu PYTHONPATH=$PWD python tests/impls/gen_boot_contract_fixture.py

CPU is pinned so the baseline does not depend on which GPU happened to be free.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from impls.agents import Agent, CRITICS
from impls.utils.config_utils import load_config

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "boot_contract_reference.npz")
ACTOR_FAMILIES = ("flowbc", "ddpgbc", "awr", "fql")
B, OBS, ACT = 8, 40, 5


def make_batch(seed=1):
    r = np.random.default_rng(seed)
    # value_goals and actor_goals MUST differ: if they were the same array, an actor
    # that drew its "value-goal" bootstrap at the actor goal would pass every test.
    g_value = jnp.asarray(r.standard_normal((B, OBS)))
    g_actor = jnp.asarray(r.standard_normal((B, OBS)))
    rh = np.random.default_rng(seed + 9000)     # hierarchical keys only; see below
    return dict(
        observations=jnp.asarray(r.standard_normal((B, OBS))),
        actions=jnp.asarray(np.tanh(r.standard_normal((B, ACT)))),
        next_observations=jnp.asarray(r.standard_normal((B, OBS))),
        next_actions=jnp.asarray(np.tanh(r.standard_normal((B, ACT)))),
        # vip's stage_loss reads initial_observations; every other critic ignores it.
        initial_observations=jnp.asarray(r.standard_normal((B, OBS))),
        value_goals=g_value, actor_goals=g_actor,
        # hierarchical keys, so hiql is covered by this gate too (HGCDataset emits these; a
        # GCDataset-shaped batch alone makes HIQLCritic.sample_cond raise). Drawn from a
        # SEPARATE rng: taking them from `r` would shift its stream and change rewards/masks
        # below, moving every recorded loss for reasons that have nothing to do with the
        # change under test. (Learned the hard way -- it moved 76 scalars.)
        # rldp's self-predictive loss needs an h-step window. Same separate-rng rule as the
        # hierarchical keys below: taken from `r` these would shift rewards/masks and move
        # every recorded loss for reasons unrelated to the change under test.
        future_observations=jnp.asarray(rh.standard_normal((B, 4, OBS))),
        future_actions=jnp.asarray(np.tanh(rh.standard_normal((B, 4, ACT)))),
        # psm keys its proto policy on the DATASET ROW id and refuses a batch without it.
        # Fixed ids, not arange(B): arange is the positional fallback the raise exists to
        # prevent, so using it here would pin the very behaviour we removed.
        index=jnp.asarray([5, 11, 23, 4, 9, 1, 6, 2][:B] * (B // 8 + 1))[:B],
        low_actor_goals=jnp.asarray(rh.standard_normal((B, OBS))),
        high_actor_goals=jnp.asarray(rh.standard_normal((B, OBS))),
        high_actor_targets=jnp.asarray(rh.standard_normal((B, OBS))),
        rewards=jnp.asarray(r.integers(0, 2, size=B) - 1.0),
        masks=jnp.asarray(r.integers(0, 2, size=B).astype(float)),
    )


def pair_statics(critic, actor):
    """(boot dict, info dict) for one critic x actor pair, at fixed seeds."""
    cfg = load_config(critic, actor, "cube_single")
    cfg["batch_size"] = B
    agent = Agent.create(critic, actor, 0, jnp.zeros((B, OBS)), jnp.zeros((B, ACT)), cfg)
    batch = make_batch()
    cond = agent.critic.sample_cond(agent.net, batch, jax.random.PRNGKey(0))
    boot = agent.actor.sample_bootstrap(agent.net, batch, cond, jax.random.PRNGKey(1))
    info = agent.compute_static(batch, cond, boot)
    return boot, info


def collect():
    out, failed = {}, []
    for critic in sorted(CRITICS):
        for actor in ACTOR_FAMILIES:
            try:
                boot, info = pair_statics(critic, actor)
            except Exception as e:                                   # noqa: BLE001
                failed.append(f"{critic}|{actor}|{type(e).__name__}")
                continue
            for k, v in boot.items():
                out[f"boot|{critic}|{actor}|{k}"] = np.asarray(v)
            for k, v in info.items():
                out[f"info|{critic}|{actor}|{k}"] = np.asarray(jnp.asarray(v).mean())
    out["__failed__"] = np.asarray(sorted(failed))
    return out


if __name__ == "__main__":
    data = collect()
    os.makedirs(os.path.dirname(FIXTURE), exist_ok=True)
    np.savez(FIXTURE, **data)
    n_pairs = len({k.split("|")[1] + "|" + k.split("|")[2] for k in data if k != "__failed__"})
    print(f"wrote {FIXTURE}: {n_pairs} pairs, {len(data) - 1} arrays, "
          f"{len(data['__failed__'])} pairs failed")
    for f in data["__failed__"]:
        print("  failed:", f)
