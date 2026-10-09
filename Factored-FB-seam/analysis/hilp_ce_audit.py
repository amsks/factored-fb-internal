"""Small source-level HILP audit. No training, environment launch, or source edits.

Run via /tmp/carry-drop-python.py to use the existing cached JAX environment.
"""

import dataclasses
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from env import make_config
from impls.actors.flowbc import FlowBCActor
from impls.critics.hilp import HILPCritic, hilp_phi_loss
from impls.critics.hilp_density import HILPDensityCritic


class LinearProbe(HILPDensityCritic):
    """Keep production losses/readouts; replace networks with explicit linear maps."""

    def _phi(self, p, x):
        return x[..., :2] @ p["matrix"]

    def _feature_net(self, p, stats, x):
        return self._phi(p, x) - stats["mean"]

    def _sf(self, p, obs, z, actions):
        return p["bias"][:, None, :] + actions[None, :, :1] * p["slope"][:, None, :]

    def _le(self, container, x):
        return x


def linear_critic(**updates):
    values = dict(hilp_phi1=None, hilp_phi2=None, hilp_sf=None, hilp_stats=None,
                  z_dim=2, num_parallel=2, discount=.99, hilp_discount=.99,
                  hilp_expectile=.5, phi_tau=.005, sf_tau=.005, norm_z=True,
                  stats_momentum=.005, q_loss=False, mix_ratio=.5,
                  head_reduce="mean", actor_head_reduce="mean", sf_loss_scale=.5,
                  actor_q_center=False)
    return LinearProbe(**(values | updates))


def max_tree_abs(tree):
    return float(max(jnp.max(jnp.abs(x)) for x in jax.tree_util.tree_leaves(tree)))


def temperature_probe():
    # Tiny architecture, same configuration loader and production critic factory.
    overrides = ["measure_density_temp=0.25", "measure_density_actor_temp=0.5",
                 "z_dim=2", "L_dim=2", "phi.hidden_dim=8", "phi.hidden_layers=2",
                 "sf.hidden_dim=8", "sf.hidden_layers=1", "sf.embedding_layers=2",
                 "hilp_left_encoder.identity=true"]
    cfg = make_config("hilp_density_tdjc", "flowbc", "cube_single", overrides)
    cfg["seed"] = 0
    critic, _, _ = HILPDensityCritic.build(jnp.zeros((3, 37)), jnp.zeros((3, 5)), cfg)
    assert cfg["measure_density_temp"] == .25 and critic.temp == 1.0
    assert cfg["measure_density_actor_temp"] == .5 and critic.actor_temp is None
    return {"requested_critic_temperature": cfg["measure_density_temp"],
            "actual_critic_temperature": critic.temp,
            "requested_actor_temperature": cfg["measure_density_actor_temp"],
            "actual_actor_temperature": critic.actor_temp,
            "scope": "Nondefault overrides silently inert; existing 1/null defaults coincide."}


def ce_and_heads_probe():
    critic = linear_critic()
    obs = jnp.array([[-1., .2], [.1, -.5], [1., .4]])
    batch = {"observations": obs, "next_observations": obs[::-1],
             "actions": jnp.array([[-.3], [.1], [.5]]), "terminals": jnp.array([0., 1., 0.])}
    params = {"hilp_phi1": {"matrix": jnp.array([[1., .2], [-.3, .8]])},
              "hilp_stats": {"mean": jnp.array([.2, -.1])},
              "hilp_sf": {"bias": jnp.array([[.2, .8], [-.4, .3]]),
                          "slope": jnp.array([[.9, -.1], [.2, .7]])}}
    target = jax.tree.map(lambda x: x * .8, params)
    cond = {"z": jnp.array([[1., 0.], [0., 1.], [1., -1.]])}
    boot = {"next_action": batch["actions"] * .5}
    loss, info = critic.stage_loss("sf", params, target, batch, cond, boot)
    f = critic._sf(params["hilp_sf"], obs, cond["z"], batch["actions"])
    b = critic._feature_net(params["hilp_phi1"], params["hilp_stats"], batch["next_observations"])
    tf = critic._sf(target["hilp_sf"], obs, cond["z"], boot["next_action"])
    tb = critic._feature_net(target["hilp_phi1"], target["hilp_stats"], batch["next_observations"])
    logp = jax.nn.log_softmax(jnp.einsum("hbd,cd->hbc", f, b), -1)
    target_p = jax.nn.softmax(jnp.einsum("hbd,cd->hbc", tf, tb), -1)
    discount = .99 * (1 - batch["terminals"][:, None])
    labels = (1 - discount) * jnp.eye(3) + discount * target_p
    expected = -.5 * jnp.mean(jnp.sum(labels * logp, -1))
    np.testing.assert_allclose(loss, expected, atol=1e-14, rtol=0)
    np.testing.assert_allclose(labels.sum(-1), 1, atol=1e-14, rtol=0)
    np.testing.assert_allclose(labels[:, 1], jnp.array([[0., 1., 0.], [0., 1., 0.]]))
    gp, gt = jax.grad(lambda p, t: critic.stage_loss("sf", p, t, batch, cond, boot)[0], (0, 1))(params, target)
    assert max_tree_abs(gp["hilp_phi1"]) == 0
    assert max_tree_abs(gt) == 0
    assert max_tree_abs(gp["hilp_sf"]) > 0
    # Target centering uses stats with tau zero; this is harmless in the softmax,
    # because subtracting any common pool mean adds only a row-constant logit.
    shifted = target | {"hilp_stats": {"mean": jnp.array([17., -13.])}}
    shifted_loss = critic.stage_loss("sf", params, shifted, batch, cond, boot)[0]
    np.testing.assert_allclose(shifted_loss, loss, atol=1e-14, rtol=0)

    cr_min = linear_critic(actor_head_reduce="min")
    hparams = {"hilp_sf": {"bias": jnp.array([[2., 0.], [-2., 0.]]),
                           "slope": jnp.zeros((2, 2))}}
    hcond = {"_pool_phi": jnp.array([[-1., 0.], [1., 0.]]),
             "z": jnp.array([[1., 0.], [1., 0.]])}
    q, heads = cr_min.q_fn(hparams, jnp.zeros((2, 2)), hcond, jnp.zeros((2, 1)))
    np.testing.assert_allclose(q, heads.mean(0))
    assert not np.allclose(q, heads.min(0))
    return {"ce": float(info["density_ce"]), "scaled_loss": float(loss),
            "manual_loss_matches": True, "row_sums": np.asarray(labels.sum(-1)).tolist(),
            "online_phi_gradient_max": max_tree_abs(gp["hilp_phi1"]),
            "online_sf_gradient_max": max_tree_abs(gp["hilp_sf"]),
            "target_gradient_max": max_tree_abs(gt), "target_mean_translation_invariant": True,
            "configured_actor_head_reduce": "min", "actual_returned_q": np.asarray(q).tolist(),
            "head_min_q": np.asarray(heads.min(0)).tolist(),
            "scope": "Head-reduction override is ignored; all examined campaign configs request mean."}


def object_alias_probe():
    obs, goals, nobs = jnp.array([[0., 0.]]), jnp.array([[0., 1.]]), jnp.array([[0., 2.]])
    apply = lambda p, x: p * x[:, :1]
    loss_fn = lambda p: hilp_phi_loss(p, p, p, p, apply, obs, goals, nobs, .99, .5)[0]
    full_loss = loss_fn(jnp.array(1.))
    sliced_loss = hilp_phi_loss(1., 1., 1., 1., apply, obs[:, :1], goals[:, :1], nobs[:, :1], .99, .5)[0]
    grad = jax.grad(loss_fn)(jnp.array(1.))
    assert float(full_loss) > .99 and float(sliced_loss) < 2e-6 and float(grad) == 0
    return {"obs_object_arm": np.asarray(obs).tolist(), "goal_object_arm": np.asarray(goals).tolist(),
            "next_object_arm": np.asarray(nobs).tolist(), "full_state_goal_test_loss": float(full_loss),
            "object_space_goal_test_loss": float(sliced_loss), "gradient": float(grad),
            "scope": "Object-only phi forces equal embeddings, but full-state equality labels the goal as unreached; affects both HILP variants with object-only inputs."}


def goal_kernel_encoding_probe():
    critic = linear_critic(goal_kernel=True, kernel_blocks=((0, 2, 1.),))
    pool = jnp.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
    net = {"hilp_phi1": SimpleNamespace(params={"matrix": jnp.eye(2)}),
           "hilp_stats": SimpleNamespace(params={"mean": jnp.zeros(2)})}
    goal = jnp.array([1., 1.])
    batch = {"observations": pool, "next_observations": pool,
             "actor_goals": jnp.broadcast_to(goal, pool.shape)}
    train = critic.sample_cond(net, batch, jax.random.PRNGKey(0))["z"]
    mapped = critic.map_eval_cond(net, {"goal": goal, "z": jnp.zeros(2)}, observations=pool)["z"]
    assert not np.allclose(train, mapped)
    np.testing.assert_allclose(mapped[-1], 0)
    return {"training_goal_z": np.asarray(train).tolist(), "evaluation_goal_z_by_current_state": np.asarray(mapped).tolist(),
            "scope": "Goal-kernel-only mismatch: train uses whitened absolute goal; inherited goal evaluation uses relative goal-current feature direction (zero at goal). Reward-inference evaluation is a different route."}


class ToyFlow(FlowBCActor):
    def _actor(self, p, obs, z, noise):
        return jnp.ones((len(obs), 1)) * p

    def _vf(self, p, obs, action, t):
        return jnp.zeros_like(action)

    def _rollout(self, p, obs, noise):
        return jnp.zeros((len(obs), 1))


class ToyReadout:
    def __init__(self, offsets, centered, scale=1.):
        self.offsets, self.centered, self.scale = jnp.asarray(offsets), centered, scale

    def cond_vec(self, cond):
        return cond["z"]

    def q_fn(self, params, obs, cond, actions, stop_encoder_grad=False):
        q = self.scale * (actions[:, 0] + self.offsets)
        return q, (q - q.mean())[None] if self.centered else jnp.stack([q, q])


def actor_normalizer_probe():
    actor = ToyFlow(None, None, 1, 3., 0.)
    batch = {"observations": jnp.zeros((2, 2)), "actions": jnp.zeros((2, 1))}
    cond = {"z": jnp.ones((2, 2))}
    boot = {k: jnp.zeros((2, 1)) for k in ("flow_x0", "flow_t", "actor_noise")}

    def evaluate(offsets, centered, scale=1.):
        critic = ToyReadout(offsets, centered, scale)
        loss_fn = lambda theta: actor.loss({"actor": theta, "actor_vf": 0.}, {}, batch, critic, cond, boot)[0]
        value, grad = jax.value_and_grad(loss_fn)(jnp.array(.1))
        number = lambda x: float(x) if bool(jnp.isfinite(x)) else str(float(x))
        return {"loss": number(value), "gradient": number(grad),
                "finite": bool(jnp.isfinite(value) & jnp.isfinite(grad))}

    raw = evaluate([10., 12.], False)
    centered = evaluate([10., 12.], True)
    scaled = evaluate([10., 12.], True, .01)
    zero_spread = evaluate([1., 1.], True)
    assert raw["gradient"] > 0 and centered["gradient"] < 0
    np.testing.assert_allclose(centered["gradient"], scaled["gradient"], atol=1e-14, rtol=0)
    assert not zero_spread["finite"]
    return {"same_values_raw_head_normalizer": raw, "same_values_centered_normalizer": centered,
            "centered_and_times_1_minus_gamma": scaled, "constant_q_centered_normalizer": zero_spread,
            "scope": "Production FlowBC.loss on explicit toy actor/readout. Centering can reverse combined actor+BC gradient; positive scaling cancels. Zero spread gives nonfinite loss/gradient. Does not establish occurrence in recorded runs."}


def main():
    report = {"jax_version": jax.__version__, "dtype": "float64", "training_steps": 0,
              "temperature_alias": temperature_probe(), "ce_and_heads": ce_and_heads_probe(),
              "object_coordinate_alias": object_alias_probe(), "goal_kernel_encoding": goal_kernel_encoding_probe(),
              "actor_normalization": actor_normalizer_probe()}
    sources = ["impls/critics/hilp.py", "impls/critics/hilp_density.py", "impls/actors/flowbc.py", "impls/actors/ddpgbc.py",
               "impls/utils/datasets.py", "impls/configs/critic/hilp_density_tdjc.yaml"]
    report["source_sha256"] = {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in sources}
    report["all_assertions_passed"] = True
    (ROOT / "analysis/hilp_ce_audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
