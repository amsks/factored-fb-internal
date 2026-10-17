"""RLDPCritic: FB critic whose backward map is trained by multi-step latent prediction."""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from impls.agents import Agent, CRITICS, _LR_KEY
from impls.critics.rldp import RLDPCritic
from impls.utils.config_utils import load_config, ConfigError
from impls.utils.datasets import make_dataset, Dataset

B, OBS, ACT, Z, L, H, P, HOR = 6, 12, 4, 5, 5, 16, 2, 4
ACTOR_FAMILIES = ("flowbc", "ddpgbc")


def _cfg(**over):
    cfg = dict(
        seed=0, batch_size=B, z_dim=Z, L_dim=L, num_parallel=P, discount=0.99,
        f_target_tau=0.005, b_target_tau=0.005, ortho_coef=1.0, train_goal_ratio=0.5,
        norm_z=True, lr_f=1e-4, lr_b=1e-4, lr_actor=1e-4,
        horizon=HOR,
        forward=dict(hidden_dim=H, hidden_layers=2, embedding_layers=2),
        backward=dict(hidden_dim=H, hidden_layers=4, norm=True),
        left_encoder=dict(hidden_dim=H, hidden_layers=4, norm=True),
        predictor=dict(hidden_dim=H, hidden_layers=2, embedding_layers=2),
        dataset_class="WindowedDataset",
    )
    cfg.update(over)
    return cfg


def _ex():
    rng = np.random.default_rng(0)
    return (jnp.asarray(rng.standard_normal((B, OBS))),
            jnp.asarray(np.tanh(rng.standard_normal((B, ACT)))))


def _batch(seed=1, b=B, h=HOR):
    rng = np.random.default_rng(seed)
    out = dict(
        observations=jnp.asarray(rng.standard_normal((b, OBS))),
        actions=jnp.asarray(np.tanh(rng.standard_normal((b, ACT)))),
        next_observations=jnp.asarray(rng.standard_normal((b, OBS))),
        rewards=jnp.asarray(np.zeros((b, 1))),
        terminals=jnp.asarray(np.zeros((b,))),
        # GC keys for actors that read them.
        value_goals=jnp.asarray(rng.standard_normal((b, OBS))),
        actor_goals=jnp.asarray(rng.standard_normal((b, OBS))),
        masks=jnp.asarray(np.ones((b,))),
    )
    if h > 1:
        out["future_observations"] = jnp.asarray(rng.standard_normal((b, h - 1, OBS)))
        out["future_actions"] = jnp.asarray(np.tanh(rng.standard_normal((b, h - 1, ACT))))
    return out


def _boot(b=B, seed=2):
    rng = np.random.default_rng(seed)
    return {"next_action": jnp.asarray(np.tanh(rng.standard_normal((b, ACT))))}


# SP loss against a numpy reconstruction
def _np_layernorm(x, scale, bias, eps=1e-5):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * scale + bias


def _np_relu(x):
    return np.maximum(x, 0.0)


def _np_phi(p, x, hidden_layers, norm):
    """PhiMap in numpy."""
    d = p["Dense_0"]
    x = x @ np.asarray(d["kernel"]) + np.asarray(d["bias"])
    ln = p["LayerNorm_0"]
    x = np.tanh(_np_layernorm(x, np.asarray(ln["scale"]), np.asarray(ln["bias"])))
    for i in range(1, hidden_layers):
        d = p[f"Dense_{i}"]
        x = _np_relu(x @ np.asarray(d["kernel"]) + np.asarray(d["bias"]))
    d = p[f"Dense_{hidden_layers}"]
    x = x @ np.asarray(d["kernel"]) + np.asarray(d["bias"])
    if norm:
        n = np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)
        x = np.sqrt(x.shape[-1]) * x / n
    return x


def _np_predictor(p, x, action, hidden_layers):
    """PredictorMap in numpy; x is [P, B, in], action is [B, adim]."""
    def g(k, h):
        return (np.asarray(p["tower"][k]["kernel"][h], np.float64),
                np.asarray(p["tower"][k]["bias"][h], np.float64))

    def ln(k, h):
        return (np.asarray(p["tower"][k]["scale"][h], np.float64),
                np.asarray(p["tower"][k]["bias"][h], np.float64))

    outs = []
    for h in range(x.shape[0]):
        xi = np.asarray(x[h], np.float64)
        w, b = g("embed_z_0", h)
        ze = np.concatenate([xi, action], -1) @ w + b
        sc, bb = ln("embed_z_ln", h)
        ze = np.tanh(_np_layernorm(ze, sc, bb))
        w, b = g("embed_z_3", h)
        ze = _np_relu(ze @ w + b)

        w, b = g("embed_s_0", h)
        se = xi @ w + b
        sc, bb = ln("embed_s_ln", h)
        se = np.tanh(_np_layernorm(se, sc, bb))
        w, b = g("embed_s_3", h)
        se = _np_relu(se @ w + b)

        y = np.concatenate([se, ze], -1)
        for i in range(hidden_layers):
            w, b = g(f"fs_{i}", h)
            y = _np_relu(y @ w + b)
        w, b = g(f"fs_{hidden_layers}", h)
        outs.append(y @ w + b)
    return np.stack(outs, 0)


def _np_sp(critic, params, target, batch, cfg):
    """Latent-prediction loss in numpy."""
    obs = np.asarray(batch["observations"])
    hl_b, norm_b = cfg["backward"]["hidden_layers"], cfg["backward"]["norm"]
    hl_p = cfg["predictor"]["hidden_layers"]

    actions = [np.asarray(batch["actions"])]
    tgt_states = [np.asarray(batch["next_observations"])]
    if "future_observations" in batch:
        fo = np.asarray(batch["future_observations"])
        fa = np.asarray(batch["future_actions"])
        tgt_states += [fo[:, i] for i in range(fo.shape[1])]
        actions += [fa[:, i] for i in range(fa.shape[1])]

    ys = [_np_phi(target["backward"], s, hl_b, norm_b) for s in tgt_states]
    curr = np.broadcast_to(_np_phi(params["backward"], obs, hl_b, norm_b)[None],
                           (cfg["num_parallel"], obs.shape[0], cfg["z_dim"]))
    sp = 0.0
    for a_t, y_t in zip(actions, ys):
        curr = _np_predictor(params["predictor"], curr, a_t, hl_p)
        sp = sp + np.mean(np.sum((curr - y_t[None]) ** 2, axis=-1))
    return sp / max(len(actions), 1)


@pytest.mark.parametrize("h", [2, 4])
def test_sp_loss_matches_the_numpy_reconstruction(h):
    cfg = _cfg(horizon=h)
    critic, params, target = RLDPCritic.build(*_ex(), cfg)
    batch = _batch(h=h)
    got, _ = critic._sp_loss(params, target, batch)
    want = _np_sp(critic, params, target, batch, cfg)
    np.testing.assert_allclose(float(got), float(want), rtol=1e-10, atol=1e-10)


def test_sp_loss_reads_every_window_column():
    """Every future window column affects the SP loss."""
    cfg = _cfg()
    critic, params, target = RLDPCritic.build(*_ex(), cfg)
    batch = _batch()
    base = float(critic._sp_loss(params, target, batch)[0])
    for k in range(HOR - 1):
        for key in ("future_observations", "future_actions"):
            bumped = dict(batch)
            arr = np.asarray(batch[key]).copy()
            arr[:, k] += 1.0
            bumped[key] = jnp.asarray(arr)
            assert float(critic._sp_loss(params, target, bumped)[0]) != base, \
                f"{key} column {k} is not read by the SP rollout"


def test_sp_pairs_on_a_real_window_have_the_right_alignment():
    """_sp_pairs on a WindowedDataset batch pairs a_{t+k-1} with target s_{t+k}."""
    h = HOR
    critic, _, _ = RLDPCritic.build(*_ex(), _cfg())
    T, n_traj = 10, 3
    n = T * n_traj
    rng = np.random.default_rng(0)
    terminals = np.zeros(n, np.float32)
    terminals[np.arange(1, n_traj + 1) * T - 1] = 1.0
    obs = rng.standard_normal((n, OBS)).astype(np.float32)
    acts = np.tanh(rng.standard_normal((n, ACT))).astype(np.float32)
    obs[:, 0] = np.arange(n)                          # row tag
    acts[:, 0] = np.arange(n)                         # row tag
    ds = make_dataset(Dataset.create(observations=obs, actions=acts, terminals=terminals),
                      _cfg())
    idxs = np.array([0, 3, 12, 2 * T + 5])            # idx + h <= trajectory end
    batch = ds.sample(len(idxs), idxs=idxs)
    actions, targets = critic._sp_pairs(batch)
    assert len(actions) == len(targets) == h
    for k in range(1, h + 1):
        np.testing.assert_allclose(np.asarray(actions[k - 1])[:, 0], idxs + k - 1,
                                   err_msg=f"rollout step {k} does not apply a_(t+{k - 1})")
        np.testing.assert_allclose(np.asarray(targets[k - 1])[:, 0], idxs + k,
                                   err_msg=f"rollout step {k} does not target s_(t+{k})")


def test_sp_targets_come_from_the_TARGET_backward_map():
    """SP targets come from the target backward map."""
    cfg = _cfg()
    critic, params, target = RLDPCritic.build(*_ex(), cfg)
    batch = _batch()
    base = float(critic._sp_loss(params, target, batch)[0])
    moved = jax.tree_util.tree_map(lambda a: a + 0.1, target)
    assert float(critic._sp_loss(params, moved, batch)[0]) != base


def test_horizon_mismatch_between_critic_and_dataset_is_a_named_error():
    """A critic/dataset horizon mismatch raises an error naming both."""
    critic, params, target = RLDPCritic.build(*_ex(), _cfg(horizon=5))
    with pytest.raises(AssertionError, match="horizon"):
        critic._sp_loss(params, target, _batch(h=3))


# Differences from FB
def test_predictor_has_no_target_network():
    """Only forward / backward / left_encoder have target networks."""
    critic, params, target = RLDPCritic.build(*_ex(), _cfg())
    assert set(critic.GROUPS) == {"forward", "backward", "left_encoder", "predictor"}
    assert set(critic.TARGETS) == {"forward", "backward", "left_encoder"}
    assert set(target) == set(critic.TARGETS)


def test_predictor_rides_the_backward_learning_rate():
    """The predictor is trained at the backward map's learning rate."""
    assert _LR_KEY["predictor"] == "lr_b"


def test_registered():
    assert CRITICS["rldp"] is RLDPCritic


def test_real_yaml_composes_and_declares_the_windowed_dataset():
    cfg = load_config("rldp", "flowbc", "cube_single")
    assert cfg["dataset_class"] == "WindowedDataset"
    assert int(cfg["horizon"]) >= 1
    assert cfg["predictor"]["hidden_dim"] > 0


@pytest.mark.parametrize("key", ["horizon", "predictor", "dataset_class"])
def test_required_keys_are_enforced(key):
    with pytest.raises(ConfigError, match=key):
        load_config("rldp", "flowbc", "cube_single", overrides=[f"~{key}"])


def test_windowed_dataset_feeds_the_critic_the_columns_it_expects():
    """make_dataset(horizon=h) yields the h-1 window columns _sp_pairs expects."""
    cfg = load_config("rldp", "flowbc", "cube_single")
    cfg["horizon"] = 3
    rng = np.random.default_rng(0)
    n, T = 40, 10
    terminals = np.zeros(n, np.float32)
    terminals[np.arange(1, n // T + 1) * T - 1] = 1.0
    base = Dataset.create(observations=rng.standard_normal((n, OBS)).astype(np.float32),
                          actions=rng.standard_normal((n, ACT)).astype(np.float32),
                          terminals=terminals)
    ds = make_dataset(base, cfg)
    b = ds.sample(8)
    assert b["future_observations"].shape == (8, 2, OBS)
    assert b["future_actions"].shape == (8, 2, ACT)


# Full update with each actor
@pytest.mark.parametrize("actor", ACTOR_FAMILIES)
def test_builds_and_updates_with_each_actor_family(actor):
    cfg = load_config("rldp", actor, "cube_single")
    cfg.update(_cfg())
    ex_obs, ex_act = _ex()
    agent = Agent.create("rldp", actor, 0, ex_obs, ex_act, cfg)
    assert set(agent.net) >= {"forward", "backward", "left_encoder", "predictor"}

    before = jax.tree_util.tree_map(np.asarray, {g: agent.net[g].params for g in agent.net})
    batch = _batch()
    agent, info = agent.update(batch)
    for k, v in info.items():
        assert np.isfinite(np.asarray(v)).all(), f"{k} not finite"
    assert "sp_loss" in info
    after = {g: agent.net[g].params for g in agent.net}
    for g in before:
        moved = any(not np.array_equal(np.asarray(a), np.asarray(b_))
                    for a, b_ in zip(jax.tree_util.tree_leaves(before[g]),
                                     jax.tree_util.tree_leaves(after[g])))
        assert moved, f"group {g!r} did not move after one update with actor {actor!r}"


@pytest.mark.parametrize("actor", ACTOR_FAMILIES)
def test_three_updates_stay_finite(actor):
    cfg = load_config("rldp", actor, "cube_single")
    cfg.update(_cfg())
    agent = Agent.create("rldp", actor, 0, *_ex(), cfg)
    for i in range(3):
        agent, info = agent.update(_batch(seed=10 + i))
    assert np.isfinite(np.asarray(info["actor_loss"])).all()
    assert np.isfinite(np.asarray(info["sp_loss"])).all()


def test_the_sp_term_is_what_trains_the_predictor():
    """Only the SP term trains the predictor."""
    cfg = _cfg()
    critic, params, target = RLDPCritic.build(*_ex(), cfg)
    batch = _batch()
    g = jax.grad(lambda p: critic._sp_loss({**params, **p}, target, batch)[0])(
        {"predictor": params["predictor"]})
    assert any(float(jnp.sum(jnp.abs(l))) > 0 for l in jax.tree_util.tree_leaves(g["predictor"]))


def test_stage_loss_is_the_sum_of_its_reported_parts():
    """fb_loss equals offdiag + diag + sp + ortho_coef * ortho."""
    cfg = _cfg()
    critic, params, target = RLDPCritic.build(*_ex(), cfg)
    batch, boot = _batch(), _boot()
    cond = {"z": jnp.asarray(np.random.default_rng(3).standard_normal((B, Z)))}
    loss, info = critic.stage_loss("fb", params, target, batch, cond, boot)
    recon = (info["fb_offdiag"] + info["fb_diag"] + info["sp_loss"]
             + cfg["ortho_coef"] * info["ortho_loss"])
    np.testing.assert_allclose(float(loss), float(recon), atol=1e-10)
    np.testing.assert_allclose(float(info["fb_loss"]), float(loss), atol=0)


def test_stage_loss_itself_sends_no_backward_gradient_through_the_fb_residual():
    """With ortho off, B's gradient comes only from the SP loss."""
    cfg = _cfg(ortho_coef=0.0)
    critic, params, target = RLDPCritic.build(*_ex(), cfg)
    batch, boot = _batch(), _boot()
    cond = critic.sample_cond({"backward": type("S", (), {"params": params["backward"]})},
                              batch, jax.random.PRNGKey(0))

    def full(p):
        return critic.stage_loss("fb", {**params, **p}, target, batch, cond, boot)[0]

    def sp_only(p):
        return critic._sp_loss({**params, **p}, target, batch)[0]

    g_full = jax.grad(full)({"backward": params["backward"]})["backward"]
    g_sp = jax.grad(sp_only)({"backward": params["backward"]})["backward"]
    for a, b in zip(jax.tree_util.tree_leaves(g_full), jax.tree_util.tree_leaves(g_sp)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=0, atol=0)

    # The residual trains F, and the SP term reaches B.
    g_fwd = jax.grad(full)({"forward": params["forward"]})["forward"]
    assert any(float(np.abs(np.asarray(l)).sum()) > 0
               for l in jax.tree_util.tree_leaves(g_fwd)), "the FB residual is dead"
    assert any(float(np.abs(np.asarray(l)).sum()) > 0
               for l in jax.tree_util.tree_leaves(g_full)), \
        "the backward map gets no gradient at all -- the comparison above is vacuous"
