"""RLDPCritic — the FB critic plus its self-predictive (SP) head.

PARITY METHOD: an INDEPENDENT NUMPY RECONSTRUCTION, not a torch->flax fixture
(precedent: tests/impls/test_density_fb.py, which reconstructs the density CE in numpy
because no cross-import parity harness exists for that critic either). `impls/utils/
torch_to_flax.py` is not usable here: it maps weights out of `fixtures/fb_reference.npz`,
and that fixture was exported from a PyTorch FB model that has no `_predictor` at all, so
there are no reference weights for the one module this critic adds. The reference is
therefore rebuilt from scratch below in numpy --
  * `_np_phi`      : PhiMap forward (Dense/LayerNorm/tanh/relu/Dense/psm_norm),
  * `_np_predictor`: one VForwardMap tower, per ensemble head,
  * `_np_sp`       : the rollout and the L2 accumulation of agents/rldp/agent.py::update_fb
-- and compared against `RLDPCritic._sp_loss` at 1e-10 under x64. The numpy code reads
only the flax parameter arrays, so a wrong module (wrong concat order, wrong embedding
input, wrong ensemble slicing) and a wrong loss (wrong reduction, wrong stop-gradient,
wrong pairing) both show up as a mismatch.

What that DOES NOT establish: nothing here runs PyTorch, so agreement with the actual
torch tensors is argued from transcription, not measured.
"""
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
ACTOR_FAMILIES = ("flowbc", "ddpgbc", "awr", "fql")


def _cfg(**over):
    cfg = dict(
        seed=0, batch_size=B, z_dim=Z, L_dim=L, num_parallel=P, discount=0.99,
        f_target_tau=0.005, b_target_tau=0.005, ortho_coef=0.01, train_goal_ratio=0.5,
        fb_pessimism_penalty=0.0, norm_z=True, lr_f=1e-4, lr_b=1e-4, lr_actor=1e-4,
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
        # GC keys so the batch also satisfies actors that reach for them.
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


# ==========================================================================
# 1. the SP loss against an independent numpy reconstruction
# ==========================================================================
def _np_layernorm(x, scale, bias, eps=1e-5):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * scale + bias


def _np_relu(x):
    return np.maximum(x, 0.0)


def _np_phi(p, x, hidden_layers, norm):
    """impls/utils/networks.py::PhiMap, in numpy."""
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
    """impls/utils/networks.py::PredictorMap, in numpy, head by head.

    Reconstructed from nn_models.VForwardMap directly: embed_z sees [x, action],
    embed_s sees x ALONE, and the trunk consumes concat([s_embedding, z_embedding]).
    `x` is [P, B, in], `action` is [B, adim] and is shared across heads."""
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
    """td_jepa's update_fb SP block (metamotivo/agents/rldp/agent.py), in numpy: prepend
    the one-step pair, then zip window columns 1:1. The columns arrive PRE-ALIGNED from
    WindowedDataset (obs column k = the state action column k reaches, s_{t+2+k}; the
    2026-08-25 off-by-one fix), so this enumeration is the corrected pairing
    (a_{t+k-1} -> s_{t+k}); the alignment itself is pinned by
    `test_sp_pairs_on_a_real_window_implement_the_reference_alignment` below."""
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


@pytest.mark.parametrize("h", [1, 2, 4])
def test_sp_loss_matches_the_numpy_reconstruction(h):
    cfg = _cfg(horizon=h)
    critic, params, target = RLDPCritic.build(*_ex(), cfg)
    batch = _batch(h=h)
    got, _ = critic._sp_loss(params, target, batch)
    want = _np_sp(critic, params, target, batch, cfg)
    np.testing.assert_allclose(float(got), float(want), rtol=1e-10, atol=1e-10)


def test_sp_loss_reads_every_window_column():
    """Perturbing ANY future column must move the SP loss. A rollout that silently
    stopped after the first step (or that indexed only column 0) would pass the shape
    tests and this is what catches it."""
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


def test_sp_pairs_on_a_real_window_implement_the_reference_alignment():
    """END-TO-END pin of the 2026-08-25 off-by-one fix: run _sp_pairs on a batch served
    by the REAL WindowedDataset over row-tagged data and check the zipped rollout is
    td_jepa's alignment -- step k applies a_{t+k-1} and targets s_{t+k}, for k = 1..h:

        actions = [a_t, a_{t+1}, ..., a_{t+h-1}]
        targets = [s_{t+1}, s_{t+2}, ..., s_{t+h}]

    The pre-fix pairing (targets [s_{t+1}, s_{t+1}, s_{t+2}, ..., s_{t+h-1}]: step 2
    re-targeting s_{t+1}, s_{t+h} never reached) fails this on every column."""
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
    idxs = np.array([0, 3, 12, 2 * T + 5])            # interior: idx + h <= traj end
    batch = ds.sample(len(idxs), idxs=idxs)
    actions, targets = critic._sp_pairs(batch)
    assert len(actions) == len(targets) == h
    for k in range(1, h + 1):
        np.testing.assert_allclose(np.asarray(actions[k - 1])[:, 0], idxs + k - 1,
                                   err_msg=f"rollout step {k} does not apply a_(t+{k - 1})")
        np.testing.assert_allclose(np.asarray(targets[k - 1])[:, 0], idxs + k,
                                   err_msg=f"rollout step {k} does not target s_(t+{k})")


def test_sp_targets_come_from_the_TARGET_backward_map():
    """The reference computes every SP target under no_grad with _target_backward_map.
    Perturbing the target params must move the loss; and the gradient must not flow
    through them (they are not a differentiable argument here, so this is asserted via
    the stop_gradient: B_target's contribution to d(sp)/d(online backward) is only
    through `curr`)."""
    cfg = _cfg()
    critic, params, target = RLDPCritic.build(*_ex(), cfg)
    batch = _batch()
    base = float(critic._sp_loss(params, target, batch)[0])
    moved = jax.tree_util.tree_map(lambda a: a + 0.1, target)
    assert float(critic._sp_loss(params, moved, batch)[0]) != base


def test_horizon_mismatch_between_critic_and_dataset_is_a_named_error():
    """A critic built for h=5 handed 2-column windows would quietly train a shorter
    rollout. That must be an error naming both numbers."""
    critic, params, target = RLDPCritic.build(*_ex(), _cfg(horizon=5))
    with pytest.raises(AssertionError, match="horizon"):
        critic._sp_loss(params, target, _batch(h=3))
    with pytest.raises(AssertionError, match="future_observations"):
        critic._sp_loss(params, target, _batch(h=1))


# ==========================================================================
# 2. the three documented divergences from impls/critics/fb.py
# ==========================================================================
def test_the_successor_residual_does_not_train_the_backward_map():
    """Divergence 1: the reference detaches B(goal) inside Ms. So with the SP and ortho
    terms switched off (ortho_coef=0, and the SP term removed by hand), the backward map
    gets zero gradient from the FB residual."""
    cfg = _cfg(ortho_coef=0.0)
    critic, params, target = RLDPCritic.build(*_ex(), cfg)
    batch, boot = _batch(), _boot()
    cond = critic.sample_cond({"backward": type("S", (), {"params": params["backward"]})},
                              batch, jax.random.PRNGKey(0))

    def fb_only(p):
        # exactly stage_loss minus the sp and ortho terms
        full = {**params, **p}
        obs, action, next_obs = batch["observations"], batch["actions"], batch["next_observations"]
        nle = critic._le(target["left_encoder"], next_obs)
        tFs = critic._F(target["forward"], nle, cond["z"], boot["next_action"])
        tB = critic._B(target["backward"], next_obs)
        tM = jnp.einsum("pbz,cz->pbc", tFs, tB)
        from impls.utils.fb_math import targets_uncertainty
        tmean, _ = targets_uncertainty(tM, P)
        target_M = jax.lax.stop_gradient(tmean)
        le = critic._le(full["left_encoder"], obs)
        Fs = critic._F(full["forward"], le, cond["z"], action)
        Bg = jax.lax.stop_gradient(critic._B(full["backward"], next_obs))
        M = jnp.einsum("pbz,cz->pbc", Fs, Bg)
        d = M - critic.discount * target_M
        off = 1.0 - jnp.eye(obs.shape[0])
        return 0.5 * jnp.sum((d * off) ** 2) / off.sum() - jnp.mean(jnp.diagonal(d, axis1=1, axis2=2)) * P

    g = jax.grad(fb_only)({"backward": params["backward"]})
    for leaf in jax.tree_util.tree_leaves(g["backward"]):
        np.testing.assert_array_equal(np.asarray(leaf), np.zeros_like(leaf))


def test_predictor_has_no_target_network():
    """The reference polyak-updates forward / backward / left_encoder only."""
    critic, params, target = RLDPCritic.build(*_ex(), _cfg())
    assert set(critic.GROUPS) == {"forward", "backward", "left_encoder", "predictor"}
    assert set(critic.TARGETS) == {"forward", "backward", "left_encoder"}
    assert set(target) == set(critic.TARGETS)


def test_predictor_rides_the_backward_learning_rate():
    """RLDPAgent.setup_training appends _predictor to the BACKWARD optimizer."""
    assert _LR_KEY["predictor"] == "lr_b"


# ==========================================================================
# 3. registry / config
# ==========================================================================
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
    """End-to-end on the composed config: make_dataset(horizon=h) must produce exactly
    the h-1 columns rldp.py::_sp_pairs asserts on."""
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


# ==========================================================================
# 4. the seam: builds and runs a real update with each actor family
# ==========================================================================
@pytest.mark.parametrize("actor", ACTOR_FAMILIES)
def test_builds_and_updates_with_each_actor_family(actor):
    cfg = load_config("rldp", actor, "cube_single")
    cfg.update(_cfg())            # small nets so this runs on CPU in seconds
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
    """The predictor appears in no other term, so zeroing the window must still move it
    (one-step pair survives) while the FB residual alone must leave it untouched."""
    cfg = _cfg()
    critic, params, target = RLDPCritic.build(*_ex(), cfg)
    batch = _batch()
    g = jax.grad(lambda p: critic._sp_loss({**params, **p}, target, batch)[0])(
        {"predictor": params["predictor"]})
    assert any(float(jnp.sum(jnp.abs(l))) > 0 for l in jax.tree_util.tree_leaves(g["predictor"]))


def test_stage_loss_is_the_sum_of_its_reported_parts():
    """fb_loss is reported as the TOTAL (matching impls/critics/fb.py's convention); pin
    that it really is offdiag + diag + sp + ortho_coef * ortho, so a term cannot be
    dropped from the objective while still being logged."""
    cfg = _cfg()
    critic, params, target = RLDPCritic.build(*_ex(), cfg)
    batch, boot = _batch(), _boot()
    cond = {"z": jnp.asarray(np.random.default_rng(3).standard_normal((B, Z)))}
    loss, info = critic.stage_loss("fb", params, target, batch, cond, boot)
    recon = (info["fb_offdiag"] + info["fb_diag"] + info["sp_loss"]
             + cfg["ortho_coef"] * info["ortho_loss"])
    np.testing.assert_allclose(float(loss), float(recon), atol=1e-10)
    np.testing.assert_allclose(float(info["fb_loss"]), float(loss), atol=0)


# ==========================================================================
# 5. the detach, tested THROUGH stage_loss  (found by mutation, audit 2026-08-23)
#
# `test_the_successor_residual_does_not_train_the_backward_map` above asserts the right
# property about the wrong function: it RE-IMPLEMENTS the FB residual in its own body,
# detach included, and never calls `critic.stage_loss`. Removing the detach from
# impls/critics/rldp.py therefore leaves it green (verified by mutation). This drives the
# real loss instead.
# ==========================================================================
def test_stage_loss_itself_sends_no_backward_gradient_through_the_fb_residual():
    """With ortho off, `stage_loss = fb + sp`. The reference detaches B(goal) inside the
    residual, so ALL of the backward map's gradient must come from the self-predictive
    term -- i.e. grad_B(stage_loss) == grad_B(_sp_loss), exactly. Both sides here are
    source functions; nothing about the residual is restated in this file."""
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

    # vacuity guards: the residual is live (it trains the forward map), and the backward
    # map is not simply gradient-free (the SP term does reach it).
    g_fwd = jax.grad(full)({"forward": params["forward"]})["forward"]
    assert any(float(np.abs(np.asarray(l)).sum()) > 0
               for l in jax.tree_util.tree_leaves(g_fwd)), "the FB residual is dead"
    assert any(float(np.abs(np.asarray(l)).sum()) > 0
               for l in jax.tree_util.tree_leaves(g_full)), \
        "the backward map gets no gradient at all -- the comparison above is vacuous"


# ── official-parity knobs: the default-off gate ───────────────────────────────
# docs/specs/2026-08-28-exorl-official-parity-validation.md item A house rule: "every knob
# default-off; parent td_jepa arm must remain bit-identical". These pin that literally.

_OFFICIAL_KNOBS = dict(
    rldp_phase="fb", rldp_pred_norm=True, rldp_actor_pessimism=0.5,
    rldp_no_left_encoder=True, rldp_sp_official_scale=True, rldp_ortho_inert=True,
)


def test_knob_defaults_are_all_off():
    """A config naming NO knob builds the td_jepa critic: joint phase, no predictor
    normalization, no actor pessimism, learned left encoder, td_jepa SP scale, paper
    ortho. If a default flips, every previously-recorded rldp cell silently changes
    meaning -- so the defaults are pinned by value, not by behavior."""
    critic, _, _ = RLDPCritic.build(*_ex(), _cfg())
    assert critic.phase == "joint"
    assert critic.pred_norm is False
    assert critic.actor_pessimism == 0.0
    assert critic.no_left_encoder is False
    assert critic.sp_official_scale is False
    assert critic.ortho_inert is False
    # ... and the phase-dependent plumbing is the shipped plumbing.
    assert critic.stage_groups("fb") == critic.GROUPS
    assert critic.FROZEN_GROUPS == ()
    assert critic.TRAIN_ACTOR is True
    assert set(critic.TARGETS) == {"forward", "left_encoder", "backward"}
    assert critic.TARGETS["backward"] == _cfg()["b_target_tau"]


def test_stage_loss_is_bit_identical_with_knobs_defaulted():
    """The parent arm's LOSS, not just its flags: building with the knob keys explicitly
    present-and-off must give the same number, to the last bit, as building without them.
    Catches a knob that reads its config key but changes the expression anyway."""
    ex = _ex()
    off = dict(rldp_phase="joint", rldp_pred_norm=False, rldp_actor_pessimism=0.0,
               rldp_no_left_encoder=False, rldp_sp_official_scale=False,
               rldp_ortho_inert=False)
    c0, p0, t0 = RLDPCritic.build(*ex, _cfg())
    c1, p1, t1 = RLDPCritic.build(*ex, _cfg(**off))
    batch, boot = _batch(), _boot()
    cond = {"z": jnp.asarray(np.random.default_rng(3).standard_normal((B, Z)))}
    l0, i0 = c0.stage_loss("fb", p0, t0, batch, cond, boot)
    l1, i1 = c1.stage_loss("fb", p1, t1, batch, cond, boot)
    assert np.asarray(l0).dtype == np.asarray(l1).dtype
    np.testing.assert_array_equal(np.asarray(l0), np.asarray(l1))
    for k in ("fb_loss", "fb_offdiag", "fb_diag", "sp_loss", "ortho_loss"):
        np.testing.assert_array_equal(np.asarray(i0[k]), np.asarray(i1[k]),
                                      err_msg=f"metric {k} moved with knobs defaulted off")


@pytest.mark.parametrize("knob,value", sorted(_OFFICIAL_KNOBS.items()))
def test_each_official_knob_actually_changes_something(knob, value):
    """The other half of the gate: a knob that silently does nothing would pass the
    default-off test and then quietly fail to replicate. Each one must move either the
    loss or the plumbing."""
    ex = _ex()
    c0, p0, t0 = RLDPCritic.build(*ex, _cfg())
    c1, p1, t1 = RLDPCritic.build(*ex, _cfg(**{knob: value}))
    batch, boot = _batch(), _boot()
    cond = {"z": jnp.asarray(np.random.default_rng(3).standard_normal((B, Z)))}
    l0, _ = c0.stage_loss("fb", p0, t0, batch, cond, boot)
    l1, _ = c1.stage_loss("fb", p1, t1, batch, cond, boot)
    plumbing_moved = (c0.stage_groups("fb") != c1.stage_groups("fb")
                      or c0.TARGETS != c1.TARGETS
                      or c0.TRAIN_ACTOR != c1.TRAIN_ACTOR
                      or type(c0.left_encoder) is not type(c1.left_encoder))
    loss_moved = not np.array_equal(np.asarray(l0), np.asarray(l1))
    if knob == "rldp_actor_pessimism":
        # q_fn, not stage_loss: pessimism only enters through the actor's critic call.
        obs = batch["observations"]
        q0, _ = c0.q_fn(p0, obs, cond, batch["actions"])
        q1, _ = c1.q_fn(p1, obs, cond, batch["actions"])
        assert not np.array_equal(np.asarray(q0), np.asarray(q1))
        return
    assert loss_moved or plumbing_moved, f"{knob}={value} changed nothing"


def test_two_phase_freezes_the_right_groups():
    """The structural claim of official RLDP: phase 1 trains B and the predictor with no
    FB loss and no actor; phase 2 trains F with B FROZEN. Freezing is by group EXCLUSION
    (no optimizer step at all), not a zeroed gradient -- Adam moves params on a zero
    gradient via momentum, so a masked-gradient implementation would not hold B fixed."""
    ex = _ex()
    mr, _, _ = RLDPCritic.build(*ex, _cfg(rldp_phase="mr"))
    fb, _, _ = RLDPCritic.build(*ex, _cfg(rldp_phase="fb"))

    assert set(mr.stage_groups("fb")) == {"backward", "predictor"}
    assert mr.TRAIN_ACTOR is False
    # every target still EXISTS in phase 1 (the SP loss reads target["backward"]) but
    # none of them polyak -- phase 1 refreshes B's target by hard copy instead.
    assert set(mr.TARGETS) == {"forward", "left_encoder", "backward"}
    assert set(mr.TARGETS.values()) == {0.0}

    assert set(fb.stage_groups("fb")) == {"forward", "left_encoder"}
    assert fb.TRAIN_ACTOR is True
    assert fb.TARGETS["backward"] == 0.0          # B frozen => its target must not drift
    assert fb.TARGETS["forward"] == _cfg()["f_target_tau"]


def test_phase1_loss_has_no_fb_term_and_phase2_has_no_sp_term():
    """The losses themselves, not just the group lists."""
    ex = _ex()
    batch, boot = _batch(), _boot()
    cond = {"z": jnp.asarray(np.random.default_rng(3).standard_normal((B, Z)))}
    mr, pm, tm = RLDPCritic.build(*ex, _cfg(rldp_phase="mr"))
    _, i_mr = mr.stage_loss("fb", pm, tm, batch, cond, boot)
    assert float(i_mr["fb_loss"]) == 0.0 and float(i_mr["fb_diag"]) == 0.0
    assert float(i_mr["sp_loss"]) != 0.0

    fb, pf, tf = RLDPCritic.build(*ex, _cfg(rldp_phase="fb"))
    _, i_fb = fb.stage_loss("fb", pf, tf, batch, cond, boot)
    assert float(i_fb["sp_loss"]) == 0.0 and float(i_fb["ortho_loss"]) == 0.0
    assert float(i_fb["fb_diag"]) != 0.0
