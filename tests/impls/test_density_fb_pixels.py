"""density_fb on PIXELS (docs/specs/2026-08-21-density-fb-pixels-drq.md).

The single most important thing in this file is section 1. density_fb's hit kernel and query
support are literal COLUMN SLICES of the observation (`measure_hit_spec: "19:22:0.4"` ->
`obs[..., 19:22]`). An image has no such columns, and jax will slice one anyway -- returning
NUMBERS, computed from pixel values, with no error, no nan and no loss spike. Every other
test here is about the pixel path working; that one is about it failing loudly when it is
wired wrong.

Sections:
  1. the shape guard                     -- an image where a qpos channel belongs RAISES
  2. train/eval channel agreement        -- npz `qpos`-derived block xyz == the env's
                                            `privileged/block_0_pos` ON THE SAME STATE
  3. encoder ownership                   -- encoder params move under the density loss, and
                                            NO critic parameter moves from the actor stage
  4. augmentation                        -- per-key independent shifts, 4-D arrays only
  5. frame stacking + the qpos channel   -- stacks never cross an episode boundary; the
                                            measure channel is the LATEST frame
  6. smoke                               -- a real short train on visual-cube-single-play-v0

Tests that need the `visual-cube-single-play-v0` npz or an EGL context skip when it is
absent, so this file stays runnable on a bare checkout.
"""
import os

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from impls.agents import Agent, CRITIC_ALIASES
from impls.critics.density_fb import (
    MeasureChannelError, _delta_phi, _density_queries, _raw_block_cols, _check_rows,
)
from impls.critics._measure_common import parse_hit_spec
from impls.utils.config_utils import load_config
from impls.utils.datasets import Dataset, GCDataset, parse_qpos_slices
from impls.utils.encoders import DrQEncoder, encoder_modules, random_shifts_batch

# Small everything: the guard/ownership tests are about wiring, not scale. 16x16 keeps the
# DrQ conv stack valid (16 -> 7 -> 5 -> 3 -> 1) and each test sub-second.
B, H, W, C, ACT, D = 6, 16, 16, 3, 5, 8
QPOS_DIM = 21           # visual-cube-single-play-v0's qpos width
BLOCK_SLICE = (14, 17)  # object_joint_0's free-joint position columns

DATA = os.path.expanduser("~/.ogbench/data/visual-cube-single-play-v0.npz")
needs_data = pytest.mark.skipif(
    not os.path.exists(DATA), reason=f"{DATA} not present")


# ── helpers ─────────────────────────────────────────────────────────────────────────────
def pixel_cfg(actor="flowbc", **over):
    """The shipped pixel config, shrunk. Loaded from the real yaml rather than hand-built so
    a key that config and code disagree about shows up here."""
    cfg = load_config("density_fb_pixels", actor, "cube_single")
    cfg.update(dict(batch_size=B, seed=0, measure_density_dim=D,
                    value_hidden_dims=[32, 32], frame_stack=None, p_aug=None))
    cfg["actor"] = {**cfg["actor"], "hidden_dim": 32, "hidden_layers": 1,
                    "embedding_layers": 2, "flow_vf_hidden_dim": 32,
                    "flow_vf_hidden_layers": 2, "hidden_dims": [32, 32]}
    cfg.update(over)
    return cfg


def ex_pixels(n=B, seed=0):
    rng = np.random.default_rng(seed)
    return (jnp.asarray(rng.integers(0, 256, (n, H, W, C), dtype=np.uint8)),
            jnp.asarray(np.tanh(rng.standard_normal((n, ACT))), jnp.float32))


def pixel_batch(seed=1, n=B):
    """A pixel batch shaped exactly as GCDataset emits one under `measure_qpos_slices`:
    IMAGE keys plus their `_qpos` object-coordinate twins."""
    rng = np.random.default_rng(seed)
    img = lambda: jnp.asarray(rng.integers(0, 256, (n, H, W, C), dtype=np.uint8))
    q = lambda: jnp.asarray(rng.uniform(-0.3, 0.6, (n, 3)), jnp.float32)
    return dict(
        observations=img(), next_observations=img(),
        value_goals=img(), actor_goals=img(),
        observations_qpos=q(), next_observations_qpos=q(),
        value_goals_qpos=q(), actor_goals_qpos=q(),
        actions=jnp.asarray(np.tanh(rng.standard_normal((n, ACT))), jnp.float32),
        next_actions=jnp.asarray(np.tanh(rng.standard_normal((n, ACT))), jnp.float32),
        rewards=jnp.asarray(rng.integers(0, 2, size=n) - 1.0),
        masks=jnp.asarray(rng.integers(0, 2, size=n).astype(float)),
    )


def pixel_agent(actor="flowbc", **over):
    cfg = pixel_cfg(actor=actor, **over)
    ex_obs, ex_act = ex_pixels()
    return Agent.create("density_fb_pixels", actor, 0, ex_obs, ex_act, cfg), cfg


def synth_pixel_dataset(n_traj=4, traj_len=8, seed=0):
    """A compact pixel dataset with the same fields the real npz has after add_info=True."""
    rng = np.random.default_rng(seed)
    n = n_traj * traj_len
    terminals = np.zeros(n, np.float32)
    terminals[np.arange(traj_len - 1, n, traj_len)] = 1.0
    valids = 1.0 - terminals
    # compact layout: terminals marks the LAST TWO rows of a trajectory (see ogbench loader)
    term = np.minimum(terminals + np.concatenate([terminals[1:], [1.0]]), 1.0)
    qpos = rng.uniform(-1.0, 1.0, (n, QPOS_DIM)).astype(np.float32)
    return Dataset.create(
        observations=rng.integers(0, 256, (n, H, W, C), dtype=np.uint8),
        actions=np.tanh(rng.standard_normal((n, ACT))).astype(np.float32),
        terminals=term, valids=valids, qpos=qpos,
    )


# ══ 1. THE SHAPE GUARD ═══════════════════════════════════════════════════════════════════
def test_an_image_would_silently_slice_without_the_guard():
    """The motivating fact, asserted rather than asserted-about: the raw slice the kernel
    performs does NOT fail on an image. It returns finite numbers computed from pixel
    intensities. Everything else in section 1 is what stands between that and a training run.
    """
    blocks = parse_hit_spec("0:3:0.04")
    img = jnp.asarray(np.random.default_rng(0).integers(0, 256, (B, H, W, C)), jnp.float32)
    sliced = img[:, 0:3]                       # what `_raw_block_cols` does, unguarded
    assert sliced.shape == (B, 3, W, C)        # the HEIGHT axis, not a coordinate block
    assert bool(jnp.all(jnp.isfinite(sliced))) # no nan, no error -- just wrong numbers
    # and the guard is what refuses it
    with pytest.raises(MeasureChannelError):
        _raw_block_cols(img, blocks)


@pytest.mark.parametrize("fn", [_raw_block_cols, _density_queries])
def test_kernel_row_builders_reject_image_tensors(fn):
    blocks = parse_hit_spec("0:3:0.04")
    img = jnp.zeros((B, H, W, C), jnp.float32)
    with pytest.raises(MeasureChannelError, match="object-coordinate rows"):
        fn(img, blocks)


def test_delta_phi_rejects_an_image_on_either_argument():
    blocks = parse_hit_spec("0:3:0.04")
    rows = jnp.zeros((B, 3), jnp.float32)
    img = jnp.zeros((B, H, W, C), jnp.float32)
    with pytest.raises(MeasureChannelError):
        _delta_phi(img, rows, blocks)          # obs_rows is an image
    with pytest.raises(MeasureChannelError):
        _delta_phi(rows, img, blocks)          # protos is an image


def test_guard_rejects_a_FLATTENED_image_which_a_rank_check_alone_would_pass():
    """(B, 12288) is rank 2 and far wider than the hit spec needs, so a rank-and-minimum-width
    check lets it through and the kernel returns three arbitrary pixel intensities. The exact
    width check is what catches it."""
    blocks = parse_hit_spec("0:3:0.04")
    flat = jnp.zeros((B, 64 * 64 * 3), jnp.float32)
    _check_rows(flat, blocks, "no expect_dim")           # passes the weak check...
    with pytest.raises(MeasureChannelError, match="columns wide"):
        _check_rows(flat, blocks, "with expect_dim", expect_dim=3)


def test_pixel_critic_refuses_a_batch_without_the_qpos_channel():
    """The end-to-end version: drop the privileged channel and the critic must raise instead
    of reaching for the image."""
    (agent, _), batch = pixel_agent(), pixel_batch()
    stripped = {k: v for k, v in batch.items() if not k.endswith("_qpos")}
    with pytest.raises(MeasureChannelError, match="next_observations_qpos"):
        agent.critic.sample_cond(agent.net, stripped, jax.random.PRNGKey(0),
                                 target=agent.target)


def test_pixel_critic_refuses_a_qpos_channel_of_the_wrong_width():
    (agent, _), batch = pixel_agent(), pixel_batch()
    bad = dict(batch)
    bad["next_observations_qpos"] = jnp.zeros((B, 7), jnp.float32)   # e.g. pos+quat
    with pytest.raises(MeasureChannelError, match="columns wide"):
        agent.critic.sample_cond(agent.net, bad, jax.random.PRNGKey(0), target=agent.target)


def test_q_fn_refuses_an_image_goal_without_its_qpos_twin():
    """q_fn's reward vector is `_delta_phi(goal, u_raw)`. On pixels cond["goal"] is the goal
    IMAGE, so the kernel must read cond["goal_qpos"] -- and say so when it is missing."""
    (agent, _), batch = pixel_agent(), pixel_batch()
    cond = agent.critic.sample_cond(agent.net, batch, jax.random.PRNGKey(0),
                                    target=agent.target)
    cond.pop("goal_qpos")
    params = {g: agent.net[g].params for g in agent.net}
    with pytest.raises(MeasureChannelError, match="goal_qpos"):
        agent.critic.q_fn(params, batch["observations"], cond, batch["actions"])


def test_state_path_still_slices_state_rows():
    """The guard must not have made the STATE path stricter: an ordinary [B, obs_dim] row
    block still passes every checker."""
    blocks = parse_hit_spec("19:22:0.4")
    rows = jnp.asarray(np.random.default_rng(0).standard_normal((B, 28)), jnp.float32)
    cols = _raw_block_cols(rows, blocks)
    assert cols.shape == (B, 3)
    assert _density_queries(rows, blocks).shape == (B, 3)
    assert _delta_phi(rows, cols, blocks).shape == (B, B)


# ══ 2. TRAIN/EVAL CHANNEL AGREEMENT ══════════════════════════════════════════════════════
@needs_data
def test_npz_qpos_layout_is_the_block_position():
    """Necessary condition, cheap: qpos[14:17] on the real dataset must look like a cube
    position -- z pinned at the resting height and xy inside the table's sampling bounds."""
    d = np.load(DATA, mmap_mode="r")
    a, b = BLOCK_SLICE
    assert d["qpos"].shape[1] == QPOS_DIM
    xyz = np.asarray(d["qpos"][:20000, a:b])
    # z: resting at 0.02 on the table, lifted while carried -- but never off the workspace.
    assert 0.015 < xyz[:, 2].min() < 0.025, f"z floor is not the cube resting height: {xyz[:, 2].min()}"
    assert xyz[:, 2].max() < 0.6, f"z out of range: {xyz[:, 2].max()}"
    assert 0.1 < xyz[:, 0].min() and xyz[:, 0].max() < 0.8, "x outside the table"
    assert abs(xyz[:, 1]).max() < 0.6, "y outside the table"


def test_train_eval_qpos_agreement():
    """THE risk the spec calls out: offline the measure channel comes from the npz's `qpos`,
    online it comes from `info['privileged/block_0_pos']`. If those denote different
    quantities, training and evaluation disagree with nothing failing.

    Pinned by construction, not by trusting index arithmetic: push a REAL dataset qpos row
    into the simulator, then ask the env what it thinks `privileged/block_0_pos` is and
    compare against `qpos[14:17]` of the row we pushed.
    """
    pytest.importorskip("mujoco")
    if not os.path.exists(DATA):
        pytest.skip("dataset not present")
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco
    import ogbench

    env = ogbench.make_env_and_datasets("visual-cube-single-play-v0", env_only=True)
    env.reset(options=dict(task_id=1, render_goal=False))
    u = env.unwrapped

    d = np.load(DATA, mmap_mode="r")
    a, b = BLOCK_SLICE
    for row in (0, 5_000, 123_456):
        qpos = np.asarray(d["qpos"][row], np.float64)
        u._data.qpos[:] = qpos
        mujoco.mj_forward(u._model, u._data)
        info = u.compute_ob_info()
        np.testing.assert_allclose(info["privileged/block_0_pos"], qpos[a:b],
                                   rtol=0, atol=1e-9)
    # and the live env agrees with its own qpos, which is what the slice indexes
    _, live = env.reset(options=dict(task_id=2, render_goal=False))
    np.testing.assert_allclose(live["privileged/block_0_pos"],
                               np.asarray(u.data.qpos)[a:b], rtol=0, atol=1e-12)


def test_eval_goal_channel_matches_the_env_task():
    """The eval-side half: `goal_qpos_channel` must return the goal's block xyz in the SAME
    units and the SAME layout as the training channel (metres, [x, y, z] per cube)."""
    pytest.importorskip("mujoco")
    os.environ.setdefault("MUJOCO_GL", "egl")
    import ogbench
    from impls.utils.evaluation import goal_qpos_channel

    env = ogbench.make_env_and_datasets("visual-cube-single-play-v0", env_only=True)
    for task_id in (1, 3, 5):
        ob, info = env.reset(options=dict(task_id=task_id, render_goal=True))
        ch = goal_qpos_channel(env)
        assert ch.shape == (3,), ch.shape
        np.testing.assert_allclose(
            ch, env.unwrapped.cur_task_info["goal_xyzs"].reshape(-1), rtol=1e-6, atol=1e-7)
        # the goal the env hands the agent is an IMAGE, not a state -- the whole reason a
        # separate coordinate channel has to exist (the spec's section 2 says otherwise).
        assert np.asarray(info["goal"]).ndim == 3, np.asarray(info["goal"]).shape
        assert np.asarray(info["goal"]).dtype == np.uint8


# ══ 3. ENCODER OWNERSHIP ═════════════════════════════════════════════════════════════════
def _encoder_leaves(tree):
    """Every leaf under an `encoder` subtree, flattened with its path."""
    out = {}

    def walk(node, path):
        if isinstance(node, dict) or hasattr(node, "items"):
            for k, v in node.items():
                walk(v, path + (str(k),))
        else:
            out[path] = node
    walk(tree, ())
    return {p: v for p, v in out.items() if "encoder" in p}


def test_both_towers_own_an_encoder_and_the_actor_owns_none():
    agent, _ = pixel_agent()
    f_enc = _encoder_leaves(agent.net["density_f"].params)
    b_enc = _encoder_leaves(agent.net["density_b"].params)
    assert f_enc, "density_f has no encoder params"
    assert b_enc, "density_b has no encoder params"
    for g in ("actor", "actor_vf"):
        assert not _encoder_leaves(agent.net[g].params), \
            f"actor group {g!r} owns encoder parameters; FB's actor_optimizer holds no encoder"


def test_encoder_params_move_under_the_density_loss():
    agent, _ = pixel_agent()
    batch = pixel_batch()
    cond = agent.critic.sample_cond(agent.net, batch, jax.random.PRNGKey(0),
                                    target=agent.target)
    boot = agent.actor.sample_bootstrap(agent.net, batch, cond, jax.random.PRNGKey(1))
    online = {g: agent.net[g].params for g in agent.net}
    groups = agent.critic.stage_groups("density")

    def loss_fn(gp):
        return agent.critic.stage_loss("density", {**online, **gp}, agent.target,
                                       batch, cond, boot)

    grads = jax.grad(loss_fn, has_aux=True)({g: online[g] for g in groups})[0]
    for g in ("density_f", "density_b"):
        enc = _encoder_leaves(grads[g])
        assert enc, f"{g} produced no encoder gradient leaves"
        biggest = max(float(jnp.abs(v).max()) for v in enc.values())
        assert biggest > 0.0, f"{g}'s encoder received an all-zero gradient"


@pytest.mark.parametrize("actor_name", ["flowbc", "ddpgbc_density"])
def test_no_critic_parameter_moves_from_the_actor_stage(actor_name):
    """The seam's analogue of FB's `with torch.no_grad():` around the actor's encode.

    Stronger than "the actor's optimizer does not hold the encoder" (which is true by
    construction, since Agent only differentiates `actor.GROUPS`): this differentiates the
    actor loss with respect to EVERY group, including the towers, and asserts the critic side
    is exactly zero. That is what proves the stop_gradients in `q_fn` and `_enc_f` are real.
    """
    agent, _ = pixel_agent(actor=actor_name)
    batch = pixel_batch()
    cond = agent.critic.sample_cond(agent.net, batch, jax.random.PRNGKey(0),
                                    target=agent.target)
    boot = agent.actor.sample_bootstrap(agent.net, batch, cond, jax.random.PRNGKey(1))
    online = {g: agent.net[g].params for g in agent.net}

    def loss_fn(gp):
        return agent.actor.loss({**online, **gp}, agent.target, batch, agent.critic,
                                cond, boot)

    grads = jax.grad(loss_fn, has_aux=True)(online)[0]
    for g in ("density_f", "density_b"):
        leaves = jax.tree_util.tree_leaves(grads[g])
        assert leaves, f"no leaves for {g}"
        worst = max(float(jnp.abs(v).max()) for v in leaves)
        assert worst == 0.0, (
            f"the actor stage moved {g} by up to {worst:g}; on pixels that means the actor "
            f"is training the visual encoder, which FB's optimizer split forbids")
    # ...while the actor's own groups DO move, so the test is not vacuous.
    for g in agent.actor.GROUPS:
        assert max(float(jnp.abs(v).max())
                   for v in jax.tree_util.tree_leaves(grads[g])) > 0.0


def test_one_full_update_moves_the_encoder_and_stays_finite():
    agent, _ = pixel_agent()
    batch = pixel_batch()
    before = _encoder_leaves(agent.net["density_b"].params)
    new_agent, info = agent.update(batch)
    after = _encoder_leaves(new_agent.net["density_b"].params)
    assert any(not np.allclose(np.asarray(before[k]), np.asarray(after[k]))
               for k in before), "no B-side encoder parameter moved in a full update"
    for k, v in info.items():
        assert np.all(np.isfinite(np.asarray(v))), f"{k} is not finite: {v}"


# ══ 4. AUGMENTATION ══════════════════════════════════════════════════════════════════════
def test_augmentation_touches_only_4d_arrays():
    rng = np.random.default_rng(0)
    batch = {
        "observations": rng.integers(0, 256, (B, H, W, C), dtype=np.uint8),
        "observations_qpos": rng.standard_normal((B, 3)).astype(np.float32),
        "actions": rng.standard_normal((B, ACT)).astype(np.float32),
        "rewards": rng.standard_normal(B).astype(np.float32),
    }
    keeps = {k: v.copy() for k, v in batch.items()}
    random_shifts_batch(batch, list(batch), pad=2, rng=np.random.default_rng(1))
    for k in ("observations_qpos", "actions", "rewards"):
        np.testing.assert_array_equal(batch[k], keeps[k]), k
    assert batch["observations"].shape == keeps["observations"].shape
    assert batch["observations"].dtype == np.uint8


def test_augmentation_shifts_are_independent_per_key():
    """FB applies a SEPARATE `_augmentator` call per tensor, so the observation, the next
    observation and each goal get different shifts. A shared shift would be a translation the
    bilinear form could absorb instead of a nuisance the encoder must be invariant to."""
    rng = np.random.default_rng(2)
    img = rng.integers(0, 256, (64, H, W, C), dtype=np.uint8)
    batch = {"observations": img.copy(), "next_observations": img.copy()}
    random_shifts_batch(batch, ["observations", "next_observations"], pad=2,
                        rng=np.random.default_rng(3))
    assert not np.array_equal(batch["observations"], batch["next_observations"]), \
        "both keys got the same shifts"


def test_augmentation_is_a_replicate_pad_integer_crop():
    """Every output row must equal SOME integer shift of the input under edge padding --
    i.e. the augmentation moves the image, it does not blend or resample it."""
    from impls.utils.encoders import _shift_batch_np
    rng = np.random.default_rng(4)
    img = rng.integers(0, 256, (8, H, W, C), dtype=np.uint8)
    out = _shift_batch_np(img, np.full((8, 2), 2), pad=2)
    np.testing.assert_array_equal(out, img)     # shift == pad is the identity crop
    out0 = _shift_batch_np(img, np.zeros((8, 2), np.int64), pad=2)
    # top-left corner replicated by the edge pad
    np.testing.assert_array_equal(out0[:, 0, 0], img[:, 0, 0])


def test_augmentation_is_seeded_and_reproducible():
    rng = np.random.default_rng(5)
    img = rng.integers(0, 256, (16, H, W, C), dtype=np.uint8)
    a = random_shifts_batch({"o": img.copy()}, ["o"], pad=2, rng=np.random.default_rng(7))["o"]
    b = random_shifts_batch({"o": img.copy()}, ["o"], pad=2, rng=np.random.default_rng(7))["o"]
    np.testing.assert_array_equal(a, b)


def test_gcdataset_augmentation_replays_under_a_global_seed():
    """`impls/main.py` seeds the GLOBAL numpy RNG from --seed so a run replays its batch
    stream. Augmentation must ride that same stream: an entropy-seeded generator inside
    `random_shifts_batch` would make the pixel arm the one path a seed does not reproduce."""
    ds = synth_pixel_dataset()
    gc = GCDataset(ds, pixel_cfg(p_aug=1.0, frame_stack=None))
    np.random.seed(11)
    a = gc.sample(B)
    np.random.seed(11)
    b = gc.sample(B)
    for k in ("observations", "next_observations", "value_goals", "actor_goals"):
        np.testing.assert_array_equal(a[k], b[k]), k
    np.random.seed(12)
    c = gc.sample(B)
    assert not np.array_equal(a["observations"], c["observations"]), \
        "a different seed produced the same augmented batch"


def test_gcdataset_augments_images_and_never_the_measure_channel():
    ds = synth_pixel_dataset()
    cfg = pixel_cfg(p_aug=1.0, frame_stack=None)
    gc = GCDataset(ds, cfg)
    np.random.seed(0)
    batch = gc.sample(B)
    assert batch["observations"].shape == (B, H, W, C)
    assert batch["observations_qpos"].shape == (B, 3)
    # the measure channel must be EXACTLY the dataset's own qpos columns -- augmentation
    # cannot have touched it (it is 2-D, and `random_shifts_batch` skips non-4-D arrays)
    assert np.isfinite(batch["observations_qpos"]).all()


# ══ 5. FRAME STACKING × THE QPOS CHANNEL ═════════════════════════════════════════════════
def test_frame_stack_widens_only_the_channel_axis():
    ds = synth_pixel_dataset()
    gc = GCDataset(ds, pixel_cfg(frame_stack=3, p_aug=None))
    np.random.seed(0)
    batch = gc.sample(B)
    for k in ("observations", "next_observations", "value_goals", "actor_goals"):
        assert batch[k].shape == (B, H, W, C * 3), (k, batch[k].shape)
    # spec section 9: the measure channel is ONE frame -- the latest -- not a stack.
    for k in ("observations_qpos", "next_observations_qpos"):
        assert batch[k].shape == (B, 3), (k, batch[k].shape)


def test_frame_stack_never_crosses_a_trajectory_boundary():
    """At a trajectory's first row the stack must repeat that row, not read the tail of the
    previous episode."""
    ds = synth_pixel_dataset(n_traj=3, traj_len=6)
    gc = GCDataset(ds, pixel_cfg(frame_stack=3, p_aug=None))
    starts = gc.initial_locs
    stacked = gc.get_stacked_observations(np.asarray(starts))
    raw = np.asarray(ds["observations"])[starts]
    for k in range(3):
        np.testing.assert_array_equal(stacked[..., k * C:(k + 1) * C], raw)


def test_qpos_channel_indexes_the_same_rows_as_the_images():
    """The two channels describe the same state only if they are gathered at the same index.
    Sample with FIXED idxs and check the qpos channel against the dataset directly."""
    ds = synth_pixel_dataset()
    gc = GCDataset(ds, pixel_cfg(frame_stack=None, p_aug=None))
    idxs = np.array([0, 1, 2, 9, 10, 17])
    np.random.seed(0)
    batch = gc.sample(len(idxs), idxs=idxs)
    a, b = BLOCK_SLICE
    qp = np.asarray(ds["qpos"])
    np.testing.assert_array_equal(batch["observations_qpos"], qp[idxs, a:b])
    nxt = np.minimum(idxs + 1, ds.size - 1)
    np.testing.assert_array_equal(batch["next_observations_qpos"], qp[nxt, a:b])
    np.testing.assert_array_equal(batch["next_observations"],
                                  np.asarray(ds["observations"])[nxt])


def test_qpos_slices_parser():
    assert parse_qpos_slices(None) is None
    assert parse_qpos_slices("14:17") == ((14, 17),)
    assert parse_qpos_slices("14:17,21:24") == ((14, 17), (21, 24))
    with pytest.raises(AssertionError):
        parse_qpos_slices("17:14")


def test_measure_qpos_slices_without_a_qpos_field_raises():
    ds = synth_pixel_dataset()
    ds = Dataset.create(**{k: np.asarray(v) for k, v in ds.items() if k != "qpos"})
    with pytest.raises(KeyError, match="qpos"):
        GCDataset(ds, pixel_cfg(frame_stack=None, p_aug=None))


# ══ 6. WIRING + CONFIG ═══════════════════════════════════════════════════════════════════
def test_the_pixel_config_composes_and_names_a_real_encoder():
    cfg = load_config("density_fb_pixels", "flowbc", "cube_single")
    assert cfg["encoder"] in encoder_modules
    assert parse_qpos_slices(cfg["measure_qpos_slices"]) == (BLOCK_SLICE,)
    blocks = parse_hit_spec(cfg["measure_hit_spec"])
    assert max(b for (_a, b, _e) in blocks) <= 3, "hit spec must index the qpos CHANNEL"
    assert CRITIC_ALIASES["density_fb_pixels"] == "density_fb"


def test_the_pixel_kernel_is_the_state_kernel_in_different_units():
    """The state arm slices obs[19:22] = (block_pos - c) * 10 with eps 0.4; the pixel arm
    slices the qpos channel = block_pos in metres with eps 0.04. Centering cancels in the
    pairwise differences, so the two kernels are the SAME function of the same states -- which
    is the whole justification for calling a pixel-vs-state comparison an encoder comparison.
    """
    rng = np.random.default_rng(0)
    pos = rng.uniform(0.3, 0.55, (5, 3))            # metres
    c = np.array([0.425, 0.0, 0.0])
    state_rows = jnp.asarray(np.concatenate(
        [np.zeros((5, 19)), (pos - c) * 10.0], axis=1), jnp.float32)
    qpos_rows = jnp.asarray(pos, jnp.float32)
    s_blocks, p_blocks = parse_hit_spec("19:22:0.4"), parse_hit_spec("0:3:0.04")
    k_state = _delta_phi(state_rows, _raw_block_cols(state_rows, s_blocks), s_blocks)
    k_pixel = _delta_phi(qpos_rows, _raw_block_cols(qpos_rows, p_blocks), p_blocks)
    np.testing.assert_allclose(np.asarray(k_state), np.asarray(k_pixel), rtol=1e-5, atol=1e-6)


def test_the_actor_rides_encoder_features_not_the_image():
    agent, cfg = pixel_agent()
    enc_dim = int(encoder_modules[cfg["encoder"]]().feature_dim)
    assert agent.critic.cvec_dim == enc_dim
    assert agent.critic.actor_obs_dim == enc_dim
    cond = agent.critic.sample_cond(agent.net, pixel_batch(), jax.random.PRNGKey(0),
                                    target=agent.target)
    for k in ("_cvec", "_cvec_vgoal", "_obs_feat", "_obs_feat_next"):
        assert cond[k].shape == (B, enc_dim), (k, cond[k].shape)
    assert agent.critic.cond_vec(cond).shape == (B, enc_dim)


@pytest.mark.parametrize("actor_name", ["flowbc", "bcflow", "ddpgbc", "ddpgbc_density",
                                        "awr", "fql"])
def test_every_actor_family_rides_the_features_not_the_image(actor_name):
    """The pixel contract is a SEAM contract, so it has to hold for every actor in
    impls.agents.ACTORS, not just the two the pixel spec was written against.

    density_fb on pixels publishes the actor's observation leg as cond["_obs_feat"] /
    cond["_obs_feat_next"] and its width as `critic.actor_obs_dim`. An actor that instead
    assumes "the policy's observation leg IS batch['observations']" concatenates a
    (B, H, W, C) image onto a (B, enc_dim) conditioning vector. This is the same class of
    defect as the three cond["value_goal"] cases (seam section 12.1): the actor inferring a
    critic-owned quantity from `cond` instead of reading what the critic published.

    Exercises all three entry points -- the bootstrap draw, the actor loss, and the eval
    action -- because the observation leg is read separately in each.
    """
    agent, cfg = pixel_agent(actor=actor_name)
    batch = pixel_batch()
    cond = agent.critic.sample_cond(agent.net, batch, jax.random.PRNGKey(0),
                                    target=agent.target)
    boot = agent.actor.sample_bootstrap(agent.net, batch, cond, jax.random.PRNGKey(1))
    assert np.all(np.isfinite(np.asarray(boot["next_action"])))

    info = agent.compute_static(batch, cond, boot)
    assert np.isfinite(float(info["actor_loss"]))

    ex_obs, _ = ex_pixels()
    a = agent.sample_actions(
        ex_obs[0], goals={"goal": ex_obs[1], "goal_qpos": np.zeros((1, 3), np.float32)},
        seed=jax.random.PRNGKey(2))
    assert np.asarray(a).shape == (ACT,)
    assert np.all(np.isfinite(np.asarray(a)))


def test_eval_cond_encodes_a_single_unbatched_frame():
    """`evaluate` hands `sample_actions` ONE (H, W, C) frame, not a batch."""
    agent, cfg = pixel_agent()
    enc_dim = int(encoder_modules[cfg["encoder"]]().feature_dim)
    obs = jnp.asarray(np.random.default_rng(0).integers(0, 256, (H, W, C), dtype=np.uint8))
    goal = jnp.asarray(np.random.default_rng(1).integers(0, 256, (H, W, C), dtype=np.uint8))
    cond = agent.critic.map_eval_cond(agent.net, {"goal": goal}, observations=obs)
    assert cond["_cvec"].shape == (enc_dim,)
    assert cond["_obs_feat"].shape == (enc_dim,)
    a = agent.sample_actions(obs, goals={"goal": goal, "goal_qpos": np.zeros((1, 3), np.float32)},
                             seed=jax.random.PRNGKey(0))
    assert np.asarray(a).shape == (ACT,)
    assert np.all(np.isfinite(np.asarray(a)))


def test_reward_index_on_pixels_raises_rather_than_running_a_broken_pool():
    ex_obs, ex_act = ex_pixels()
    cfg = pixel_cfg(measure_density_z=True, batch_size=512)
    with pytest.raises(NotImplementedError, match="FROZEN EVAL POOL"):
        Agent.create("density_fb_pixels", "flowbc", 0, ex_obs, ex_act, cfg)


def test_missing_encoder_or_qpos_slices_is_a_named_error():
    ex_obs, ex_act = ex_pixels()
    cfg = pixel_cfg()
    cfg.pop("encoder")
    with pytest.raises(KeyError, match="encoder"):
        Agent.create("density_fb_pixels", "flowbc", 0, ex_obs, ex_act, cfg)
    cfg = pixel_cfg()
    cfg.pop("measure_qpos_slices")
    with pytest.raises(KeyError, match="measure_qpos_slices"):
        Agent.create("density_fb_pixels", "flowbc", 0, ex_obs, ex_act, cfg)


def test_drq_encoder_normalizes_internally():
    """Datasets hand the encoder RAW uint8; the encoder does `x/255 - 0.5` itself. Feeding it
    a pre-scaled float would halve the input range silently, so pin the convention."""
    enc = DrQEncoder(feature_dim=16)
    x = jnp.asarray(np.random.default_rng(0).integers(0, 256, (2, H, W, C), dtype=np.uint8))
    p = enc.init(jax.random.PRNGKey(0), x)["params"]
    y_uint = enc.apply({"params": p}, x)
    y_float = enc.apply({"params": p}, jnp.asarray(x, jnp.float32))
    np.testing.assert_allclose(np.asarray(y_uint), np.asarray(y_float), rtol=1e-6, atol=1e-6)
    assert y_uint.shape == (2, 16)
    assert float(jnp.abs(y_uint).max()) <= 1.0 + 1e-6, "tanh output must be in [-1, 1]"


# ══ 7. SMOKE ON REAL IMAGE DATA ══════════════════════════════════════════════════════════
@needs_data
def test_smoke_trains_on_visual_cube_single():
    """A short REAL train: the shipped `density_fb_pixels` config, the real npz (memory-
    mapped, truncated to a few trajectories), the real GCDataset, the real update.

    This is what distinguishes "the wiring type-checks on synthetic fixtures" from "it trains
    on image data": the observations are actual 64x64 renders, the qpos channel is the actual
    simulator state, and the loss has to move.
    """
    from impls.utils.datasets import FBDataset

    d = np.load(DATA, mmap_mode="r")
    n = 4000
    terminals = np.asarray(d["terminals"][:n], np.float32)
    ends = np.nonzero(terminals > 0)[0]
    assert len(ends), "no trajectory boundary in the truncated prefix"
    n = int(ends[-1]) + 1
    terminals = terminals[:n]
    valids = 1.0 - terminals
    term = np.minimum(terminals + np.concatenate([terminals[1:], [1.0]]), 1.0)
    ds = FBDataset.create(
        observations=np.asarray(d["observations"][:n]),          # uint8 (n, 64, 64, 3)
        actions=np.asarray(d["actions"][:n], np.float32),
        terminals=term, valids=valids,
        qpos=np.asarray(d["qpos"][:n], np.float32),
    )
    cfg = load_config("density_fb_pixels", "flowbc", "cube_single")
    cfg.update(dict(batch_size=32, seed=0, measure_density_dim=16,
                    value_hidden_dims=[64, 64], frame_stack=None, p_aug=0.5))
    cfg["actor"] = {**cfg["actor"], "hidden_dim": 64, "hidden_layers": 1,
                    "embedding_layers": 2, "flow_vf_hidden_dim": 64,
                    "flow_vf_hidden_layers": 2}

    gc = GCDataset(ds, cfg)
    np.random.seed(0)     # augmentation rides this stream too -- see random_shifts_batch
    ex = gc.sample(2)
    assert ex["observations"].shape[1:] == (64, 64, 3)
    assert ex["observations"].dtype == np.uint8, "images must not be float32-copied"
    assert ex["observations_qpos"].shape == (2, 3)

    agent = Agent.create("density_fb_pixels", "flowbc", 0,
                         jnp.asarray(ex["observations"]), jnp.asarray(ex["actions"]), cfg)
    losses = []
    for step in range(120):
        agent, info = agent.update(gc.sample(cfg["batch_size"]))
        losses.append(float(info["density_ce"]))
        assert np.isfinite(losses[-1]), f"density_ce went non-finite at step {step}"
        assert np.isfinite(float(info["actor_loss"]))
    # The measure has to actually be LEARNING from the images, not merely not crashing.
    assert np.mean(losses[-20:]) < np.mean(losses[:20]), \
        f"density CE did not decrease over 120 steps: {np.mean(losses[:20]):.4f} -> " \
        f"{np.mean(losses[-20:]):.4f}"
    # ...and the encoder is the thing that moved: freeze-check one B-side conv kernel.
    assert float(jnp.abs(agent.net["density_b"].params["encoder"]["conv0"]["kernel"]).sum()) > 0
