"""Tests for agents/fb/measure/agent.py — the two-phase bounded successor
measure on the torch FB stack (the jax gciql_flow recipe in FB's own house).

Mirrors the tests/test_onestep_fb.py conventions: CPU `_tiny` factory (obs
Box(24,) so the cube slice 19:22 is in range, z_dim 8, batch 16), plain torch
batches, no dataset/GPU.
"""
import copy

import numpy as np
import pytest
import torch


def _tiny_measure(**extra):
    import gymnasium
    from agents.fb.measure.agent import FBMeasureFlowBCAgent
    from nn_models import (IdentityNNConfig, NoiseConditionedActorArchiConfig,
                           SimpleVectorFieldArchiConfig)
    from normalizers import IdentityNormalizerConfig
    return FBMeasureFlowBCAgent(
        obs_space=gymnasium.spaces.Box(low=-1, high=1, shape=(24,), dtype=np.float32),
        action_dim=2, z_dim=8, L_dim=8, batch_size=16,
        actor_cfg=NoiseConditionedActorArchiConfig(),
        actor_vf_cfg=SimpleVectorFieldArchiConfig(),
        obs_normalizer_cfg=IdentityNormalizerConfig(),
        rgb_encoder_cfg=IdentityNNConfig(),
        augmentator_cfg=IdentityNNConfig(),
        ortho_coef=0.0, q_loss_coef=0.0,
        device="cpu", **extra,
    )


def _tiny_flowbc_stock(**extra):
    import gymnasium
    from agents.fb.flow_bc.agent import FBFlowBCAgent
    from nn_models import (IdentityNNConfig, NoiseConditionedActorArchiConfig,
                           SimpleVectorFieldArchiConfig)
    from normalizers import IdentityNormalizerConfig
    return FBFlowBCAgent(
        obs_space=gymnasium.spaces.Box(low=-1, high=1, shape=(24,), dtype=np.float32),
        action_dim=2, z_dim=8, L_dim=8, batch_size=16,
        actor_cfg=NoiseConditionedActorArchiConfig(),
        actor_vf_cfg=SimpleVectorFieldArchiConfig(),
        obs_normalizer_cfg=IdentityNormalizerConfig(),
        rgb_encoder_cfg=IdentityNNConfig(),
        augmentator_cfg=IdentityNNConfig(),
        device="cpu", **extra,
    )


def _batch(B=16, seed=0, obs_dim=24, act_dim=2, cluster=None):
    """cluster=(k, xyz): plant the first k next-obs cube slices at xyz (tight),
    the rest spread far apart -- the column-mass pattern for hit/nhat tests."""
    rng = np.random.default_rng(seed)
    nobs = rng.normal(size=(B, obs_dim)).astype(np.float32)
    if cluster is not None:
        k, xyz = cluster
        nobs[:k, 19:22] = np.asarray(xyz, np.float32) + \
            rng.normal(0, 0.05, (k, 3)).astype(np.float32)
        nobs[k:, 19:22] = rng.uniform(5, 50, (B - k, 3)).astype(np.float32)
    return {
        "observation": torch.as_tensor(rng.normal(size=(B, obs_dim)).astype(np.float32)),
        "action": torch.as_tensor(rng.uniform(-1, 1, (B, act_dim)).astype(np.float32)),
        "next": {
            "observation": torch.as_tensor(nobs),
            "terminated": torch.zeros(B, 1),
            "physics": torch.zeros(B, 20),
        },
    }


# ============================ 1. construction + update ===========================

@pytest.mark.parametrize("mode", ["first_passage", "visitation"])
def test_update_runs_finite(mode):
    torch.manual_seed(0)
    a = _tiny_measure(measure_mode=mode)
    m = a.update(_batch(), step=0)
    for k in ("fb_loss", "measure_loss", "ms_abs_mean", "hit_rate", "actor_loss"):
        assert k in m, (mode, sorted(m))
        assert np.isfinite(m[k]), (mode, k, m[k])
    assert 0.0 <= m["hit_rate"] <= 1.0
    # a second step still finite (optimizers really stepped)
    m2 = a.update(_batch(seed=1), step=1)
    assert np.isfinite(m2["measure_loss"])


def test_bad_mode_refused():
    with pytest.raises(AssertionError):
        _tiny_measure(measure_mode="banana")


# ============================ 2. hit matrix ======================================

def test_hit_matrix_matches_numpy():
    a = _tiny_measure()
    rng = np.random.default_rng(0)
    raw = rng.normal(size=(16, 24)).astype(np.float32) * 0.3
    got = a.hit_matrix(torch.as_tensor(raw)).numpy()
    cube = raw[:, 19:22]
    d = np.linalg.norm(cube[:, None] - cube[None, :], axis=-1)
    exp = (d <= 0.4).astype(np.float32)
    np.testing.assert_array_equal(got, exp)
    assert (np.diag(got) == 1.0).all()            # self-columns always hit
    # eps and slice are honored
    a2 = _tiny_measure(hit_eps=1.0, hit_slice=(0, 3))
    got2 = a2.hit_matrix(torch.as_tensor(raw)).numpy()
    d2 = np.linalg.norm(raw[:, None, 0:3] - raw[None, :, 0:3], axis=-1)
    np.testing.assert_array_equal(got2, (d2 <= 1.0).astype(np.float32))


# ============================ 3. fp vs visitation semantics ======================

def test_first_passage_masks_bootstrap_visitation_does_not():
    """hit == all-ones (every next-obs cube identical) => the FP residual
    Ms - 1 - g*(1-hit)*tM is INDEPENDENT of the target maps, the visitation
    residual Ms - 1 - g*tM is not: perturbing the target forward map must
    leave the FP measure_loss bit-identical and move the visitation one."""
    b = _batch()
    b["next"]["observation"][:, 19:22] = torch.tensor([1.0, 2.0, 3.0])  # all hit
    for mode, expect_same in (("first_passage", True), ("visitation", False)):
        torch.manual_seed(0)
        a = _tiny_measure(measure_mode=mode)
        a2 = copy.deepcopy(a)
        with torch.no_grad():
            for p in a2.model._target_forward_map.parameters():
                p.mul_(7.0)
        torch.manual_seed(3)
        m1 = a.update(copy.deepcopy(b), step=0)
        torch.manual_seed(3)
        m2 = a2.update(copy.deepcopy(b), step=0)
        if expect_same:
            assert m1["measure_loss"] == m2["measure_loss"], mode
        else:
            assert m1["measure_loss"] != m2["measure_loss"], mode


# ============================ 4. every entry squared =============================

def test_full_loss_matches_numpy_reference_no_linear_term():
    """Capture Fs/B/target_M via hooks + a probe subclass and rebuild the loss
    in numpy: mean((Ms - hit - g*tM)^2) over ALL entries -- the diagonal is a
    squared residual (hit_eps=-1 => hit==0 so the diag term is pure Ms - g*tM;
    FB's linear -mean(diag) would give a different number)."""
    from agents.fb.measure.agent import FBMeasureFlowBCAgent

    captured = {}

    class Probe(FBMeasureFlowBCAgent):
        def get_targets_uncertainty(self, t, p):
            out = super().get_targets_uncertainty(t, p)
            captured.setdefault("target_M", out[2].detach().clone())
            return out

    import gymnasium
    from nn_models import (IdentityNNConfig, NoiseConditionedActorArchiConfig,
                           SimpleVectorFieldArchiConfig)
    from normalizers import IdentityNormalizerConfig
    torch.manual_seed(0)
    a = Probe(
        obs_space=gymnasium.spaces.Box(low=-1, high=1, shape=(24,), dtype=np.float32),
        action_dim=2, z_dim=8, L_dim=8, batch_size=16,
        actor_cfg=NoiseConditionedActorArchiConfig(),
        actor_vf_cfg=SimpleVectorFieldArchiConfig(),
        obs_normalizer_cfg=IdentityNormalizerConfig(),
        rgb_encoder_cfg=IdentityNNConfig(),
        augmentator_cfg=IdentityNNConfig(),
        measure_mode="visitation", hit_eps=-1.0,   # hit == 0 everywhere
        scale_norm=False, device="cpu",
    )
    Fs_calls, B_calls = [], []
    a.model._forward_map.register_forward_hook(
        lambda m, i, o: Fs_calls.append(o.detach().clone()))
    a.model._backward_map.register_forward_hook(
        lambda m, i, o: B_calls.append((i[0].detach().clone(), o.detach().clone())))
    b = _batch()
    m = a.update(copy.deepcopy(b), step=0)
    # online update_fb call: first _forward_map call; the _backward_map call
    # whose INPUT is the goal (= raw next_obs under identity normalizer/encoder)
    Fs = Fs_calls[0]
    nobs = b["next"]["observation"]
    B = next(o for (inp, o) in B_calls if inp.shape == nobs.shape
             and torch.equal(inp, nobs))
    Ms = torch.matmul(Fs, B.T)                                   # [P,B,B]
    tM = captured["target_M"]
    ref = (Ms - 0.99 * tM).pow(2).mean().item()                  # hit == 0
    assert m["measure_loss"] == pytest.approx(ref, rel=1e-5)
    # and it is NOT FB's linear-diag construction
    P = Ms.shape[0]
    off = 1 - torch.eye(16)
    diffm = Ms - 0.99 * tM
    fb_style = (0.5 * (diffm * off).pow(2).sum() / off.sum()
                - torch.diagonal(diffm, dim1=1, dim2=2).mean() * P).item()
    assert abs(m["measure_loss"] - fb_style) > 1e-8


# ============================ 5. freeze_b ========================================

def test_freeze_b_keeps_backward_map_byte_identical():
    torch.manual_seed(0)
    a = _tiny_measure(freeze_b=True)
    assert a.backward_optimizer is None
    b_before = [p.detach().clone() for p in a.model._backward_map.parameters()]
    f_before = [p.detach().clone() for p in a.model._forward_map.parameters()]
    for i in range(3):
        a.update(_batch(seed=i), step=i)
    for p0, p1 in zip(b_before, a.model._backward_map.parameters()):
        assert torch.equal(p0, p1), "frozen B moved"
    assert any(not torch.equal(p0, p1)
               for p0, p1 in zip(f_before, a.model._forward_map.parameters())), \
        "forward map did not train"


# ============================ 6. nhat head =======================================

def test_nhat_loss_decreases_on_planted_pattern():
    """Half the batch's next-obs cubes clustered (column mass 0.5), the rest
    isolated (1/16): the head must fit this pattern -- loss decreases over 50
    updates on the same batch, predictions stay positive (softplus)."""
    torch.manual_seed(0)
    a = _tiny_measure(nhat_head=True)
    b = _batch(cluster=(8, [1.0, 2.0, 0.0]))
    first = last = None
    for i in range(50):
        m = a.update(copy.deepcopy(b), step=i)
        assert "nhat_loss" in m and np.isfinite(m["nhat_loss"])
        assert m["nhat_pred_mean"] > 0.0
        if first is None:
            first = m["nhat_loss"]
        last = m["nhat_loss"]
    assert last < first, (first, last)


def test_nhat_off_no_head_no_metric():
    torch.manual_seed(0)
    a = _tiny_measure()
    assert not hasattr(a, "_nhat_head")
    m = a.update(_batch(), step=0)
    assert "nhat_loss" not in m


# ============================ 7. warm start ======================================

def test_warm_start_restores_shared_and_fresh_inits_nhat(tmp_path):
    torch.manual_seed(0)
    stock = _tiny_flowbc_stock()
    ckpt = tmp_path / "step_300.pt"
    torch.save(stock.state_dict(), str(ckpt))
    torch.manual_seed(1)                       # different init on purpose
    a = _tiny_measure(nhat_head=True, warm_start_ckpt=str(ckpt))
    # shared modules restored bit for bit
    for (k0, p0), (k1, p1) in zip(stock.model.state_dict().items(),
                                  a.model.state_dict().items()):
        assert k0 == k1
        assert torch.equal(p0, p1), f"model param {k0} not warm-started"
    # nhat head exists and is fresh (present in the measure agent's state only)
    assert hasattr(a, "_nhat_head")
    s = a.state_dict()
    assert "nhat_head" in s
    # and the warm-started agent trains
    m = a.update(_batch(), step=0)
    assert np.isfinite(m["measure_loss"]) and np.isfinite(m["nhat_loss"])


def test_warm_start_state_dict_roundtrip(tmp_path):
    """The measure agent's own checkpoints round-trip (nhat head included)."""
    torch.manual_seed(0)
    a = _tiny_measure(nhat_head=True)
    a.update(_batch(), step=0)
    ckpt = tmp_path / "m.pt"
    torch.save(a.state_dict(), str(ckpt))
    torch.manual_seed(2)
    b = _tiny_measure(nhat_head=True)
    b.load_state_dict(torch.load(str(ckpt)))
    for p0, p1 in zip(a._nhat_head.parameters(), b._nhat_head.parameters()):
        assert torch.equal(p0, p1)
