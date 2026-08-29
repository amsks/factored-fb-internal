"""agents/dfb — the torch Density-FB port, tested against the math and the
tabular ground truth (NOT against the jax code: the numpy references here are
assembled independently from the documented semantics, so a shared bug cannot
hide in a port-to-port comparison).

Five groups (the port directive's contract):
  1. target composition — t rows on the simplex; hard k = I; soft k = the
     query-set-normalized Gaussian kernel (independent numpy twin);
  2. per-head softmax normalization of w;
  3. stop-grad discipline — t/a_plus/kg/distill-target carry no grad; the actor
     step cannot move the density towers and vice versa;
  4. the zero-shot de-biased readout vs the EXACT tabular successor measure
     (reusing scripts/spec30_tabular_mdp.py, the same ground truth the jax
     suite is pinned to);
  5. fixed-seed numerical parity of the LOSS VALUE against a hand-built numpy
     reference (loss ASSEMBLY reimplemented from the math; only the tower
     forward passes come from the model, because they are the parameterization).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import gymnasium  # noqa: E402

from agents.dfb.agent import (  # noqa: E402
    DFBAgent,
    compute_density_primary,
    density_weights,
)
from agents.dfb.hit_spec import (  # noqa: E402
    delta_phi,
    parse_hit_spec,
    raw_block_cols,
    scaled_block_cols,
)
from nn_models import NoiseConditionedActorArchiConfig, SimpleVectorFieldArchiConfig  # noqa: E402
from scripts.spec30_tabular_mdp import (  # noqa: E402
    band_random_walk,
    density_td_fixed_point,
    eps_kernel,
    exact_successor_measure,
    measure_integral,
    normalized_measure,
)

OBS_DIM, ACT_DIM, BATCH = 6, 3, 8
HIT_SPEC = "2:4:0.5,5:6:0.8"          # two blocks, widths 2 + 1, distinct eps
GAMMA = 0.97


def make_agent(hard: bool, actor_kind: str = "flow", seed: int = 0) -> DFBAgent:
    torch.manual_seed(seed)
    space = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)
    return DFBAgent(
        space, ACT_DIM,
        measure_hit_spec=HIT_SPEC,
        measure_density_dim=8,
        measure_density_hard=hard,
        hidden_dims=(32, 32),
        actor_kind=actor_kind,
        actor_cfg=(NoiseConditionedActorArchiConfig(hidden_dim=32, hidden_layers=1,
                                                    embedding_layers=2)
                   if actor_kind == "flow" else None),
        actor_vf_cfg=SimpleVectorFieldArchiConfig(hidden_dim=32, hidden_layers=2),
        discount=GAMMA,
        flow_steps=3,
        bc_coeff=1.0,
        device="cpu",
    )


def make_batch(seed: int = 1) -> dict:
    g = torch.Generator().manual_seed(seed)
    return {
        "observations": torch.randn(BATCH, OBS_DIM, generator=g),
        "next_observations": torch.randn(BATCH, OBS_DIM, generator=g),
        "actions": torch.randn(BATCH, ACT_DIM, generator=g).clamp(-1, 1),
        "value_goals": torch.randn(BATCH, OBS_DIM, generator=g),
        "actor_goals": torch.randn(BATCH, OBS_DIM, generator=g),
    }


def numpy_kernel_rows(x_full: np.ndarray, protos_raw: np.ndarray, blocks) -> np.ndarray:
    """Independent numpy twin of delta_phi: block-sliced, eps-scaled, softmaxed."""
    xs, ps, off = [], [], 0
    for (a, b, e) in blocks:
        xs.append(x_full[:, a:b] / e)
        ps.append(protos_raw[:, off:off + (b - a)] / e)
        off += b - a
    xc, pc = np.concatenate(xs, -1), np.concatenate(ps, -1)
    logits = -0.5 * ((xc[:, None, :] - pc[None]) ** 2).sum(-1)
    z = np.exp(logits - logits.max(-1, keepdims=True))
    return z / z.sum(-1, keepdims=True)


# ───────────────────────── 1. target composition ─────────────────────────────

def test_target_rows_are_on_the_simplex_hard_and_soft():
    batch = make_batch()
    for hard in (True, False):
        agent = make_agent(hard)
        u_raw = raw_block_cols(batch["next_observations"], agent.blocks)
        u_sc = scaled_block_cols(batch["next_observations"], agent.blocks)
        torch.manual_seed(7)
        t = agent.density_targets(batch["observations"], batch["next_observations"],
                                  batch["value_goals"], u_raw, u_sc)
        assert t.shape == (2, BATCH, BATCH)
        assert torch.allclose(t.sum(-1), torch.ones(2, BATCH), atol=1e-6), hard
        assert not t.requires_grad


def test_hard_immediate_term_is_the_identity():
    """Same seed => same bootstrap draw, so t_hard - t_soft isolates the k swap:
    it must equal (1-gamma) * (I - k_soft) exactly."""
    batch = make_batch()
    hard_agent = make_agent(True, seed=3)
    soft_agent = make_agent(False, seed=3)
    # identical tower/actor weights => identical w_bar under the same seed
    soft_agent.model.load_state_dict(hard_agent.model.state_dict())
    args = (batch["observations"], batch["next_observations"], batch["value_goals"],
            raw_block_cols(batch["next_observations"], hard_agent.blocks),
            scaled_block_cols(batch["next_observations"], hard_agent.blocks))
    torch.manual_seed(11)
    t_hard = hard_agent.density_targets(*args)
    torch.manual_seed(11)
    t_soft = soft_agent.density_targets(*args)
    k_soft = delta_phi(batch["next_observations"],
                       raw_block_cols(batch["next_observations"], hard_agent.blocks),
                       hard_agent.blocks)
    expected = (1.0 - GAMMA) * (torch.eye(BATCH) - k_soft)
    assert torch.allclose(t_hard - t_soft, expected[None].expand(2, -1, -1), atol=1e-6)


def test_soft_kernel_matches_independent_numpy_twin():
    blocks = parse_hit_spec(HIT_SPEC)
    x = torch.randn(5, OBS_DIM, generator=torch.Generator().manual_seed(2))
    protos = raw_block_cols(torch.randn(7, OBS_DIM,
                                        generator=torch.Generator().manual_seed(4)), blocks)
    got = delta_phi(x, protos, blocks).numpy()
    ref = numpy_kernel_rows(x.numpy(), protos.numpy(), blocks)
    assert np.abs(got - ref).max() < 1e-6
    assert np.abs(got.sum(-1) - 1.0).max() < 1e-6


# ─────────────────────── 2. softmax normalization per head ───────────────────

def test_online_weights_are_per_head_simplex_rows():
    agent = make_agent(False)
    batch = make_batch()
    with torch.no_grad():
        u_sc = scaled_block_cols(batch["next_observations"], agent.blocks)
        b = agent.model.density_b(u_sc)
        f = agent.model.density_f(agent.model.f_input(
            batch["observations"], batch["actions"], batch["value_goals"]))
        w = torch.softmax(torch.einsum("hbd,md->hbm", f, b) / agent.temp, dim=-1)
    assert w.shape == (2, BATCH, BATCH)
    assert torch.allclose(w.sum(-1), torch.ones(2, BATCH), atol=1e-6)
    # the two heads are genuinely different networks (parallel params), not copies
    assert not torch.allclose(w[0], w[1])


# ─────────────────────────── 3. stop-grad discipline ─────────────────────────

def test_density_update_never_touches_target_or_actor_params():
    agent = make_agent(False)
    batch = make_batch()
    tgt_before = {k: v.clone() for k, v in agent.model.target_density_f.state_dict().items()}
    actor_before = {k: v.clone() for k, v in agent.model._actor.state_dict().items()}
    ce, _ = agent.density_loss(batch)
    agent.f_optimizer.zero_grad(set_to_none=True)
    agent.b_optimizer.zero_grad(set_to_none=True)
    ce.backward()
    # frozen targets: no grads at all
    assert all(p.grad is None for p in agent.model.target_density_f.parameters())
    assert all(p.grad is None for p in agent.model.target_density_b.parameters())
    # bootstrap action was drawn under no_grad: actor got no gradient either
    assert all(p.grad is None for p in agent.model._actor.parameters())
    agent.f_optimizer.step()
    agent.b_optimizer.step()
    for k, v in agent.model.target_density_f.state_dict().items():
        assert torch.equal(v, tgt_before[k]), k
    for k, v in agent.model._actor.state_dict().items():
        assert torch.equal(v, actor_before[k]), k


def test_actor_update_cannot_move_the_density_towers():
    agent = make_agent(False)
    batch = make_batch()
    f_before = {k: v.clone() for k, v in agent.model.density_f.state_dict().items()}
    b_before = {k: v.clone() for k, v in agent.model.density_b.state_dict().items()}
    loss, _ = agent.actor_loss(batch)
    agent.actor_optimizer.zero_grad(set_to_none=True)
    agent.actor_vf_optimizer.zero_grad(set_to_none=True)
    loss.backward()
    agent.actor_optimizer.step()
    agent.actor_vf_optimizer.step()
    for k, v in agent.model.density_f.state_dict().items():
        assert torch.equal(v, f_before[k]), k
    for k, v in agent.model.density_b.state_dict().items():
        assert torch.equal(v, b_before[k]), k


def test_actor_gradient_flows_through_the_one_shot_action():
    """The q-term must train the actor THROUGH the towers (input path), even
    though the tower parameters never step on this loss."""
    agent = make_agent(False)
    batch = make_batch()
    loss, _ = agent.actor_loss(batch)
    loss.backward()
    grads = [p.grad for p in agent.model._actor.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)


def test_full_update_runs_and_moves_targets_by_tau():
    agent = make_agent(False)
    batch = make_batch()
    online_before = {k: v.clone() for k, v in agent.model.density_f.state_dict().items()}
    tgt_before = {k: v.clone() for k, v in agent.model.target_density_f.state_dict().items()}
    info = agent.update(batch, step=0)
    assert np.isfinite(info["density_ce"]) and np.isfinite(info["actor_loss"])
    moved = any(not torch.equal(v, online_before[k])
                for k, v in agent.model.density_f.state_dict().items())
    assert moved
    for k, v in agent.model.target_density_f.state_dict().items():
        if v.dtype.is_floating_point:
            expected = (1 - agent.target_tau) * tgt_before[k] \
                + agent.target_tau * agent.model.density_f.state_dict()[k]
            assert torch.allclose(v, expected, atol=1e-7), k


# ──────────────── 4. tabular ground truth for the zero-shot readout ──────────

N_STATES, S0, A0 = 25, 12, 0


@pytest.fixture(scope="module")
def toy():
    rng = np.random.default_rng(0)
    P = band_random_walk(N_STATES, 8, 16, seed=3)
    pi = rng.dirichlet(np.ones(2), size=N_STATES)
    M = exact_successor_measure(P, pi, GAMMA)
    coords = np.arange(N_STATES, dtype=np.float64)[:, None]
    return dict(P=P, pi=pi, M=M, nu=normalized_measure(M, GAMMA),
                coords=coords, r=rng.normal(size=N_STATES))


def test_readout_recovers_exact_measure_integral(toy):
    """Hard TD fixed point + full pool + Dirac kappa: the ported readout equals
    the matrix-inverse answer M^T r to float64 precision."""
    w = density_td_fixed_point(toy["P"], toy["pi"], GAMMA, np.eye(N_STATES), iters=50000)
    kappa = eps_kernel(toy["coords"], toy["coords"], 0.02)
    assert np.abs(kappa - np.eye(N_STATES)).max() == 0.0
    for r in (toy["r"], np.arange(N_STATES, dtype=np.float64)):
        truth = float(measure_integral(toy["M"], r)[S0, A0])
        got = compute_density_primary(r[S0], w[S0, A0], kappa[S0], r, GAMMA)
        assert abs(got - truth) <= 1e-11 * max(1.0, abs(truth))


@pytest.mark.parametrize("gamma,const", [(0.5, 1.0), (0.99, -2.5)])
def test_constant_reward_pins_the_normalization(toy, gamma, const):
    w = density_td_fixed_point(toy["P"], toy["pi"], gamma, np.eye(N_STATES), iters=50000)
    kappa = eps_kernel(toy["coords"], toy["coords"], 0.02)
    r = np.full(N_STATES, const)
    got = compute_density_primary(r[S0], w[S0, A0], kappa[S0], r, gamma)
    assert abs(got - const / (1 - gamma)) <= 1e-10 * abs(const / (1 - gamma))


def test_debias_pair_cancels_in_the_dirac_limit_and_halves_break_it(toy):
    w = density_td_fixed_point(toy["P"], toy["pi"], GAMMA, np.eye(N_STATES), iters=50000)
    kappa = eps_kernel(toy["coords"], toy["coords"], 0.02)
    r = toy["r"]
    truth = float(measure_integral(toy["M"], r)[S0, A0])
    paired = compute_density_primary(r[S0], w[S0, A0], kappa[S0], r, GAMMA)
    bare = float(w[S0, A0] @ r) / (1 - GAMMA)
    assert paired == pytest.approx(bare, abs=1e-12)
    dropped_r0 = compute_density_primary(0.0, w[S0, A0], kappa[S0], r, GAMMA)
    dropped_kp = compute_density_primary(r[S0], w[S0, A0], np.zeros(N_STATES), r, GAMMA)
    assert dropped_r0 - truth == pytest.approx(-r[S0] / (1 - GAMMA), rel=1e-9)
    assert dropped_kp - truth == pytest.approx(+r[S0] / (1 - GAMMA), rel=1e-9)


def test_double_q_min_over_heads_is_subnormalized():
    """The shipped eval convention, recorded: min of two simplex vectors loses
    mass when the heads disagree, and Qhat scales by exactly that mass on a
    constant reward."""
    rng = np.random.default_rng(3)
    f_emb = rng.normal(size=(2, 4)) * 1.5
    b_pool = rng.normal(size=(8, 4))
    w = density_weights(f_emb, b_pool, temp=1.0, double_q=True)
    w0 = density_weights(f_emb[:1], b_pool, temp=1.0, double_q=False)
    assert abs(w0.sum() - 1.0) < 1e-12
    mass = float(w.sum())
    assert mass < 1.0 - 1e-3
    c, P = 2.0, 8
    got = compute_density_primary(c, w, np.full(P, 1.0 / P), np.full(P, c), GAMMA)
    assert got == pytest.approx(mass * c / (1 - GAMMA), rel=1e-10)


def test_agent_zero_shot_q_end_to_end_smoke():
    """The nn-backed readout path: finite, and exactly linear in r (the PRIMARY
    readout is a fixed dot product against r — the structural property the
    linearity split rests on)."""
    agent = make_agent(True)
    batch = make_batch()
    pool = batch["next_observations"]
    obs, act, goal = batch["observations"][0], batch["actions"][0], batch["value_goals"][0]
    rng = np.random.default_rng(0)
    r1, r2 = rng.normal(size=BATCH), rng.normal(size=BATCH)
    a_, b_ = 1.7, -0.3
    q1 = agent.zero_shot_q(obs, act, goal, pool, r1, 0.5)
    q2 = agent.zero_shot_q(obs, act, goal, pool, r2, -1.0)
    q12 = agent.zero_shot_q(obs, act, goal, pool, a_ * r1 + b_ * r2, a_ * 0.5 + b_ * (-1.0))
    assert np.isfinite(q12)
    assert q12 == pytest.approx(a_ * q1 + b_ * q2, rel=1e-5)


# ──────────── 5. fixed-seed numpy parity of the loss value ───────────────────

def _np_log_softmax(x):
    m = x.max(-1, keepdims=True)
    lse = m + np.log(np.exp(x - m).sum(-1, keepdims=True))
    return x - lse


@pytest.mark.parametrize("hard", [True, False])
def test_density_loss_matches_hand_built_numpy_reference(hard):
    """Tower outputs come from the model (they ARE the parameterization); the
    ENTIRE loss assembly — logits, softmaxes, kernel, TD target, CE — is
    rebuilt in float64 numpy from the documented math and must agree."""
    agent = make_agent(hard, seed=5)
    batch = make_batch(seed=6)
    torch.manual_seed(42)
    ce, _ = agent.density_loss(batch)
    ce = ce.detach()

    # replicate the single rng consumption (the bootstrap noise) under the same seed
    torch.manual_seed(42)
    noise = torch.randn((BATCH, ACT_DIM))
    with torch.no_grad():
        obs, nobs = batch["observations"], batch["next_observations"]
        acts, goals = batch["actions"], batch["value_goals"]
        u_raw = raw_block_cols(nobs, agent.blocks)
        u_sc = scaled_block_cols(nobs, agent.blocks)
        b = agent.model.density_b(u_sc).double().numpy()
        f = agent.model.density_f(agent.model.f_input(obs, acts, goals)).double().numpy()
        a_plus = agent.model.one_shot_action(nobs, goals, noise)
        tb = agent.model.target_density_b(u_sc).double().numpy()
        tf = agent.model.target_density_f(
            agent.model.f_input(nobs, a_plus, goals)).double().numpy()

    logits = np.einsum("hbd,md->hbm", f, b) / agent.temp
    logp = _np_log_softmax(logits)
    tgt_logits = np.einsum("hbd,md->hbm", tf, tb) / agent.temp
    e = np.exp(tgt_logits - tgt_logits.max(-1, keepdims=True))
    w_bar = e / e.sum(-1, keepdims=True)
    if hard:
        k = np.eye(BATCH)
    else:
        k = numpy_kernel_rows(nobs.numpy().astype(np.float64),
                              u_raw.numpy().astype(np.float64), agent.blocks)
    t = (1.0 - GAMMA) * k[None] + GAMMA * w_bar
    ce_ref = -(t * logp).sum(-1).mean()
    assert float(ce) == pytest.approx(float(ce_ref), rel=1e-5), (hard, float(ce), ce_ref)


def test_ddpgbc_variant_runs_one_full_update():
    agent = make_agent(False, actor_kind="ddpgbc")
    info = agent.update(make_batch(), step=0)
    assert np.isfinite(info["density_ce"]) and np.isfinite(info["actor_loss"])
