"""Mass-conservation penalty + smooth hit kernel on the bounded successor
measure (`FBMeasureFlowBCAgent.mass_coef` / `hit_kernel`, added 2026-07-29,
ledger §9.15).

Why (spec 13): the D1 diagnostic showed the ~50% pointwise negativity carries
24-29% of the object's absolute MASS and is worst in high-density regions --
truncated-basis ringing of a rank-50 fit to a boxcar kernel. The hinge alone
(§9.8/9.9) fixed sign but inflated set mass 2.5-3.6x because nothing conserves
total mass. Two knobs attack this:

  mass_coef   per-row total-mass anchor. Identity: for the visitation fixed
              point, E_{g~rho}[M(s,g)/rho(B_eps(g))] = sum_t gamma^t = 1/(1-gamma)
              (the cube is always somewhere). In-batch estimator: N_j = column
              mean of the hit matrix (clamped at 2/B), R_i = mean_j M_ij/N_j,
              penalty mean_i ((1-gamma) R_i - 1)^2.
  hit_kernel  'ball' (0/1, the historical kernel) | 'quartic' (biweight
              (1-(d/h)^2)^2, C^1, compact support h=hit_eps) -- less
              high-frequency content => less ringing at fixed rank.

The load-bearing properties, in order:
  1. defaults (mass_coef=0, hit_kernel='ball') are BIT-IDENTICAL to the
     previous objective -- every launched checkpoint stays reproducible.
  2. the mass penalty is exactly zero on a perfectly calibrated M, and its
     gradient pushes an inflated M down / a deflated M up (the anchor the
     hinge lacked).
  3. the quartic kernel is a valid kernel: 1 on the diagonal, 0 outside h,
     strictly inside (0,1) in between, monotone in distance.
  4. both knobs must actually REACH the agent through train.py (the `shared`
     whitelist lesson -- a recorded config proves nothing on its own).
"""
import torch


GAMMA = 0.99


def _mass_penalty(Ms, hit, gamma=GAMMA, clamp_frac=2.0):
    """The penalty assembly as in FBMeasureFlowBCAgent.update_fb."""
    B = hit.shape[-1]
    nhat = hit.mean(dim=0).clamp_min(clamp_frac / B)          # [B]
    row_mass = (Ms / nhat).mean(dim=-1)                       # [P,B]
    return ((1.0 - gamma) * row_mass - 1.0).pow(2).mean()


def _quartic(dist, h):
    u2 = (dist / h).pow(2)
    return (1.0 - u2).clamp_min(0.0).pow(2)


# --------------------------------------------------------------- mass_coef ----

def test_mass_penalty_zero_on_calibrated_M():
    """M_ij = N_j / (1-gamma) has exactly the right row mass => zero penalty."""
    torch.manual_seed(0)
    hit = (torch.rand(32, 32) < 0.3).double()
    hit.fill_diagonal_(1.0)
    nhat = hit.mean(dim=0).clamp_min(2.0 / 32)
    Ms = (nhat / (1.0 - GAMMA)).expand(2, 32, 32)
    pen = _mass_penalty(Ms, hit)
    assert pen.item() < 1e-20, f"calibrated M must not be taxed, got {pen}"


def test_mass_penalty_positive_and_monotone_in_inflation():
    torch.manual_seed(0)
    hit = (torch.rand(16, 16) < 0.3).double()
    hit.fill_diagonal_(1.0)
    nhat = hit.mean(dim=0).clamp_min(2.0 / 16)
    M_ok = (nhat / (1.0 - GAMMA)).expand(1, 16, 16)
    p2 = _mass_penalty(2.0 * M_ok, hit)
    p3 = _mass_penalty(3.0 * M_ok, hit)
    assert p2.item() > 0 and p3.item() > p2.item()
    # analytic check: scale k => penalty (k-1)^2
    assert abs(p2.item() - 1.0) < 1e-9
    assert abs(p3.item() - 4.0) < 1e-9


def test_mass_penalty_gradient_direction():
    """Inflated mass must be pushed DOWN, deflated mass UP."""
    hit = torch.eye(8, dtype=torch.float64)
    for scale, sign in ((2.0, -1.0), (0.5, +1.0)):
        nhat = hit.mean(dim=0).clamp_min(2.0 / 8)
        Ms = (scale * nhat / (1.0 - GAMMA)).expand(1, 8, 8).clone().requires_grad_(True)
        _mass_penalty(Ms, hit).backward()
        g = Ms.grad.sum().item()
        assert g * sign < 0, (f"scale={scale}: descending must move mass "
                              f"{'down' if sign < 0 else 'up'}, grad sum {g}")


def test_mass_penalty_clamp_guards_sparse_columns():
    """A column whose only hit is the diagonal (N=1/B) must not dominate: the
    clamp bounds the importance weight at B/2."""
    B = 16
    hit = torch.eye(B, dtype=torch.float64)      # every column maximally sparse
    nhat = hit.mean(dim=0).clamp_min(2.0 / B)
    assert (nhat == 2.0 / B).all(), "clamp must bind at 2/B"


# --------------------------------------------------------------- hit_kernel ----

def test_quartic_kernel_properties():
    h = 0.6
    d = torch.tensor([0.0, 0.2, 0.4, 0.59, 0.6, 0.61, 5.0], dtype=torch.float64)
    w = _quartic(d, h)
    assert w[0].item() == 1.0, "diagonal (d=0) must be exactly 1"
    assert (w[1:4] > 0).all() and (w[1:4] < 1).all()
    assert (w[4:] == 0).all(), "compact support: zero at and beyond h"
    assert (w[:-1] >= w[1:]).all(), "monotone non-increasing in distance"


def test_ball_kernel_unchanged_by_default():
    """hit_kernel='ball' must reproduce the historical cdist<=eps matrix
    byte-for-byte on the same inputs."""
    torch.manual_seed(0)
    cube = torch.randn(64, 3)
    eps = 0.4
    ref = (torch.cdist(cube, cube) <= eps).float()
    d = torch.cdist(cube, cube)
    ball = (d <= eps).float()
    assert torch.equal(ball, ref)
    # and the quartic differs (it had better -- the whitelist lesson)
    assert not torch.equal(_quartic(d, 0.6).float(), ref)


# ------------------------------------------------------- agent + train.py ----

def test_agent_hit_matrix_quartic_and_ball():
    """End-to-end through the agent's hit_matrix: ball path byte-identical to
    the pre-knob computation, quartic path correct on the cube slice."""
    import gymnasium
    from hydra import compose, initialize_config_dir
    from pathlib import Path
    import train as train_mod

    cfg_dir = str(Path(train_mod.__file__).resolve().parent / "configs")
    obs = torch.randn(32, 28)

    def build(**ov):
        with initialize_config_dir(config_dir=cfg_dir, version_base=None):
            cfg = compose(config_name="train",
                          overrides=["agent=fb_measure_flowbc", "domain=cube_single",
                                     "device=cpu"] + [f"{k}={v}" for k, v in ov.items()])
        return train_mod.make_agent(cfg, gymnasium.spaces.Box(-1, 1, (28,)), 5)

    ball = build().hit_matrix(obs)
    cube = obs[:, 19:22]
    assert torch.equal(ball, (torch.cdist(cube, cube) <= 0.4).float())

    q = build(hit_kernel="quartic", hit_eps=0.6).hit_matrix(obs)
    d = torch.cdist(cube, cube)
    assert torch.allclose(q, ((1.0 - (d / 0.6) ** 2).clamp_min(0.0) ** 2).float())
    assert torch.allclose(torch.diagonal(q), torch.ones(32))


def test_train_py_propagates_the_knobs():
    """The `shared`-whitelist lesson, applied to the measure branch: both knobs
    must reach the constructed agent."""
    import gymnasium
    from hydra import compose, initialize_config_dir
    from pathlib import Path
    import train as train_mod

    cfg_dir = str(Path(train_mod.__file__).resolve().parent / "configs")
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = compose(config_name="train",
                      overrides=["agent=fb_measure_flowbc", "domain=cube_single",
                                 "device=cpu", "mass_coef=0.5",
                                 "hit_kernel=quartic", "hit_eps=0.6"])
    agent = train_mod.make_agent(cfg, gymnasium.spaces.Box(-1, 1, (28,)), 5)
    assert agent.mass_coef == 0.5
    assert agent.hit_kernel == "quartic"
    assert agent.hit_eps == 0.6
    # defaults stay off
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg0 = compose(config_name="train",
                       overrides=["agent=fb_measure_flowbc", "domain=cube_single",
                                  "device=cpu"])
    agent0 = train_mod.make_agent(cfg0, gymnasium.spaces.Box(-1, 1, (28,)), 5)
    assert agent0.mass_coef == 0.0
    assert agent0.hit_kernel == "ball"


# --------------------------------------------------------------- hit_spec ----

def _build_measure_agent(**ov):
    import gymnasium
    from hydra import compose, initialize_config_dir
    from pathlib import Path
    import train as train_mod
    cfg_dir = str(Path(train_mod.__file__).resolve().parent / "configs")
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = compose(config_name="train",
                      overrides=["agent=fb_measure_flowbc", "domain=cube_single",
                                 "device=cpu"] + [f"{k}={v}" for k, v in ov.items()])
    return train_mod.make_agent(cfg, gymnasium.spaces.Box(-1, 1, (46,)), 5)


def test_hit_spec_default_off_is_byte_identical():
    """hit_spec='' must reproduce the single-ball hit matrix exactly."""
    torch.manual_seed(0)
    obs = torch.randn(32, 46)
    ref = _build_measure_agent().hit_matrix(obs)
    cube = obs[:, 19:22]
    assert torch.equal(ref, (torch.cdist(cube, cube) <= 0.4).float())


def test_hit_spec_single_block_equals_slice_path():
    torch.manual_seed(1)
    obs = torch.randn(32, 46)
    a = _build_measure_agent(**{"hit_spec": "'19:22:0.4'"})
    assert a.hit_blocks == ((19, 22, 0.4),)
    ref = _build_measure_agent().hit_matrix(obs)
    assert torch.equal(a.hit_matrix(obs), ref)


def test_hit_spec_conjunction():
    """Multi-block spec must be the AND of per-block ball hits (jax E7 semantics)."""
    torch.manual_seed(2)
    obs = torch.randn(48, 46)
    a = _build_measure_agent(**{"hit_spec": "'19:22:0.4,28:31:0.4'"})
    assert a.hit_blocks == ((19, 22, 0.4), (28, 31, 0.4))
    got = a.hit_matrix(obs)
    c1, c2 = obs[:, 19:22], obs[:, 28:31]
    want = ((torch.cdist(c1, c1) <= 0.4) & (torch.cdist(c2, c2) <= 0.4)).float()
    assert torch.equal(got, want)
    # conjunction is sparser than either block alone
    assert got.sum() <= (torch.cdist(c1, c1) <= 0.4).float().sum()
    assert torch.equal(torch.diagonal(got), torch.ones(48))


def test_domain_configs_do_not_clobber_the_agent_group():
    """`agent=fb_measure_flowbc domain=cube_double` must construct the MEASURE
    agent. cube_double.yaml/scene.yaml used to set a literal `agent:` key that
    silently reverted the CLI group choice to stock FB while the recorded
    config still carried the measure knobs (caught 2026-07-30)."""
    import gymnasium
    from hydra import compose, initialize_config_dir
    from pathlib import Path
    import train as train_mod
    from agents.fb.measure.agent import FBMeasureFlowBCAgent

    cfg_dir = str(Path(train_mod.__file__).resolve().parent / "configs")
    for dom, ndim in (("cube_double", 46), ("scene", 46)):
        with initialize_config_dir(config_dir=cfg_dir, version_base=None):
            cfg = compose(config_name="train",
                          overrides=[f"domain={dom}", "agent=fb_measure_flowbc",
                                     "device=cpu", "hit_spec='19:22:0.4,28:31:0.4'"])
        assert cfg.agent == "fb_measure_flowbc", f"{dom}: {cfg.agent}"
        agent = train_mod.make_agent(cfg, gymnasium.spaces.Box(-1, 1, (ndim,)), 5)
        assert isinstance(agent, FBMeasureFlowBCAgent), f"{dom} built {type(agent)}"
        assert agent.hit_blocks == ((19, 22, 0.4), (28, 31, 0.4))
        # and with no agent override the domain default must still be stock FB
        with initialize_config_dir(config_dir=cfg_dir, version_base=None):
            cfg0 = compose(config_name="train", overrides=[f"domain={dom}", "device=cpu"])
        assert cfg0.agent == "fb_flowbc", f"{dom} default broke: {cfg0.agent}"
