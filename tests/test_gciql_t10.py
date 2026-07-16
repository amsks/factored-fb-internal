"""T10 -- centered (advantage) successor measure (PAPER/rescuing-fb/04-centered-measure-spec.md).

New default-off knobs on gciql_flow.py: centered_measure in {off, aux, only},
center_b_samples, center_b_coef, centered_measure_coef, centered_split_bootstrap.

  aux  (T10-a): stock scalar-IQL grounding + centered-measure TD on
                DeltaM = (phi(s,a) - sg[b(s)]).psi(s')/sqrt(d) as an auxiliary;
                b(s) regressed (scale-normed) to sg[mean_{a~mu} phi(s,a)] over K
                flow samples from the goal-agnostic BC flow.
  only (T10-b): NO scalar TD; base leg = behavior-anchored measure TD with the
                DATASET next action (SARSA, batch['next_actions']) + the centered
                DeltaM leg with a pi bootstrap (split bootstrap).

Contract under test:
  1. default-off is bit-identical: the centered path is never entered, the param
     tree has no center_b module (every existing checkpoint keeps loading).
  2. both arms train end-to-end with finite losses and the pre-registered
     instrumentation (b_fidelity, psi/phi norm-sq, q_mean/q_max) present.
  3. stop-gradients: b(s) is trained ONLY by its regression (the DeltaM leg sends
     it no gradient); under 'only' the IQL value net receives NO gradient.
  4. the assert lattice refuses invalid compositions.

Runs in the jax venv subprocess (mirrors tests/test_gciql_psi_sg.py).
"""
import textwrap

from tests.jax_subprocess import run_jax, OGBENCH_IMPLS

_COMMON = """
import jax, jax.numpy as jnp, numpy as np
from gciql_flow import GCIQLFlowAgent, get_config

def make_cfg(**over):
    cfg = dict(get_config())
    cfg["encoder"] = None; cfg["frame_stack"] = None
    cfg["value_arch"] = "bilinear"; cfg["value_latent_dim"] = 8
    cfg.update(over)
    return cfg

def make_batch(B=16, with_next_actions=False, seed=0):
    rng = np.random.default_rng(seed)
    batch = dict(
        observations=rng.normal(size=(B, 11)).astype(np.float32),
        next_observations=rng.normal(size=(B, 11)).astype(np.float32),
        actions=np.clip(rng.normal(size=(B, 5)), -1, 1).astype(np.float32),
        value_goals=rng.normal(size=(B, 11)).astype(np.float32),
        actor_goals=rng.normal(size=(B, 11)).astype(np.float32),
        rewards=rng.integers(0, 2, size=B).astype(np.float32) - 1.0,
        masks=rng.integers(0, 2, size=B).astype(np.float32),
    )
    if with_next_actions:
        batch["next_actions"] = np.clip(rng.normal(size=(B, 5)), -1, 1).astype(np.float32)
    return batch

ex_obs = jnp.zeros((1, 11), jnp.float32); ex_act = jnp.zeros((1, 5), jnp.float32)
"""


def _run(code):
    proc = run_jax(_COMMON + textwrap.dedent(code), cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_default_off_is_inert():
    """REGRESSION GUARD: with centered_measure='off' (the default) the centered path
    must never be entered, the param tree must have NO center_b module, and update()
    must run exactly as stock -- every bridge checkpoint on disk depends on this."""
    _run("""
        assert dict(get_config())["centered_measure"] == "off"
        # booby-trap the new code path: default-off must never call it
        def _boom(self, *a, **k):
            raise RuntimeError("centered_measure_loss called on the default-off path")
        GCIQLFlowAgent.centered_measure_loss = _boom
        ag = GCIQLFlowAgent.create(0, ex_obs, ex_act, make_cfg())
        assert "modules_center_b" not in ag.network.params, \\
            "default-off must not add center_b to the param tree (checkpoint compat)"
        batch = make_batch()
        ag, info = ag.update(batch)
        assert not any(k.startswith("centered/") for k in info), sorted(info)
        for k, v in info.items():
            assert np.isfinite(float(v)), f"{k} not finite: {v}"
        print("OK")
    """)


def test_aux_arm_trains_end_to_end():
    """T10-a: scalar IQL stays on (value/critic losses present) + the centered aux
    trains, with the pre-registered instrumentation logged and finite."""
    _run("""
        cfg = make_cfg(centered_measure="aux", center_b_samples=2, centered_measure_coef=1.0)
        ag = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
        assert "modules_center_b" in ag.network.params
        batch = make_batch()
        for _ in range(3):
            ag, info = ag.update(batch)
        # scalar grounding still on
        assert "value/value_loss" in info and "critic/critic_loss" in info
        # T10 instrumentation (spec risk 3/4 + the T2 early-warning set)
        for k in ("centered/b_loss", "centered/b_fidelity", "centered/dm_loss",
                  "centered/dm_offdiag", "centered/dm_diag", "centered/dm_msq",
                  "centered/phi_norm_sq", "centered/psi_norm_sq",
                  "centered/q_mean", "centered/q_max"):
            assert k in info, (k, sorted(info))
        assert "centered/base_loss" not in info, "aux must have NO base measure leg"
        for k, v in info.items():
            assert np.isfinite(float(v)), f"{k} not finite: {v}"
        a = ag.sample_actions(observations=batch["observations"], goals=batch["actor_goals"],
                              seed=jax.random.PRNGKey(0), temperature=0.0)
        assert np.all(np.isfinite(np.asarray(a)))
        print("OK")
    """)


def test_only_arm_trains_and_drops_scalar_td():
    """T10-b: no scalar TD anywhere -- no value/critic loss keys; base (SARSA) leg +
    DeltaM leg + b regression all present and finite."""
    _run("""
        cfg = make_cfg(centered_measure="only", center_b_samples=2)
        ag = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
        batch = make_batch(with_next_actions=True)
        for _ in range(3):
            ag, info = ag.update(batch)
        assert "value/value_loss" not in info and "critic/critic_loss" not in info, \\
            "T10-b must have NO scalar TD"
        for k in ("centered/base_loss", "centered/base_offdiag", "centered/base_diag",
                  "centered/base_msq", "centered/dm_loss", "centered/b_loss",
                  "centered/b_fidelity", "centered/q_mean", "centered/q_max",
                  "centered/psi_norm_sq"):
            assert k in info, (k, sorted(info))
        for k, v in info.items():
            assert np.isfinite(float(v)), f"{k} not finite: {v}"
        print("OK")
    """)


def test_only_requires_next_actions():
    """T10-b + split bootstrap must refuse a batch without next_actions (the SARSA
    base leg would silently be wrong otherwise)."""
    _run("""
        cfg = make_cfg(centered_measure="only", center_b_samples=2)
        ag = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
        try:
            ag.update(make_batch(with_next_actions=False))
            raise SystemExit("should have refused a batch without next_actions")
        except AssertionError as e:
            assert "next_actions" in str(e)
        print("OK")
    """)


def test_stop_gradients():
    """b(s) is trained ONLY by its regression: with center_b_coef=0 the center_b
    params get ZERO gradient (the DeltaM leg stop-grads b on both TD sides); with
    center_b_coef=1 they get a real one. Under 'only', the value net gets ZERO
    gradient (no scalar TD anywhere)."""
    _run("""
        def total_grad(cfg, batch):
            ag = GCIQLFlowAgent.create(0, ex_obs, ex_act, cfg)
            def f(p):
                return ag.total_loss(batch, p, jax.random.PRNGKey(1))[0]
            return jax.grad(f)(ag.network.params)
        def tree_absmax(t):
            return max(float(jnp.max(jnp.abs(l))) for l in jax.tree_util.tree_leaves(t))

        batch = make_batch(with_next_actions=True)
        g0 = total_grad(make_cfg(centered_measure="aux", center_b_samples=2,
                                 center_b_coef=0.0), batch)
        assert tree_absmax(g0["modules_center_b"]) == 0.0, \\
            "DeltaM leg must send NO gradient into b (b trains only by regression)"
        g1 = total_grad(make_cfg(centered_measure="aux", center_b_samples=2,
                                 center_b_coef=1.0), batch)
        assert tree_absmax(g1["modules_center_b"]) > 0.0, "b regression must train b"
        gv = total_grad(make_cfg(centered_measure="only", center_b_samples=2), batch)
        assert tree_absmax(gv["modules_value"]) == 0.0, \\
            "T10-b: the IQL value net must receive NO gradient"
        assert tree_absmax(gv["modules_critic"]) > 0.0, "measure legs must train the critic"
        print("OK")
    """)


def test_assert_lattice():
    """Invalid compositions must fail loudly at construction."""
    _run("""
        bad = [
            dict(centered_measure="aux", value_arch="mlp"),
            dict(centered_measure="aux", measure_coef=1.0, measure_scale_norm=True),
            dict(centered_measure="only", measure_only=True),
            dict(centered_measure="aux", actor_cond="z"),
            dict(centered_measure="aux", actor_vf_cond="goal"),
            dict(centered_measure="aux", fb_forward_z=True),
            dict(centered_measure="aux", actor_kind="ddpgbc"),
            dict(centered_measure="nope"),
        ]
        for over in bad:
            try:
                GCIQLFlowAgent.create(0, ex_obs, ex_act, make_cfg(**over))
                raise SystemExit(f"lattice failed to refuse {over}")
            except AssertionError:
                pass
        print("OK")
    """)
