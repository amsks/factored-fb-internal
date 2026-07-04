"""Scalar-gated in-sample Forward-Backward (SG-ISFB) agent -- PAPER/fb-modified/isfb.tex.

ISFB imports IQL's in-sample improvement into the FB successor measure:
  * action measure  M_ij = <F(s_i,a_i,z_i), B(s'_j)>/sqrt(d)
  * value measure   N_ij = <F_V(s_i,z_i),  B(s'_j)>/sqrt(d)   (shares B=psi with M)
  * scalar gate     q_i=<F,z>/sqrt d, v_i=<F_V,z>/sqrt d; V trained by IQL expectile,
                    advantage A_i=sg(q_i-v_i) -> weights w_i (hard expectile or AWR)
  * weighted vector regression  N_ij -> M_ij (weight w_i)  [coherent: convex combo of rows]
  * measure TD      M_ij bootstraps through gamma*N^-(s'_i, s'_j)  (no a+~pi)

Runs in the jax venv subprocess (mirrors tests/test_gciql_two_tower_critic.py).
"""
from tests.jax_subprocess import run_jax, OGBENCH_IMPLS


def test_isfb_module_returns_M_N_shared_B():
    """GCISFBValue exposes phi=F(s,a,z), fv=F_V(s,z), psi=B(g), z -- so M and N share B."""
    code = """
        import jax, jax.numpy as jnp
        from isfb_flow import GCISFBValue
        m = GCISFBValue(hidden_dims=(32, 32), latent_dim=8, layer_norm=True, ensemble=True)
        obs = jnp.ones((4, 11)); acts = jnp.ones((4, 5))
        bg = jnp.ones((4, 11)); zg = jnp.zeros((4, 11))
        params = m.init(jax.random.PRNGKey(0), obs, bg, acts, zg)['params']
        v, phi, psi, fv, z = m.apply({'params': params}, obs, bg, acts, zg, info=True, return_fv=True)
        assert phi.shape == (2, 4, 8), phi.shape
        assert fv.shape == (2, 4, 8), fv.shape
        assert psi.shape == (2, 4, 8), psi.shape
        # forward map F depends on the action; value map F_V does NOT (state/z only).
        v2, phi2, psi2, fv2, z2 = m.apply({'params': params}, obs, bg, acts * 0.0, zg, info=True, return_fv=True)
        assert not jnp.allclose(phi, phi2), "F(s,a,z) must depend on a"
        assert jnp.allclose(fv, fv2), "F_V(s,z) must NOT depend on a"
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_isfb_scalar_gate_hard_trains_finite():
    """SG-ISFB with hard expectile weights: L_V, L_N, L_M all finite over a few steps."""
    code = """
        import jax, jax.numpy as jnp
        from isfb_flow import ISFBAgent, get_config
        cfg = dict(get_config())
        cfg["encoder"] = None; cfg["frame_stack"] = None
        cfg["value_latent_dim"] = 8
        cfg["measure_insample"] = "scalar_gate_hard"
        ex_obs = jnp.zeros((4, 11)); ex_act = jnp.zeros((4, 5))
        agent = ISFBAgent.create(0, ex_obs, ex_act, cfg)
        k = jax.random.PRNGKey(0)   # realistic batch: a successor-measure objective needs
        o = jax.random.normal(k, (8, 11))                       # DISTINCT successors, else the
        no = jax.random.normal(jax.random.fold_in(k, 1), (8, 11))  # measure matrix is degenerate
        a = jax.random.normal(jax.random.fold_in(k, 2), (8, 5))
        batch = dict(observations=o, next_observations=no, actions=a,
                     rewards=jnp.zeros((8,)), masks=jnp.ones((8,)),
                     value_goals=no, actor_goals=no)
        for _ in range(5):
            agent, info = agent.update(batch)
        for k in ("isfb/L_V", "isfb/L_N", "isfb/L_M", "actor/actor_loss"):
            assert k in info, list(info)
            assert jnp.isfinite(info[k]), (k, info[k])
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_isfb_scalar_gate_awr_trains_finite():
    """SG-ISFB with AWR (exp-advantage) weights trains finite."""
    code = """
        import jax, jax.numpy as jnp
        from isfb_flow import ISFBAgent, get_config
        cfg = dict(get_config())
        cfg["encoder"] = None; cfg["frame_stack"] = None
        cfg["value_latent_dim"] = 8
        cfg["measure_insample"] = "scalar_gate_awr"; cfg["awr_beta"] = 3.0
        ex_obs = jnp.zeros((4, 11)); ex_act = jnp.zeros((4, 5))
        agent = ISFBAgent.create(0, ex_obs, ex_act, cfg)
        k = jax.random.PRNGKey(0)   # realistic batch: a successor-measure objective needs
        o = jax.random.normal(k, (8, 11))                       # DISTINCT successors, else the
        no = jax.random.normal(jax.random.fold_in(k, 1), (8, 11))  # measure matrix is degenerate
        a = jax.random.normal(jax.random.fold_in(k, 2), (8, 5))
        batch = dict(observations=o, next_observations=no, actions=a,
                     rewards=jnp.zeros((8,)), masks=jnp.ones((8,)),
                     value_goals=no, actor_goals=no)
        for _ in range(5):
            agent, info = agent.update(batch)
        assert jnp.isfinite(info["isfb/L_N"]), info["isfb/L_N"]
        assert jnp.isfinite(info["actor/actor_loss"]), info["actor/actor_loss"]
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_isfb_coord_baseline_trains_finite():
    """The coordinatewise expectile baseline (Claude's original IS-FB) also trains finite."""
    code = """
        import jax, jax.numpy as jnp
        from isfb_flow import ISFBAgent, get_config
        cfg = dict(get_config())
        cfg["encoder"] = None; cfg["frame_stack"] = None
        cfg["value_latent_dim"] = 8
        cfg["measure_insample"] = "coord"
        ex_obs = jnp.zeros((4, 11)); ex_act = jnp.zeros((4, 5))
        agent = ISFBAgent.create(0, ex_obs, ex_act, cfg)
        k = jax.random.PRNGKey(0)   # realistic batch: a successor-measure objective needs
        o = jax.random.normal(k, (8, 11))                       # DISTINCT successors, else the
        no = jax.random.normal(jax.random.fold_in(k, 1), (8, 11))  # measure matrix is degenerate
        a = jax.random.normal(jax.random.fold_in(k, 2), (8, 5))
        batch = dict(observations=o, next_observations=no, actions=a,
                     rewards=jnp.zeros((8,)), masks=jnp.ones((8,)),
                     value_goals=no, actor_goals=no)
        for _ in range(5):
            agent, info = agent.update(batch)
        assert jnp.isfinite(info["isfb/L_N"]), info["isfb/L_N"]
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_isfb_measure_td_respects_masks():
    """Delta = M - gamma*mask*N^-: masking the bootstrap (episodic goal-termination,
    consistent with GCIQL's masked value TD) must change L_M vs unmasked."""
    code = """
        import jax, jax.numpy as jnp
        from isfb_flow import ISFBAgent, get_config
        cfg = dict(get_config()); cfg["encoder"]=None; cfg["frame_stack"]=None; cfg["value_latent_dim"]=8
        ex_obs=jnp.zeros((4,11)); ex_act=jnp.zeros((4,5))
        agent=ISFBAgent.create(0, ex_obs, ex_act, cfg)
        k=jax.random.PRNGKey(0)
        o=jax.random.normal(k,(8,11)); no=jax.random.normal(jax.random.fold_in(k,1),(8,11))
        a=jax.random.normal(jax.random.fold_in(k,2),(8,5))
        base=dict(observations=o,next_observations=no,actions=a,rewards=jnp.zeros((8,)),
                  value_goals=no,actor_goals=no)
        _, i1 = agent.isfb_loss(dict(base, masks=jnp.ones((8,))),  agent.network.params, k)
        _, i0 = agent.isfb_loss(dict(base, masks=jnp.zeros((8,))), agent.network.params, k)
        assert not jnp.allclose(i1["L_M"], i0["L_M"]), (i1["L_M"], i0["L_M"])
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_isfb_sample_actions_shapes():
    """sample_actions (flow actor, z-conditioned) returns action-shaped output, batched + single."""
    code = """
        import jax, jax.numpy as jnp
        from isfb_flow import ISFBAgent, get_config
        cfg = dict(get_config())
        cfg["encoder"] = None; cfg["frame_stack"] = None
        cfg["value_latent_dim"] = 8
        ex_obs = jnp.zeros((4, 11)); ex_act = jnp.zeros((4, 5))
        agent = ISFBAgent.create(0, ex_obs, ex_act, cfg)
        a = agent.sample_actions(ex_obs[0], ex_obs[0], seed=jax.random.PRNGKey(0))
        assert a.shape == (5,), a.shape
        print("OK")
    """
    proc = run_jax(code, cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout
