"""ogbench-sota value_arch cells: latefusion (item1), phi_norm_cap (item4b),
bilinear_sg_bounded (item5). All opt-in (new archs / default-off knob) => existing
mlp/bilinear configs stay byte-identical. Runs in the jax venv subprocess.
"""
import textwrap
from tests.jax_subprocess import run_jax, OGBENCH_IMPLS

COMMON = """
    import jax, jax.numpy as jnp, numpy as np, flax
    from gciql_flow import GCIQLFlowAgent, get_config

    OBS_D = 17
    EX_OBS = jnp.zeros((1, OBS_D), jnp.float32)
    EX_ACT = jnp.zeros((1, 5), jnp.float32)

    def cfg(**kw):
        c = dict(get_config()); c["encoder"] = None; c["frame_stack"] = None
        c.update(kw); return c

    def batch(B=16, seed=0):
        r = np.random.default_rng(seed)
        return dict(
            observations=r.normal(size=(B, OBS_D)).astype(np.float32),
            next_observations=r.normal(size=(B, OBS_D)).astype(np.float32),
            actions=np.clip(r.normal(size=(B, 5)), -1, 1).astype(np.float32),
            value_goals=r.normal(size=(B, OBS_D)).astype(np.float32),
            actor_goals=r.normal(size=(B, OBS_D)).astype(np.float32),
            rewards=r.integers(0, 2, size=B).astype(np.float32) - 1.0,
            masks=r.integers(0, 2, size=B).astype(np.float32),
        )

    def flat_sd(a):
        sd = flax.serialization.to_state_dict(a.network.params)
        return [(jax.tree_util.keystr(p), np.asarray(l))
                for p, l in jax.tree_util.tree_flatten_with_path(sd)[0]]
"""


def _run(code):
    proc = run_jax(textwrap.dedent(COMMON) + textwrap.dedent(code), cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_new_knobs_default_off_bit_identical_bilinear():
    """A bilinear config that PREDATES the sota knobs and one carrying them at default
    give the identical param tree AND bit-identical update() losses."""
    _run("""
        old = cfg(value_arch="bilinear", value_latent_dim=8)
        for k in ("value_fusion_hidden", "psi_bound_scale", "phi_norm_cap", "value_split_factors"):
            old.pop(k, None)
        new = cfg(value_arch="bilinear", value_latent_dim=8)
        assert new["phi_norm_cap"] is False and float(new["psi_bound_scale"]) == 0.0
        ao = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, old)
        an = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, new)
        fo, fn = flat_sd(ao), flat_sd(an)
        assert [k for k,_ in fo] == [k for k,_ in fn], "param tree changed"
        for (k,lo),(_,ln) in zip(fo,fn):
            assert lo.shape==ln.shape and (lo==ln).all(), f"param {k} differs"
        b = batch()
        for _ in range(3):
            ao, io = ao.update(b); an, ino = an.update(b)
            for k in io: assert float(io[k])==float(ino[k]), (k, float(io[k]), float(ino[k]))
        print("OK")
    """)


def test_item1_latefusion_shapes_goaldep_and_update():
    """latefusion: MLP readout on concat[phi;psi]; twin heads => critic Q is [2,B];
    Q depends on the goal; update() finite; param tree has the fusion head."""
    _run("""
        ag = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, cfg(value_arch="latefusion", value_latent_dim=8))
        assert any("fusion" in k for k,_ in flat_sd(ag)), "no fusion head in the param tree"
        b = batch()
        v, phi, psi = ag.network.select("critic")(
            jnp.asarray(b["observations"]), jnp.asarray(b["value_goals"]),
            jnp.asarray(b["actions"]), info=True)
        assert v.shape == (2, 16) and phi.shape == (2,16,8) and psi.shape == (2,16,8), (v.shape, phi.shape)
        # Q is goal-dependent (the fusion readout mixes psi(g)).
        v2 = ag.network.select("critic")(jnp.asarray(b["observations"]),
             jnp.asarray(batch(seed=9)["value_goals"]), jnp.asarray(b["actions"]))
        assert float(jnp.abs(v - v2).max()) > 1e-4, "latefusion Q must depend on the goal"
        for _ in range(3):
            ag, info = ag.update(b)
        for k,val in info.items(): assert np.isfinite(float(val)), f"{k}={val}"
        print("OK")
    """)


def test_item5_bounded_psi_is_bounded_and_requires_scale():
    """bilinear_sg_bounded: |psi| <= scale everywhere; refuses without psi_bound_scale."""
    _run("""
        SC = 2.0
        ag = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT,
            cfg(value_arch="bilinear_sg_bounded", value_latent_dim=8, psi_bound_scale=SC))
        b = batch()
        _, _, psi = ag.network.select("critic")(jnp.asarray(b["observations"]),
            jnp.asarray(b["value_goals"]), jnp.asarray(b["actions"]), info=True)
        assert float(jnp.abs(psi).max()) <= SC + 1e-4, f"psi exceeded the tanh bound: {float(jnp.abs(psi).max())}"
        for _ in range(3):
            ag, info = ag.update(b)
        for k,val in info.items(): assert np.isfinite(float(val)), f"{k}={val}"
        # refuses without a positive bound.
        try:
            GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, cfg(value_arch="bilinear_sg_bounded", value_latent_dim=8))
        except AssertionError as e:
            assert "psi_bound_scale>0" in str(e), str(e)
        else:
            raise SystemExit("expected refusal without psi_bound_scale")
        print("OK")
    """)


def test_item4b_phi_norm_cap_caps_and_composes():
    """phi_norm_cap on plain bilinear: caps ||phi|| to sg[p99 over batch]; changes the
    loss vs uncapped; refuses with fb_forward_z."""
    _run("""
        capped = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT,
            cfg(value_arch="bilinear", value_latent_dim=8, phi_norm_cap=True))
        plain  = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT,
            cfg(value_arch="bilinear", value_latent_dim=8, phi_norm_cap=False))
        # same param tree (the cap is a forward-pass op, no new params).
        assert [k for k,_ in flat_sd(capped)] == [k for k,_ in flat_sd(plain)]
        b = batch(B=64)
        # capped & plain share identical init (the cap adds no params), so plain's phi is the
        # UNCAPPED phi and its p99 is exactly the internal cap c. Capped max must be <= c.
        oo, gg, aa = (jnp.asarray(b["observations"]), jnp.asarray(b["value_goals"]),
                      jnp.asarray(b["actions"]))
        _, phi_c, _ = capped.network.select("critic")(oo, gg, aa, info=True)
        _, phi_p, _ = plain.network.select("critic")(oo, gg, aa, info=True)
        c = float(np.quantile(np.asarray(jnp.linalg.norm(phi_p, axis=-1)), 0.99))
        assert float(jnp.linalg.norm(phi_c, axis=-1).max()) <= c + 1e-4, \
            f"capped phi norm {float(jnp.linalg.norm(phi_c, axis=-1).max())} exceeds cap {c}"
        # the cap changes training vs uncapped.
        lc, _ = capped.critic_loss(b, capped.network.params)
        lp, _ = plain.critic_loss(b, plain.network.params)
        assert float(lc) != float(lp), "phi_norm_cap must change the critic loss"
        for _ in range(2):
            capped, info = capped.update(b)
        for k,val in info.items(): assert np.isfinite(float(val)), f"{k}={val}"
        # refuses with fb_forward_z.
        try:
            GCIQLFlowAgent.create(0, EX_OBS, EX_ACT,
                cfg(value_arch="bilinear", value_latent_dim=8, phi_norm_cap=True, fb_forward_z=True))
        except AssertionError as e:
            assert "incompatible with fb_forward_z" in str(e), str(e)
        else:
            raise SystemExit("expected refusal phi_norm_cap+fb_forward_z")
        print("OK")
    """)


def test_item7_bilinear_split_factored():
    """bilinear_split: K per-factor goal towers; Q depends on each factor; K towers in the
    param tree; update() finite; explicit value_split_dims must sum to goal_dim."""
    _run("""
        ag = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT,
            cfg(value_arch="bilinear_split", value_latent_dim=4, value_split_factors=3))
        # 3 psi towers (psi_0/psi_1/psi_2) present.
        keys = [k for k,_ in flat_sd(ag)]
        for t in ("psi_0", "psi_1", "psi_2"):
            assert any(t in k for k in keys), f"missing tower {t}"
        b = batch()
        v, phi, psi = ag.network.select("critic")(jnp.asarray(b["observations"]),
            jnp.asarray(b["value_goals"]), jnp.asarray(b["actions"]), info=True)
        assert v.shape == (2, 16), v.shape
        # goal-dependent.
        v2 = ag.network.select("critic")(jnp.asarray(b["observations"]),
            jnp.asarray(batch(seed=7)["value_goals"]), jnp.asarray(b["actions"]))
        assert float(jnp.abs(v - v2).max()) > 1e-4
        for _ in range(3):
            ag, info = ag.update(b)
        for k,val in info.items(): assert np.isfinite(float(val)), f"{k}={val}"
        # explicit split dims (comma-string) must sum to goal_dim (17).
        ag2 = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT,
            cfg(value_arch="bilinear_split", value_latent_dim=4, value_split_dims="8,9"))
        assert any("psi_1" in k for k,_ in flat_sd(ag2)) and not any("psi_2" in k for k,_ in flat_sd(ag2)), \
            "string split_dims '8,9' should make exactly 2 towers"
        try:
            GCIQLFlowAgent.create(0, EX_OBS, EX_ACT,
                cfg(value_arch="bilinear_split", value_latent_dim=4, value_split_dims="5,5"))
        except AssertionError as e:
            assert "must sum to goal_dim" in str(e), str(e)
        else:
            raise SystemExit("expected refusal for bad value_split_dims")
        print("OK")
    """)


def test_bilinear_mlpv_split_v_and_q():
    """bilinear_mlpv: MLP V (no psi tower) + bilinear Q (phi/psi). V goal-dependent via MLP,
    Q via the inner product; update finite; refuses actor_cond='z' (MLP V has no psi=z)."""
    _run("""
        ag = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, cfg(value_arch="bilinear_mlpv", value_latent_dim=8))
        keys = [k for k,_ in flat_sd(ag)]
        # critic has bilinear phi/psi; value net (MLP) has NO psi tower.
        assert any("modules_critic" in k and "psi" in k for k in keys), "critic missing bilinear psi"
        assert not any("modules_value" in k and "psi" in k for k in keys), "value should be plain MLP (no psi)"
        b = batch()
        # bilinear Q is ensembled [2,B]; MLP V is [B].
        v = ag.network.select("value")(jnp.asarray(b["observations"]), jnp.asarray(b["value_goals"]))
        q = ag.network.select("critic")(jnp.asarray(b["observations"]), jnp.asarray(b["value_goals"]),
                                        jnp.asarray(b["actions"]))
        assert v.shape == (16,) and q.shape == (2,16), (v.shape, q.shape)
        for _ in range(3):
            ag, info = ag.update(b)
        for k,val in info.items(): assert np.isfinite(float(val)), f"{k}={val}"
        # refuses z-conditioning (no psi=z on the MLP V).
        try:
            GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, cfg(value_arch="bilinear_mlpv", value_latent_dim=8, actor_cond="z"))
        except AssertionError as e:
            assert "bilinear" in str(e), str(e)
        else:
            raise SystemExit("expected refusal for bilinear_mlpv + actor_cond='z'")
        print("OK")
    """)


def test_occupancy_shared_B_update_and_lattice():
    """occupancy: ONE shared B (F_V/F_crit/B in the critic, NO separate value module); V and Q
    both read it; occupancy reward + whitening logged; update finite; lattice refuses."""
    _run("""
        ag = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, cfg(value_arch="occupancy", value_latent_dim=8))
        keys = [k for k,_ in flat_sd(ag)]
        assert not any("modules_value" in k for k in keys), "occupancy must have NO separate value module (B is shared in the critic)"
        for tow in ("F_V", "F_crit", "B"):
            assert any(("modules_critic" in k) and (tow in k) for k in keys), f"missing {tow}"
        b = batch()
        v = ag.network.select("critic")(jnp.asarray(b["observations"]), jnp.asarray(b["value_goals"]))
        q = ag.network.select("critic")(jnp.asarray(b["observations"]), jnp.asarray(b["value_goals"]),
                                        jnp.asarray(b["actions"]))
        assert v.shape == (16,) and q.shape == (2,16), (v.shape, q.shape)   # V single head, Q twin
        v2 = ag.network.select("critic")(jnp.asarray(b["observations"]), jnp.asarray(batch(seed=9)["value_goals"]))
        assert float(jnp.abs(v - v2).max()) > 1e-5, "V must depend on the goal (shared B)"
        for _ in range(3):
            ag, info = ag.update(b)
        assert any("occ_ortho" in k for k in info) and any("occ_reward" in k for k in info), \
            f"occupancy critic info missing: {list(info)}"
        for k,val in info.items(): assert np.isfinite(float(val)), f"{k}={val}"
        # lattice.
        try:
            GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, cfg(value_arch="occupancy", value_latent_dim=8, measure_coef=1.0))
        except AssertionError as e:
            assert "incompatible with measure_coef" in str(e), str(e)
        else:
            raise SystemExit("expected occupancy+measure_coef refusal")
        print("OK")
    """)


def test_assert_lattice_latefusion_bounded():
    """new archs refuse the bilinear z-task paths."""
    _run("""
        def refuses(frag, **kw):
            try: GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, cfg(value_latent_dim=8, **kw))
            except AssertionError as e:
                assert frag in str(e), (frag, str(e)); return
            raise SystemExit(f"expected refusal ({frag}) for {kw}")
        # actor_cond='z' hits the earlier "requires value_arch='bilinear'" guard for both archs.
        refuses("requires value_arch='bilinear'", value_arch="latefusion", actor_cond="z")
        refuses("incompatible with the measure", value_arch="latefusion", measure_coef=1.0)
        refuses("requires value_arch='bilinear'", value_arch="bilinear_sg_bounded",
                psi_bound_scale=1.0, actor_cond="z")
        print("OK")
    """)
