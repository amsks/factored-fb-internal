"""warm_start_ckpt (2026-07-21): splice a trained scalar-Q checkpoint into a measure
cell, keep it TRAINABLE, fresh-init the measure-only towers. Tests the mechanism the
warm-start experiment needs (does the Q basin survive the measure objective).

obs_dim = 29 (the measure cell reads cube xyz at obs[19:22]). Jax venv subprocess.
"""
import textwrap
from tests.jax_subprocess import run_jax, OGBENCH_IMPLS

COMMON = """
    import jax, jax.numpy as jnp, numpy as np, flax, tempfile, pickle, os
    from gciql_flow import GCIQLFlowAgent, get_config
    OBS_D = 29
    EX_OBS = jnp.zeros((1, OBS_D), jnp.float32); EX_ACT = jnp.zeros((1, 5), jnp.float32)
    def cfg(**kw):
        c = dict(get_config()); c["encoder"]=None; c["frame_stack"]=None
        c["value_arch"]="bilinear"; c["value_latent_dim"]=8; c.update(kw); return c
    def make_batch(B=16, seed=0):
        r=np.random.default_rng(seed)
        return dict(observations=r.normal(size=(B,OBS_D)).astype(np.float32),
            next_observations=r.normal(size=(B,OBS_D)).astype(np.float32),
            actions=np.clip(r.normal(size=(B,5)),-1,1).astype(np.float32),
            value_goals=r.normal(size=(B,OBS_D)).astype(np.float32),
            actor_goals=r.normal(size=(B,OBS_D)).astype(np.float32),
            rewards=r.integers(0,2,size=B).astype(np.float32)-1.0,
            masks=r.integers(0,2,size=B).astype(np.float32))
    def flat_sd(a):
        sd=flax.serialization.to_state_dict(a.network.params)
        return {jax.tree_util.keystr(p): np.asarray(l)
                for p,l in jax.tree_util.tree_flatten_with_path(sd)[0]}
"""


def _run(code):
    proc = run_jax(textwrap.dedent(COMMON) + textwrap.dedent(code), cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_warm_start_splices_trainable_and_fresh_towers():
    _run("""
        # source = a trained-ish plain bilinear (the scalar Q basin).
        src = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, cfg())
        b = make_batch()
        for _ in range(3): src, _ = src.update(b)
        tmp = tempfile.mkdtemp()
        with open(os.path.join(tmp, "params_0.pkl"), "wb") as f:
            pickle.dump({"agent": {"network": {"params": src.network.params}}}, f)

        # warm target = a first-passage + expectile measure cell (measure_v_phi tower is extra).
        warm_cfg = cfg(measure_only=True, measure_coef=1.0, measure_legs="critic", measure_first_passage=True,
                       measure_expectile=True, measure_ortho_coef=1.0,
                       warm_start_ckpt=tmp, warm_start_epoch=0)
        warm = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, warm_cfg)
        # cold control (no warm-start) — its critic must DIFFER from the source.
        cold = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT,
                cfg(measure_only=True, measure_legs="critic", measure_first_passage=True,
                    measure_expectile=True, measure_ortho_coef=1.0))

        sv, wv, cv = flat_sd(src), flat_sd(warm), flat_sd(cold)
        crit_keys = [k for k in wv if "modules_critic" in k or "modules_value" in k]
        assert crit_keys, "no value/critic keys?"
        # (a) warm critic/value == source (spliced); cold != source (fresh).
        for k in crit_keys:
            assert k in sv and np.array_equal(sv[k], wv[k]), f"warm not spliced at {k}"
        assert any(not np.array_equal(sv[k], cv[k]) for k in crit_keys), "cold unexpectedly == source"
        # (b) measure_v_phi present in warm, fresh (source had none).
        assert any("measure_v_phi" in k for k in wv), "measure_v_phi missing"
        assert not any("measure_v_phi" in k for k in sv), "source unexpectedly had measure_v_phi"
        # (c) trainable: an update MOVES the spliced critic (not frozen).
        p0 = {k: v.copy() for k, v in wv.items() if "modules_critic" in k}
        warm2, info = warm.update(make_batch(seed=1))
        p1 = {k: v for k, v in flat_sd(warm2).items() if "modules_critic" in k}
        assert any(not np.array_equal(p0[k], p1[k]) for k in p0), \
            "warm-start critic did not move — must stay TRAINABLE"
        for k, v in info.items(): assert np.isfinite(float(v)), f"{k}={v}"
        print("OK")
    """)


def test_warm_start_default_off_and_lattice():
    _run("""
        # default off => no warm_start knobs touched, plain build works.
        a = GCIQLFlowAgent.create(0, EX_OBS, EX_ACT, cfg())
        assert a.config["warm_start_ckpt"] == "" and int(a.config["warm_start_epoch"]) == 0
        # mutual exclusion with freeze_critic.
        try:
            GCIQLFlowAgent.create(0, EX_OBS, EX_ACT,
                cfg(warm_start_ckpt="/tmp/nope", freeze_critic_ckpt="/tmp/nope"))
        except AssertionError as e:
            assert "mutually exclusive" in str(e), str(e)
        else:
            raise SystemExit("expected mutual-exclusion refusal")
        print("OK")
    """)
