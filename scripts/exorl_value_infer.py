"""Model-based (zero-shot) task-vector inference for an ExoRL Density-FB checkpoint.

The paper protocol infers the task vector z by a ridge regression of reward samples on the
centered backward features (`critic.infer_cond`). That readout can only express rewards that
are LINEAR in B(u); ExoRL's speed-threshold rewards are not, and an oracle search over z
scored by environment rollouts (scripts/exorl_zsearch.py) finds better task vectors than the
ridge z on cheetah. This script replaces the ridge with a search that never touches the
environment: the critic itself evaluates ANY reward exactly on a pool of reward-labelled
next states (paper Eq. density-value),

    q_r(s, a, z) = min_h  sum_j  p_h(u_j | s, a, z) r(u_j),
    p_h(. | s, a, z) = softmax_j( F_h(s, a, z) . B(u_j) / tau ),

so the objective  J(z) = mean_{s in S} q_r(s, pi(s, z), z)  is the model's own estimate of
how good the frozen actor is at z under the TRUE task reward. J is maximised on the sqrt(d)
sphere by (a) cross-entropy search seeded at the ridge z and (b) Adam ascent through the
actor and the F tower. Both are then scored on the paper protocol (10 episodes,
InferredCondAgent) next to the ridge z, and next to the oracle z from exp/zsearch/*.jsonl
when one exists for this (seed, task).

The readout is built from the critic's OWN tower calls (`_F`, `_B`, `_b_query`, `temp`), not
a re-implementation; `--check` asserts that with the latent proxy reward r_z(u) = (B_zenc(u) -
mu)^T z / sqrt(d) it reproduces `q_fn` exactly before the true rewards are swapped in.

    CUDA_VISIBLE_DEVICES=6 /dev/shm/.venv-jax/bin/python -m scripts.exorl_value_infer \
        --restore_path exp/exorl_sqt/dmc-cheetah/dmc-cheetah_density_fb_z_ddpgbc_seed_2 \
        --restore_epoch 1000000 --seed 2 --override exorl_task=walk \
        --out_jsonl exp/zsearch/value_infer.jsonl
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from absl import app, flags
import jax
import jax.numpy as jnp
import numpy as np
import optax

from impls.agents import Agent
from impls.utils.datasets import make_dataset
from env import get_adapter, make_config
from impls.utils.flax_utils import resolve_run_dir, restore_agent
from impls.critics.base import infer_cond_rows
from impls.critics.density_fb import _raw_block_cols
from impls.utils.eval_cond import InferredCondAgent


def _recorded_config_overrides(run_dir):
    """Copy of scripts/exorl_zsearch.py::_recorded_config_overrides (importing that module
    would define its absl flags a second time)."""
    path = os.path.join(resolve_run_dir(run_dir), "config.json")
    if not os.path.isfile(path):
        return []
    with open(path) as f:
        recorded = json.load(f).get("config", {})
    skip = {"seed", "save_dir", "restore_path", "restore_epoch", "train_steps",
            "eval_episodes", "eval_interval", "save_interval", "log_interval",
            "exorl_task", "wandb", "run_group", "online"}
    out = []

    def walk(prefix, node):
        for k, v in node.items():
            if not prefix and k in skip:
                continue
            key = f"{prefix}{k}"
            if isinstance(v, dict):
                walk(f"{key}.", v)
            elif isinstance(v, str):
                out.append(f'{key}="{v}"')
            else:
                out.append(f"{key}={json.dumps(v)}")

    walk("", recorded)
    return out


def sphere(z, d):
    z = np.asarray(z, np.float64)
    return z * np.sqrt(d) / max(np.linalg.norm(z), 1e-12)


FLAGS = flags.FLAGS
flags.DEFINE_string("critic", "density_fb_z", "")
flags.DEFINE_string("actor", "ddpgbc", "")
flags.DEFINE_string("domain", "dmc-cheetah", "")
flags.DEFINE_integer("seed", 0, "")
flags.DEFINE_string("restore_path", None, "run dir")
flags.DEFINE_integer("restore_epoch", 1000000, "")
flags.DEFINE_multi_string("override", [], "dotlist overrides, CLI wins over the recorded config")
flags.DEFINE_integer("infer_cond_episodes", 50000, "reward-labelled rows (the ridge z's input)")
flags.DEFINE_integer("pool", 4096, "P: candidate pool for the value readout, drawn from the rows")
flags.DEFINE_integer("n_states", 512, "|S|: dataset observations the objective averages over")
flags.DEFINE_integer("eval_episodes", 10, "paper-protocol episodes per scored z")
flags.DEFINE_integer("pop", 64, "candidates per CEM round")
flags.DEFINE_integer("rounds", 4, "CEM rounds")
flags.DEFINE_integer("elite", 8, "elites per round")
flags.DEFINE_float("sigma0", 0.6, "initial per-coordinate CEM noise (unit-sphere scale)")
flags.DEFINE_float("sigma_min", 0.15, "CEM noise in the last round")
flags.DEFINE_integer("n_random", 16, "uniform sphere draws mixed into round 1")
flags.DEFINE_integer("grad_steps", 200, "Adam ascent steps from the ridge z (0 = skip)")
flags.DEFINE_float("grad_lr", 0.05, "Adam learning rate on the pre-projection vector")
flags.DEFINE_bool("check", True, "assert the readout reproduces q_fn under the latent proxy reward")
flags.DEFINE_bool("skip_env", False, "no environment rollouts (objective and search only)")
flags.DEFINE_string("oracle_jsonl", "exp/zsearch/cheetah.jsonl",
                    "exorl_zsearch rows; the matching (seed, task) z_best is scored by J too")
flags.DEFINE_string("tag", "", "")
flags.DEFINE_string("out_jsonl", "", "append the result row here")
flags.DEFINE_integer("rng_seed", 0, "search / subsample rng")


def build_objective(agent, pool_obs, pool_r, states):
    """-> (J, q_at) with J(z) = mean_s q_r(s, pi(s, z), z) on the sqrt(d) sphere, both jitted.

    Uses the critic's own tower calls exactly as `q_fn` does: online F at (s, a, z), online B
    on the eps-scaled query rows of the pool, softmax at `temp` (or `actor_temp`), the two
    heads combined by min AFTER the reward integration."""
    critic, actor = agent.critic, agent.actor
    net = agent.net
    f_p = net["density_f"].params
    b_p = net["density_b"].params
    a_p = net["actor"].params
    u_sc = critic._b_query(net, pool_obs, pool_obs)                    # [P, d_c]
    b_emb = jax.lax.stop_gradient(critic._B(b_p, u_sc))                # [P, d]
    tau = critic.temp if critic.actor_temp is None else float(critic.actor_temp)
    d = int(critic.z_dim)
    sobs = critic._sobs(net, states)

    def q_at(z, r):
        """q_r(s, pi(s,z), z) for every s in S, one z [d] and one reward vector r [P]."""
        zt = jnp.broadcast_to(z[None], (states.shape[0], d))
        acts = jnp.clip(actor._dist(a_p, sobs, zt).mode(), -1, 1)      # the eval-time action
        f_emb = critic._F(f_p, sobs, acts, zt)                          # [2, n, d]
        w = jax.nn.softmax(jnp.einsum('hnd,pd->hnp', f_emb, b_emb) / tau, -1)
        q_h = jnp.einsum('hnp,p->hn', w, r)                            # [2, n]
        if critic.actor_head_reduce == "min":
            return jnp.minimum(q_h[0], q_h[1])
        return 0.5 * (q_h[0] + q_h[1])

    def J(z):
        return jnp.mean(q_at(z, pool_r))

    # the un-jitted readout goes to the parity check so a precision context applies to it
    return jax.jit(J), q_at, u_sc


def check_against_q_fn(agent, pool_obs, pool_r, u_sc, states, z):
    """The readout with the LATENT proxy reward must equal the critic's q_fn (which reads
    that proxy under measure_density_z_actor_reward='latent_centered')."""
    critic, net = agent.critic, agent.net
    d = int(critic.z_dim)
    st = critic._zstats(net)
    zenc_u = jax.lax.stop_gradient(critic._B(critic._zenc(net), u_sc) - st["mu"][None])
    r_lat = zenc_u @ z / jnp.sqrt(float(d))                              # [P]
    n = states.shape[0]
    zt = jnp.broadcast_to(z[None], (n, d))
    cond = {"z": zt, "_density_u_raw": _raw_block_cols(pool_obs, critic.blocks),
            "_density_u_sc": u_sc, "_density_zenc_u": zenc_u,
            "_density_tb_u": jax.lax.stop_gradient(critic._B(agent.target["density_b"], u_sc)),
            "_z_hindsight": jnp.ones((n, 1))}
    params = {g: net[g].params for g in net}
    if critic.z_actor_reward != "latent_centered":
        raise ValueError(f"the parity check assumes 'latent_centered'; got {critic.z_actor_reward!r}")
    # exact f32 matmuls: on Ampere+ the default GPU precision is TF32, and two einsums with
    # different contraction orders then disagree at ~1e-3 relative, which would hide a bug.
    with jax.default_matmul_precision("highest"):
        # rebuilt INSIDE the context: build_objective embeds the pool once, and q_fn
        # re-embeds it, so both must run at the same precision to be comparable.
        _, q_at, _ = build_objective(agent, pool_obs, pool_r, states)
        acts = jnp.clip(agent.actor._dist(net["actor"].params, critic._sobs(net, states), zt).mode(), -1, 1)
        q_ref, _ = critic.q_fn(params, states, cond, acts)
        q_mine = q_at(z, r_lat)
    err = float(jnp.max(jnp.abs(q_ref - q_mine)))
    scale = float(jnp.max(jnp.abs(q_ref)))
    print(f"[value_infer] parity vs q_fn (latent proxy): max|dq| {err:.3e} on |q| <= {scale:.3e}", flush=True)
    assert err <= 1e-4 * max(scale, 1e-6) + 1e-6, "readout does not reproduce q_fn"
    return err


def cem(J, z0, d, rng, pop, rounds, elite, sigma0, sigma_min, n_random):
    """CEM on the sqrt(d) sphere, seeded at z0, scored by J only. -> (best_z, best_J, history)."""
    mean, sigma = z0.copy(), sigma0
    decay = (sigma_min / sigma0) ** (1.0 / max(rounds - 1, 1))
    best_z, best_j, hist = z0.copy(), float(J(jnp.asarray(z0))), []
    for rnd in range(rounds):
        cands = [sphere(mean + sigma * rng.normal(size=d), d) for _ in range(pop)]
        if rnd == 0:
            cands += [sphere(rng.normal(size=d), d) for _ in range(n_random)]
        js = np.array([float(J(jnp.asarray(z))) for z in cands])
        k = int(np.argmax(js))
        if js[k] > best_j:
            best_j, best_z = float(js[k]), cands[k].copy()
        order = np.argsort(js)[::-1]
        mean = sphere(np.stack([cands[i] for i in order[:elite]]).mean(0), d)
        hist.append(dict(round=rnd, best=best_j, round_max=float(js.max()), round_mean=float(js.mean()),
                         elite_mean=float(js[order[:elite]].mean()), sigma=sigma,
                         cos_best_to_seed=float(best_z @ z0 / (np.linalg.norm(best_z) * np.linalg.norm(z0)))))
        print(f"[value_infer] CEM round {rnd}: max {js.max():.4f} mean {js.mean():.4f} "
              f"best {best_j:.4f} sigma {sigma:.2f}", flush=True)
        sigma = max(sigma_min, sigma * decay)
    return best_z, best_j, hist


def grad_ascent(J, z0, d, steps, lr):
    """Adam ascent of J through the actor and F, on v with z = sqrt(d) v / ||v||.
    Returns the best z seen along the trajectory."""
    def obj(v):
        z = v * jnp.sqrt(float(d)) / jnp.linalg.norm(v)
        return J(z)
    grad = jax.jit(jax.value_and_grad(obj))
    opt = optax.adam(lr)
    v = jnp.asarray(z0)
    state = opt.init(v)
    best_v, best_j, hist = v, float(obj(v)), []
    for t in range(steps):
        j, g = grad(v)
        upd, state = opt.update(-g, state, v)
        v = optax.apply_updates(v, upd)
        jt = float(obj(v))
        if jt > best_j:
            best_j, best_v = jt, v
        if t % 25 == 0 or t == steps - 1:
            hist.append(dict(step=t, J=jt, best=best_j))
            print(f"[value_infer] grad step {t}: J {jt:.4f} best {best_j:.4f}", flush=True)
    z = np.asarray(best_v, np.float64)
    return sphere(z, d), best_j, hist


def main(_):
    t0 = time.monotonic()
    base = _recorded_config_overrides(FLAGS.restore_path)
    cfg = make_config(FLAGS.critic, FLAGS.actor, FLAGS.domain, base + list(FLAGS.override))
    np.random.seed(FLAGS.seed)
    adapter = get_adapter(cfg)
    env, train_ds, _ = adapter.make_env_and_datasets(cfg)
    train_ds = make_dataset(train_ds, cfg)
    ex = train_ds.sample(1)
    agent = Agent.create(FLAGS.critic, FLAGS.actor, FLAGS.seed,
                         jnp.asarray(ex["observations"]), jnp.asarray(ex["actions"]), cfg)
    agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch, params_only=True)
    task = cfg.get("exorl_task")

    # 1. the paper-protocol ridge z, from exactly the rows infer_cond gets (same draw order
    #    as scripts/exorl_zsearch.py, so z_inf matches its z_inferred)
    rb = train_ds.sample(FLAGS.infer_cond_episodes)
    rows = infer_cond_rows(agent.critic)
    inferred = agent.infer_eval_cond(jnp.asarray(rb[rows]), jnp.asarray(rb["rewards"]))
    z_inf = np.asarray(inferred.cond_eval["z"], np.float64)
    d = z_inf.shape[-1]
    r_all = np.asarray(rb["rewards"], np.float32).reshape(-1)
    assert r_all.shape[0] == rb[rows].shape[0], (r_all.shape, rb[rows].shape)

    # 2. the model-based objective: pool U (P of the reward rows) and state set S
    rng = np.random.default_rng(FLAGS.rng_seed)
    pi = rng.choice(r_all.shape[0], size=min(FLAGS.pool, r_all.shape[0]), replace=False)
    pool_obs = jnp.asarray(np.asarray(rb[rows], np.float32)[pi])
    pool_r = jnp.asarray(r_all[pi])
    sb = train_ds.sample(FLAGS.n_states)
    states = jnp.asarray(np.asarray(sb["observations"], np.float32))
    J, q_at, u_sc = build_objective(agent, pool_obs, pool_r, states)
    print(f"[value_infer] task {task} seed {FLAGS.seed}: P={pool_obs.shape[0]} |S|={states.shape[0]} "
          f"pool reward mean {float(pool_r.mean()):.4f} frac>0 {float((pool_r > 0).mean()):.3f} "
          f"(all rows: mean {r_all.mean():.4f}); ridge |z|={np.linalg.norm(z_inf):.3f} d={d}", flush=True)
    parity = check_against_q_fn(agent, pool_obs, pool_r, u_sc, states, jnp.asarray(z_inf)) if FLAGS.check else None

    j_inf = float(J(jnp.asarray(z_inf)))
    j_neutral = float(J(jnp.asarray(sphere(np.ones(d), d))))
    print(f"[value_infer] J(ridge z) = {j_inf:.4f}   J(1_d) = {j_neutral:.4f}", flush=True)

    # oracle z from the env-scored search, if one exists for this (seed, task)
    oracle = None
    if FLAGS.oracle_jsonl and os.path.isfile(FLAGS.oracle_jsonl):
        for line in open(FLAGS.oracle_jsonl):
            r = json.loads(line)
            if int(r["seed"]) == FLAGS.seed and r.get("exorl_task") == task \
                    and r.get("restore_path", "").rstrip("/") == FLAGS.restore_path.rstrip("/"):
                zo = sphere(np.asarray(r["z_best"]), d)
                oracle = dict(J=float(J(jnp.asarray(zo))), return_env=float(r["return_best"]),
                              return_inferred_env=float(r["return_inferred"]),
                              cos_to_ridge=float(zo @ z_inf / (np.linalg.norm(zo) * np.linalg.norm(z_inf))))
                print(f"[value_infer] oracle z (env-searched): J = {oracle['J']:.4f}, "
                      f"env return {oracle['return_env']:.1f}", flush=True)

    # 3a. CEM on the sphere scored by J only
    t1 = time.monotonic()
    z_cem, j_cem, cem_hist = cem(J, z_inf, d, rng, FLAGS.pop, FLAGS.rounds, FLAGS.elite,
                                 FLAGS.sigma0, FLAGS.sigma_min, FLAGS.n_random)
    t_cem = time.monotonic() - t1
    cos_cem = float(z_cem @ z_inf / (np.linalg.norm(z_cem) * np.linalg.norm(z_inf)))
    print(f"[value_infer] CEM: J {j_inf:.4f} -> {j_cem:.4f}, cos(z_cem, ridge) {cos_cem:.3f} ({t_cem:.0f}s)", flush=True)

    # 3b. Adam ascent from the ridge z
    z_grad, j_grad, grad_hist, t_grad, cos_grad = None, None, None, 0.0, None
    if FLAGS.grad_steps > 0:
        t1 = time.monotonic()
        z_grad, j_grad, grad_hist = grad_ascent(J, z_inf, d, FLAGS.grad_steps, FLAGS.grad_lr)
        t_grad = time.monotonic() - t1
        cos_grad = float(z_grad @ z_inf / (np.linalg.norm(z_grad) * np.linalg.norm(z_inf)))
        print(f"[value_infer] grad: J {j_inf:.4f} -> {j_grad:.4f}, cos(z_grad, ridge) {cos_grad:.3f} ({t_grad:.0f}s)", flush=True)

    # 4. the paper protocol on each z
    task_id = adapter.eval_tasks(env, cfg)[0]

    def score(z):
        ag = InferredCondAgent(agent.replace(cond_eval={"z": jnp.asarray(sphere(z, d))}))
        row, _ = adapter.evaluate(ag, env, task_id, cfg, FLAGS.eval_episodes, 0)
        return float(row["episode_return"])

    rets = {}
    if not FLAGS.skip_env:
        for name, z in (("ridge", z_inf), ("cem", z_cem), ("grad", z_grad)):
            if z is None:
                continue
            t1 = time.monotonic()
            rets[name] = score(z)
            print(f"[value_infer] env return ({name} z, {FLAGS.eval_episodes} eps): {rets[name]:.1f} "
                  f"({time.monotonic()-t1:.0f}s)", flush=True)

    rec = dict(tag=FLAGS.tag, domain=FLAGS.domain, exorl_task=task, seed=FLAGS.seed,
               restore_path=FLAGS.restore_path, restore_epoch=FLAGS.restore_epoch,
               pool=int(pool_obs.shape[0]), n_states=int(states.shape[0]),
               pool_reward_mean=float(pool_r.mean()), parity_err=parity,
               J_ridge=j_inf, J_neutral=j_neutral, J_cem=j_cem, J_grad=j_grad,
               cos_cem_to_ridge=cos_cem, cos_grad_to_ridge=cos_grad,
               return_ridge=rets.get("ridge"), return_cem=rets.get("cem"), return_grad=rets.get("grad"),
               oracle=oracle, eval_episodes=FLAGS.eval_episodes,
               pop=FLAGS.pop, rounds=FLAGS.rounds, elite=FLAGS.elite, sigma0=FLAGS.sigma0,
               grad_steps=FLAGS.grad_steps, grad_lr=FLAGS.grad_lr,
               cem_history=cem_hist, grad_history=grad_hist, t_cem_s=t_cem, t_grad_s=t_grad,
               z_ridge=z_inf.tolist(), z_cem=z_cem.tolist(),
               z_grad=None if z_grad is None else z_grad.tolist(), elapsed_s=time.monotonic() - t0)
    orc = "-" if oracle is None else f"{oracle['return_env']:.1f} (J {oracle['J']:.4f})"
    print(f"VALUE_INFER tag={FLAGS.tag} task={task} seed={FLAGS.seed} "
          f"J ridge/cem/grad = {j_inf:.4f}/{j_cem:.4f}/{'-' if j_grad is None else f'{j_grad:.4f}'} "
          f"return ridge/cem/grad = {rets.get('ridge')}/{rets.get('cem')}/{rets.get('grad')} "
          f"oracle={orc}", flush=True)
    if FLAGS.out_jsonl:
        os.makedirs(os.path.dirname(FLAGS.out_jsonl) or ".", exist_ok=True)
        with open(FLAGS.out_jsonl, "a") as f:
            f.write(json.dumps(rec, default=float) + "\n")


if __name__ == "__main__":
    flags.mark_flag_as_required("restore_path")
    app.run(main)
