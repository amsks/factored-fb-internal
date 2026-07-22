"""campaign/pretrain_teacher.py — pretrain a standalone per-task policy head on a small
mixed dataset, NO critic. Two policy types:

  --policy dp    : flow-matching (diffusion) BC. velocity field v(obs, a, t) trained by
                   rectified-flow regression; action = Euler-integrate the ODE from FIXED
                   noise (seed 128), so eval is deterministic.
  --policy l2bc  : plain MLP obs->action, trained by MSE (deterministic).

This is the "teacher" the flowbc / fql actors distill from, trained in isolation so the
head + its checkpoint exist independently of any RL run. Single task => policy is pi(a|s)
(obs-only; no goal/z). Eval uses the unified protocol: success everywhere, integer at-goal
ticks on Box2D only. Saves a flax-params checkpoint and writes the per-seed eval jsonl the
agg daemon already consumes.
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

FB = "/raid/tongle/rl/Factored-FB"
if FB not in sys.path:
    sys.path.insert(0, FB)
from impls.adapters.unified_adapter import make_task_env, seeded_reset, load_unified, _flag  # noqa: E402
from impls.teacher.nets import MLP, FlowVF  # noqa: E402

_FAMILY = {"b2d": "Box2D", "p2d": "Box2D", "mw": "Metaworld", "ms": "ManiSkill"}




def _append_agg(agg_dir, group, seed, step, row, task):
    os.makedirs(agg_dir, exist_ok=True)
    rec = dict(seed=int(seed), step=int(step), success_rate=float(row["success"]),
               at_goal_ticks=float(row.get("at_goal_ticks", float("nan"))),
               family=_FAMILY.get(task.split("-")[0]))
    with open(os.path.join(agg_dir, f"{group}__s{seed}.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--policy", choices=["dp", "l2bc"], required=True)
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--steps", type=int, default=50_000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--flow_steps", type=int, default=10)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_interval", type=int, default=10_000)
    ap.add_argument("--eval_episodes", type=int, default=50)
    ap.add_argument("--agg_dir", default=None)
    ap.add_argument("--agg_group", default=None)
    args = ap.parse_args()

    d = load_unified(args.data_dir)
    obs = np.asarray(d["observations"], np.float32)
    act = np.clip(np.asarray(d["actions"], np.float32), -1.0, 1.0)
    odim, adim = obs.shape[1], act.shape[1]
    hidden = tuple([args.hidden] * args.layers)
    rng = jax.random.PRNGKey(args.seed)
    rng, init_rng = jax.random.split(rng)
    print(f"[pretrain] {args.policy} {args.task}: {len(obs)} transitions obs={odim} act={adim} "
          f"(data_dir={args.data_dir})", flush=True)

    if args.policy == "l2bc":
        net = MLP(hidden, adim)
        params = net.init(init_rng, jnp.zeros((1, odim)))

        def loss_fn(p, ob, ac, key):
            return jnp.mean((net.apply(p, ob) - ac) ** 2)

        def act_fn(p, ob):
            return np.clip(np.asarray(net.apply(p, ob)), -1.0, 1.0)
    else:
        net = FlowVF(hidden, adim)
        params = net.init(init_rng, jnp.zeros((1, odim)), jnp.zeros((1, adim)), jnp.zeros((1, 1)))

        def loss_fn(p, ob, ac, key):
            k1, k2 = jax.random.split(key)
            x0 = jax.random.normal(k1, ac.shape)
            t = jax.random.uniform(k2, (ac.shape[0], 1))
            xt = (1 - t) * x0 + t * ac
            return jnp.mean((net.apply(p, ob, xt, t) - (ac - x0)) ** 2)

        fs = args.flow_steps

        @jax.jit
        def _flow_action(p, ob):
            # full Euler rollout from FIXED noise (seed 128), as ONE jitted call --
            # the per-step unjitted version made eval ~100x slower and stalled runs.
            a0 = jax.random.normal(jax.random.PRNGKey(128), (ob.shape[0], adim))

            def body(i, a):
                t = jnp.full((ob.shape[0], 1), i / fs)
                return a + net.apply(p, ob, a, t) / fs

            return jnp.clip(jax.lax.fori_loop(0, fs, body, a0), -1.0, 1.0)

        def act_fn(p, ob):
            return np.asarray(_flow_action(p, jnp.asarray(ob, jnp.float32)))

    state = TrainState.create(apply_fn=None, params=params, tx=optax.adam(args.lr))

    @jax.jit
    def update(state, ob, ac, key):
        loss, grads = jax.value_and_grad(loss_fn)(state.params, ob, ac, key)
        return state.apply_gradients(grads=grads), loss

    env, horizon = make_task_env(args.task, render=bool(args.task.startswith(("b2d-", "p2d-"))))
    is_box2d = args.task.startswith(("b2d-", "p2d-"))

    def evaluate():
        succ, ticks = [], []
        for i in range(args.eval_episodes):
            ob, info = seeded_reset(env, args.task, i)
            s = _flag(info.get("success", 0))
            ag = int(s)
            for _ in range(horizon):
                a = act_fn(state.params, np.asarray(ob, np.float32)[None])[0]
                ob, _r, term, trunc, info = env.step(np.clip(a, -1.0, 1.0))
                now = _flag(info.get("success", 0))
                s = s or now
                ag += int(now)
                if _flag(term) or _flag(trunc):
                    break
            succ.append(float(s))
            ticks.append(ag)
        out = {"success": float(np.mean(succ))}
        if is_box2d:
            out["at_goal_ticks"] = float(np.mean(ticks))
        return out

    def do_eval(step):
        r = evaluate()
        print(f"[pretrain] step {step}: {r}", flush=True)
        if args.agg_dir and args.agg_group:
            _append_agg(args.agg_dir, args.agg_group, args.seed, step, r, args.task)

    N = len(obs)
    for step in range(1, args.steps + 1):
        rng, k = jax.random.split(rng)
        idx = np.random.randint(0, N, args.batch)
        state, _l = update(state, obs[idx], act[idx], k)
        if step == 1 or step % args.eval_interval == 0:
            do_eval(step)

    os.makedirs(args.save_dir, exist_ok=True)
    ckpt = dict(params=jax.device_get(state.params), policy=args.policy, task=args.task,
                odim=odim, adim=adim, hidden=list(hidden), flow_steps=args.flow_steps,
                data_dir=args.data_dir, note="teacher head pretrained on the mixed 80/10/10 set")
    with open(os.path.join(args.save_dir, "policy.pkl"), "wb") as f:
        pickle.dump(ckpt, f)
    print(f"[pretrain] DONE {args.policy} {args.task} -> {args.save_dir}/policy.pkl", flush=True)


if __name__ == "__main__":
    main()
