"""Oracle task-vector search on an ExoRL Density-FB checkpoint (EVAL-ONLY).

Question: is the cheetah gap task INFERENCE (the z the reward samples produce) or the CRITIC /
ACTOR (no z reaches a good policy)? For one checkpoint and one task this script
  1. scores the INFERRED z (the paper protocol: 50k reward-labelled rows -> critic.infer_cond),
  2. runs a cross-entropy-method search over z on the sqrt(d) sphere, seeded at the inferred z
     plus uniform sphere draws, scoring every candidate on the SAME `--search_episodes` seeded
     episodes (env/unified.py::evaluate resets with seeded_reset(i + seed_off), so every
     candidate sees the same starts), and
  3. re-scores the best candidate on the full `--eval_episodes` protocol.
If (3) >> (1) the gap is inference; if (3) ~ (1) the actor cannot do better with any z.

Loading mirrors scripts/reeval_checkpoint.py exactly (recorded config + CLI overrides, Agent.create,
restore_agent(params_only), adapter.evaluate) and rolls z through InferredCondAgent so no goal
frame can overwrite it.

    CUDA_VISIBLE_DEVICES=0 /dev/shm/.venv-jax/bin/python -m scripts.exorl_zsearch \
        --restore_path exp/exorl_sqt/dmc-cheetah/dmc-cheetah_density_fb_z_ddpgbc_seed_2 \
        --restore_epoch 1000000 --seed 2 --override exorl_task=run ... --out_jsonl exp/zsearch/cheetah.jsonl
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from absl import app, flags
import jax.numpy as jnp
import numpy as np

from impls.agents import Agent
from impls.utils.datasets import make_dataset
from env import get_adapter, make_config
from impls.utils.flax_utils import resolve_run_dir, restore_agent
from impls.critics.base import infer_cond_rows
from impls.utils.eval_cond import InferredCondAgent


def _recorded_config_overrides(run_dir):
    """Copy of scripts/reeval_checkpoint.py::_recorded_config_overrides (importing that module
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

FLAGS = flags.FLAGS
flags.DEFINE_string("critic", "density_fb_z", "")
flags.DEFINE_string("actor", "ddpgbc", "")
flags.DEFINE_string("domain", "dmc-cheetah", "")
flags.DEFINE_integer("seed", 0, "")
flags.DEFINE_string("restore_path", None, "run dir")
flags.DEFINE_integer("restore_epoch", 1000000, "")
flags.DEFINE_multi_string("override", [], "dotlist overrides, CLI wins over the recorded config")
flags.DEFINE_integer("infer_cond_episodes", 50000, "reward-labelled rows for the inferred z")
flags.DEFINE_integer("eval_episodes", 10, "episodes for the inferred z and the final best z")
flags.DEFINE_integer("search_episodes", 2, "episodes per candidate during the search")
flags.DEFINE_integer("pop", 24, "candidates per CEM round")
flags.DEFINE_integer("rounds", 3, "CEM rounds")
flags.DEFINE_integer("elite", 6, "elites per round")
flags.DEFINE_float("sigma0", 0.6, "initial per-coordinate noise, in units of sqrt(d)/sqrt(d) (unit-sphere scale)")
flags.DEFINE_integer("n_random", 8, "uniform sphere draws mixed into round 1")
flags.DEFINE_string("tag", "", "")
flags.DEFINE_string("out_jsonl", "", "append the result row here")
flags.DEFINE_integer("rng_seed", 0, "")


def sphere(z, d):
    z = np.asarray(z, np.float64)
    return z * np.sqrt(d) / max(np.linalg.norm(z), 1e-12)


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

    # 1. the paper-protocol inferred z
    rb = train_ds.sample(FLAGS.infer_cond_episodes)
    rows = infer_cond_rows(agent.critic)
    inferred = agent.infer_eval_cond(jnp.asarray(rb[rows]), jnp.asarray(rb["rewards"]))
    z_inf = np.asarray(inferred.cond_eval["z"], np.float64)
    d = z_inf.shape[-1]
    task_id = adapter.eval_tasks(env, cfg)[0]

    def score(z, episodes):
        ag = InferredCondAgent(agent.replace(cond_eval={"z": jnp.asarray(sphere(z, d))}))
        row, _ = adapter.evaluate(ag, env, task_id, cfg, episodes, 0)
        return float(row["episode_return"])

    r_inf = score(z_inf, FLAGS.eval_episodes)
    print(f"[zsearch] inferred z: return {r_inf:.1f} over {FLAGS.eval_episodes} eps "
          f"(|z|={np.linalg.norm(z_inf):.3f}, d={d}, {time.monotonic()-t0:.0f}s)", flush=True)

    # 2. CEM on the sphere, seeded at the inferred z
    rng = np.random.default_rng(FLAGS.rng_seed)
    mean, sigma = z_inf.copy(), FLAGS.sigma0
    best_z, best_r, history = z_inf.copy(), score(z_inf, FLAGS.search_episodes), []
    r_inf_search = best_r
    print(f"[zsearch] inferred z on the {FLAGS.search_episodes} search episodes: {best_r:.1f}", flush=True)
    for rnd in range(FLAGS.rounds):
        cands = [sphere(mean + sigma * rng.normal(size=d), d) for _ in range(FLAGS.pop)]
        if rnd == 0:
            cands += [sphere(rng.normal(size=d), d) for _ in range(FLAGS.n_random)]
        rets = []
        for k, z in enumerate(cands):
            r = score(z, FLAGS.search_episodes)
            rets.append(r)
            if r > best_r:
                best_r, best_z = r, z.copy()
        order = np.argsort(rets)[::-1]
        elites = np.stack([cands[i] for i in order[:FLAGS.elite]])
        mean = sphere(elites.mean(0), d)
        sigma = max(0.15, 0.7 * sigma)
        cos = float(best_z @ z_inf / (np.linalg.norm(best_z) * np.linalg.norm(z_inf)))
        history.append(dict(round=rnd, best=best_r, round_max=float(max(rets)),
                            round_mean=float(np.mean(rets)), elite_mean=float(np.mean([rets[i] for i in order[:FLAGS.elite]])),
                            cos_best_to_inferred=cos, sigma=sigma))
        print(f"[zsearch] round {rnd}: max {max(rets):.1f} mean {np.mean(rets):.1f} "
              f"best-so-far {best_r:.1f} cos(best, inferred) {cos:.3f} ({time.monotonic()-t0:.0f}s)", flush=True)

    # 3. the best candidate on the full protocol
    r_best = score(best_z, FLAGS.eval_episodes)
    cos = float(best_z @ z_inf / (np.linalg.norm(best_z) * np.linalg.norm(z_inf)))
    rec = dict(tag=FLAGS.tag, domain=FLAGS.domain, exorl_task=cfg.get("exorl_task"), seed=FLAGS.seed,
               restore_path=FLAGS.restore_path, restore_epoch=FLAGS.restore_epoch,
               return_inferred=r_inf, return_inferred_search_eps=r_inf_search,
               return_best_search_eps=best_r, return_best=r_best, cos_best_to_inferred=cos,
               eval_episodes=FLAGS.eval_episodes, search_episodes=FLAGS.search_episodes,
               pop=FLAGS.pop, rounds=FLAGS.rounds, history=history,
               z_inferred=z_inf.tolist(), z_best=best_z.tolist(), elapsed_s=time.monotonic() - t0)
    print(f"ZSEARCH tag={FLAGS.tag} task={cfg.get('exorl_task')} inferred={r_inf:.1f} "
          f"best={r_best:.1f} (search-eps {best_r:.1f}) cos={cos:.3f}", flush=True)
    if FLAGS.out_jsonl:
        os.makedirs(os.path.dirname(FLAGS.out_jsonl) or ".", exist_ok=True)
        with open(FLAGS.out_jsonl, "a") as f:
            f.write(json.dumps(rec, default=float) + "\n")


if __name__ == "__main__":
    flags.mark_flag_as_required("restore_path")
    app.run(main)
