"""Probe: why does fb_state_encmatch eval at 0% while fb_state_plain ~0.5?

Loads both final checkpoints, pushes a batch of real cube obs through, and
measures (a) _fw_encoder feature diversity across states + tanh saturation,
(b) whether the actor emits diverse actions across states (degenerate eval
policy => ~0% success). CPU only.
"""
import sys
import numpy as np
import torch

from evals.analysis import load_cfg, build_env_and_agent, load_checkpoint
from data.ogbench import load_ogbench_dataset

RUNS = "/dev/shm/factored-fb/runs"
CASES = {
    "plain":    f"{RUNS}/20260607_070225__fb_state_plain__s0",
    "encmatch": f"{RUNS}/20260607_062100__fb_state_encmatch__s0",
}
DATA = "/dev/shm/factored-fb/datasets"


def probe(name, rundir, batch):
    cfg = load_cfg(f"{rundir}/.hydra/config.yaml", device="cpu")
    env, agent, obs_space, action_dim = build_env_and_agent(cfg)
    load_checkpoint(agent, f"{rundir}/checkpoints/final.pt")
    m = agent.model
    m.train(False)

    obs = torch.as_tensor(batch["observation"], dtype=torch.float32)
    with torch.no_grad():
        nobs = m._normalize(obs)
        feat = m._fw_encoder(nobs)                       # actor input (actor_encode_obs=False)
        # random z on the sphere, fixed across the batch
        z = torch.randn(1, m.z_dim).repeat(obs.shape[0], 1)
        z = m.project_z(z)
        act = m.act(obs, z, mean=True)

    # --- z-sensitivity: SAME obs, different z. Goal-directed policy => actions
    #     change with z. z-ignoring policy => actions ~identical across z. ---
    with torch.no_grad():
        fixed_obs = obs[:64]
        zs = m.project_z(torch.randn(8, m.z_dim))           # 8 distinct tasks
        acts_per_z = []
        for zi in zs:
            zz = zi.unsqueeze(0).repeat(fixed_obs.shape[0], 1)
            acts_per_z.append(m.act(fixed_obs, zz, mean=True).numpy())
        acts_per_z = np.stack(acts_per_z)                    # [8, 64, act_dim]
    # per (state, action-dim) std ACROSS the 8 z's, averaged
    z_sens = acts_per_z.std(axis=0).mean()

    # --- backward-map goal separation: do distinct goals map to distinct z? ---
    with torch.no_grad():
        goals = m._normalize(obs[:256])
        zg = m.backward_map(obs[:256])      # B(goal), already projected
        zg = zg.numpy()
    # mean pairwise cosine similarity of inferred z (1.0 => all goals collapse to one z)
    zn = zg / (np.linalg.norm(zg, axis=1, keepdims=True) + 1e-8)
    cos = zn @ zn.T
    off = cos[~np.eye(len(cos), dtype=bool)]

    # --- Q-ranking: does the actor's action beat dataset/random under its own Q? ---
    # Working FB: Q(actor) >~ Q(dataset) > Q(random). actor < random => optimization
    # bug. actor >= others but eval=0 => Q misaligned with true reward (representation).
    with torch.no_grad():
        obs_q = obs[:256]
        z_q = m.project_z(torch.randn(256, m.z_dim))
        a_actor = m.act(obs_q, z_q, mean=True)
        a_data = torch.as_tensor(batch["observation"], dtype=torch.float32)  # placeholder
        a_rand = torch.empty_like(a_actor).uniform_(-1, 1)

        def qval(a_):
            Fs = m.forward_map(obs_q, z_q, a_)              # [P, B, z_dim] or [B, z_dim]
            if Fs.dim() == 3:
                Fs = Fs.mean(0)
            return (Fs * z_q).sum(-1)                        # [B]
        q_actor = qval(a_actor).mean().item()
        q_rand = torch.stack([qval(torch.empty_like(a_actor).uniform_(-1, 1))
                              for _ in range(8)]).mean().item()

    f = feat.numpy()
    a = act.numpy()
    print(f"\n===== {name} =====")
    print(f"obs dim={obs.shape[1]}  fw_encoder out dim={f.shape[1]}  act dim={a.shape[1]}")
    print(f"feat: |x|>0.99 frac = {(np.abs(f) > 0.99).mean():.3f}   (tanh saturation)")
    print(f"feat: mean per-dim std ACROSS states = {f.std(axis=0).mean():.4f}   "
          f"(0 => encoder ignores the state)")
    print(f"feat: mean |value| = {np.abs(f).mean():.4f}")
    print(f"act : per-dim std ACROSS states = {np.array2string(a.std(axis=0), precision=3)}")
    print(f"act : mean per-dim std = {a.std(axis=0).mean():.4f}   "
          f"(0 => same action regardless of state)")
    print(f"act : |a|>0.99 frac = {(np.abs(a) > 0.99).mean():.3f}   (action saturation)")
    print(f"act : range [{a.min():.3f}, {a.max():.3f}]")
    print(f"Z-SENSITIVITY: mean action std ACROSS 8 z (same obs) = {z_sens:.4f}   "
          f"(0 => policy IGNORES the task z)")
    print(f"B(goal) sep: mean |pairwise cos| = {np.abs(off).mean():.4f}   "
          f"(1.0 => all goals collapse to one z); B_norm mean = {np.linalg.norm(zg,axis=1).mean():.3f}")
    print(f"Q-RANK: Q(actor)={q_actor:.3f}  Q(random)={q_rand:.3f}  "
          f"=> actor {'BEATS' if q_actor > q_rand else 'LOSES TO'} random "
          f"({'maximizes its Q ok' if q_actor > q_rand else 'ACTOR NOT MAXIMIZING Q'})")


def main():
    # one shared batch of real obs (same indices for both via fixed seed)
    torch.manual_seed(0)
    buf = load_ogbench_dataset(domain="cube-single-play-v0", data_path=DATA,
                               obs_type="state", load_n_episodes=50)
    batch = buf.sample(512)
    batch = {k: (v.cpu().numpy() if torch.is_tensor(v) else v)
             for k, v in batch.items() if k == "observation"}
    for name, rundir in CASES.items():
        probe(name, rundir, batch)


if __name__ == "__main__":
    main()
