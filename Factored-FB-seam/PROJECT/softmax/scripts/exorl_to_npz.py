"""ExoRL buffer (episode_*.npz with observation[T+1], action[T+1]) -> npz {observations, terminals}."""
import sys, glob, numpy as np
src, out = sys.argv[1], sys.argv[2]; maxe = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
obs, term = [], []
for f in sorted(glob.glob(f"{src}/episode_*.npz"))[:maxe]:
    d = np.load(f); o = d["observation"].astype(np.float32)
    obs.append(o); t = np.zeros(len(o), bool); t[-1] = True; term.append(t)
obs = np.concatenate(obs); term = np.concatenate(term)
np.savez_compressed(out, observations=obs, terminals=term); print(out, obs.shape, int(term.sum()))
