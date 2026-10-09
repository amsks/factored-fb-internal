# Spec — Isolate why FB's goal vector z=B(g) is coarse (regularization vs allocation vs encoder)

**Run on a GPU instance.** State `cube-single-play-v0`, FB (FlowBC), PyTorch `.venv`.

## Context (what's already established — don't re-derive)
FB fails at cube **transport/placement**, not reach/grasp. The chain, all measured:
- Value-alignment by phase: approach FB 98% ≈ GCIQL 99%; **transport FB 83% < GCIQL 94%** (% trajectories where value tracks the right target).
- Within transport, the gap is **only near the goal** (far/mid: FB=GCIQL; near: FB 80% vs 94%) → **fine terminal resolution**.
- Component isolation (`privileged_z_probe`, frozen s3): forward path F *has* the geometry (`left_enc→d` R²=0.53); with the actual `z=B(g)` the value's near-goal Spearman(V,−d) is **0.038** (chance), with the best free `z*` it's **0.201** (~5×). ⇒ **bottleneck = the goal vector z=B(g)**, not F.
- Representation: R²(B→d)=**0.20** (state, MLP≈linear ⇒ geometry *absent from B*); pixel-B=0.76.
- Data: near-goal states are **abundant but proportionally rare** — ~35k states/goal within 5 cm (~3.5% of ρ), 14% within 10 cm. **No absolute data lack.**
- Frozen proxy: ortho=1000 makes state-B **near-isotropic** (effective rank 47.6/50) vs pixel-B (ortho=1.0) structured (18.2/50).

**Open question this spec answers:** *why* is B(g) coarse — (A) the ortho regularizer whitening B over the bulk, (B) ρ allocating ~no gradient to the rare terminal, or (C) the encoder. The ortho=100 checkpoints were **not saved**, so this needs retrains.

## Baseline to beat (the paper's state run, ortho=1000)
`results/Factored-FB-cube-run/2026-05-14_21-44-19__cube-single-play-v0__fb_flowbc__s3`: R²(B→d)≈0.20, privileged-z near-goal 0.04, success ≈0.50 (GCIQL 0.68).

## Shared measurement protocol (run on EVERY checkpoint produced)
For run dir `$RUN` (`$RUN/.hydra/config.yaml`, `$RUN/checkpoints/final.pt`):
```bash
# 1) representation: R^2(B->d)  (headline)
MUJOCO_GL=glfw .venv/bin/python -m scripts.factorization_probe \
  --config $RUN/.hydra/config.yaml --checkpoint $RUN/checkpoints/final.pt \
  --data-path datasets --out analysis/factorization_probe/<tag> --tasks all --device cuda
# 2) goal-vector quality: z=B(g) vs best free z* near goal
MUJOCO_GL=glfw .venv/bin/python -m scripts.privileged_z_probe \
  --config $RUN/.hydra/config.yaml --checkpoint $RUN/checkpoints/final.pt \
  --data-path datasets --task 1 --device cuda
# 3) behaviour: final eval success (from the run's eval logging / wandb)
```
Record per run: **R²(B→d)** (transport, B-MLP and B-linear), **privileged-z near-goal (z=B(g) and z\*)**, **success**.

---

## Experiment A — ortho sweep (regularization?)  [config only, no code]
**Hypothesis:** the 1000× orthonormality clamp whitens B over the bulk ρ → can't concentrate on the terminal.
```bash
for OC in 1 10 100 1000; do
  .venv/bin/python train.py domain=cube_single ortho_coef=$OC lr_b=1e-4 \
    num_train_steps=1000000 seed=1 device=cuda use_wandb=true \
    wandb_run_name=ortho${OC}_s1
done
```
Run the measurement protocol on each. **Decision:** R²(B→d) rises monotonically as ortho↓ → **regularization is a cause** (fix = lower/anneal ortho). Flat → **ortho ruled out**. (Note: lower ortho *hurt success* in the old sweep — if it *raises* R²(B→d), report the B-resolution-vs-success tension explicitly; it constrains the fix.)

---

## Experiment B — coverage-balanced ρ reweight (allocation?)  [small code change, flag-gated]
**Hypothesis:** B's resolution is allocated by ρ's *density* — densely-visited cube configs get fine resolution, the rare tail (where placements live, ~3.5% of ρ) gets none. Up-weight the **under-sampled** regions so B resolves *any* potential goal uniformly.

> ⚠️ **DO NOT use the eval/task goals to define the reweight.** That leaks the test tasks into reward-free pretraining and collapses FB into goal-conditioning on the eval goals. The reweight must be **goal-agnostic / self-supervised** — derived only from the offline data's own cube-position density. (Using the eval goals to *measure* R²(B→d) afterward is fine; using them in *training* is not.)

### B0 — precompute the goal-agnostic density (no eval goals)
A density over cube configuration = the goal-relevant projection of ρ; rare cells are where B is starved.
```bash
.venv/bin/python - <<'PY'
import numpy as np, glob
cubes=np.concatenate([np.load(f)['physics'][:,14:17] for f in
    sorted(glob.glob('datasets/cube-single-play-v0/buffer/episode_*.npz'))]).astype(np.float64)
nb=(16,16,8)
edges=[np.linspace(cubes[:,i].min(),cubes[:,i].max(),nb[i]+1) for i in range(3)]
H,_=np.histogramdd(cubes,bins=edges); H=H/H.sum()
np.savez('analysis/cube_density.npz', H=H, ex=edges[0], ey=edges[1], ez=edges[2])
print('density grid',H.shape,'nonzero cells',int((H>0).sum()))
PY
```

### Code change (default-off ⇒ byte-identical to current FB)
**`configs/agent/fb.yaml`** — add:
```yaml
reweight_alpha: 0.0                       # 0 = off (current FB exactly); try 0.5, 1.0
reweight_clip: 10.0                       # cap per-sample weight (rare-outlier guard)
reweight_density_path: analysis/cube_density.npz
```
**`agents/fb/agent.py::__init__`** — load the self-supervised density (no eval goals):
```python
self.reweight_alpha = float(getattr(cfg, "reweight_alpha", 0.0))
self.reweight_clip = float(getattr(cfg, "reweight_clip", 10.0))
self._rho = None
if self.reweight_alpha > 0:
    z = np.load(getattr(cfg, "reweight_density_path"))
    self._rho = (torch.tensor(z["H"], dtype=torch.float32, device=self.device),
                 [torch.tensor(z[k], dtype=torch.float32, device=self.device) for k in ("ex", "ey", "ez")])
```
**`agents/fb/agent.py::update`** — inverse-density weight for s⁺ (cube xyz from the offline buffer's `physics`):
```python
diag_w = None
if self.reweight_alpha > 0 and self._rho is not None:
    cube = batch["next"]["physics"][:, 14:17].to(self.device).float()        # s+ cube xyz
    H, (ex, ey, ez) = self._rho
    ix = (torch.bucketize(cube[:, 0], ex) - 1).clamp(0, H.shape[0] - 1)
    iy = (torch.bucketize(cube[:, 1], ey) - 1).clamp(0, H.shape[1] - 1)
    iz = (torch.bucketize(cube[:, 2], ez) - 1).clamp(0, H.shape[2] - 1)
    dens = H[ix, iy, iz] + 1e-6
    w = (dens.median() / dens) ** self.reweight_alpha                        # rare -> >1, common -> <1
    w = w.clamp(max=self.reweight_clip)
    diag_w = (w / w.mean()).detach()                                         # mean-preserving
metrics = self.update_fb(obs=obs, action=action, discount=discount,
                         next_obs=next_obs, goal=goal, z=z,
                         q_loss_coef=q_loss_coef, clip_grad_norm=clip_grad_norm,
                         diag_w=diag_w)
```
> If `batch["next"]["physics"]` is absent in the training batch, plumb it through the buffer sampler (it IS stored offline — the probes read `buffer.sample()["next"]["physics"]`).

**`agents/fb/agent.py::update_fb`** — add `diag_w=None`; weight only the diagonal (positive, per-s⁺ resolution) term. `diag_w` is already mean-normalized:
```python
diag = torch.diagonal(diff, dim1=1, dim2=2)                  # [n_parallel, batch]
if diag_w is not None:
    fb_diag = -(diag * diag_w).mean() * Ms.shape[0]          # diag_w broadcasts over n_parallel
else:
    fb_diag = -diag.mean() * Ms.shape[0]                     # == current behaviour
fb_loss = fb_offdiag + fb_diag
```

### Test (TDD — confirm isolation + goal-agnosticism)
`tests/test_fb_reweight.py`:
```python
# (1) reweight_alpha=0 -> diag_w=None -> fb_diag == -diagonal(diff).mean()*Ms.shape[0] (byte-identical).
# (2) alpha>0 -> diag_w.mean() ~= 1, all finite, and low-density (rare) s+ get weight > high-density s+.
# (3) the reweight reads ONLY analysis/cube_density.npz + physics -- assert no task/eval-goal input is used.
```
Run: `.venv/bin/python -m pytest tests/test_fb_reweight.py -v`

### Sweep
```bash
for A in 0 0.5 1.0; do
  .venv/bin/python train.py domain=cube_single reweight_alpha=$A ortho_coef=1000 \
    lr_b=1e-4 num_train_steps=1000000 seed=1 device=cuda use_wandb=true \
    wandb_run_name=reweight_a${A}_s1
done
```
Measurement protocol on each. **Decision:** R²(B→d) rises with α **and** privileged-z `z=B(g)` near-goal sharpens (gap to z\* closes) **and** success→GCIQL ⇒ **allocation is the cause** (coverage-balancing — goal-agnostic — fixes it) → **PURSUE this as the Tier-2 fix.** No change ⇒ not allocation.

---

## Experiment C — encoder capacity & architecture-vs-signal  [config + small script]
**C1 (config):** state-B with the pixel-style shallow-wide encoder (td_jepa `left_encoder=shallow`):
```bash
.venv/bin/python train.py domain=cube_single L_dim=256 'left_encoder.hidden_layers=0' \
  ortho_coef=1000 lr_b=1e-4 num_train_steps=1000000 seed=1 device=cuda use_wandb=true
```
Probe R²(B→d). Rises ⇒ encoder shape matters.

**C2 (decisive — supervised B-ceiling):** train B's *exact architecture* with a supervised MSE head to predict cube xyz from the offline states (no FB loss). New script `scripts/b_supervised_ceiling.py` (build `model._backward_map`∘`model._bw_encoder` like `agents/fb/model.py`, fit to `physics[:,14:17]`, held-out R²).
- supervised-B **achieves high R²** ⇒ architecture is capable ⇒ cause is the **FB training signal** (ρ/ortho/objective) → Experiments A/B.
- supervised-B **also fails** ⇒ the **encoder architecture** is the bottleneck → C1 / wider / object-centric / conv.

---

## Decision tree → which Tier-2 fix
| outcome | cause | fix |
|---|---|---|
| A: R²(B→d) ↑ as ortho↓ | ortho whitening | lower/anneal ortho (mind success tradeoff) |
| B: R²(B→d) ↑ with reweight_alpha, z=B(g) sharpens | ρ allocation (density starves the tail) | coverage-balanced (goal-agnostic) ρ reweight |
| C1: R²(B→d) ↑ with shallow-wide / C2 supervised-B capable | training signal, not encoder | A/B (+ aux geometry head on B) |
| C2: supervised-B also fails | encoder architecture | wider / conv / object-centric B encoder |

The winner is whatever **raises R²(B→d) AND sharpens z=B(g) (privileged-z) AND raises success** — all three, since geometry-in-B is necessary but the goal vector + behaviour are what we ultimately need.

## Isolation guarantee
Every change is flag-gated (`terminal_reweight_k=0`, stock `ortho_coef`/`L_dim`); with defaults the training path is byte-identical to the current FB. The C2 supervised script and the probes only read checkpoints/data — they modify nothing.
