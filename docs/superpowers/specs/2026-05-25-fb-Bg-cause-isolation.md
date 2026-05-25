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

## Experiment B — terminal-reweighted ρ (allocation?)  [small code change, flag-gated]
**Hypothesis:** B gets only ~3.5% of its gradient at the (abundant) near-goal s⁺; up-weighting them lets B resolve the terminal.

### Code change (default-off ⇒ byte-identical to current FB)

**`configs/agent/fb.yaml`** — add:
```yaml
terminal_reweight_k: 0.0          # 0 = off (current FB exactly); try 5, 20
terminal_reweight_eps: 0.05       # "near a placement goal" radius (m)
terminal_goals:                   # cube-single task goal xyz (placement targets)
  - [0.425, -0.1, 0.02]
  - [0.5,    0.0, 0.02]
  - [0.35,   0.0, 0.02]
  - [0.5,    0.2, 0.02]
  - [0.5,   -0.2, 0.02]
```

**`agents/fb/agent.py::__init__`** — store config + goal set:
```python
self.terminal_reweight_k = float(getattr(cfg, "terminal_reweight_k", 0.0))   # adapt to how cfg is read here
self.terminal_reweight_eps = float(getattr(cfg, "terminal_reweight_eps", 0.05))
gs = getattr(cfg, "terminal_goals", None)
self.goal_set = (torch.tensor(np.asarray(gs), dtype=torch.float32, device=self.device)
                 if gs is not None else None)   # [n_goals, 3]
```

**`agents/fb/agent.py::update`** — compute the near-goal mask for s⁺ and pass it down (the offline buffer stores `physics`; the cube xyz is `physics[:, 14:17]`):
```python
near = None
if self.terminal_reweight_k > 0 and self.goal_set is not None:
    cube_next = batch["next"]["physics"][:, 14:17].to(self.device).float()   # s+ cube xyz
    near = (torch.cdist(cube_next, self.goal_set).min(dim=-1).values
            < self.terminal_reweight_eps).float()                            # [batch]
metrics = self.update_fb(obs=obs, action=action, discount=discount,
                         next_obs=next_obs, goal=goal, z=z,
                         q_loss_coef=q_loss_coef, clip_grad_norm=clip_grad_norm,
                         near=near)
```
> If `batch["next"]["physics"]` is absent in the training batch, plumb it through the buffer sampler (it IS stored offline — the probes read `buffer.sample()["next"]["physics"]`).

**`agents/fb/agent.py::update_fb`** — add `near=None` param; weight only the diagonal (positive, per-s⁺ resolution) term, mean-preserving:
```python
diag = torch.diagonal(diff, dim1=1, dim2=2)                  # [n_parallel, batch]
if near is not None and self.terminal_reweight_k > 0:
    w = 1.0 + self.terminal_reweight_k * near                # [batch], broadcasts over n_parallel
    fb_diag = -(diag * w).mean() * Ms.shape[0] / w.mean()
else:
    fb_diag = -diag.mean() * Ms.shape[0]                     # == current behaviour
fb_loss = fb_offdiag + fb_diag
```

### Test (TDD — confirm isolation)
`tests/test_fb_terminal_reweight.py`:
```python
# build the FB agent twice on a fixed batch+seed: terminal_reweight_k=0 vs default.
# assert update_fb fb_diag with near=None equals -diagonal(diff).mean()*Ms.shape[0]
#   (k=0 reproduces current loss exactly), and with near + k>0 it differs and is finite.
```
Run: `.venv/bin/python -m pytest tests/test_fb_terminal_reweight.py -v`

### Sweep
```bash
for K in 0 5 20; do
  .venv/bin/python train.py domain=cube_single terminal_reweight_k=$K ortho_coef=1000 \
    lr_b=1e-4 num_train_steps=1000000 seed=1 device=cuda use_wandb=true \
    wandb_run_name=reweight${K}_s1
done
```
Measurement protocol on each. **Decision:** R²(B→d) rises with K **and** privileged-z `z=B(g)` near-goal sharpens (gap to z\* closes) **and** success→GCIQL ⇒ **allocation/sampling is the cause** (the abundant terminal data just needed up-weighting) → **PURSUE this as the Tier-2 fix.** No change ⇒ not sampling.

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
| B: R²(B→d) ↑ with reweight_k, z=B(g) sharpens | ρ allocation (terminal starvation) | terminal-reweighted ρ |
| C1: R²(B→d) ↑ with shallow-wide / C2 supervised-B capable | training signal, not encoder | A/B (+ aux geometry head on B) |
| C2: supervised-B also fails | encoder architecture | wider / conv / object-centric B encoder |

The winner is whatever **raises R²(B→d) AND sharpens z=B(g) (privileged-z) AND raises success** — all three, since geometry-in-B is necessary but the goal vector + behaviour are what we ultimately need.

## Isolation guarantee
Every change is flag-gated (`terminal_reweight_k=0`, stock `ortho_coef`/`L_dim`); with defaults the training path is byte-identical to the current FB. The C2 supervised script and the probes only read checkpoints/data — they modify nothing.
