# Spec — Coverage-balanced FB (consistent goal-side ρ′ reweight)

**Run on a GPU instance.** State `cube-single-play-v0`, FB (FlowBC), PyTorch `.venv`.
**This supersedes Experiment B of `2026-05-25-fb-Bg-cause-isolation.md`** — that version reweighted *only* the diagonal term, which is the "mixed objective" the idea doc (`PAPER/fb-cube-failure/reference_measure_idea.md` §7) explicitly says to avoid. This spec reweights **every goal-side ρ-term consistently**, so the trained model is a clean FB factorization in L²(ρ′_G).

---

## UPDATE 2026-05-25 (post code-review + first empirical results)

**Diagonal-weighting correction (important).** The FB linear/diagonal term `−E[F(s,a)ᵀB(s′)]` is the Bellman/Dirac term, *not* an integral over an independently sampled goal `g∼ρ`. Under a pure change of reference measure ρ→ρ′ its density ratio **cancels** (the immediate successor is `δ_{s′}`, and the `1/ρ′(s′)` cancels the `ρ′` in the L²(ρ′) integral), so **a pure ρ′-FB leaves the diagonal UNWEIGHTED**. Weighting it (as §(1) below originally prescribed, per idea-doc §7.3) is a *different*, additional intervention = extra TD pressure on rare achieved next-states (**coverage-PRIORITIZED FB**, a hybrid). This is now a flag, not a fixed choice:

- `weight_diag: false` (default) → **pure ρ′-FB** (off-diag + ortho + reward-cov weighted; diagonal unweighted).
- `weight_diag: true` → **coverage-prioritized FB** (also weights the diagonal). Reproduces the original §(1) below.

**New flags / hardening (implemented in `agents/fb/agent.py`):**
- `weight_diag` (above) and `weight_z` (resample goal-derived z by `w` so F/actor get rare-goal `z=B(g)` exposure — the idea-doc §8.2 hindsight-z lever, here as a separate arm).
- `coverage_weights` now two-sided-clips `w∈[1/clip, clip]` (dense cells keep mass), then mean-normalizes.
- `__init__` **fails loud** if `reweight_alpha>0` and no `reweight_density_path` (no silent disable).
- Q-loss covariance solve hardened (FP32 + 1e-4 ridge); inactive while `q_loss_coef=0`.
- Logged metrics: `reweight_w_{mean,min,max,std}`, `reweight_ess_frac` (ESS/N; 1=uniform, →0=few dominate).

**Ablation matrix** (`scripts/run_fb_reweight.sh` knobs `WEIGHT_DIAG`/`WEIGHT_Z`; α∈{0.25,0.5}, `clip∈{5,10}`; avoid α=1 until ESS/cov-cond are healthy):

| variant | offdiag | diag | ortho | reward-cov | weight_z |
|---|---|---|---|---|---|
| baseline | – | – | – | – | – |
| pure ρ′-FB | ✓ | – | ✓ | ✓ | – |
| pure ρ′-FB + weighted-z | ✓ | – | ✓ | ✓ | ✓ |
| prioritized ρ′-FB | ✓ | ✓ | ✓ | ✓ | – |
| **prioritized + weighted-z** (sweep priority) | ✓ | ✓ | ✓ | ✓ | ✓ |

**First result (prioritized α=0.5, 10 seeds @600k) — see `2026-05-25-privileged-z-findings.md`:** `z*` near-goal rose to **0.28±0.03** (vs baseline 0.20, 10/10 seeds) but the *actual* goal vector `z=B(g)` stayed **≈ chance (0.056±0.060 vs 0.038)**. So the reweight lifted the forward-path ceiling but did **not** sharpen `z=B(g)` → predicts no terminal-success gain; binding constraint looks like B's goal-encoding capacity and/or the bilinear (z*≪ F's 0.53), i.e. rank/architecture, not ρ alone. This is the spec's pre-registered **kill** direction; the ablation above tests whether pure-ρ′ or weighted-z changes it.

---

## Context (established — don't re-derive)
FB fails at cube **transport/placement**, not reach/grasp. The measured chain:
- Phase value-alignment: approach FB 98% ≈ GCIQL 99%; **transport FB 83% < GCIQL 94%**.
- The gap is **only near the goal** (far/mid: FB=GCIQL; near: FB 80% vs 94%) → fine **terminal resolution**.
- Component isolation (`privileged_z_probe`, frozen s3): forward path F *has* the geometry (`left_enc→d` R²=**0.53**); with the actual goal vector `z=B(g)` near-goal Spearman(V,−d)=**0.038** (chance), with the best free `z*`=**0.201** (~5×). ⇒ bottleneck = the goal vector **z=B(g)**.
- Representation: R²(B→d)=**0.20** (state, MLP≈linear ⇒ geometry *absent from B*); pixel-B=0.76.
- Data: near-goal states are **abundant but proportionally rare** — ~35k states/goal within 5 cm (~3.5% of ρ). **No absolute data lack.**
- Frozen proxy: ortho=1000 makes state-B **near-isotropic** (eff. rank 47.6/50). ⇒ the **orthonormality whitening is computed under the bulk-dominated ρ**, so it isotropizes B over the densely-visited region and starves the rare terminal tail.

**Hypothesis (allocation):** B's resolution is allocated by the goal measure ρ_G. The whitening `E_ρ[BBᵀ]=I` and the successor-measure regression are both integrated under ρ_G, which is dominated by behavior-dense (non-terminal) cube configs. Re-project FB under a **goal-agnostic coverage-balanced** measure ρ′_G (up-weight under-sampled cube configs) so B resolves *any* potential goal uniformly. Because the same ρ′_G is applied to **all** goal-side terms, this stays a valid FB in L²(ρ′_G) (no theory break).

> **Why diag-only (the old Experiment B) is the wrong lever:** the diagnosis blames the orthonormality whitening, which the diagonal reweight does not touch. The whitening (`E_ρ[BBᵀ]=I`) is the term that isotropizes B over the bulk; it **must** be reweighted to ρ′_G. This spec does that.

## Baseline to beat (the paper's state run, ortho=1000)
`results/Factored-FB-cube-run/2026-05-14_21-44-19__cube-single-play-v0__fb_flowbc__s3`: R²(B→d)≈0.20, privileged-z near-goal z=B(g) 0.04 / z\* 0.20, success ≈0.50 (GCIQL DrQ 0.68).

---

## The measure ρ′_G — goal-agnostic, no eval leak
ρ′_G = w(g)·ρ_G with `w(g) = clip((ρ̂_G(g)+ε)^{-α}, 0, w_max)`, mean-normalized so `E_{ρ_G}[w]=1`.
`g` = cube configuration; `ρ̂_G` = density of cube configs in the **offline data only**.

> ⚠️ **DO NOT use the eval/task goals to define the reweight.** That leaks the test tasks into reward-free pretraining and collapses FB into goal-conditioning on the eval goals. ρ̂_G is derived only from the offline data's own cube-position density. (Using eval goals to *measure* R²(B→d) afterward is fine; using them in *training* is not.)
>
> **Orientation note (doc/impl consistency):** `reference_measure_idea.md` §2/§6 define `g=φ(s)` to include cube *orientation*. This spec balances cube **xyz** only (the quantity the diagnosis is about: cube→goal distance). If the xyz reweight helps but orientation-sensitive placements remain, re-run B0 with `g=(xyz, quat)` and a 6-D histogram. Start xyz-only.

### B0 — precompute the goal-agnostic density (no eval goals)
```bash
mkdir -p analysis
.venv/bin/python - <<'PY'
import numpy as np, glob
cubes=np.concatenate([np.load(f)['physics'][:,14:17] for f in
    sorted(glob.glob('datasets/cube-single-play-v0/buffer/episode_*.npz'))]).astype(np.float64)
nb=(16,16,8)
edges=[np.linspace(cubes[:,i].min(),cubes[:,i].max(),nb[i]+1) for i in range(3)]
H,_=np.histogramdd(cubes,bins=edges); H=H/H.sum()
np.savez('analysis/cube_density.npz', H=H, ex=edges[0], ey=edges[1], ez=edges[2])
print('density grid',H.shape,'nonzero cells',int((H>0).sum()),
      'min/median/max nonzero',float(H[H>0].min()),float(np.median(H[H>0])),float(H.max()))
PY
```
The "nonzero cells" and the min/median spread tell you how peaked ρ_G is — the bigger the median/min ratio, the more the tail is starved.

---

## Code changes (default-off ⇒ byte-identical to current FB)

### `configs/agent/fb.yaml` — add
```yaml
reweight_alpha: 0.0                       # 0 = off (current FB exactly); try 0.5, 1.0
reweight_clip: 10.0                       # cap per-sample weight (rare-outlier guard)
reweight_density_path: analysis/cube_density.npz
```

### `agents/fb/agent.py::__init__` — accept + store (add params to the signature, default-safe)
Add to `__init__` signature: `reweight_alpha: float = 0.0, reweight_clip: float = 10.0, reweight_density_path: str = ""`, then in the body (after `self.device = device`):
```python
self.reweight_alpha = float(reweight_alpha)
self.reweight_clip = float(reweight_clip)
self._rho = None
if self.reweight_alpha > 0 and reweight_density_path:
    import numpy as _np
    z = _np.load(reweight_density_path)
    self._rho = (torch.tensor(z["H"], dtype=torch.float32, device=device),
                 [torch.tensor(z[k], dtype=torch.float32, device=device) for k in ("ex", "ey", "ez")])
```
> If `FBFlowBCAgent` (or the Hydra builder) forwards `**kwargs` to `FBAgent.__init__`, the three new keys flow through automatically. If it has an explicit signature, add the three params there too and pass them up.

### `agents/fb/agent.py::__init__` — new helper method (anywhere on the class)
Goal-agnostic per-sample weight from the cube-config density. Indexed by the **goal/next-state** axis (the ρ-integrated variable), so it aligns with `goal[j]`, the diagonal `i`, and `B`.
```python
def _coverage_weights(self, batch) -> torch.Tensor | None:
    """Inverse-density weight w(g) over cube xyz of next-state (the ρ_G variable).
    Goal-agnostic: reads ONLY the precomputed data density + physics. mean(w)=1."""
    if self.reweight_alpha <= 0 or self._rho is None:
        return None
    cube = batch["next"]["physics"][:, 14:17].to(self.device).float()   # s+ cube xyz
    H, (ex, ey, ez) = self._rho
    ix = (torch.bucketize(cube[:, 0], ex) - 1).clamp(0, H.shape[0] - 1)
    iy = (torch.bucketize(cube[:, 1], ey) - 1).clamp(0, H.shape[1] - 1)
    iz = (torch.bucketize(cube[:, 2], ez) - 1).clamp(0, H.shape[2] - 1)
    dens = H[ix, iy, iz] + 1e-6
    w = (dens.median() / dens) ** self.reweight_alpha                   # rare -> >1, common -> <1
    w = w.clamp(max=self.reweight_clip)
    return (w / w.mean()).detach()                                     # mean-preserving, batch-normalized
```

### `agents/fb/agent.py::update` — compute weights, pass through
After `z = self.sample_mixed_z(train_goal=goal).clone()`:
```python
w = self._coverage_weights(batch)        # None when reweight_alpha==0  -> byte-identical
```
and add `w=w,` to the `self.update_fb(...)` call kwargs.

### `agents/fb/agent.py::update_fb` — add `w` and reweight EVERY goal-side ρ-term
Add `w: torch.Tensor | None = None,` to the signature. Replace the three loss blocks:

**(1) FB successor loss.** ⚠️ **SUPERSEDED by the UPDATE block above** — the diagonal is now flag-gated (`weight_diag`): pure ρ′-FB leaves it **unweighted** (the Dirac density-ratio cancels); the weighted-diagonal form below is the `weight_diag=true` *prioritized* variant. Shipped in `agents/fb/agent.py::fb_successor_terms(..., weight_diag)`:
```python
diff = Ms - discount * target_M  # num_parallel x batch x batch
if w is None:
    fb_offdiag = 0.5 * (diff * self.off_diag).pow(2).sum() / self.off_diag_sum
    fb_diag = -torch.diagonal(diff, dim1=1, dim2=2).mean() * Ms.shape[0]
else:
    wcol = w.view(1, 1, -1)                                            # weight goal/column j ~ rho'
    fb_offdiag = 0.5 * ((diff * self.off_diag).pow(2) * wcol).sum() / self.off_diag_sum
    diag = torch.diagonal(diff, dim1=1, dim2=2)
    fb_diag = (-(diag * w.view(1, -1)).mean() if weight_diag       # prioritized
               else -diag.mean()) * Ms.shape[0]                    # pure rho' (default)
fb_loss = fb_offdiag + fb_diag
```
> `mean(w)==1` ⇒ the off-diag denominator and diag `.mean()` scale are unchanged; only the *distribution* over goals shifts. **Off-diagonal is always weighted** (g~ρ′). The diagonal is weighted only under `weight_diag=true`; idea-doc §7.3's row–goal coupling is that (prioritized) variant, not the pure measure change.

**(2) Orthonormality** (currently lines 280–283) — the centerpiece: whiten under ρ′ via `B̃=√w·B`:
```python
if w is None:
    Cov = torch.matmul(B, B.T)
else:
    Bw = B * w.sqrt().unsqueeze(-1)                                    # E_rho'[B B^T] = (1/n) sum_i w_i B_i B_i^T
    Cov = torch.matmul(Bw, Bw.T)
orth_loss_diag = -Cov.diag().mean()
orth_loss_offdiag = 0.5 * (Cov * self.off_diag).pow(2).sum() / self.off_diag_sum
orth_loss = orth_loss_offdiag + orth_loss_diag
fb_loss += self.ortho_coef * orth_loss
```
> Identity: with `B̃_i=√w_i B_i`, `Coṽ[i,j]=√(w_i w_j)⟨B_i,B_j⟩` ⇒ `mean_i Coṽ[i,i]=trace(E_rho'[BBᵀ])` and `mean_{i≠j} Coṽ[i,j]²` = off-diagonal of `‖E_rho'[BBᵀ]‖_F²`. So the loss minimum is `E_rho'[BBᵀ]=I`. This is the term diag-only never touched.

**(3) Reward-inference covariance in the q-loss path** (currently line 293, inside `if q_loss_coef is not None`) — only matters if `q_loss_coef>0`:
```python
with autocast(device_type=self.device, dtype=self.amp_dtype, enabled=False):
    if w is None:
        cov = torch.matmul(B.T, B) / B.shape[0]                        # z_dim x z_dim
    else:
        cov = torch.matmul((B * w.unsqueeze(-1)).T, B) / B.shape[0]    # E_rho'[B B^T]
```
> Check the run's config: FlowBC typically has `q_loss_coef=0` ⇒ this branch is inactive and irrelevant. If it's >0, this keeps implicit-reward inference consistent with ρ′. **Eval-time** `model.reward_inference` is unchanged: for point goals `z=B(g)` up to scaling, so reweighting it is a no-op; it matters only for dense rewards (out of scope).

> Leave `sample_mixed_z` (z-sampling) **unchanged**. Reweighting *which z's are sampled* is the separate hindsight-z lever (idea doc §8.2) — a later ablation, not this experiment.

---

## Tests (TDD — confirm isolation, correctness, goal-agnosticism)
`tests/test_fb_reweight.py`:
```python
# (1) BYTE-IDENTICAL: reweight_alpha=0 -> _coverage_weights returns None -> update_fb with w=None
#     produces fb_loss/fb_diag/fb_offdiag/orth_loss identical (allclose, atol=0) to a w-absent call
#     on the same fixed (obs,action,next_obs,goal,z,discount). Run with a seeded model.
# (2) WEIGHT VALIDITY: with a toy density, _coverage_weights -> mean()~=1, all finite, and a rare
#     (low-density) cube xyz gets weight > a common (high-density) one; clip caps at reweight_clip.
# (3) ORTHONORMALITY IS REWEIGHTED: build B (batch x z_dim) and w; assert the spec's
#     Cov=(√w·B)(√w·B)^T equals the weighted second-moment form, i.e.
#     Cov.diag().mean()  ==  (w * (B*B).sum(-1)).mean()       (trace term under rho')
#     and that this differs from the unweighted Cov when w!=1 (guards against a no-op).
# (4) GOAL-AGNOSTIC: monkeypatch/inspect that _coverage_weights reads ONLY batch["next"]["physics"]
#     and self._rho (the data density) -- assert no task/eval-goal tensor is referenced.
```
Run: `.venv/bin/python -m pytest tests/test_fb_reweight.py -v`

---

## Sweep
```bash
for A in 0 0.5 1.0; do
  .venv/bin/python train.py domain=cube_single \
    reweight_alpha=$A reweight_clip=10.0 reweight_density_path=analysis/cube_density.npz \
    ortho_coef=1000 lr_b=1e-4 num_train_steps=1000000 seed=1 device=cuda use_wandb=true \
    wandb_run_name=covbal_a${A}_s1
done
```
`reweight_alpha=0` reproduces the baseline exactly (sanity that the plumbing is inert). Start mild (`0.5`); strong inverse-density (`1.0`) can ill-condition the covariance and sacrifice bulk performance — that tension is part of the result, report it.

---

## Measurement protocol (run on EVERY checkpoint)
For run dir `$RUN` (`$RUN/.hydra/config.yaml`, `$RUN/checkpoints/final.pt`):
```bash
# 1) representation: R^2(B->d) (headline) — transport, B-MLP and B-linear
MUJOCO_GL=glfw .venv/bin/python -m scripts.factorization_probe \
  --config $RUN/.hydra/config.yaml --checkpoint $RUN/checkpoints/final.pt \
  --data-path datasets --out analysis/factorization_probe/covbal_a<A> --tasks all --device cuda
# 2) goal-vector quality: z=B(g) vs best free z* near goal
MUJOCO_GL=glfw .venv/bin/python -m scripts.privileged_z_probe \
  --config $RUN/.hydra/config.yaml --checkpoint $RUN/checkpoints/final.pt \
  --data-path datasets --task 1 --device cuda
# 3) behaviour: final eval success (run's eval logging / wandb)
```
Record per α: **R²(B→d)** (B-MLP, B-linear), **privileged-z near-goal (z=B(g) and z\*)**, **success**, and **B effective rank** (from factorization_probe embeddings dump, to confirm de-isotropization).

---

## Decision tree + kill criterion
| outcome | reading | action |
|---|---|---|
| R²(B→d) ↑ with α **and** z=B(g) near-goal sharpens (gap to z\* closes) **and** success → GCIQL | allocation was the cause; coverage-balancing (goal-agnostic) fixes it | **PURSUE as the Tier-2 fix**; write up coverage-balanced FB |
| R²(B→d) ↑ and z=B(g)→z\* **but success flat** | **forward-path ceiling is binding** (recall z\*=0.20 ≪ left_enc→d=0.53): the bilinear `V=Fᵀz` loses geometry F has; reweighting B alone can't lift it | the fix is insufficient as-is → next: raise rank (`z_dim`/`L_dim`), or aux geometry head on F, or confirm retrained-F (which this *does* train) moved z\* upward — re-run privileged-z and check if z\* itself rose |
| R²(B→d) flat with α | not allocation (or weights too clipped / density too coarse) | rule out: check `B` eff-rank moved; if not, allocation ruled out → ortho-anneal (cause-isolation Exp A) or encoder (Exp C) |
| success ↓ as α↑ (bulk sacrificed) | reallocation is zero-sum at fixed rank; tail gained, bulk lost | lower α / tighten clip; report the resolution-vs-coverage tradeoff curve |

**Pre-registered kill condition (Carlini):** if the consistent reweight raises R²(B→d) and closes the z=B(g)→z\* gap but **does not** raise success, the headline claim ("coverage-balanced FB fixes transport") is falsified — the binding constraint is the forward-path/bilinear ceiling, not the goal measure. Report that honestly; the fallback is rank/architecture, not ρ.

## Isolation guarantee
Every change is flag-gated on `reweight_alpha` (default `0.0`). With the default, `_coverage_weights` returns `None`, `update_fb` takes the `w is None` branches, and the training path is **byte-identical** to current FB (test 1 enforces this). The density precompute and probes only read data/checkpoints — they modify nothing.
