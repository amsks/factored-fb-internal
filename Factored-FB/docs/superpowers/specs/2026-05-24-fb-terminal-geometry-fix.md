# Tier-2 Fix Spec — make state-FB's backward map B encode terminal geometry

**Status:** design, pending approval before implementation (changes the training loop + needs GPU).

## What the diagnosis says the fix must do
The factorization probe established (transport phase, R²(rep→cube-to-goal distance d), held-out):

| rep | R²(→d) | reading |
|---|---|---|
| FB **state**-B | **0.25** (MLP≈linear) | geometry **absent** from B (not merely nonlinear) |
| FB pixel-B | 0.76 | FB factorization **can** hold the geometry given the right featurization/pressure |
| FB forward `left_enc` (state) | 0.58 | the forward branch keeps geometry; **only B fails** |
| GCIQL state value-φ | 0.93 | a value rep that succeeds **does** keep it |

So the fix is **not** a bigger value head or higher rank (ruled out: MLP≈linear on state-B ⇒ info isn't in B at all). It must **inject geometric supervision into B** so B(s) linearly encodes object position. Success = R²(B→d) rises from 0.25 toward ≥0.6–0.76 **and** state-cube transport success rises toward GCIQL's 0.68 (stretch: zero-shot FB ≥ GCIQL-DrQ).

## Why B is starved (mechanism)
B(s) is goal-independent; the goal enters via z=B(g), and the FB loss only pressures the F⊤B inner product to match the successor measure. Under a uniform sampling measure ρ, near-terminal states are ~2% of samples (Goal Frac ≈ 0.022), and the orthonormality penalty (state ortho_coef=1000) pushes B toward an isotropic basis — together B never gets a signal to *resolve object position*, least of all near the terminal set. The forward `left_enc` keeps geometry because the actor/Q path needs it; B doesn't, so it drops it.

## Primary fix (recommended first): auxiliary geometry head on B
Add a small read-out head `g(·)` trained to predict the object position from `B(s)`, with an MSE auxiliary loss that backprops **into B**. This directly forces B to retain the geometry the probe says is missing. The target (cube xyz) is already in the state observation — **not privileged**, merely unused by the standard FB loss.

- **Hook:** `agents/fb/agent.py::update_fb`. After `B = self.model._backward_map(goal)` (B is over the s⁺/goal batch), add:
  `aux = geom_coef * MSE( geom_head(B), cube_pose(goal) )` and `fb_loss += aux`.
- **`geom_head`:** new small module (e.g. `Linear(z_dim→3)` or 1-hidden MLP) in `agents/fb/model.py`, built only when `geom_coef>0`.
- **`cube_pose(goal)`:** the object xyz for each s⁺ in the batch. State obs already contains it; extract by the known slice (mirror `CUBE_SLICE`/the buffer's physics) or pass alongside the batch. Predict **absolute** cube xyz (B is goal-independent; the inner product with z=B(g) recovers relative geometry).
- **Flag:** `geom_coef: float = 0.0` in `configs/agent/fb.yaml` — default 0.0 ⇒ existing training path is byte-identical (isolation preserved).

### Secondary levers (sweep if primary partially works)
- **Terminal-reweighted ρ (C1):** weight the FB loss / aux toward near-terminal s⁺ (oversample states with cube near any dataset goal) so B gets resolution where placement is decided. Flag `terminal_reweight_k`.
- **Lower the ortho clamp:** state ortho_coef=1000 may over-isotropize B; sweep {1000, 100, 10} *with* the aux loss (note: ortho=1000 was best WITHOUT the aux — it may change once B has a geometry signal).
- **Contrastive near-miss (C2):** if the regression head helps R²(B→d) but not near-miss separation, add a triplet/InfoNCE on B with hard near-miss negatives.

## Falsifiable first run (1–2 weeks → days of compute)
Two tasks × 2 seeds, state, ~1e6 steps, vs the `geom_coef=0` baseline:
```bash
# .venv (torch). Sweep the new coef; 0.0 reproduces current FB exactly.
.venv/bin/python train.py domain=cube_single geom_coef=1.0 \
  ortho_coef=1000 lr_b=1e-4 num_train_steps=1000000 seed=1 device=cuda use_wandb=true
```
Then re-probe the checkpoint:
```bash
MUJOCO_GL=glfw .venv/bin/python -m scripts.factorization_probe \
  --config <run>/.hydra/config.yaml --checkpoint <run>/checkpoints/final.pt \
  --data-path datasets --out analysis/factorization_probe/geomfix_s1 --tasks all --device cpu
```

### Decision rule
- **R²(B→d) rises (0.25 → ≥0.6) AND state success rises toward 0.68** ⇒ fix works; scale seeds/tasks, then test zero-shot vs GCIQL-DrQ. **PURSUE.**
- **R²(B→d) rises but success does NOT** ⇒ geometry in B is necessary-not-sufficient; the bottleneck moves downstream (z=B(g) inference or the FlowBC actor) — pivot to the privileged-z / actor-swap probes (already scoped in the diagnosis eval).
- **R²(B→d) does NOT rise** (aux can't coexist with the FB successor objective, or trades it off) ⇒ the factorization genuinely can't hold object geometry in state without the pixel pipeline ⇒ stronger negative result; report and reconsider (object-centric encoder, etc.).

## Verification / guardrails
- `geom_coef=0.0` must reproduce current results bit-for-bit (run the existing tests + one short baseline).
- Watch the FB successor loss doesn't blow up when the aux is added (aux should *complement*, not dominate — sweep coef ∈ {0.1, 1, 10}).
- The win condition is **both** R²(B→d) and downstream success — geometry alone isn't the goal.

## Files (all additive / flag-gated)
- `agents/fb/model.py` — add optional `geom_head` (built iff `geom_coef>0`).
- `agents/fb/agent.py::update_fb` — add the flag-gated aux loss; expose `geom_coef`.
- `configs/agent/fb.yaml` — `geom_coef: 0.0`.
- `tests/` — unit test: `geom_coef=0` leaves the loss/graph unchanged; `geom_coef>0` adds a finite aux term and a head whose readout R²(B→cube) climbs on a synthetic fit.
