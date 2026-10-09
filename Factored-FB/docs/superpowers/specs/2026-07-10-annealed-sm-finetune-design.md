# Annealed successor-measure finetuning from converged bilinear GCIQL

**Date:** 2026-07-10
**Branch:** `gciql-fb-bridge`
**Living doc:** `PAPER/fb-gciql-bridge/sf-design-space.md`
**Status:** design approved, not implemented. Queued behind the `tasknorm` + `simplesf` campaigns.

---

## 1. The question

Every measure-vs-IQL result we have is a **cold start**. This asks a different question: given an
*already-converged* value geometry (bilinear GCIQL, **0.904 ± 0.019**), does growing the successor-
measure loss slowly destroy it, and at what dose?

That separates two explanations we have never distinguished:

- **Fixed-point incompatibility** — the measure objective's optimum simply is not a good value
  function, so it destroys the geometry however you approach it.
- **Optimization path** — the cold-start collapse is an artifact of learning the measure *from
  scratch*, and a good geometry, once found, absorbs it.

### 1.1 The cold-start priors (E2, constant λ)

| cell | λ | @500k |
|---|---|---|
| shared legs | 0.01 / 0.1 / 1 / 10 | 0.897 / 0.881 / 0.873 / **0.917** |
| stopgrad-φ | 0.1 / 1 | 0.891 / 0.857 |
| stopgrad-φ | **3** | **0.176** |
| stopgrad-φ | **10** | **0.024** |
| stopgrad-φ | 30 | 0.001 (uninterpretable, q-band violation) |

Shared legs absorb even λ=10; stopgrad-φ has a **cliff between λ=1 and λ=3**. The cliff is the
only place a warm start can say something new.

---

## 2. The schedules

```
t       = network.step                       # starts at 0 -- see the prep script, section 4
lam(t)  = lam_max * clip(t / anneal_ramp, 0, 1)      # ramp, then HOLD
lr(t)   = lr * clip(t / anneal_lr_warmup, 0, 1)      # short warmup, then constant

L = L_IQL  +  lam(t) * L_SM                  # L_IQL untouched throughout
```

**Ramp 300k, then hold at `lam_max` for 100k.** Without the hold we would only ever see the
transient, never the steady state at full dose.

**The two ramps do not confound each other**, because they live on different timescales. At
`t = 10k`, when the LR reaches full value, `lam = lam_max/30` — for the sgphi cell that is λ ≈ 0.33,
an order of magnitude below its cold cliff at 3. The entire dose-response through the interesting
region therefore happens at constant LR.

**The LR warmup is not optional, and it cannot be faked by scaling the loss.** Adam is
approximately scale-invariant per parameter (`m/√v`), so multiplying the total loss by a constant
is a no-op. λ works because it changes the *relative* weight of `L_SM` against `L_IQL`, not the
global scale. A genuine LR ramp requires `optax.adam(learning_rate=<callable>)`.

Dense evals (`eval_interval = 25_000`, 16 points) turn each run into a **success-vs-λ dose curve**,
warm-started — one run traces what E2 needed a whole grid to sample.

---

## 3. Two facts about `restore_agent` that dictate the architecture

Verified against the actual checkpoint
(`~/bridge_ckpt_backup/gciql_outputs/factored-fb/GCIQL-State-Flow-Bilinear-d50/sd000_*/params_1000000.pkl`):

1. **`network.step` restores as `1_000_001`, and `opt_state` restores too.** `restore_agent` calls
   `flax.serialization.from_state_dict(agent, ...)` over the whole `TrainState`. A schedule keyed
   on `network.step` would therefore begin at 1M — fully ramped — and the experiment would silently
   be a constant-λ run.
2. **An LR schedule changes the `opt_state` tree.** `optax.adam(learning_rate=3e-4)` ends in
   `EmptyState`; `optax.adam(learning_rate=<callable>)` ends in `ScaleByScheduleState(count)`.
   Measured: the two trees differ. So a checkpoint saved with a constant LR **cannot** be restored
   into a scheduled optimizer — `from_state_dict` will raise.

`config` is a `nonpytree_field`, so it is *not* in the state dict: our new knobs survive restore,
and the param tree matches exactly (the measure loss adds no parameters).

---

## 4. `scripts/prepare_anneal_ckpt.py` — checkpoint surgery

Both problems in §3 dissolve if we write a **finetune-ready checkpoint** instead of fighting the
restored state. The script:

1. Builds a fresh agent from the *finetune* config (bilinear d50, scheduled `tx`).
2. Loads the anchor checkpoint and copies **`network.params` only** into it.
3. Leaves `opt_state` at its **fresh** `tx.init(params)` value — matching the scheduled optimizer,
   and giving Adam clean moments.
4. Sets `network.step = 0`.
5. Writes `params_0.pkl` into a new directory; the finetune config points `restore_path` there with
   `restore_epoch: 0`.

Consequences: every schedule starts at 0 (no `anneal_t0` offset, no trap), optimizer structures
match by construction, and **no `third_party/` file is touched**.

**Fresh Adam moments are a deliberate choice, not an accident.** Restored moments would be tiny at
convergence, so the first SM gradient would be *amplified* by `1/√v` — a concrete mechanism by
which one large early step could destroy the geometry and manufacture a spurious "the measure
objective is incompatible" result. Fresh moments + LR warmup removes that failure mode. This is the
standard finetuning recipe and it is what makes the dose-response attributable to λ.

---

## 5. Three cells × 3 seeds = 9 runs

All restore from the `GCIQL-State-Flow-Bilinear-d50` anchor, seeds 0/1/2, and inherit the E2 mix
config verbatim: `measure_only=false`, `measure_legs=critic`, `measure_ortho_coef=0`,
**`measure_scale_norm=true`**, `bc_coeff=3.0`, `value_arch=bilinear`, `value_latent_dim=50`,
**no `fb_forward_z`, no `project_z`** (both change the param tree and would break restore).

| cell | legs | λ ramp | LR warmup | cold-start reference |
|---|---|---|---|---|
| **Anneal-Zero** (control) | shared | `lam_max = 0` | 10k | — |
| **Anneal-SgPhi** | `measure_stopgrad_phi=true` | 0 → **10** | 10k | λ1 = 0.857, **λ3 = 0.176**, λ10 = 0.024 |
| **Anneal-Shared** | shared | 0 → **100** | 10k | λ10 = **0.917** (absorbed; cliff unknown) |

`Anneal-Zero` does double duty: it isolates "400k more IQL steps" from "the SM loss," and it
validates the restore pipeline end-to-end. Its first eval (`i = 1`) should read ≈0.90.

Every anneal run gets the same free restore check, because `lam(0) = 0` by construction: **if the
first eval is not ≈0.90, the restore is broken and the run must be killed.**

---

## 6. Knobs and guards

```python
config.anneal_measure   = False      # enable the lambda ramp; measure_coef is reinterpreted as lam_max
config.anneal_ramp      = 300_000    # linear ramp length, then HOLD at lam_max
config.anneal_lr_warmup = 0          # linear LR warmup length (0 = constant LR, today's behaviour)
```

`measure_coef` is `lam_max`. There is **no `anneal_t0`**: the prep script resets `step` to 0.

Guards in `create()`, fired whenever `anneal_measure`:

1. `not measure_only` (we are grafting onto the IQL losses, not replacing them).
2. `measure_scale_norm` (raw measure grafts blew the Q scale and crashed 18/18; audit-20260706 §1).
3. `measure_legs == 'critic'`.
4. `measure_coef >= 0` and `anneal_ramp > 0`.
5. Param-tree identity with the checkpoint: `value_arch == 'bilinear'`, `psi_base == 'learned'`,
   and **not** any of `fb_forward_z`, `fwd_two_tower`, `isfb`, `sf_feature_td`,
   `value_arch=='simplesf'`, `measure_test_states != 'rho'`, `measure_test_rank != 'full'`.

`anneal_lr_warmup > 0` is independent of `anneal_measure` (the control cell uses it too).

---

## 7. Tests (TDD)

In `tests/test_gciql_anneal.py`, jax-subprocess style.

1. **λ(t) is exact and holds.** With `anneal_ramp=100`, `measure_coef=10`: λ(0)=0, λ(50)=5,
   λ(100)=10, λ(200)=10 (hold). Read from the `anneal/lambda` info key.
2. **λ(0) = 0 contributes no gradient.** At `step=0` the gradient of `total_loss` w.r.t. the critic
   params equals that of `L_IQL` alone (compare against `measure_coef=0`), to `< 1e-8`.
3. **The measure loss is still *computed* at λ=0** (static shapes under jit) but scaled to zero —
   assert `measure/measure_loss` is present and finite in `info`.
4. **LR warmup is live.** Build the tx with `anneal_lr_warmup=10`, and assert the effective LR at
   counts 0, 5, 10, 20 is `0, lr/2, lr, lr`.
5. **The scheduled optimizer's `opt_state` tree differs from the constant-LR one** — the fact that
   makes the prep script necessary. Assert it explicitly, so a future optax upgrade that unifies
   them is caught.
6. **Prep script round-trip.** Write a finetune-ready checkpoint from a tiny agent, restore it, and
   assert: params equal the source's, `step == 0`, `opt_state` matches a fresh `tx.init`, and the
   restored agent takes an update without error.
7. **Guards** reject all invalid combinations from §6 and accept the three cells.
8. **Backward compatibility:** `anneal_measure=False` and `anneal_lr_warmup=0` reproduce today's
   `total_loss` and today's `tx` exactly.

---

## 8. Diagnostics and the pre-registered interpretability check

Log `anneal/lambda`, `anneal/t`, `anneal/lr`, the `measure/*` terms, and `critic/q_mean`.

**PRE-REGISTERED (from the E2 audit).** `critic/q_mean` must stay in the band **[−21, −8]**. The
raw, non-scale-normed measure grafts blew the Q scale and crashed 18/18 runs; every surviving E2
cell sat in that band. **If `q_mean` leaves the band, the cell is a scale blowup and its success
number is uninterpretable** — report that before reporting success, exactly as with the SimpleSF
`reward_loss` check.

**Restore check.** `anneal/lambda` must be ≈0 at the first logged step, and the first eval must be
≈0.90. Either failing means the restore or the schedule is wrong; kill the run rather than let it
burn 5 hours.

---

## 9. Reading it

- **SgPhi survives past λ=3** (where cold gives 0.176) ⇒ the cold collapse was an **optimization
  path**, not the fixed point. Annealing rescues the measure objective, and the bridge's central
  claim weakens to "the measure objective is hard to optimize from scratch."
- **SgPhi collapses near λ≈3 anyway** ⇒ the **fixed point itself** is incompatible with a good value
  geometry. This is the strongest possible form of the bridge's thesis: no schedule can save it.
- **Shared absorbs to λ=100** ⇒ φ co-adaptation has an enormous margin, and the sgphi cliff is
  entirely about denying φ that freedom. Combined with mdose (φ co-adaptation ≥10× margin), that
  localizes the conflict to the ψ leg.

The λ=0 control must land at ≈0.90. If it drifts, 400k extra IQL steps are themselves changing the
geometry and every anneal number needs re-baselining against it rather than against 0.904.

---

## 10. Anchors

| cell | @500k |
|---|---|
| bilinear GCIQL d50 (the warm-start source) | **0.904 ± 0.019** |
| E2-mix shared λ=10 | 0.917 ± 0.041 |
| E2-mix shared λ=1 | 0.873 ± 0.042 |
| E2-mix sgphi λ=1 | 0.857 ± 0.017 |
| **E2-mix sgphi λ=3** | **0.176 ± 0.047** |
| **E2-mix sgphi λ=10** | **0.024 ± 0.047** |

---

## 11. Cost and sequencing

9 runs × 400k ≈ 4–5 hours, plus a few minutes for the prep script (3 checkpoints). Queued **after**
`tasknorm` + `simplesf` free the GPUs.

After collection: add the three groups to `scripts/results_map.py`, run
`compile_results_csvs.py --refresh`, and fold the dose curves into
`PAPER/fb-gciql-bridge/sf-design-space.md`.

## 12. Out of scope

- Ramping the LR and λ on the same timescale (confounds the dose; see §2).
- Warm-transplanting the anchor's Adam moments (see §4 — it manufactures a failure mode).
- Annealing on antmaze (the mdose result says the measure is domain-conditional; a separate cell).
- Non-linear λ schedules (exponential, cosine). Linear first; revisit only if the cliff is sharp
  enough that the ramp resolution matters.

One change at a time.
