# FB ⇄ GCIQL bridge: untried design-space catalog

_Written 2026-07-01. Revised same day after a systematic audit against the full wandb run
history (`amsks/factored-fb`, 784 runs / 123 groups). A prioritized map of design-decision
combinations we have **not** yet run, scoped to **FB and GCIQL only** (CRL and the GCIQL
ortho-coefficient sweep removed per direction — they are not part of the FB↔GCIQL comparison)._

Numbers are cube-single-state unless noted. GCIQL (jax) metric = `evaluation/overall_success`
@500k; FB (torch) metric = `success/overall`. All in % (×100). `±` = 95% normal CI.

Scope note: this is the **coherent frontier**, not a literal cross-product. §4 lists deliberate
omissions. **§0 records what the run history already covers** so nothing below re-proposes a
completed experiment.

---

## 0. What the run history already covers (audited 2026-07-01)

Two morph directions have already been run, and they must anchor everything below:

**FB-side bridge — start from FB, import GCIQL's value objective (`fb_bridge_*`, `fb_cube_*`, n=3):**
| Run | success | meaning |
|-----|--------:|---------|
| `fb_state_plain` | **48.0 ±5** | plain FB (measure), the FB anchor |
| `fb_state_iqlcritic_fix` | 23.6 ±6 | FB-IQL critic at the old incidentals (lr1e-4, ortho1000) |
| `fb_bridge_r0_base` | 58.7 ±8 | FB-IQL, goal-match + GCIQL mix (sane incidentals) |
| `fb_bridge_r1_decond` | 58.0 ±13 | + de-condition F→φ |
| `fb_bridge_r2_dnorm` | 65.3 ±6 | + √d normalization |
| `fb_bridge_r3_allon` | 56.7 ±2 | + both |
| **`fb_bridge_r3_bilinearv`** | **54.0 ±8** | **+ bilinear-V head — already tested; does NOT beat 57** |
| `fb_bridge_a0_cubexyz` | 62.7 ±7 | privileged cube-xyz reward |
| `fb_bridge_idB_dnorm` | 65.3 ±6 | identity-B (z=g) + √d |
| `fb_bridge_idB_decond_dnorm` | 2.7 ±1 | identity-B + de-cond + √d → collapse |
| `fb_cube_zg_iql` | 44.0 ±5 | z=g + IQL |
| `fb_cube_iql_learnedB_r1` | 20.7 ±15 | learned-B + IQL |
| `fb_cube_decouple{,_iql,_zher_traj}` | 39.3 / 20.7 / 18.0 | decouple actor-z |

**Takeaway that constrains this catalog:** on the FB-IQL base (~59), **every structural knob
stays in 54–65 — none reach GCIQL's 90.** The lever is the base/objective, not FB's structure.
The bilinear-V head is already refuted as the missing piece (54 < 57).

**GCIQL-side bridge — start from GCIQL, impose FB restrictions (jax, @500k):**
| Run | success | meaning |
|-----|--------:|---------|
| `GCIQL-State-Flow-Bilinear-d50` | **90.7 ±2** | the winning bilinear critic (the base for §2) |
| `GCIQL-State-Flow-Bilinear-d128` | 91.5 ±2 | d128 also works |
| `GCIQL-State-Flow-Bilinear-d5` | **9.6 ±4** | **d5 is a dead floor** (matters for §2 d-crossing) |
| `GCIQL-State-Flow-Bilinear-d50-TwoTower` | 86.4 ±2 | + FB two-tower forward map (nearly free) |
| **`GCIQL-…-B1_fbz_iql`** | **84.8** | **B1 DONE: `fb_forward_z` under the IQL objective (0.848/0.836/0.848) — z-forward structure is nearly free on the IQL base** |
| **`GCIQL-…-C2_measure_twotower`** | **7.6–8.4** | **C2 DONE: measure + two-tower forward map — no better than plain measure** |
| `GCIQL-Bil-FBz-O1000` | 25.5 | measure + z-conditioned forward map (C1) |
| `GCIQL-Bil-MeasureOnly{,-O1000}` | 9.9 / 8.8 | measure objective on the bilinear critic |
| `GCIQL-Bil-MeasureSA-*` | **0.0** | (s,a)-measure variants — all dead |

**Zero-shot GCIQL (E1a, 2026-07-07):** the 90 IQL critic is zero-shot via FB's API —
dataset-based moment `z=E[r·ψ]` = **95.5 = exact anchor**; the "inferred-z→7" G4 gap was
an on-policy sampling artifact, not the actor/inference formula. This partly answers the
Cluster-D "actor/critic z-source mismatch" question from the value side: with a well-sampled
z the value-net-ψ readout already conditions the actor to full success
([[zeroshot-gciql-e1a-works]]).

---

## 1. Axis inventory (tried ✓ / untried ○)

| # | Axis | Knob | Tried (result) | Untried ○ |
|---|------|------|----------------|-----------|
| 1 | Value objective | `measure_only` / IQL | ✓ FB measure (~9, `fb_forward_z`→25.5) · ✓ IQL goal-match (90) | — |
| 2 | Reward signal | reward src | ✓ dataset goal-match · ✓ privileged cube-xyz (63) · ✓ measure | — |
| 3 | Critic form | `value_arch` / `fwd_two_tower` | ✓ MLP (73) · ✓ bilinear (90) · ✓ two-tower alone (86) · ✓ bilinear-V on FB-IQL (54) | ○ two-tower **stacked with z-forward / sphere-norm** on the 90 critic |
| 4 | Fwd-map z-cond | `fb_forward_z` | ✓ flat · ✓ de-cond F→φ (58) · ✓ `fb_forward_z` on **measure** (25.5) | ○ `fb_forward_z` on the **IQL** critic (needs assert relaxed) |
| 5 | Ortho penalty | `iql_ortho_coef` | ✓ 0 (90) · ✓ 1000 (7, collapse) · ✓ FB ortho on FB side (in r-ladder) | ○ only at FB's own value (=1.0) inside §2 stacks — **not** a GCIQL sweep |
| 6 | Normalization | √d scale / sphere-Norm | ✓ soft √d scale (FB-IQL 65) | ○ **hard sphere-`Norm`(z,B) + ortho=1.0** on the 90 critic (not built) |
| 7 | Actor cond | `actor_cond` | ✓ `goal` · ✓ `z`-from-value-net · ✓ decouple actor-z (39/21/18) | ○ **`z`-from-critic-ψ** (audit #2; not built) |
| 8 | Double-Q / expectile | `double_q` / `expectile` | ✓ single-Q (88) · ✓ tau {0.5,0.95} (89,89; flat) | ○ crossed with §2 stacks (low value) |
| 9 | Latent dim | `value_latent_dim` | ✓ 5 (**9.6, dead**) · ✓ 50 (90) · ✓ 128 (91) · ✓ 512 (94@1M) | ○ new §2 knobs only at d50; cross at **d∈{128,512}** (NOT d5) |
| 10 | Domain | dataset | ✓ cube-state · ~pixel (partial) · ✓ antmaze · ✓ scene | ○ new §2 knobs only on cube-state |

---

## 2. Untried combinations, grouped as coherent experiments

Each entry: **config delta · hypothesis · discriminates · cost · tier**.
Cost unit = "evening" ≈ 3 cells × 3 seeds × 500k on 3 GPUs. Prior from §0 in _italics_.

### Cluster B — Full FB structure on the winning IQL critic ("FB skeleton, IQL blood")  **[P1]**
Progressively bolt the FB restrictions **not yet tested on the 90 critic** (two-tower is done = 86;
de-cond / √d / bilinear-V were tested only on the FB-IQL base, §0):
- **B1. DONE = 84.8** (`fb_forward_z` on the IQL critic, z-conditioned forward map, IQL-trained;
  0.848/0.836/0.848). A genuine z-indexed forward family is **nearly free** under the IQL backup
  (vs 25.5 under measure) → the z-forward structure is fine; the objective is the break.
  ([[audit-sf-measure-20260706]].)
- **B2.** `+ two-tower + fb_forward_z` combined.
- **B3.** `+ hard sphere-Norm(z,B) + ortho=1.0` (FB's true normalization on the 90 critic).
  - _Code prereq:_ sphere-`Norm` knob (#3) + drop-√d knob (#5).
- **B4.** **Full stack:** two-tower + z-forward + sphere-Norm + ortho=1.0, IQL objective.
  - _Hypothesis:_ _two-tower(86) already says FB structure survives on the 90 base; the rest should
    stay ~86–90._ If so, the entire FB↔GCIQL gap is objective, not structure — decisively.
  - _Discriminates:_ structure vs objective, from the GCIQL side. **This is the payoff cell.**
  - _Cost:_ ~2 evenings incl. code. _Tier:_ **P1**.

### Cluster C — Rebuild real FB *inside* the GCIQL code (close the bridge from the measure side)  **[P1]**
Start from `measure_only` and add FB restrictions back one rung at a time, targeting plain-FB=48:
- **C1.** measure + `fb_forward_z` — **done, 25.5** (§0).
- **C2. DONE = 7.6–8.4** (`+ two-tower`) — the two-tower forward map does **not** help the
  measure objective (no better than plain measure 9.9). ([[audit-sf-measure-20260706]].)
- **C3.** `+ sphere-Norm(z,B)`.
- **C4.** `+ ortho=1.0` (FB's true measure coef).
  - _Hypothesis:_ each rung recovers structure the graft dropped; C4 → ~plain-FB (48).
  - _Discriminates:_ if the reconstruction reaches 48, the 9%→48 deficit was **missing FB
    restrictions in the graft**, not anything GCIQL-specific — and the residual 48→90 is purely
    objective. Symmetrical bookend to Cluster B. _Prior:_ `MeasureSA-*`=0.0 warns the (s,a) measure
    form is fragile; keep the (s)-form.
  - _Cost:_ ~2 evenings (shares code prereqs with B). _Tier:_ **P1**.

### Cluster D — Actor z-from-critic-ψ (audit #2)  **[P1]**
- **D1.** `actor_cond='z'` sourced from the **critic ψ** (not the value net) for the best measure
  and best IQL-z-forward cells.
  - _Code prereq:_ route actor z from critic ψ; today z comes from `_value_psi`, untrained under
    `measure_only`. (Distinct from the `decouple` runs in §0, which separate optimization, not source.)
  - _Hypothesis:_ real FB conditions actor **and** critic on the *same* z; matching this closes the
    last conditioning mismatch.
  - _Discriminates:_ how much of the graft's deficit is actor/critic z-source mismatch vs the
    representation itself. _Cost:_ 1 evening + code. _Tier:_ **P1**.

### Cluster E — Cross-products (transfer / robustness)  **[P2]**
- **E1.** Two-tower at **d∈{128,512}** (NOT d5 — d5 is already a 9.6% floor, §0).
  - _Discriminates:_ is two-tower's free-ness dim-invariant on a *working* base? _Tier:_ **P2**.
- **E2.** Best bridge cells (B4 / C4 / two-tower) transferred to **antmaze / scene** (+ one **pixel**
  cell only).
  - _Discriminates:_ do bridge conclusions generalize off cube? _Prior:_ antmaze killer is the flow
    actor (0.22) not the bilinear value (0.88 w/ ddpgbc); scene FB 0.05 ≪ GCIQL 0.29 — so the open
    question is whether two-tower/structure behaves the same across domains. _Cost:_ per-domain
    evening; pixel is DrQ host-bound → **one** cell max. _Tier:_ **P2**.

---

## 3. Priority-ordered launch queue (if/when we run)

| Order | Experiment | Tier | Blocks on | Code needed |
|-------|-----------|------|-----------|-------------|
| ✓ | **B1 fb_forward_z-under-IQL** — **DONE = 84.8** | P1 | — | done |
| ✓ | **C2 measure+two-tower** — **DONE = 7.6–8.4** | P1 | — | done |
| 1 | D1 actor z-from-ψ | P1 | — | route actor z from critic ψ |
| 2 | B2→B4 further FB-structure-on-IQL | P1 | — | sphere-Norm; drop-√d |
| 3 | C3→C4 rebuild-FB-in-GCIQL | P1 | shares B's code | (same) |
| 4 | two-tower d∈{128,512} | P2 | — | none |
| 5 | domain transfer | P2 | B4/C4 pick winners | none |

Ranking rule = cheapness × open-question-closed. B/C are the scientific payoff (structure-vs-objective
decided from both directions), gated on the three shared code knobs. Note: the §0 FB-side evidence
already suggests B's likely outcome (structure survives on a strong base) — B/C mainly *confirm* the
structure-vs-objective split rather than open a new unknown.

---

## 4. Deliberately omitted (YAGNI / out of scope)

- **CRL and the GCIQL ortho-coefficient sweep** — removed per direction; not part of the FB↔GCIQL
  comparison. (The ortho *value* 1.0 still appears in B/C as one of FB's restrictions, not as a sweep.)
- **Anything on the d5 base** — d5 = 9.6% is a dead floor; structural knobs cannot be read there.
- **Bilinear-V head as a "gap-closer"** — already refuted (`fb_bridge_r3_bilinearv` = 54 < 57).
  _(Also fix the ladder-tex line that still lists it as an open candidate.)_
- **Full `{d}×{tau}×{double_q}×{ortho}` grid** — tau/double-Q shown flat (§1); cross only the best cell.
- **Pixel for every knob** — DrQ pixel is host-bound; only 1–2 winning cube-state cells transfer (E2).

---

## 5. Code prerequisites (shared by B/C/D)

Three default-OFF knobs in `gciql_flow.py` (all TDD):
1. **Relax `assert measure_only`** on `fb_forward_z` → allow the z-conditioned forward map under IQL.
2. **Sphere-`Norm`(z,B) knob** (#3) — hard ‖z‖=‖B‖=√d projection, FB's actual normalization.
3. **Drop-√d knob** (#5) + **route actor z from critic ψ** (#2).

E1 / E2 need **no** new code.
