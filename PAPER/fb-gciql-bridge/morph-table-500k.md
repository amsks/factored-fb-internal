# FB ↔ GCIQL Bridge — Consolidated Morph Table

**Domain:** cube-single-play (state). **Metric:** `success/overall`, **@500k**, IQM/mean ±
sample SD over seeds, pulled fresh from wandb `amsks/factored-fb` (2026-06-30).
**Actor:** FlowBC ("flow") held fixed on **both** FB and GCIQL in the primary
comparison; a few **DDPG+BC** rows are the *actor-isolation control only* (not the
FB-vs-GCIQL comparison).

> **@500k vs @1M caveat.** The paper's grids report @1M. Several FB-side methods
> **peak near 500k and degrade by 1M** (zHER 46.6→32, q_loss=1 49.2→38.8), while the
> FB-IQL-critic *improves* (18.8→23.6). GCIQL bilinear is flat (~90→~93). So @500k
> flatters the FB family; cross-check the @1M column before quoting.

---

## 1. The morph axes (what each knob changes)

A configuration is a point in this knob space; FB sits at one corner, GCIQL at the other.

| Axis | FB end | GCIQL end | Knob(s) |
|---|---|---|---|
| **Objective** | successor-**measure** TD (density) | **IQL** value-TD (reward backup + expectile max) | `critic_mode` / `measure_only` / `measure_coef` |
| **Forward map** | `F(s,a,z)` task-conditioned | `φ(s,a)` z-blind | `fb_forward_z` (GCIQL side) / `decondition_F` (FB side) |
| **Backward map** | learned `B(s')` | raw goal `g` input / `B=I, z=g` | `fixed_b` (none\|identity) |
| **Critic form** | bilinear `F·B/√d` | bilinear `φ·ψ/√d` or MLP | `value_arch`, `value_latent_dim` (rank d) |
| **Task entry / inference** | post-hoc `z=E[rB]` (zero-shot) | exact goal `g` throughout | `actor_cond` (goal\|z), eval z-rule |
| **Ortho constraint** | `‖E[BBᵀ]−I‖`, λ=1000 | none (λ=0) | `ortho_coef` / `measure_ortho_coef` |
| **√d normalization** | yes | — | `d_norm` |
| **Reward** | none (density) | sparse goal-match | `iql_reward` (goal_match\|cube-xyz) |
| **Goal sampling** | random `s'∼ρ` | traj/cur/rand mix `[.5,.2,.3]` | `train_goal_ratio`, mix samplers, `zher`, `iwr` |
| **Actor** | flow | flow (primary) / Gaussian DDPG+BC (control) | `actor_kind` |

Anchors @500k: **FB ≈ 34** (measure, learned B, zero-shot z) → **GCIQL-bilinear ≈ 90**.

---

## 2. Direction A — GCIQL → FB (start at GCIQL, add FB ingredients)

Flow actor; cube-single; @500k. Top = pure GCIQL, descending = more FB structure.

| Config | obj | form | fwd map | actor | @500k success | n | wandb group |
|---|---|---|---|---|---|---|---|
| stock GCIQL (MLP) | IQL | MLP | — | DDPG+BC | **70.3 ± 4.9** | 10 | `GCIQL-State-DDPGBC` |
| GCIQL + flow (MLP) | IQL | MLP | — | flow | 65.5 ± 22.2† | 11 | `GCIQL-State-Flow` |
| **bilinear d512 + flow** | IQL | bilin | φ(s,a) | flow | **90.4 ± 2.7** | 10 | `GCIQL-State-Flow-Bilinear` |
| bilinear d128 + flow | IQL | bilin | φ(s,a) | flow | 90.7 ± 3.2 | 13 | `…-Bilinear-d128` |
| bilinear d50 + flow | IQL | bilin | φ(s,a) | flow | 90.4 ± 3.5 | 13 | `…-Bilinear-d50` |
| bilinear **d5** + flow | IQL | bilin | φ(s,a) | flow | **12.4 ± 10.3** | 13 | `…-Bilinear-d5` (rank cliff) |
| bilinear d50, **τ=0.5** | IQL | bilin | φ(s,a) | flow | 92.0 ± 2.8 | 13 | `…-d50-tau05` |
| bilinear d50, **τ=0.95** | IQL | bilin | φ(s,a) | flow | 87.4 ± 3.4 | 13 | `…-d50-tau095` |
| bilinear d50, **single-Q** | IQL | bilin | φ(s,a) | flow | 86.7 ± 3.8 | 14 | `…-d50-singleq` |
| bilinear d50 + **DDPG+BC** ‡ | IQL | bilin | φ(s,a) | DDPG+BC | 51.3 ± 8.3 | 13 | `…-d50-ddpgbc` |
| bilinear d512, **z-cond actor (G4)** | IQL | bilin | φ(s,a) | flow, z=ψ(g) | 92.8 ± 1.8 | 13 | `gciql_g4_zcond` |
| **↓ swap IQL value → FB measure loss (objective flip)** | | | | | | | |
| bilinear d50 + **measure-only** (λ=0) | measure | bilin | φ(s,a) | flow | **9.9 ± 5.3** | 3 | `GCIQL-Bil-MeasureOnly` |
| + **ortho λ=1000** | measure | bilin | φ(s,a) | flow | 8.8 ± 4.2 | 3 | `…-MeasureOnly-O1000` |
| + **`fb_forward_z`** (z-cond fwd map) ★ | measure | bilin | **F(s,a,z)** | flow | **25.5 ± 4.6** | 3 | `GCIQL-Bil-FBz-O1000` |

†one dead seed (0.0); healthy seeds ~64–80. ‡actor-isolation control. ★ committed `2812892` (2026-06-30).

**Reading A.** Going GCIQL→FB: the IQL **bilinear** form is the strong regime (~90,
flat over d∈{50,512}, robust to τ and single-Q). Swapping IQL's value for FB's
**measure loss** is the cliff — it drops bilinear-GCIQL from ~90 to **~9**. The
single most recovering structural fix is **`fb_forward_z`** (task-conditioning the
forward map back to `F(s,a,z)·B(s')`): 8.8 → **25.5** (~2.9×). The forward map's
missing task index is the dominant gap, but it only recovers a third of the way —
the measure objective is still far below the IQL value.

---

## 3. Direction B — FB → GCIQL (start at FB, add GCIQL ingredients)

Flow actor; cube-single; @500k. Top = pure FB measure, descending = more GCIQL structure.

| Config | obj | B map | reward | extra | @500k success | n | wandb group |
|---|---|---|---|---|---|---|---|
| **plain FB** (measure, zero-shot z) | measure | learned | none | ortho1000 | **~34**§ (48 @1M) | 10 | `fb_state_plain` |
| FB d128 | measure | learned | none | ortho1000 | 37§ (28 @1M) | 10 | `fb_z128_ortho1000` |
| FB + **zHER** (traj-z sampling) | measure | learned | none | | **46.6 ± 10.2** | 10 | `fb_state_zher` |
| FB + **IWR** (interaction-weighted) | measure | learned | none | | 22.7 ± 9.2 | 3 | `fb_state_iwr` |
| FB + IWR + matched-z | measure | learned | none | | 16.7 ± 5.0 | 3 | `fb_state_iwr_wz` |
| FB + **q_loss=1** (goal-TD added, measure kept) | measure+TD | learned | goal | | **49.2 ± 12.1** | 10 | `fb_state_iql` |
| **↓ replace measure → IQL critic (objective flip)** | | | | | | | |
| FB-IQL critic (old, env-term mask) | IQL | learned | goal | diverged | 2.7 ± 2.3 | 3 | `fb_state_iqlcritic` |
| FB-IQL critic (goal-mask fix) | IQL | learned | goal | ortho1000 | **18.8 ± 6.1** | 10 | `fb_state_iqlcritic_fix` |
| FB-IQL on-traj (ratio=1) + flow | measure+TD | learned | goal | | 50.7 ± 3.1 | 3 | `fb_iql_ontraj_flow` |
| FB-IQL on-traj (ratio=1) + DDPG+BC ‡ | measure+TD | learned | goal | | 26.0 ± 8.0 | 3 | `fb_iql_ontraj_ddpgbc` |
| **↓ Part-1 bridge ladder (IQL critic, goal_match reward, GCIQL mix, λ=0)** | | | | | | | |
| **r0** base (FB-IQL + GCIQL mix) | IQL | learned | goal_match | | **58.7 ± 8.3** | 3 | `fb_bridge_r0_base` |
| r1 + decondition F→φ | IQL | learned | goal_match | decond | 58.0 ± 14.0 | 3 | `fb_bridge_r1_decond` |
| r2 + √d norm | IQL | learned | goal_match | dnorm | **65.3 ± 6.1** | 3 | `fb_bridge_r2_dnorm` |
| r3 + both (decond+dnorm) | IQL | learned | goal_match | both | 56.7 ± 2.3 | 3 | `fb_bridge_r3_allon` |
| r3 + bilinear V head | IQL | learned | goal_match | +bilin-V | 54.0 ± 9.2 | 3 | `fb_bridge_r3_bilinearv` |
| **idB** + dnorm (B=I, z=g) | IQL | identity | goal_match | dnorm | **65.3 ± 6.4** | 3 | `fb_bridge_idB_dnorm` |
| idB + decond + dnorm | IQL | identity | goal_match | decond+dnorm | **2.7 ± 1.2** | 3 | `fb_bridge_idB_decond_dnorm` (collapse) |
| a0 privileged cube-xyz reward | IQL | learned | cube-xyz | | 62.7 ± 8.1 | 3 | `fb_bridge_a0_cubexyz` |
| **↓ z=g / learned-B isolation (ratio=1)** | | | | | | | |
| z=g, IQL critic | IQL | identity | goal | ortho1000 | **50.7 ± 15.1** | 3 | `fb_cube_zg_iql` |
| z=g, measure | measure | identity | none | ortho1000 | 24.7 ± 10.1 | 3 | `fb_cube_zg_her` |
| learned-B, IQL, ratio=1 | IQL | learned | goal | ortho1000 | 20.0 ± 5.3 | 3 | `fb_cube_iql_learnedB_r1` |
| learned-B, measure, ratio=1 | measure | learned | none | ortho1000 | 28.0 ± 8.7 | 3 | `fb_cube_her_learnedB_r1` |
| z=g, measure, z-renorm | measure | identity | none | renorm | 27.3 ± 5.0 | 3 | `fb_cube_her_idnorm_r1` |
| **↓ decouple actor-z** | | | | | | | |
| decouple actor-z, measure, perm goals | measure | learned | none | decouple | 44.0 ± 3.5 | 3 | `fb_cube_decouple` |
| decouple actor-z, IQL | IQL | learned | goal | decouple | 20.0 ± 5.3 | 3 | `fb_cube_decouple_iql` |
| decouple actor-z, measure, traj goals | measure | learned | none | decouple | 38.7 ± 3.1 | 3 | `fb_cube_decouple_zher_traj` |

§`fb_state_plain`/`fb_z128` @500k via the Part-1 ladder snapshot (history-access
holdout for the @500k pull); @1M anchors 48 / 28 are wandb-confirmed. ‡actor control.

**Reading B.** The **objective is the lever**: FB measure (~34) → FB-IQL with
GCIQL's goal-match reward + mix (**r0 = 59**) nearly doubles it, flow actor and
bilinear/learned-B held fixed. The structural knobs on top (decond F→φ, √d norm,
bilinear-V) are **flat** (54–65, all overlapping at n=3). Two cliffs: (i) **identity-B
+ decondition together collapses** (idB_decond_dnorm = 2.7) — you can drop the learned
B *or* decondition F, not both; (ii) the **naive FB-IQL critic with learned B** stays
low (18.8) — dropping B to identity (z=g) recovers it to ~44–51, confirming the
**learned backward map is FB's bottleneck**. Nothing in the measure framework lifts
FB above its own ~48 ceiling toward GCIQL's ~90.

---

## 4. Off-cube controls & sibling GC methods (@500k)

| Config | @500k success | n | group |
|---|---|---|---|
| GCIVL stock (MLP, AWR) | 49.6 ± 2.5 | 3 | `GCIVL-State-Stock` |
| GCIVL bilinear d50 (AWR) | 67.5 ± 10.6 | 3 | `GCIVL-State-Bilinear-d50` |
| GCIVL + flow, stock | 29.6 (bimodal 0/60) | 6 | `GCIVL-State-Flow-Stock` |
| GCIVL + flow, bilinear d50 | 26.4 (bimodal 0/79) | 9 | `GCIVL-State-Flow-Bilinear-d50` |
| HIQL stock | 23.5 ± 2.4 | 3 | `HIQL-State-Stock` |
| HIQL bilinear d50 | 15.9 ± 1.2 | 3 | `HIQL-State-Bilinear-d50` |
| HIQL flow low / high / both (bilin d50) | 18.1 / 19.7 / 22.8 | 3 ea | `HIQL-State-Flow-{low,high,both}-Bilinear-d50` |
| CRL flow | 31.1 ± 4.0 | 10 | `CRL-State-Flow` |
| CRL DDPG+BC ‡ | 21.2 ± 4.4 | 13 | `crl_state_ddpgbc` |

GCIVL+flow is **bimodal** (seeds either ~0 or ~79) — the flow actor destabilizes
GCIVL/HIQL the way it does on AntMaze. Note GCIVL-bilinear's healthy seeds (~79)
beat its native AWR head.

---

## 5. Code modifications shipped (last 2 weeks)

**FB side** (torch, `agents/fb`): `critic_mode=iql` (goal-cond IQL critic replacing
measure); goal-mask the IQL Bellman bootstrap; `zHER` (traj-future z sampling) + `IWR`
(interaction-weighted measure); bridge knobs `decondition_F`, `d_norm`,
`fixed_b=identity`, `iql_reward=goal_match`, `ortho_coef`, `decouple_actor_z`,
`bilinear_v`; z-inference ladder (B3) + MLP matched-readout (B2).

**GCIQL side** (jax, `tools/wandb_mode_shim/gciql_flow.py`): `gciql_flow` agent (IQL
critic + flow actor); `actor_kind=flow|ddpgbc` + `double_q`; `value_arch=bilinear` +
`value_latent_dim` (rank d∈{5,50,128,512}); `actor_cond=z` (G4); FB-measure graft
`measure_coef`/`measure_legs`/`measure_only`; `measure_ortho_coef` + **`fb_forward_z`**
(`GCFBValue`, commit `2812892`).

**Sibling GC methods:** bilinear value head for HIQL + GCIVL; flow actor for HIQL
(low/high/both) and GCIVL.

---

## 6. Bottom line

Both directions agree on the same two-layer story:
1. **Objective is the binding ingredient.** IQL value-TD on a bilinear form ≈ **90**;
   FB measure on the *same* form ≈ **9–34**. The flip costs ~55–80 points in either
   direction, with actor/form/rank/τ held fixed.
2. **The forward-map task index (`fb_forward_z`) is the largest single structural
   recovery on the FB side** (8.8 → 25.5), and **the learned backward map B is FB's
   bottleneck** (z=g recovers 18.8 → ~50). But neither closes the objective gap: the
   measure framework tops out near FB's own ~48 ceiling.

**Still missing / sparse (n=0 or unfilled cells):** `fb_forward_z` × {value-leg,
ortho sweep, larger d}; measure-graft as an *added* term to IQL (the `…-Measure-mc*`
runs crashed <340k); z=g × `fb_forward_z`; AntMaze/Scene versions of the measure
graft. These are the natural next runs to fill the grid.
