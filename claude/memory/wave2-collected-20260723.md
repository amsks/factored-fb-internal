---
name: wave2-collected-20260723
description: "2026-07-23 — BALL-VISITATION measure WORKS (0.952/0.928 @500k, scale=MC truth 24.7); density-TD diverges on frozen ψ (3 seeds); FP verified 0.972/0.984; retrieval k=1 dirac 0.89–0.97; wave-2 occupancy dial dead; external research note two-phase-successor-measure-note.tex"
metadata: 
  node_type: memory
  type: project
  originSessionId: 401ebfbc-594d-45c0-8053-dc7f704f9f71
  modified: 2026-07-23T18:30:28.848Z
---

2026-07-23 (bridge track). Collection of the [[spec07-launched-20260722]] wave-2 campaign. Box idle, all s07* tmux sessions gone.

**Deconfounded occupancy dial (no-ortho + measure_scale_norm fix) — DEAD, confound excuse eliminated:**
- Corner-NoOrtho 0.000/0.012/0.004; Expectile-FP-NoOrtho 0.000/0.000; CellW030-NoOrtho s1 0.000.
- ⇒ ortho was NOT the only toxin for occupancy/expectile cells. Adding expectile+target_min+hindsight ON TOP of FP kills the 0.967 cell outright. First-passage semantics (plain, tau-less) is the necessary ingredient.
- CellW identity anomaly UNRESOLVED (theorem says w→0 ≡ GC-IQL 0.904; practice ≤0.17). Forensics' actor-centering fix (M=1+(1−γ)Q offset dilutes greedy signal ~3.8×, actor q/sg|q| norm not affine-invariant) was NEVER applied — still the prime suspect.

**CriticMix (7 seeds s0–s6):** finals 0.33/0.24/0.16/0.00/0.00/0.09/0.40 — every healthy seed shows mid-training excursions toward the anchor then collapse ⇒ z-in-critic linearity conflict confirmed as a slow-acting toxin (pre-registered prediction held).

**NEVER RAN (logs stop at launch command, tmux killed):** zctl_s0/s1 (FP-NoOrtho + z-actor single-difference attribution control — still the decider for the TPActor failure), w01no_s0, w03no_s0. GPUs all idle; relaunch is free.

**Inference-only zero-shot on frozen 0.967 ckpts (goal-proxy decode):**
- plain-z dirac means 0.416/0.408/0.568 (bimodal per-task); mixtures 0.94–1.00 on ALL seeds; renewal z-weighting ≈ no change on dirac.
- **RETRIEVAL decode: s0 dirac 0.680 (default), k=1 → 0.888, k=16 → 0.904** — near-anchor zero-shot WITHOUT any z-actor. s1 default 0.680, s2 default 0.456; k-sweep NOT yet run on s1/s2.

**Still open:** 0.967 verification gate (independent eval + extra seeds) — all claims provisional; results_map/CSV refresh for wave-2 groups NOT done ([[results-csv-pipeline]]); DECISIONS.md has no entry past 2026-07-21.

**How to apply:** the story is now "FP measure = trainable core; ortho, occupancy semantics, expectile-on-FP, and z-actors are four separately-demonstrated toxins; capability enters at inference (retrieval decode + GPI/mixtures) and occupancy enters as the renewal factor R." Prioritize: verification gate → retrieval k-sweep s1/s2 → dead-job relaunch → actor-centering CellW retest.

## LAUNCHED 2026-07-23 (~05:50, this session)
- **tmux `s07w3`** (`launch_spec07_wave3.sh`): the 4 segfaulted jobs relaunched — zctl_s0/s1 (z-actor attribution control), w01no_s0, w03no_s0. Cause of death was a first-eval segfault under 12-job contention; MAX=6 now.
- **tmux `s07p2`** (`launch_phase2_occupancy.sh`, queues behind w3): B `CellW0-NoOrtho-Centered` ×3 (new `actor_q_center` knob — centered q-normalizer, the forensics dilution fix); C1 `Occ-Phase2-FrozenPsi` ×3 (warm-start FP-NoOrtho 500k ckpt per seed + new `measure_stopgrad_psi` knob + plain occupancy TD); C2 `Occ-Phase2-TrainPsi` ×2 (ψ trainable control).
- Both knobs default-off bit-identical; tests `tests/test_gciql_phase2_occupancy.py` 3/3 + measure_axes/measure_scale/warmstart 13/13 green. Warm-start splice smoke-verified (all 5 modules from the FP ckpt). NOT committed.
- Read bands: B ≥0.85 ⇒ theorem cell realized; C1 works ⇒ occupancy failed by representation corruption, C1 floors ⇒ occupancy target unlearnable on play data.
- DECISIONS.md entry written (2026-07-23). CSV refresh + verification gate + retrieval k-sweep s1/s2 STILL OPEN.

## 8h-DEADLINE PIVOT (2026-07-23 ~06:00–08:00, user: "working successor measure in ≤8h")
- **C1 verdict came FAST and negative-mechanistic:** occfz (FB-form occupancy on frozen ψ) DIVERGES — q_abs 154→5.4e4 by 65k, eval 0.980(step-1, warm ckpt)→0.600@50k→0.416@100k (s0); s1 same shape (0.180@50k). The linear Dirac-diagonal attraction is unanchored EVEN with a perfect frozen backward map ⇒ FB occupancy estimator intrinsically divergent, independent of representation. s0/s1 KILLED after 100k (evidence banked in train.csv), s2 left running as the documentation seed.
- **THE REPAIR (launched ~07:50): ball-visitation measure** — new knob `measure_fp_absorbing=false` (with measure_first_passage): drop the (1−hit) absorption ⇒ M = E[Σ γ^t 1[ball(s+)]] — GENUINE occupancy of the goal ball (revisits counted, = the FP×R object) with FP's estimator hygiene: regressed 0/1 reward, γ-contraction, fixed point ≤ 1/(1−γ). Runs `vis_s0`(GPU3)/`vis_s1`(GPU7), warm-start FP ckpt + frozen ψ + actor_q_center=true (dilution defense — visitation M carries a shared R-mass baseline). Group `GCIQL-State-Flow-Measure-BallVisit-FrozenPsi-d50`. Tests: test_gciql_phase2_occupancy.py now 4/4 (absorbing-knob bit-identity + algebra) + measure_axes 6/6.
- **Gates running:** verify (independent 100-ep re-evals, seeds 100–102, tmux s07ver; occfz step-1 evals 0.980×2 already corroborate); retrieval k-sweep (tmux s07ret): **k=1 is the robust setting — s1 dirac 0.96@k1 vs 0.536@k16; s0 0.888@k1**. Monitor `bvfgu4l16` watches vis evals + q-inflation(>500) + gates.
- **Research note (user-requested, external-audience):** `PAPER/rescuing-fb/two-phase-successor-measure-note.tex` (+pdf, tectonic-clean) — standard-RL terminology ONLY (no "first-passage"/internal names; says "goal-reaching value E[γ^T]" / "visitation measure"), no overclaiming, algorithm boxes for both phases, success-rate table (phase1 0.967, +ortho 0.225, density-TD divergence, vis@50–250k), policy-extraction section (goal-cond / retrieval k=1 / GPI mixtures / honest failures: region+dense+linear ≈ chance, z-actors collapse). UPDATE its table when vis 500k + verify numbers land.
- **Deliverable framing for the deadline:** (i) FP measure verified, (ii) zero-shot via retrieval k=1 ≈0.89–0.96, (iii) occupancy = FP×R product (validated) + ball-visitation TD arm live, (iv) FB-form occupancy divergence = the mechanism negative.

## FINAL (2026-07-23 ~09:00): BALL-VISITATION COMPLETE — the working occupancy successor measure
- **vis_s0/vis_s1 BOTH complete 500k: finals 0.952/0.928, curves 0.89–0.99, zero instability, critic scale plateau ≈24 ≈ MC renewal truth 24.7.** Phase-2 recipe: warm-start FP-NoOrtho ckpt, freeze ψ (`measure_stopgrad_psi`), `measure_fp_absorbing=false`, `actor_q_center=true`. Group `...-BallVisit-FrozenPsi-d50`.
- Verification gate closed: fresh-seed re-evals 0.972 (s0, 100ep) / 0.984 (s1, 50ep); s2 pending. occfz_s2 confirms 3-seed density-TD divergence (0.652@50k decaying).
- Centered CellW0: 0.000 flat through 150k — centering fix trending DEAD for the expectile identity cell; ball-visitation supersedes it.
- External note FINALIZED with figures (pipeline + ball) + §2 baseline section; compiles clean. DECISIONS.md entry written.
- STILL PENDING: verify_s2, occtr, wave-3 (zctl/w01/w03), w0cent 250k verdict, **zero-shot eval on the vis ckpts (the real payoff of phase 2 — dense/region/linear rewards through ∫r dM)**, results_map/CSV refresh, COMMIT (knobs + tests + scripts + note all uncommitted).

## REVIEW-RESPONSE ROUND (2026-07-23 later)
- **zctl ATTRIBUTION FLIP: z-actor WORKS on FP — 0.960/0.996 @500k.** TPActor 0.395 was the z-PRIOR, not z-conditioning. π(a|s,z) is viable.
- **Dense rewards: retrieval k=1 solves the single-peak dense family on BOTH phases (phase1 0.976/0.992, phase2 vis 0.93/0.93) — it's the DECODE, goal-reducibility, not occupancy semantics.** The real open case = non-goal-reducible rewards (multi-region compromise, penalties, linear ≈ random-z control). Never claim "dense fails" unqualified again — route-specific.
- External review absorbed into the note (retitled "...Curriculum for Goal-Conditioned Visitation Values"; diagonal-slice §object; (z,x) program §dense; off-by-one E[γ^{T−1}] fixed; factorization = empirical hypothesis; capacity claim narrowed; reward-free framing qualified).
- **(z,x) ARM COMPLETE: 0.904/0.960 @500k, cold start, no collapse (criticmix mechanism ABSENT — matrix-residual z-grounding is the stable form).** Direct-z zero-shot eval (mode=z, no goal decode) running → `analysis/rescuing-fb/zeroshot_zx_s*_directz.json`. Next: visitation phase-2 on the zx form, then non-goal-reducible rewards. Was: `GCIQL-State-Flow-Measure-FP-ZX-d50` zx_s0/s1 — fb_forward_z + measure_z_random_frac=0.5 + z-actor, cold start, goal-reaching residual with z-indexed rows (z-consistent bootstrap). Spec06 single-axis assert RELAXED for exactly this composition (lattice test updated, 6/6). If stable → phase-2 visitation on top → π(a|s,z_r) direct, no goal decode.
- Other finals: occtr collapses both seeds (0.072/0.056@50k — trainable ψ dies faster); w0cent 0.000 through 200k (centering fix refuted for the identity cell); verify s1 formal 0.988.
- Monitor `bzyin29ru` watches zx evals + q-inflation + verify_s2 + visz.
- **DIRECT-Z EXTRACTION (zx ckpts, harness FIXED — two bugs: fb_forward_z critics need z_goals/z_vec in make_q_at_z, AND psi must come from the CRITIC for measure_only z-actors, not the untrained value net; both patched in t6_gpi_eval.py + eval_zeroshot_rewards.py):** dirac 0.90/0.92, mixture 0.87/0.89, **dense 0.90 vs 0.13 SEED-SPLIT (s1 inference misaims — open forensics)**; cos(z_r, ψ(g*))≈0.96. Naive-embedding route was 0.41–0.57 partly BECAUSE it used value-net ψ (random under measure_only) — recheck old plain-z numbers if ever cited.
- **Meeting report** `reports/2026-07-23-successor-measures.md` restructured intuition-first (play data → reach-first → algorithm → strips → aggregate incl. GC-IQL baseline); figures via `reports/figures/make_report_figures.py` (dataviz palette), rollout strips via `scripts/rescuing_fb/render_rollout_strips.py` (MUJOCO_GL=egl, all 5 tasks successful with vis_s0), OGBench play video (only cube-QUADRUPLE hosted) + strip. Note gained the formal/intuitive z-extraction paragraphs + direct-z numbers.
- **AUTONOMOUS OPERATING MODE (user directive 2026-07-23): FRUGAL — act only on decisive events, batch all writing, no speculative work, one wrap-up at the end. EXCEPTION: if zx crashes/diverges, drop frugal, brainstorm the next arm properly, launch it, then return to frugal.**
