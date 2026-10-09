---
name: rewardz-launched-20260729
description: "2026-07-29 reward-z law campaign LAUNCHED (tmux rzlaw, 6 jobs) — the training-side cell for L2's broad-reward failure; plus the train.py `shared` whitelist trap that silently drops yaml knobs"
metadata: 
  node_type: memory
  type: project
  originSessionId: 7631d170-5136-4555-b389-3990c137f9c5
  modified: 2026-07-29T23:21:04.832Z
---

**Bridge track.** ⚠ UPDATE 2026-07-29 ~23:00: COLLECTED — verdict in ledger
**§9.11b**: WIN (primary half-met: ball1.6 0.07→0.47-0.55 clears, smooth0.8
0.33 misses the 0.4 bar; both guards PASS with success 0.887→0.948-0.950;
NULL not triggered; rzR25 shows it's not family-circular). **rzK50 adopted as
the default arm.** The 100-step neg_frac smoke signal did NOT survive to 500k
(all arms ~0.45) — refuted, do not cite. Continuation: [[validmeasure-campaign-20260729]].

The cell [[phase2-audit-20260728]] queued: extend the training
z law with synthetic-reward inference z's, because L2w localized the broad-reward
failure to the ACTOR's z-manifold rather than the inference rule.

`agents/fb/agent.py::_draw_reward_z` adds a third `sample_mixed_z` component —
`z = project_z(Σᵢ rᵢ B(s'ᵢ))` for a random synthetic reward over the batch's
next-states, i.e. exactly what the zero-shot readout produces. `reward_z_ratio=0.0`
is byte-identical (test-pinned + reproduced: `fb_loss=0.1662` @ step 100).

**tmux `rzlaw`**, 6 jobs, GPUs 0–4,7 (5/6 left for the other session), ~2.5h each
then 3 chained evals (L2 stage B → stage A → L1). Phase 1 REUSED from `p1_s4{2,3}`,
so the only delta vs control `p2_s4{2,3}` is phase 2's z law.
Arms: `rzK25`/`rzK50` (kernel = eval families' FORM, ratio .25/.50) and `rzR25`
(rff = `relu(cos(w·c+φ))`, a DIFFERENT form ⇒ non-circular zero-shot claim).
Runs land in `~/exp_spec10/torchfb/p2_rz*`, JSONs in `results/spec10_torchfb/`.

**Pre-registration is in `PAPER/rescuing-fb/PHASE2-LEDGER.md` §9.11** (written
before results): primary = broad-family gap_frac >0.4 while concentrated hold;
guard 1 = success within a seed-sd of 0.887; guard 2 = L1 stays 1.1–1.2×/ρ≥0.7;
**NULL specified** = if even the matched-form arm fails, the z-manifold is NOT the
defect and the next suspect is F's capacity at averaged z (bilinear rank bound).

⚠ **TRAP — `train.py`'s `shared` dict is an explicit whitelist.** A knob added to
`configs/agent/fb.yaml` but NOT to `shared` is accepted by hydra, written into
`.hydra/config.yaml`, and **silently dropped before it reaches the agent** — the
run looks correctly configured and is not. Caught only because two smoke runs with
different `reward_z_family` gave byte-identical losses. Now pinned by
`tests/test_reward_z_law.py::test_train_py_propagates_the_knobs`.
**Any past cell whose knob was added to a yaml without touching `shared` is
suspect and worth auditing** — a recorded config proves nothing on its own.

Early smoke signal (100 steps, not a result): `neg_frac` 0.374 control → 0.354 K25
→ 0.301 R50, i.e. broader z coverage eases the §9.9 negativity defect WITHOUT the
nonneg hinge's 2.5–3.6× mass inflation. Check at 500k.

Still owed after this: results-CSV refresh ([[results-csv-pipeline]]), the
uncommitted torch measure agents + spec 12 + this cell.
