---
name: exorl-validation-campaign-20260828
description: "2026-08-28 NEW CAMPAIGN (user-directed): validate density objective × RLDP/HILP on ExoRL with PAPER-EXACT evals (not our T3 route) — vanilla replication gates the density arms; PSM decision deferred to results. Work home = ~/git/Factored-FB-seam (branch density-fb, fast-forwarded to 3bf818a with WIP kept); rldp_density/psm_density DRAFTS + cube smokes already exist untracked; HILP = fresh port from seohongpark/HILP, approved to run in parallel"
metadata: 
  node_type: memory
  type: project
  originSessionId: 74028a31-0c8e-407a-a040-e7761054a936
  modified: 2026-08-28T05:26:55.172Z
---

Measure track, follows [[spec31-zsearch-results-20260826]] ("keep our
interface" decided — encoder stays, search demoted).

**User's prescribed process (2026-08-28, follow in order):**
1. subagent line-by-line + logical validation of implementations,
2. extract the papers' OFFICIAL eval setups (the ones behind reported numbers),
3. per-env hyperparameters set correctly,
4. run vanilla + density variant per method — RLDP + HILP FIRST, vanilla
   replication must succeed BEFORE density arms run; report → user decides
   whether PSM joins.

**Branch/box state:**
- ~/git/Factored-FB-seam = checkout of density-fb, fast-forwarded
  29cca45→3bf818a (stash→ff→pop, one auto-merged __init__.py). WIP KEPT:
  modified rldp.py (SP off-by-one CORRECTION vs torch port, claims td_jepa
  alignment; + backward_input=hit_spec knob), psm.py, _measure_common.py,
  datasets.py (windowed sampling), extended seam tests.
- UNTRACKED drafts already exist: impls/critics/{rldp_density,psm_density}.py
  + config yamls (incl. _oc/_repair/_gk variants) + tests + launch scripts +
  exp/ cube-single smokes (08-24..26). The density-variant design is DRAFTED,
  not validated.
- ExoRL data LOCAL at ~/exorl_data (point_mass_maze/walker/cheetah RND 10k
  eps); no slurm on this box, 8 idle 23GB GPUs; the branch's slurm scripts
  are from another host — adapt to local tmux.
- Compendium docs/2026-08-25-exorl-humenv-compendium.md: prior arms — walker
  density_fb REPLICATES Touati (arch-shape 756.1±45.6 vs 760±19), cheetah
  does NOT (111.7 vs 247); psm/rldp under OUR T3 eval look broken/unstable
  (psm maze 0.34, rldp maze 129.7) — motivating paper-exact validation.
  ⚠ T3 (reward-inferred) vs T4 (goal-frame) are different evals, never mix.

**Investigation DONE 2026-08-28, spec COMMITTED (198eab1 on density-fb):**
`docs/specs/2026-08-28-exorl-official-parity-validation.md` + three evidence
notes (rldp-audit, rldp-paper-eval, hilp-paper-eval). Headline findings:
- Seam rldp.py = faithful **td_jepa-variant**, NOT official RLDP: official
  is TWO-PHASE (≈SP-only → B frozen for FB), z_dim 512 / batch 1024 / γ .98,
  pred-norm, actor pessimism .5 — six default-off parity knobs specced (§1).
  WIP off-by-one fix CORRECT vs both refs; rldp_density invariants verified
  (CE trains F+enc only); 61/61 seam tests pass in /dev/shm/.venv-jax.
- Official RLDP eval: z from 50k RE-SIMULATED-reward transitions, 16 tasks /
  4 domains / 4 seeds; RLDP Table 1 includes FB/HILP/PSM columns = the one
  replication target table (in rldp-paper-eval note). Official-code ortho
  term is INERT vs its own paper (footnote knob specced).
- HILP: z = LS regression over 10k transitions, √d-norm (our convention),
  1M steps, RND 5000 eps (jaco 20k) — local buffers are 10k eps ⇒ HILP arms
  subsample to 5000. ⚠ HILP's own Table-3 numbers UNTRUSTED (WebFetch) —
  re-extract from PDF via Read tool (spec §6.1).
- Gate: per domain ≥3/4 tasks within 1σ of official; on failure flip
  recorded ambiguity flags one at a time, never tune toward the number.
**User is dispatching an executor subagent on the spec.** Executor must
commit the WIP first (spec §5), work only in Factored-FB-seam.
