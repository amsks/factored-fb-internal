---
name: e3-sf-rewreg-campaign-20260709
description: "E3 corrected SF cell: v1 (RewReg) and v2 (RewReg-Shift) BOTH invalid -- they regressed the reward on psi(s') while GCDataset ties reward+mask to the CURRENT index. Bug fixed 2026-07-09 (commit fb83d68) along with a new psi_base scalar-TD control; NOTHING is running, GPUs idle, E3 needs a fresh 500k."
metadata: 
  node_type: memory
  type: project
  originSessionId: 1578c60e-8303-4d93-bbf9-fa70237631fc
---

**Bridge track.** The corrected SF cell from [[audit-sf-measure-20260706]] §6 (E3): reward IN
the SF loop via a LEARNED base. Design doc = `PAPER/fb-gciql-bridge/sf-design-space.md`
(untracked; the living map of what's tried / implemented / open).

## What it is
`sf_reward_regress` in `tools/wandb_mode_shim/gciql_flow.py`:
- `GCLearnedBValue` (sf_base='learned'): ψ trainable, trained ONLY by the bilinear reward
  model regressed on `batch['rewards']`. φ still trained ONLY by the SF vector TD (base
  stop-gradded in the target). Guards: regress ⇔ learned base.
- Discriminator: keeps Bellman-on-the-representation but grounds the basis in the
  TASK-WEIGHTED reward instead of the measure's ρ-weighted density.
- Anchors: bilinear GCIQL 0.90 (scalar-TD positive control), zg_iql 0.44,
  confounded SF band 0.09–0.16 (retracted, [[audit-sf-measure-20260706]]).

## 2026-07-09 audit — 3 points, all now handled (commit `fb83d68`, 21 tests green)
1. *Fixed bases optimize the wrong reward* — already the retraction; E3 IS the fix. No action.
2. **REAL BUG (fixed).** `r_hat` used ψ(s′), but GCDataset sets
   `successes = (idxs == value_goal_idxs)` and derives BOTH `rewards = successes-1` and
   `masks = 1-successes` from the CURRENT index — and the SF target's immediate feature is
   `b(s)`. So it was off-by-one against reward, mask, AND base at once. Now
   `r̂(s,g)=⟨ψ(s),ψ(g)⟩/√d` (pins ‖ψ(g)‖²=r_at_goal+shift at the goal).
   **⇒ v1 AND v2 are both invalid. Neither number is citable.**
3. *Goal-dependent mask under task-agnostic φ* — new config `..._rewreg_shift_cond`
   (`fb_forward_z=true`). The other remedy (goal-independent continuation mask =
   behaviour-SF) changes the fixed point ⇒ separate, unimplemented cell.

## Dead run history (do NOT read as results)
- v1 `RewReg` (`6dde39e`): ψ=0 saddle ({−1,0} rewards × symmetric bilinear model). Killed @260k.
- v2 `RewReg-Shift` (`sf_reward_shift=1.0`, `8afc7fc`): saddle escaped (sf_reward_loss ≈0.027
  ≪ 0.21 class prior ⇒ reward WAS expressible), but success 0.0 @100k/200k/300k. Killed @~333k.
  Carried bug #2. Logs `/dev/shm/sfrewreg2_logs/`.

## NEW: psi_base — the readout-vs-TD crux (design doc item 1b), IMPLEMENTED, NOT RUN
`config.psi_base ∈ {'learned' (default), 'frozen', 'identity'}` runs the NORMAL scalar IQL
expectile TD with the goal leg crippled exactly as the SF cells cripple it. Lifts the
`GCFrozenBValue`/`GCIdentityBValue` dispatch out of the `sf_feature_td` branch and applies it
symmetrically to the value net AND the critic. Tests `tests/test_gciql_psi_base.py` (4).
- If scalar-TD+frozen-ψ ≈ 0.9 while SF+frozen-ψ ≈ 0.1 ⇒ SF failure pinned on the VECTOR TD.
- If it also collapses ⇒ the frozen readout was part of SF's handicap; the SF cells never
  isolated the TD. Torch priors: idB_dnorm 65.3 vs idB_decond_dnorm **2.7 collapse**.

## RUNNING: tmux `psibase`, launched 2026-07-09 ~20:25
`scripts/launch_sf_psi_base.sh` (commit `1cec779`), 15 runs = 5 cells × 3 seeds @500k, all 8
GPUs, logs `/dev/shm/psibase_logs/`, dispatch log `/dev/shm/psibase_dispatch.log`.
Verified: 15 compute procs on GPU, no crashes, steps advancing.
1. `..._bilinear_d50_frozen_psi` — 1b frozen arm (vs 0.90 control)
2. `..._bilinear_identity_psi` — 1b identity arm (ψ(g)=g, d=28)
3. `..._bilinear_d28` — rank control for arm 2
4. `..._sf_feature_td_rewreg_shift` — E3 re-run, corrected reward semantics
5. `..._sf_feature_td_rewreg_shift_cond` — E3 + audit pt 3 (fb_forward_z)

**Group collision (cost 35k steps):** cell 4 was originally launched into the SAME wandb group
as the crashed pre-fix v2 runs. `results_map` keys on the group ⇒ the compiler would have
averaged buggy and corrected seeds. Group renamed to `...-RewReg-Shift-Fix`, those 3 seeds
relaunched (tmux `sfrewfix`), old groups marked `csv=E`.

**Monitor:** `.venv/bin/python scripts/psibase_monitor.py [--step 100000]` — live wandb vs the
living-CSV baseline at the same step, paired on the CONFIGURED SEED (the d50 baseline has 13
runs; positional pairing would compare seed 0 to seed 7).

**Slow first eval is normal, not a hang:** the step-1 eval runs a random-init flow policy, which
flails and makes mujoco's contact solver crawl (~19 s/episode × 5 tasks × 51 eps ≈ 80 min) while
its GPU shows 0%. `py-spy dump` shows `utils/evaluation.py:evaluate`. Don't kill it.

**Launch gotcha (cost one false start):** `PY` for these launch scripts must be the repo
`.venv/bin/python`. `run_gciql.py` is a *hydra launcher* that runs in the repo venv and
subprocesses ogbench `main.py` into `.venv-jax` itself. Pointing `PY` at `/dev/shm/.venv-jax`
(per [[jax-venv-location]], which is about the INNER process) dies on `No module named 'hydra'`
— silently, since the tmux session survives and only the per-run logs show it.

## First read @100k (2026-07-09 ~21:30, partial seeds; campaign still running)
- **Launch is healthy.** `bilinear_d28` tracks the d50 control seed-for-seed: s0 0.632 vs 0.672,
  s1 0.660 vs 0.688 (|Δ| ≈ 0.03). Rank 50→28 costs almost nothing at 100k.
- **BOTH 1b ARMS COLLAPSE under the ordinary scalar IQL TD.** identity-ψ 0.008/0.012 @100k
  (flat 0.01–0.04 through 400k); frozen-ψ 0.016 / 0.028 @100k (n=2, s1/s2; paired baseline
  0.688 / 0.728, per-seed Δ −0.672 / −0.700).
  ⇒ **the 8-cell SF grid never isolated the vector TD** — every cell in it used a frozen or
  identity ψ, and that choice alone kills the critic. The 0.08–0.16 band is NOT evidence against
  representation-level TD. Only the learned-base (E3) cells can speak to it.
- The two arms fail DIFFERENTLY (no numerical blowup in either; Q stays in [−20,−10]):
  - frozen-ψ **underfits**: critic_loss 4.9–7.0 vs control 1.3–1.9, q_mean −10 vs −19.
    A fixed random goal basis can't express Q(·,·,g).
  - identity-ψ **fits and still has no policy**: critic_loss 1.5–1.7, q_mean −17.4 ≈ control —
    healthy value regression, useless argmax over `a`. The audit's "linear proxy r=⟨s,g⟩ has the
    wrong argmax", demonstrated under the SCALAR backup.
- Caveat: value net and critic each get their OWN frozen random ψ (separate flax modules).
- Eval wall-time is itself a collapse proxy: a flailing policy makes mujoco's solver crawl
  (frozen-ψ 100k eval ≈ 20 s/episode vs d28's 0.23 s/episode).
- E3 corrected cells: success 0.000 (Fix, s2) / 0.024–0.032 (Cond) @100k. The fix DID take
  effect — seed 2 @80k: reward_loss 0.0200 vs pre-fix 0.0303, sf_loss 0.0974 vs 0.2336
  (metric keys are `training/sf/*`). reward_loss ≈ 0.02 ≪ 0.21 class prior ⇒ the reward IS
  expressible in the base; the zero success is a genuine SF failure, not a saddle artifact.
- frozen_psi has no 100k eval yet (its step-1 eval is the slow one).

## NEXT
Collect @500k (`evaluation/overall_success`, 3-seed mean). Watch `sf/sf_reward_loss` (does it
fit? vs the 0.21 class prior for {0,1}) and `sf/sf_rhat_mean`. Then results_map +
`compile_results_csvs.py --refresh` ([[results-csv-pipeline]]); fold the SF row into the
fb-gciql-bridge tex (tex owned by the parallel session — coordinate).

**Reading 1b:** frozen-ψ ≈ 0.9 under scalar TD ⇒ the SF collapse is the VECTOR TD.
Frozen-ψ also collapses ⇒ the frozen readout was part of SF's handicap and the whole 8-cell
grid never isolated the TD. Identity arm must be read against `d28`, not `d50`.

[[handoff-sf-ddpgbc-20260706]] [[handoff-sf-featuretd-20260704]] [[wandb-entity-amsks]]
[[always-run-in-tmux]] [[jax-venv-location]] [[commit-no-claude-coauthor]]
[[tmux-hassession-prefix-match-gotcha]] [[mdose-campaign-20260708]]
