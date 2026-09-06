# Delta audit — 2026-09-03 (read-only)

Scope: everything committed to `density-fb` since the 2026-08-28 critic audit
(`docs/notes/2026-08-28-rldp-audit.md`) and the 2026-08-30 network audit
(`docs/notes/2026-08-30-rldp-network-audit.md`), plus the hyperparameter configs.
Commits covered: 09c1ed1 (hilp_tdj divergence fixes), 732014e (Density-HILP),
ab0906d (rldp_density_tdj + cube_double), 54a0635 (HILP backward_input / oc substrate),
66799d2 / 42453d7 / 9a4fe8a (eval), fc50936 / 2d4327e (launcher clobber fixes),
608db03 (checkpoint/resume merge), cde57fd (z-coverage analysis).

References: official HILP = `~/git/baselines/HILP/hilp_zsrl` (sf.py, fb_modules.py,
utils.py, train_offline.py, in_memory_replay_buffer.py); td_jepa @ 840a745
(`metamotivo/agents/{sf,hilp}/`, `scripts/train/proprio/launch_{sf,rldp}_dmc.py`);
official RLDP = `~/git/baselines/rldp`. Nothing re-audited from the two prior notes.

Working-tree state at audit time: 5 dirty files from the other track
(`impls/critics/{fb,density_fb}.py`, `impls/utils/datasets.py`, `impls/agents/__init__.py`,
`PAPER/...`) plus ~15 untracked yamls/tests (`fb_oc`, `hilp_cube`, `density_fb_state`,
`rldp_portalign`, ...). Committed state audited via `git show HEAD:...` where it differed;
the test run (§9) is on the working tree as found.

---

## 1. impls/critics/hilp.py vs OFFICIAL HILP (hilp_zsrl/url_benchmark/agent/sf.py)

Line-by-line, both sides read in full.

**VERIFIED (loss arithmetic and readouts):**

* `hilp_value` clamp-inside-sqrt: hilp.py:74-82 == sf.py:139-142 (min=1e-6 on the square).
* `expectile_loss`: hilp.py:85-91 == sf.py:150-152; shipped expectile 0.5
  (hilp.yaml:35 == SFAgentConfig.hilp_expectile, sf.py:69) — the "inert asymmetry"
  trap is reproduced, not "fixed".
* `hilp_phi_loss` (hilp.py:94-134) == sf.py:154-175: indicator reward/mask, the THREE
  distinct reductions (min for adv's q, mean-of-target-heads for its baseline, per-head
  regression targets), target heads detached, online heads for v1/v2. Exact.
* SF stage (hilp.py:267-315) == sf.py:600-637: target psi at (s', z, a'), target_phi =
  ONLINE centered feature net detached (trap 4; sf.py:622), per-sample argmin-over-Q
  head pick (trap 5b; sf.py:627-628), scalar-Q regression under q_loss (trap 5a;
  sf.py:632-635 — seam's `sum(mean(...,axis=-1))` over [P,B] equals mse1+mse2), vector
  branch `sum(mean(axis=(-1,-2)))` == mse(F1)+mse(F2) at sf_loss_scale 1.0.
* Actor q: min over heads, -Q.mean() (hilp.py:318-330 == sf.py:690-694), no BC — the
  launcher delivers alpha=0, sampled action std 0.2/clip 0.3, motivo_archi, 1024-wide,
  q_normalize=false (launch_exorl_official.sh:106-111).
* z sampling + mix_ratio whitening (hilp.py:338-372 == sf.py:726-759): per-batch
  cov = phiᵀphi/N on the permuted CENTERED phi(next_obs), pinv, project, Bernoulli row
  mask. Equivalent to upstream's index-gather form.
* `infer_cond` (hilp.py:374-384): lstsq(phi_rows, r) then √d-normalize == sf.py:535-552,
  and `INFER_COND_MARGINAL = "next_observations"` MATCHES official's actual eval call —
  train_offline.py:155 passes `batch.next_obs` as the `obs` argument. Verified against
  the call site, not the parameter name.
* Running-mean centering at read (hilp.py:239-247 == sf.py:120-124); EMA 0.995/0.005
  from PRE-step online phi on `observations` (update_stats + Agent passing `self.net`
  as pre-step params, impls/agents/__init__.py:289-299 == sf.py:180-183). `running_std`
  carried, never read — matches upstream.
* Networks: `PhiMap(norm=False)` == `mlp(obs, h, "ntanh", h, "relu", d)` exactly
  (networks.py:56-75 vs fb_modules mlp/_nl); `HILPForwardMap` (networks.py:898-933) ==
  `ForwardMap(preprocess=True, add_trunk=False)` (fb_modules.py:148-193): two separate
  embedding branches shared across heads, `[obs_action, obs_z]` concat order, one
  ReLU trunk layer per head, feature_dim its own knob (512); init orthogonal gain 1.0
  + zero bias == utils.weight_init (utils.py:76-82). The 08-30-style network gap the
  RLDP port had does NOT exist here.
* Target-init quirk reproduced under a flag: `hilp_independent_target_init`
  (hilp.py:500-514) reproduces sf.py:107-114's re-init-after-copy; hilp.yaml:92 sets it
  true. Only the phi targets are independent, matching upstream (the successor target
  is a copy, sf.py:481 — and stays one in the seam).
* Stage/step ordering: `stages_from_pre_step_params: true` (hilp.yaml:89) makes both
  stage grads come from pre-step params, stepped together after — official's
  backward-backward-then-step (sf.py:663-675). Actor sees post-critic-step params
  (Agent apply_update) == sf.py:761-764 ordering.
* HILPFutureDataset (datasets.py:640-715) vs in_memory_replay_buffer.py:208-252:
  geometric offset ≥1, clip to the episode's last observation row (both sides stop one
  short of the final next_obs), then p_currgoal replacement at threshold
  `p_currgoal/(1-p_randomgoal)` FOLLOWED by p_randomgoal replacement — order and odd
  threshold reproduced exactly. Uniform-row random goals == upstream's
  random-episode/random-step on fixed-1000-step ExoRL.
* HP defaults vs SFAgentConfig (sf.py:26-69) and train_offline.py: batch 1024, z_dim 50,
  phi 512x2, sf hidden 1024 / feature_dim 512, sf_target_tau 0.01, phi tau 0.005
  (hard-coded upstream, sf.py:177-178), stats momentum 0.005, q_loss true,
  mix_ratio 0.5 (hilp.yaml:78), discount 0.98 (train_offline.py:68),
  hilp_discount 0.98, phi lr 5e-4 via `group_lr` (== lr_coef 5 × lr, sf.py:37,490),
  actor lr 1e-4 routed through group_lr so the actor yaml cannot clobber it,
  p_randomgoal 0.375 + per-domain hilp_discount/q_loss delivered by the launcher's
  `hilp*` branch (walker 0.96/false etc., matching the README commands). All VERIFIED.

**DISCREPANCIES (official arm), all LOW:**

* D1. Phi-target polyak ordering: official soft-updates the target phis BEFORE the
  optimizer step (inside `forward`, sf.py:177-178, called before backward+step at
  sf.py:669-675); the seam polyaks AFTER the step (agents/__init__.py:286-287), i.e.
  target mixes post-step params — a one-step lead. The hilp_phi_loss docstring
  (hilp.py:100-105) claims this "preserves the upstream ordering"; it does for the
  running mean (pre-step params ✓) but not for the targets. Numerically negligible at
  tau 0.005; the docstring is wrong, the code is a benign one-step variant. Seam side
  needs the doc fixed, not the code.
* D2. Discount: official's SF target uses the batch's per-row `discount` tensor
  (buffer γ·discount column, in_memory_replay_buffer.py:244); the seam uses scalar
  `self.discount` (hilp.py:297). No-op on ExoRL (no terminations), same latent
  divergence class as rldp item (c)-2 in the 08-28 audit.
* D3 (doc only). The module docstring's "KNOWN DIVERGENCES FROM OFFICIAL"
  block (hilp.py:42-57) is STALE — every listed item has since been fixed: mix_ratio
  is implemented (sample_cond) and set 0.5 in the yaml; phi lr is 5e-4 via group_lr;
  the successor net is HILPForwardMap (shared embeddings, gain 1.0); p_randomgoal /
  per-domain settings are expressible and delivered. The block now describes a critic
  that no longer exists.

## 2. hilp_tdj vs td_jepa's HILP (metamotivo/agents/{sf,hilp})

**The four 09c1ed1 fixes are correct**, each checked against the reference line:

* `hilp_q_loss` → false: SFAgentTrainConfig.q_loss defaults False (sf/agent.py:32);
  neither BASE_AGENT_CFG["hilp"] (launch_sf_dmc.py:74-81) nor sweep_hilp_walker
  overrides it. ✓
* `hilp_head_reduce` → mean: sf/agent.py:287-291 uses get_targets_uncertainty(next_SFs,
  sf_pessimism_penalty=0) = plain head mean. ✓
* `hilp_actor_head_reduce` → mean: sf/agent.py:324-327, penalty 0. ✓
* `hilp_sf_loss_scale` → 0.5: `0.5·P·mse over P·B·d` == 0.5 × the seam's
  sum-of-per-head-means; pinned numerically by test_hilp_tdjepa_parity.py:77-87. ✓

**Yaml knobs VERIFIED against launch_sf_dmc.py BASE_CFG + BASE_AGENT_CFG["hilp"]:**
sf_target_tau 0.001 / phi_target_tau 0.001 (launch:64-65), lr_features 1e-4 via
group_lr, expectile 0.5, prob_random_goal 0.375, hilp_future 0.99 (data.future),
discount = hilp_discount = 0.98 (td_jepa uses ONE train.discount for both the phi loss,
hilp/agent.py:102, and the SF target), batch 1024, z_dim 50, q_loss false,
mix rate 0.5 (train_goal_ratio default, sf/agent.py:28), stats EMA 0.005
(sf/model.py:126, center_features true per BASE_AGENT_CFG:79). The screen doc's
"both taus 0.001 ... future 0.99" line is accurate.

**UNLISTED divergences found (not in the screen doc's NOT-matched list):**

* U1 (MEDIUM-LOW). `hilp_independent_target_init: true` in hilp_tdj.yaml:69 (and
  hilp_density_tdj.yaml:56) reproduces OFFICIAL HILP's re-init-after-copy target bug —
  but td_jepa does NOT have that bug: setup_training applies weight_init FIRST and
  deepcopies targets after ("ensure that target nets are initialized after applying the
  weights", sf/agent.py:78-82; hilp/model.py:35-37 likewise for _target_features2).
  td_jepa's phi targets start as COPIES. The td_jepa-replication arm therefore
  bootstraps its Hilbert value against independent random target nets for the first
  ~1/tau ≈ few-thousand steps. Transient at 500k, but it is a knowing official-HILP-ism
  inside an arm whose stated purpose is td_jepa parity. td_jepa is "right" for this arm.
* U2 (LOW). z-mixing mechanism: td_jepa whitens with a RUNNING EMA covariance `_z_cov`
  (momentum 0.995, built from RAW phi at obs rows, sf/model.py:121-127) inverted with a
  1e-6 ridge (sf/agent.py:138); the seam uses official HILP's per-batch
  cov-of-centered-phi(next_obs) + pinv (hilp.py:363-368). Same rate (0.5), same
  projection; different covariance estimator. Distributionally close at B=1024, d=50,
  but it is a third mechanism difference on top of the disclosed tower differences.
* U3 (info). Random-goal source: td_jepa mixes future_obs with the batch's own permuted
  next_obs (sf/agent.py:179-186); the seam draws uniform dataset rows. Equivalent in
  distribution under iid batch sampling; noted for completeness.
* U4 (doc). The screen doc's NOT-matched item 1 — "Our predictor carries an
  embedding_layers: 2 stage; theirs has no such knob" — is WRONG: td_jepa's
  VForwardArchiConfig has `embedding_layers: int = 2` (nn_models.py:283) and the DMC
  BASE_CFG leaves it at 2, so the two sides MATCH on this. A disclosed divergence that
  is actually a match; the doc (and rldp_tdj.yaml's comment near the predictor block)
  should be corrected before any writeup cites it as a caveat.

Everything else the screen doc lists as NOT matched (towers 256x2-vs-512x2, missing
left_encoder, actor family, jax-vs-torch) is confirmed accurate; additionally td_jepa's
SF net and actor consume the LEFT-ENCODED obs (actor_encode_obs=True, sf/model.py:83-84,
sf/agent.py:278-285, 321-323) while the seam's consume raw obs — part of the same
disclosed tower-family divergence but worth naming explicitly.

## 3. Density-HILP (732014e, impls/critics/hilp_density.py)

* **What was swapped**: only the "sf" stage — vector/scalar SF regression replaced by
  the per-head categorical CE `-(t·logp).sum(-1).mean()` with
  `t = (1-d)·I + d·softmax(t_psiᵀ t_phi/τ)` (hilp_density.py:76-120); the phi stage is
  `super().stage_loss` verbatim (:78-79). q_fn becomes reward integration
  `Σ_j p_j·(phi(u_j)ᵀz)` at hard-frozen towers, grads through actions only, head mean
  AFTER integration, optional centering (:130-145) — same discipline as
  rldp_density/density_fb. VERIFIED.
* **Gradient ownership**: CE reads phi through stop_gradient (:89-90); pinned by
  test_hilp_density.py::test_density_ce_sends_no_gradient_into_phi with a non-vacuous
  companion assert that hilp_sf DOES receive gradient (:48-68). stage_groups("sf") =
  ("hilp_sf",) inherited. The isolation claim ("phi's training signal bit-identical
  across the pair") holds structurally. VERIFIED.
* **Parent path with the knob off**: there is no knob — HILPDensityCritic is a separate
  class + alias; hilp_tdj's code path is untouched by 732014e (the commit adds a new
  module, registry entries, a yaml, and 2 eval-map lines). Config inheritance is pinned
  by test_config_is_the_substrate_plus_density_knobs (the four parity knobs must equal
  the substrate's). The yaml is hilp_tdj.yaml verbatim + exactly
  temp/actor_temp/actor_q_center (diff checked). VERIFIED.
* **Invariant tests**: simplex/terminal-one-hot target rows (test:71-91), q_loss=true
  refused at build (HilpDensityConfigError, test:38-45). Present and meaningful.
* **Subtlety, benign (noted so nobody "fixes" it into a bug)**: the target pool reads
  `target["hilp_stats"]`, whose polyak tau is 0.0, so the target running mean is frozen
  at its init of ZERO — the target softmax pool is effectively UNCENTERED while the
  online pool is centered. This is invariant-benign: centering subtracts the same
  vector from every pool row, the induced logit shift `-psiᵀm` is constant across the
  pool axis, and softmax over that axis is exactly invariant. q_fn's centered pool adds
  a b-constant to Q (actor argmax invariant, and actor_q_center centers anyway).
  No action needed; a one-line comment would prevent a future "fix".
* **Observation (LOW)**: `sf_loss_scale` (0.5 in the tdj yaml) multiplies the CE
  (:107). rldp_density's CE carries no such factor, so the two density arms run their
  CEs at different effective lr on the successor tower. Internally consistent per arm;
  flag it if cross-density comparisons are ever quoted.

## 4. rldp_density_tdj.yaml + tdj family configs vs td_jepa BASE_CFG; launcher delivery

(Full knob table verified against launch_rldp_dmc.py BASE_CFG :15-74 +
FBAgentTrainConfig; every checked knob matches: batch 1024, z_dim 50, L_dim 256,
discount 0.98, taus 0.001, lr_f/lr_b 1e-4, ortho_coef 0.01 = the sweep's middle value
(BASE_CFG's 1 is never what the walker column reports), train_goal_ratio 0.5, norm_z,
horizon 5 via WindowedDataset, forward 1024x1, backward 256x2 norm, left_encoder 256x0
norm, predictor 1024x1 + embedding_layers 2 (which MATCHES VForwardArchiConfig — see
U4), predictor riding lr_b, num_parallel 2, joint single-phase, pessimism 0.
rldp_density_tdj.yaml = rldp_tdj.yaml verbatim + exactly temp/actor_temp/actor_q_center
— the ab0906d message says "five density knobs" but three keys were added; cosmetic.)

**F1 — DISCREPANCY (MEDIUM, live in the finished runs): rldp_tdj / rldp_density_tdj
trained with actor lr 3e-4, not td_jepa's 1e-4.** `impls/configs/actor/ddpgbc.yaml:1`
sets top-level `lr_actor: 3.0e-4`; the actor group merges AFTER the critic
(env/config.py:35-39, the contract test_config_merge_order pins), so it silently beats
rldp_tdj.yaml:29's 1e-4, and ACTOR_OV never overrides lr_actor. Confirmed from the
recorded config.json of every rldp_tdj / rldp_density_tdj seed (`lr_actor: 0.0003`).
The HILP arms escape only because hilp_tdj.yaml routes the actor lr through
`group_lr: {actor: 1.0e-4}`, which wins over the _LR_KEY lookup. The
rldp_tdj-vs-rldp_density_tdj ablation is internally clean (both clobbered equally), but
the rldp_tdj-vs-td_jepa replication ran the actor 3x hot. td_jepa is right. Fix:
`group_lr: {actor: 1.0e-4}` in both rldp tdj yamls (the hilp yamls' pattern).

**F2 — DISCREPANCY (HIGH for the comparison, config-delivery): the on-disk
hilp_density_tdj runs predate 2d4327e and are contaminated.** All three seeds
(exp/exorl-tdj/dmc-walker/hilp_density_tdj/*seed_{0,1,2}, launched Aug 31 22:49; the
glob fix landed Sep 1 06:25) recorded the old `hilp*` branch overrides:
hilp_discount **0.96** (yaml: 0.98) and the **1M** budget (vs hilp_tdj's 500k).
The script is fixed; the existing hilp_density_tdj-vs-hilp_tdj comparison is broken on
both discount and budget, and no post-fix relaunch exists under exp/. Do not quote the
pair until relaunched.

**F3 — DISCREPANCY (latent, LOW now): measure_hit_spec arrives sexagesimal.** The
escaped quotes in `--override "measure_hit_spec=\"${HIT[$DOM]}\""`
(launch_exorl_official.sh:193) are eaten when `${OV[*]}` is flattened into the tmux CMD
string, so runs record `measure_hit_spec: 1455.9` (OmegaConf parses `0:24:15.9` as
sexagesimal) — the exact footgun env/domains/dmc-walker.yaml warns about. Inert for
every current tdj arm (nothing reads the key at backward_input=state /
goal_kernel=false), but this path ships a garbage float to any future arm that reads it.

**F4 — minor latents**: (a) the launcher's DISC table has no dmc-quadruped → an empty
`--override discount=` on that domain; (b) pointmass would also need td_jepa's
lr_actor 1e-6, unhandled; (c) in-TRAINING monitoring evals use 50k inference samples
for all arms including tdj (launch_exorl_official.sh:50) — headline gate evals are
correctly 10k (below), but tdj training CURVES are z-inferred under the 50k protocol.

Launcher routing itself is clean post-fix: case order `hilp*tdj*` → `hilp*` → `*`;
rldp_density_tdj correctly falls to `*` (discount 0.98 no-op + inert rldp_phase1_steps
+ inert hit spec + ACTOR_OV); no other CRITICS name contains "tdj", so the glob cannot
over-match. Step budget 500k-vs-3M is the screen's deliberate choice, not a bug.

## 5. Eval harness consistency per arm family

* **Inference samples VERIFIED**: exorl_paper_eval.py:85-93 maps rldp/rldp_official/
  rldp_density_official/orthoa/orthob → 50k; hilp, hilp_density, and ALL FOUR tdj arms
  → 10k; fallback 50k with a printed disclosure (:271-274). 732014e added the two
  density_tdj entries BEFORE ab0906d made those arms launchable, so no eval could have
  run in a missing-entry window. 10k matches both td_jepa launchers and official
  HILP's own num_inference_steps=10000 (sf.py:43).
* **DISCREPANCY (episodes)**: `--episodes` defaults to 50 for EVERY critic
  (exorl_paper_eval.py:257) — there is no per-critic episode table. td_jepa's protocol
  is 10 episodes. A tdj arm evaluated without an explicit `--episodes 10` silently runs
  50. Variance/protocol deviation, not bias; `eval_episodes` is recorded per jsonl row,
  so past runs are auditable.
* **DISCREPANCY (task set)**: `--task-set` defaults to `paper` regardless of critic
  (:261-263); nothing couples `*_tdj` to the tdjepa table, so an unflagged tdj eval
  scores walker FLIP instead of SPIN. Visible in the output rows (task name embedded)
  but unguarded at launch. TDJEPA_TASKS matches td_jepa's ALL_TASKS exactly.
* **9a4fe8a VERIFIED**: the only reader of `rldp_official_nets` is RLDPCritic.build
  (rldp.py:470), reached by RLDPDensityCritic via super(); every alias routing to those
  classes starts with "rldp", and the pin's (key, "rldp") prefix covers exactly that
  set. No arm that needs the pin lost it; HILP no longer gets the spurious LEGACY-ARCH
  line. Legacy rldp_density/rldp_tdj checkpoints predating the key still get the pin.
* Discount at eval forced 0.98 for dmc-walker across families — consistent with all
  tdj yamls.

## 6. backward_input=hit_spec / oc substrate (54a0635)

* **Choke point VERIFIED**: hilp `_phi_in` (hilp.py:220-228) is routed by `_phi`,
  `_phi2` AND the value-loss `apply_phi` closure (:263) — no bypass. Same pattern as
  rldp `_b_in` (rldp.py:235-243) and psm `_phi_in` (psm.py:160-166).
* **Which critics carry the knob at HEAD** (correcting the brief): rldp, psm, hilp
  only. `git show HEAD:impls/critics/fb.py` has NO backward_input; density_fb's B
  unconditionally consumes `_density_queries` (hit_spec IS its baseline). fb and
  density_fb gain the knob only in the UNCOMMITTED working tree (other track), both
  with behavior-preserving defaults ("state" identity for fb; "hit_spec" for
  density_fb = its pre-existing behavior).
* **Default-off byte-identity VERIFIED** structurally (identity return, no extra RNG
  splits — the 4-way split at hilp.py:474 and the +7717 target split unchanged in
  count/order, new dataclass fields static) AND empirically: the boot-contract gate
  run on CPU → 11/11 passed against the fixture baselined at 87e2058 (an ANCESTOR of
  54a0635 — the correct discipline: baseline predates the change, the pass is the
  byte-identity proof). Coverage: 13 critics × 4 actor families including fb, hilp,
  density_fb.
* **Gap (LOW)**: `hilp_density` entered CRITICS at 732014e, AFTER the last fixture
  regen, so the gate computes but does not constrain it (no baseline rows). Regenerate
  the fixture when convenient.
* **Slice/scaling parity VERIFIED exact**: `scaled_hit_cols`
  (_measure_common.py:63-74) == density_fb `_density_queries` (density_fb.py:320-326
  at HEAD): same parse_hit_spec, same divide-by-eps, same block order. density_fb's
  trace-time `_check_rows` is replaced by the identical build-time max-column assert
  on the hilp/rldp/psm side. Shared latent footgun on BOTH sides: parse_hit_spec
  allows eps == 0 (→ inf); never exercised by any shipped spec.

## 7. Checkpoint/resume merge (608db03)

* **auto_resume × two-phase RLDP VERIFIED**: the phase is a pure function of the
  restored global step — `start_step = restore_epoch + 1` (impls/main.py:334-356),
  `phase_of(step)` recomputed and applied via set_phase before the loop (:395-415); the
  critic's `phase` is a static nonpytree attr, deliberately not serialized. Resumes
  mid-phase-1, mid-phase-2, and across the boundary all land correctly; the phase flip
  has no side effect beyond the static swap.
* **Hard-copy B-target VERIFIED**: keyed on the global loop index `i % hard_b_every`
  (main.py:432-437), which continues from the restored step — modulo alignment
  survives resume, matching official's global-step keying (rldp agent.py:196-197,
  250-step interval matches). The copied target lives in `agent.target`, a pytree
  field, serialized by save_agent and restored; the hard copy precedes both save paths
  in the loop, so a checkpoint at a copy step is post-copy. No python-side counter
  exists to reset.
* Optimizer state (opt_state + TrainState.step) and Agent.rng restored
  (params_only=False path); the numpy data-sampling RNG is deliberately re-seeded from
  (seed, restored step) — documented as non-bit-exact in docs/resuming_runs.md.
  Online-mode state (buffer, collector, both RNGs) checkpointed separately with a hard
  refusal when missing.
* Atomic save VERIFIED for `params_N.pkl` (tmp + os.replace; the latest-checkpoint
  regex cannot match a leftover .tmp; numeric max, not lexicographic). NOT atomic for
  online-mode `online_state_N.npz` (bare np.savez) — a mid-write kill poisons that
  epoch's resume loudly, not silently. Minor.
* **DISCREPANCY (minor, wandb)**: the jax path's wandb.init has no resume/id, so each
  resubmission creates a NEW wandb run under the same name; docs/resuming_runs.md's
  "continues the same wandb run" is true only for the torch path. After a hard kill
  (no signal checkpoint) the window between last checkpoint and the kill is retrained
  and re-logged — duplicate step rows in the CSVs and a two-run overlap in the group:
  the exact contamination class the `exclude_runs` memory note records. Signal-path
  resumes have no overlap.

## 8. Config hygiene across critic yamls (ortho_coef and domain calibrations)

32 yamls at HEAD + 6 untracked (other track). Current ortho_coef per yaml, against
what each reference actually ran per domain:

| yaml | ortho_coef | discount / batch / z_dim | calibration | reference verdict |
|---|---|---|---|---|
| fb.yaml | **1.0** | 0.99 / 256 / 50 | cube | **DISCREPANCY (HIGH)** — td_jepa's cube FB sweep is {100, 1000} (launch_fb_ogbench.py:91-92); 1.0 is the unswept BASE_CFG no published cube cell ran. Still live at HEAD; remediation exists only as UNTRACKED fb_ref.yaml (1000), whose header quantifies the cost (0.11 vs 0.42-0.53) |
| fb_oc.yaml (untracked) | 1.0 | cube | inherits the same bad value — will confound "oc input" with "wrong ortho" | DISCREPANCY (MED-HIGH, other track) |
| rldp.yaml + oc/gk variants | 0.01 | 0.99 / 256 / 50 | cube | ok — inside td_jepa's cube RLDP sweep {0.01, 0.1} (launch_rldp_ogbench.py:113) |
| rldp_tdj / rldp_density_tdj | 0.01 | 0.98 / 1024 / 50, L 256 | walker | ok — mid of DMC sweep {0.001, 0.01, 0.1} |
| rldp_official / density_official / orthoa | 1.0 (inert / live) | 0.98 / 1024 / 512 | ExoRL | ok — official ships 1.0 with the dead regularizer, reproduced faithfully; orthoa/b are labeled live-ortho probes (b: 1e-4) |
| psm family | 1.0 | 0.99 / 256 / 128 | cube | ok — PSM reference's configurable coef is DEAD code (`0*ortho_coef`, PSM agent/psm.py:519) and the proto branch is hardcoded `1*` (:464); psm.yaml's live 1.0 reproduces the effective reference coefficient |
| psm_repair pair | 1000 | 0.98 / 1024 | cube | own torch sweep winner, documented — never a "reference PSM" claim |
| hilp family | n/a (no ortho) | see §§1-2 | | ok |
| hilp_oc (+ untracked cube variants) | n/a | 0.99 / 1024 / 50, phi lr 5e-4 | cube | LOW — phi lr 5e-4 is official's ExoRL value; td_jepa's cube HILP swept {1e-4, 1e-5}; provenance split documented in the registry comment |

Cross-calibration: no cube value leaks into ExoRL arms (launchers override batch /
discount / z_dim / taus per domain; the unoverridden fb ortho 1.0 is inside td_jepa's
DMC FB sweep {0.1, 1, 10}); no ExoRL value leaks into cube arms (separate files).
Family-internal consistency EXACT: every density pair differs only by
temp/actor_temp/actor_q_center; every oc pair only by backward_input +
measure_hit_spec; orthoa/b only by the ortho flags. Note: the brief's "official RLDP
paper lambda=0.01" is the PAPER's value; the official CODE ships 1.0 with the
regularizer inert — the yamls follow the code and label the difference.

**cube_double.yaml (ab0906d) VERIFIED against ogbench cube_env.py:766-794**: obs 37 =
19 proprio (6+6+3+2+1+1) + 2×9 per block (pos 3 + quat 4 + cos/sin yaw 2); block pos
slices 19:22 and 28:31 correct; eps 0.4 = 0.04 m success radius × xyz_scaler 10.

## 9. Test run

{PENDING-TESTS}

## 10. z-coverage (cde57fd) — spot check

Eval-only as claimed: z_coverage_walker.py loads checkpoints, reads
RLDPCritic._B / HILPCritic._feature_net through the agent's own params, batches the
exact readout formulas, and GATES them against a live `infer_cond` call
(fidelity_check, refusing to report on mismatch). Writes figures/json only; no
training-path imports mutated. impls/utils/z_coverage_nulls.py is pure numpy. The
docstring's internal line refs (hilp.py:344-354) have drifted a few lines; content
correct.

---

## Verdicts

* **rldp-tdj**: FIX-FIRST — F1: the finished rldp_tdj / rldp_density_tdj runs trained
  the actor at lr 3e-4 (ddpgbc.yaml clobbers the critic yaml's 1e-4; confirmed in every
  recorded config.json). The internal density ablation is clean (both arms equally
  clobbered); the vs-td_jepa replication is not. Add `group_lr: {actor: 1.0e-4}` to
  both rldp tdj yamls (the hilp pattern) before quoting the replication.
* **hilp (official arm)**: SAFE — faithful transcription incl. all five traps, the
  networks, the buffer semantics, the launcher's per-domain overrides; two benign
  one-step/discount variants (D1, D2) and one stale docstring block (D3) to clean up.
* **hilp_tdj**: SAFE (with disclosures) — the four 09c1ed1 fixes are correct and
  pinned; every matchable optimizer knob is delivered. Two undisclosed divergences to
  add to the screen doc: independent target-phi init (td_jepa copies its targets — U1)
  and the z-mix covariance mechanism (running EMA + ridge vs batch pinv — U2). Also
  correct the doc's predictor-embedding_layers claim (U4: it is a MATCH, not a
  divergence).
* **density-hilp**: SAFE as code (estimator swap isolated, gradient ownership pinned,
  simplex invariants tested, config = substrate verbatim + 3 knobs) — but FIX-FIRST as
  data: F2, the three on-disk hilp_density_tdj seeds predate the 2d4327e glob fix and
  trained at hilp_discount 0.96 / 1M budget. Relaunch before quoting the
  hilp_tdj-vs-hilp_density_tdj pair.
* **rldp_density_tdj**: SAFE as config (rldp_tdj verbatim + 3 density knobs; knob table
  matches BASE_CFG throughout) — carries F1 like its substrate, equally on both sides
  of its own ablation.
* **eval harness**: SAFE with two silent-default hazards — episodes default to 50 for
  every critic (td_jepa protocol is 10; no per-critic table like INFERENCE_SAMPLES) and
  --task-set defaults to `paper` for tdj arms (flip scored instead of spin if
  unflagged). Both auditable after the fact from the jsonl rows; neither biases a run
  that passed the right flags. The 9a4fe8a pin scoping is exactly right. In-training
  monitoring evals use 50k samples for tdj arms (curves only, gate evals correct).
* **configs**: FIX-FIRST on one item — fb.yaml still ships cube ortho_coef 1.0, which
  no published cube FB cell ever ran (td_jepa swept {100, 1000}); the remediation
  (fb_ref.yaml, 1000) is untracked, and untracked fb_oc.yaml re-inherits 1.0. All other
  families check out against their references; family-internal deltas are exact;
  cube_double's slices verified against the env source.
* **resume/checkpoint**: SAFE — phase and hard-copy counters derive from the restored
  global step; state fully serialized; params save atomic. Minor: online npz save not
  atomic (fails loudly), and jax-path resumes open a NEW wandb run (docs overclaim
  "same run"; hard-kill resumes re-log the overlap window — the known exclude_runs
  contamination class).
