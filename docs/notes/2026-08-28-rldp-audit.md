# RLDP seam audit — 2026-08-28 (read-only correctness audit)

Scope: line-by-line validation of `impls/critics/rldp.py`, `impls/critics/rldp_density.py`
(untracked draft), and the supporting WIP (`impls/critics/_measure_common.py`,
`impls/utils/datasets.py::WindowedDataset`, the three test files) against:

1. **OFFICIAL** — `/home/adityamohan/git/baselines/rldp` (authors' PyTorch code +
   `RLDP_full_paper.pdf`, ICLR-2026 submission "Regularized Latent Dynamics Prediction is a
   Strong Baseline for Behavioral Foundation Models"). This is the ExoRL replication target.
2. **td_jepa** — `/home/adityamohan/git/baselines/td_jepa/metamotivo/agents/rldp/{agent,model}.py`
   (+ `buffers/parallel.py`, `data_loading/base.py`, `nn_models.py::VForwardMap`,
   `scripts/train/proprio/launch_rldp_dmc.py`).
3. **torch port** — `Factored-FB@gciql-fb-bridge:agents/rldp/{agent,model}.py` (the
   intermediate the seam claims to correct).

Headline: **the seam is a faithful transcription of td_jepa's RLDP, and the WIP off-by-one
fix is correct against BOTH references. But OFFICIAL RLDP is a structurally different
algorithm (two-phase, frozen-B) from td_jepa's joint-training variant, and the seam follows
td_jepa.** Every "vs official" discrepancy below should be read with that frame.

---

## The one structural finding that frames everything

**OFFICIAL RLDP is two-phase; td_jepa (and the seam) are single-phase joint.**

Official `rldp/rldp/agent.py::update` (lines 167–243):

* `step < representation_steps` (default 1M in `TrainConfig:35`; the ExoRL example sets 3M,
  `examples/rldp_train_dmc.py:142`): only `update_mr` runs (agent.py:245–298) — the SP
  rollout + (inert, see item d) ortho term, optimized by `encoder_optimizer` over
  **backward-map params only** (AdamW, hard-coded `weight_decay=1e-4`, agent.py:66–71).
  No F, no actor, no polyak; the SP target net is refreshed by a **hard copy of the `.zs`
  submodule every 250 steps** (agent.py:196–197).
* `step >= representation_steps`: `update_fb` + `update_actor` run; the FB TD loss detaches
  B (agent.py:321) **and only `forward_optimizer.step()` is called** (agent.py:351–356) with
  the ortho term multiplied by `0*` (agent.py:334) — so **B is completely frozen** in phase 2.
  Forward target polyak `fb_target_tau=0.01` (agent.py:240).

This matches the paper exactly: Algorithm 1 (appendix A.6) trains φ,g for `N_repr` steps,
then trains ψ,π with φ frozen; §4.2 "The RLDP representations are kept frozen in the
successor measure parameterization".

td_jepa `agents/rldp/agent.py::update` (67–128) runs **one joint update**: FB TD (B
detached, :158) + SP (:166–181) + ortho (:183–188), `forward_optimizer` +
`backward_optimizer` (which carries B, bw_encoder, predictor, :36–43) both stepped each
update, all three target nets polyak'd each step (:122–126). The seam's
`stage_loss` (rldp.py:215–256) is this update, verbatim.

**Consequence:** in the seam, F chases a moving B (B keeps learning from SP+ortho while F
trains); in official, F regresses onto a frozen B. Both are published designs (td_jepa is
Meta's own baseline harness), but they are *different algorithms*; results from the seam
replicate td_jepa's RLDP row, not the official repo's.

---

## Checklist

### (a) SP-pair alignment — VERIFIED (WIP correction is right; the torch port was wrong)

Reference enumeration for horizon h, transition anchored at t (action row i = action taken
at s_i in all three data layouts — official example loader `rldp_train_dmc.py:106–108`,
td_jepa `data_loading/base.py:100`, seam `env/exorl_env.py:378–387`):

* OFFICIAL `update_mr` (agent.py:255, 258, 269–271): `pred_zs = B(obs[:,0]) = B(s_t)`;
  loop i=0..H-1: apply `action[:,i] = a_{t+i}`, match `B_target(next_obs[:,i]) = s_{t+i+1}`.
  ⇒ step k (1-indexed) applies **a_{t+k-1}** and targets **s_{t+k}**.
* td_jepa (`buffers/parallel.py:100–108` + `agent.py:167–181`): prepend (a_t → s_{t+1});
  `traj_action` col i = `action[idx+1+i]` = a_{t+1+i}; `traj_observation` col i =
  obs row idx+2+i = s_{t+2+i}. Zip ⇒ identical pairing (a_{t+k-1} → s_{t+k}).
* Paper Eq 6 / Algorithm 1 line 5–6: h_{t+1} = g(h_t, a_t), matched to φ̄(s_t) at index t
  ⇒ same pairing.
* Seam `rldp.py::_sp_pairs` (:148–190) + `WindowedDataset` (datasets.py:576–587):
  `future_actions` col k = row t+1+k, `future_observations` col k = row t+2+k, prepended
  with (`actions`, `next_observations`). ⇒ **matches OFFICIAL and td_jepa (they agree).**
* Torch port `agents/rldp/agent.py::update`: `future_obs = batch["observation"][:, 1:]`
  (= s_{t+1}..s_{t+h-1}) where td_jepa reads the **next**-observation sequence
  (s_{t+2}..s_{t+h}). Off-by-one CONFIRMED: step 2 re-targeted s_{t+1} and the rollout
  never reached s_{t+h}. The seam docstring's history note (rldp.py:162–171,
  datasets.py:506–513) is accurate.

Pinned end-to-end by `tests/impls/test_rldp_seam.py::test_sp_pairs_on_a_real_window_implement_the_reference_alignment`
(:209–240) on row-tagged data through the real WindowedDataset — passes.

### (b) B detachment / which params each loss trains — VERIFIED vs td_jepa; official differs by schedule

* Seam: `Bg = stop_gradient(B(goal))` in the TD term (rldp.py:238) = td_jepa
  `B = self._model._backward_map(goal).detach()` (agent.py:158) = official phase-2
  (agent.py:321). TD trains forward + left_encoder only; SP trains backward + predictor
  (rldp.py:206–212); ortho trains backward (:249–252). Predictor rides `lr_b`
  (impls/agents/__init__.py:79–82) = td_jepa's backward_optimizer membership (:36–43).
  Pinned by `test_the_successor_residual_does_not_train_the_backward_map`,
  `test_the_sp_term_is_what_trains_the_predictor`,
  `test_stage_loss_itself_sends_no_backward_gradient_through_the_fb_residual` — all pass.
* vs OFFICIAL: same detach, but official never trains B and F **simultaneously** (see the
  structural finding). DISCREPANCY of schedule, not of stop-grad placement.

### (c) FB TD loss form — VERIFIED vs td_jepa; three official deltas

Residual `diff = M − γ·target_M`, `fb_off = 0.5·Σ(diff·off)²/off_sum`,
`fb_diag = −mean(diag(diff))·P` (rldp.py:239–243) is identical in all three
(official agent.py:324–327; td_jepa agent.py:161–164). Target side: mean-minus-uncertainty
over heads with `fb_pessimism_penalty` (seam `targets_uncertainty`, fb_math.py:12–17 ==
official `get_targets_uncertainty`, agent.py:404–419), penalty 0 everywhere. Deltas vs
official:

1. Official phase-2 `target_B = self._model._backward_map(goal)` under no_grad — the
   **online** B (agent.py:315), not a target net (equivalent there because B is frozen);
   seam/td_jepa use `B_target(goal)` (rldp.py:228; td_jepa :150). Benign given the
   respective schedules.
2. Official/td_jepa discount is the row vector `γ·~terminated` (official agent.py:208,
   td_jepa :76); seam uses scalar γ (rldp.py:219, and WindowedDataset emits
   `terminals = 0`, datasets.py:573). **No-op on ExoRL** (no terminations in the data;
   official's own phase-1 comments out termination handling, agent.py:180) but a latent
   divergence on terminating domains. (`rldp_density` does handle terminals — see h.)
3. Actor-side aggregation: official actor uses `mean − 0.5·unc`
   (`actor_pessimism_penalty=0.5`, agent.py:392 + example :80); seam `q_fn` is mean-only
   (rldp.py:264–267, "actor_pessimism_penalty is 0 in scope") = td_jepa's default 0.0
   (fb/agent.py:33). DISCREPANCY vs official.

### (d) Orthonormality regularizer — DISCREPANCY vs official code AND paper form; VERIFIED vs td_jepa

* Seam (rldp.py:249–254): `Cov = B(obs)@B(obs)ᵀ`; loss `ortho_coef·(0.5·Σ(Cov·off)²/off_sum
  − mean(diag))`, added to the joint loss. Identical to td_jepa (:183–188, on
  `B(obs_bw)`), and the anchor at B(obs) (not B(goal)) matches both official
  (`pred_zs = B(obs[:,0])`, agent.py:258–263) and td_jepa.
* OFFICIAL code: computes both terms but the phase-1 loss keeps **only the diag term**
  (`loss = ortho_coef * orth_loss_diag`, agent.py:267 — offdiag dropped), and with
  `b.norm=True` (default) the diag is the constant z_dim ⇒ **the official code's ortho
  regularizer is effectively inert**; phase 2 multiplies the whole term by `0*`
  (agent.py:334). This contradicts the official paper, whose Eq 7 is the **offdiag**
  cosine-similarity penalty `L_r = E[φ(s)ᵀφ(s′)]` (linear, not squared) with λ=0.01
  (§4.1 / Appendix A.4.2). So: seam ≠ official code (which appears buggy vs its own
  paper), seam ≈ paper intent but with FB's squared-offdiag form (td_jepa's choice), and
  the seam coefficient 0.01 equals the paper's λ and sits inside td_jepa's own sweep
  {0.001, 0.01, 0.1} (`launch_rldp_dmc.py:83`).
* Minor: official normalizes `pred_zs` before Cov when `b.norm=False` (agent.py:259–262);
  the seam has no such fallback if `backward.norm: false` were set (it is true in scope).

### (e) Predictor P architecture / input convention / normalization — VERIFIED vs td_jepa; DISCREPANCY vs official

* Seam `PredictorMap`/`_PredictorTower` (networks.py:186–247): ensembled (P=2), per-head
  slice `x[p]` with shared action (`in_axes=(0, None)`), `embed_z([x, action])` +
  `embed_s(x)` → concat → trunk → Dense(z_dim), **no output normalization**. This is
  exactly td_jepa's `VForwardMap` (nn_models.py:290–321) with `(obs=z-embedding,
  z=action)` as built by td_jepa `rldp/model.py:33`, including the expand-broadcast
  semantics (documented at networks.py:224–232). The torch port builds the same module.
* OFFICIAL predictor is `EncoderMap.model_unroll` (nn_models.py:324–332): **single-head**,
  `za = elu(Linear(action→256))`, `zsa = BaseMLP([zs,za]→z_dim, elu+layernorm)`,
  `Linear(z_dim)`, and — with `norm=True` — the output is **√d-hypersphere-normalized
  every step**. The paper explicitly ablates hyperspherical normalization on g and reports
  "consistent improvements" (§4.1). The seam predictor output is unnormalized ⇒
  DISCREPANCY vs official/paper-preferred design.
* B itself: seam `PhiMap` + `psm_norm` (networks.py:49–75) == official `BaseMLP` norm /
  `Norm()` (√d·F.normalize, eps 1e-12) == td_jepa BackwardMap. VERIFIED (depth differs —
  see j).
* SP loss scale: official per step `F.mse_loss(...).mean()` = mean over (B, z), **summed**
  over H (agent.py:269–271, no ÷H, no ensemble); seam/td_jepa: `sum over z, mean over
  (P,B)`, then **÷h** (rldp.py:212–213; td_jepa :180–181). Constant-factor difference
  z_dim/h (≈10 at the seam config) in the SP:ortho weighting vs official. DISCREPANCY vs
  official; VERIFIED vs td_jepa.
* SP target-net convention: official hard-copies `_target_backward_map.zs` every 250 steps
  (agent.py:196–197); seam/td_jepa polyak (`b_tau` each step, rldp.py:99–102 TARGETS,
  td_jepa :124). Predictor has no target anywhere — VERIFIED
  (`test_predictor_has_no_target_network` passes).

### (f) z sampling / task inference — VERIFIED against both references

* Train mix (rldp.py:273–284): z = 50/50 mask of `project_z(N(0,I))` and
  `project_z(B(next_obs[perm]))`, online B, no grad. Identical to official
  `sample_mixed_z` (agent.py:117–127) and td_jepa (fb/agent.py:145–156);
  `project_z` (fb_math.py:5–9) == official `model.project_z` (model.py:117–120,
  √d·normalize). `train_goal_ratio=0.5` matches both.
* Eval, reward-based: seam `infer_cond` z = project_z((rᵀB)/N) on `next_observations`
  rows (rldp.py:286–289, `INFER_COND_MARGINAL="next_observations"` :96). Official
  `reward_inference` is project_z(Σ rᵢ·B(next_obsᵢ)) (model.py:128–136; example eval
  :315–334 uses relabelled rewards at `batch["next"]["observation"]`); paper Alg 1 line 26
  uses the mean. Sum vs mean is erased by the projection under `norm_z=True` (the only
  configuration in scope). VERIFIED. (Official's softmax-weighted `reward_wr_inference`
  is a humenv-only extra, not the DMC eval path.)
* Goal eval: seam `map_eval_cond` z = project_z(B(goal)) (rldp.py:111–119) == official
  `goal_inference` (model.py:141–143). VERIFIED.

### (g) Windowed dataset sampling — alignment VERIFIED; boundary handling DISCREPANCY (minor)

* Column semantics (datasets.py:582–587): `future_actions` col k = actions row t+1+k,
  `future_observations` col k = observations row t+2+k; seam ExoRL loader guarantees
  actions[i] = action taken at observations[i] (exorl_env.py:378–387, same `[:-1]/[1:]`
  shifts as td_jepa base.py:47–52 and official example :106–108). Together with (a):
  VERIFIED, and `test_observation_rows_lead_action_rows_by_one_step` +
  `test_window_contents_are_the_actual_next_h_rows` pin it.
* Episode boundaries: both references **exclude** window starts that would cross an
  episode end (official samples `start ∈ [0, L−H]`, agent.py:151; td_jepa propagates the
  done flag back H−1 steps, parallel.py:48–52, so the deepest window still reaches the
  episode's true last state). The seam instead samples **every** start and clamps both
  windows to the anchor's terminal transition row (datasets.py:582–585). Two effects:
  (i) clamped windows train the predictor on repeated (a_final → s_final) self-loop pairs
  the reference never sees; (ii) the clamp ceiling is `observations[final] = s_final`, so
  even the repeat convention stops one state short of the trajectory's true last state
  `next_observations[final] = s_{final+1}`, and the unclamped windows can never target it
  either (the reference can). Magnitude: (h−1)/1000 ≈ 0.4% of samples at h=5 on ExoRL's
  fixed 1000-step episodes. Severity: LOW for ExoRL replication; the docstring
  (datasets.py:515–523) honestly documents the clamp as a local convention choice.
* `terminals` hard-zeroed (datasets.py:573): matches FBDataset's always-γ convention and is
  correct for ExoRL; see (c)-2 for the latent divergence.

### (h) rldp_density.py invariants and CE construction — VERIFIED

* Inheritance: `_sp_loss` inherited and re-summed with the ortho term verbatim
  (rldp_density.py:142–149 vs rldp.py:246–254); B-detach preserved
  (`Bg = stop_gradient(B(goal))`, :127). The gradient-isolation test
  (`test_density_ce_trains_exactly_what_the_parent_successor_term_trains`,
  test_rldp_density.py:137–157) proves backward+predictor gradients are bit-identical to
  the parent's at equal params and that the CE does train F — passes.
* CE: `logits = FᵀB(goal)/τ`, `logp = log_softmax(axis=-1)` (:128–129); target
  `t = sg[(1−d)·I + d·softmax(F̄ᵀB̄/τ)]` per head, `d = γ·(1−terminal)` reshaped [B,1]
  (:101–110, :133–139); `ce = −(t·logp).sum(-1).mean()` (:140). Rows stay on the simplex,
  terminal rows collapse to one-hot — pinned by
  `test_target_rows_are_simplex_and_terminal_rows_collapse` (also checks targets stay
  head-wise; the build refuses `fb_pessimism_penalty≠0` on distributions, :221–227).
  Consistent with `density_fb.py`'s estimator (softmax over the last/pool axis
  :836–876, temp default 1.0, per-head `w_bar`, `t = sg[(1−γ)k + γ·w_bar]` :876) — the
  row-discount is a strict superset of density_fb's scalar γ (equal when terminals=0,
  which WindowedDataset guarantees).
* q_fn: reward integration `Σ_j w_j·(B(u_j)ᵀz)` at hard-frozen towers, grads only
  through actions; ensemble mean AFTER integration; `actor_q_center` centers only the
  second return (:192–219). Matches its own spec and density_fb's discipline; pinned by
  `test_q_fn_matches_manual_integration_and_freezes_towers`. The goal_kernel /
  backward_input=hit_spec gates and their build refusals behave as documented (tests
  :247–455 all pass). Note: this critic is a seam-native research arm — there is no
  external reference to check the CE against beyond density_fb; internal consistency is
  what was auditable, and it holds.

### (i) Test run

`.venv` does not exist in Factored-FB-seam; the Makefile names the interpreter
(`VENV ?= /dev/shm/.venv-jax`). Ran:

```
JAX_PLATFORMS=cpu /dev/shm/.venv-jax/bin/python -m pytest \
  tests/impls/test_rldp_seam.py tests/impls/test_rldp_density.py \
  tests/impls/test_windowed_dataset.py -x -q
→ 61 passed, 1 warning in 100.84s
```

The suites are substantive: numpy loss reconstructions at 1e-10, end-to-end alignment pin
through the real dataset, gradient-isolation checks, named-error lattice checks.

### (j) Hyperparameter defaults, side by side

| knob | seam rldp.yaml / rldp_density.yaml | OFFICIAL (ExoRL example) | td_jepa (DMC rldp launch) | flag |
|---|---|---|---|---|
| batch_size | 256 | 1024 (example :55) | 1024 | **MISMATCH both** |
| z_dim | 50 | 512 (example :58; paper sweeps 64–1024, reports best) | **50** (:82) | matches td_jepa only |
| L_dim / left_encoder | 50; PhiMap 512×4, norm | **no left_encoder at all** (F(s,z,a) on raw obs) | 256, "shallow" (hidden_layers 0) or deep 50/2 | **MISMATCH both** |
| discount | 0.99 | 0.98 (walker/cheetah/quadruped), 0.99 pointmass (:82–85) | 0.98 (0.99 pointmass) | **MISMATCH (ExoRL wants 0.98)** |
| f_target_tau / b_target_tau | 0.005 / 0.005 | fb_target_tau 0.01 (F only; B target = hard `.zs` copy /250 steps) | 0.001 / 0.001 | MISMATCH both (structure matches td_jepa) |
| ortho_coef | 0.01 | example 1.0 — but code applies diag-only ⇒ inert; **paper λ = 0.01** | 1.0 base, swept {0.001, **0.01**, 0.1} | defensible (= paper λ, in td_jepa sweep) |
| train_goal_ratio | 0.5 | 0.5 | 0.5 | ok |
| fb_pessimism_penalty | 0.0 | 0.0 | 0.0 | ok |
| actor pessimism | (mean-only q_fn) | **0.5** (example :80) | 0.0 | MISMATCH vs official |
| lr_f / lr_b / lr_actor | 1e-4 each | 1e-4 each (pointmass lr_b/actor 1e-6) | 1e-4; lr_b swept {1e-4,1e-5} | ok |
| weight_decay | none | encoder AdamW wd=1e-4 hard-coded (agent.py:70) | 0.0 | minor, matches td_jepa |
| horizon | 5 | enc_horizon 5 | 5 | ok |
| forward net | 512 hid × 2 layers | 1024 × 1 (example :64,66) | 1024 × 1 | **MISMATCH both** |
| backward net | 512 × 4, norm | 256 × 2 (BaseMLP depth fixed), norm | 256 × 2, norm | **MISMATCH both** |
| predictor | 512 × 2, P=2 ensemble, unnormalized out | model_unroll: za 256 + BaseMLP(hdim 256) + linear, single head, **√d-normalized out** | VForwardMap 1024 × 1, P=2 | MISMATCH both |
| num_parallel (F) | 2 | 2 | 2 | ok |
| norm_z | true | true | true | ok |
| schedule | joint every step | 3M repr + 3M FB (frozen B), 6M total | joint, 3M total | **structural, see top** |
| q_loss_coef / clip_grad | not carried / none | 0 / 0 | 0 / 0 | ok (documented drop) |
| rldp_density extras: temp 1.0, actor_temp null, actor_q_center true, fb_pess lattice-refused | — | n/a (no density variant exists upstream) | n/a | seam-native |

The seam config note says the values are "critic/fb.yaml at cube-single launch values" —
i.e. they are deliberately the torch-port cube calibration, not either reference's ExoRL
calibration. That is fine for the cube control arms; it is **not** an ExoRL replication
config.

---

## Verdict

* As a transcription of **td_jepa's RLDP** (which is what the docstring claims): correct.
  Alignment fix (a) right against both references; stop-grads, ortho anchor, predictor
  module, z law, inference, and the density variant's invariants all verified; 61/61 tests
  pass and the tests genuinely pin the claims.
* As a replication of **official baselines/rldp on ExoRL**: not yet — the two-phase
  schedule, the frozen-B successor phase, predictor output normalization, actor pessimism
  0.5, discount 0.98, batch 1024, and the net widths all differ (and the official code's
  own ortho term is inert relative to its paper, which any comparison writeup should
  footnote).
