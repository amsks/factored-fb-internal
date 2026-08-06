# Spec 20 — Dwell-balanced FB implementation contract (detector + torch-FB row weights)

Audience: the implementing agent. Self-contained; science context in
`specs/19-ors-program.md` (this is priority P2+P3 — the SECONDARY
hypothesis; do not start P3 before P2's validation gate is reviewed).
TDD; default-off must be BIT-identical.

## 0. What is being built

The proposal's dwell-balancing arm: downweight SETTLED post-transport
dwell transitions in the torch FB loss so the finite-rank representation
spends less capacity on repeated "the cube is still here" rows. Two
deliverables: (1) an offline detector producing per-transition weights +
a validation report (P2, no training); (2) a torch-FB `reweight_mode="dwell"`
consuming them, with a cs/cd/scene wave against plain-FB anchors (P3).

Pre-registered stakes (spec 19 §3): spec 18 already shows the same-rank
bilinear class learns manipulation with dwell fully in its targets when
the estimator is normalized — so a NULL here is an informative result
(estimator, not sample weighting, was the bottleneck), not a failure to
be patched.

## 1. Detector — `scripts/rescuing_fb/make_dwell_weights.py` (P2)

Builds on `scripts/rescuing_fb/dwell_stats.py` (spec 15 — read it first;
reuse its segmentation primitives, do not re-derive).

CLI: `--dataset <npz> --hit_spec '<start:stop:eps,...>' --p 0.5
--alpha_min 0.02 --v_eps <default: 0.05*min_block_eps per step>
--settle_k 5 --out <npy> --report <json>`.

Definition (pin exactly; the proposal's §4 warning is the whole game):
- Per trajectory, per hit-spec block b: speed_b[t] = ‖x_b[t+1] − x_b[t]‖.
- A transition t is GLOBALLY STATIONARY iff speed_b[t] < v_eps for ALL
  blocks (a cube resting while another is manipulated is sequential
  skill — NEVER downweighted; this multi-object rule is load-bearing).
- A SETTLED RUN is a maximal globally-stationary run of length ≥ settle_k
  that starts AFTER at least one non-stationary frame earlier in the same
  trajectory (pre-manipulation rest at trajectory start counts too —
  approach happens during it; therefore: a settled run additionally must
  FOLLOW a frame where some block moved, i.e. post-motion only. Runs at
  the very start of a trajectory, before any object has ever moved, are
  NOT downweighted — that is where approach/grasp lives).
- Weight: rows inside a settled run of length L get
  α = max(alpha_min, L^(−p)); all other rows get 1.0. Save float32 [N]
  aligned to the npz transition order, then mean-normalize to E[α]=1
  over the dataset (record the normalizer in the report).

Report JSON (the P2 HARD GATE artifact — reviewed before any training):
fraction of transitions downweighted; distribution of settled-run
lengths; per-phase breakdown using a simple phase tag (pre-first-motion /
moving / settled) with counts; effective sample size Σα/N; 10 example
trajectory segmentations (indices + tags) for spot-checking; per-block
stationarity rates (sanity vs spec 15: cs ≈ 48% stationary, cd ≈ 72%
per-cube — large disagreement = detector bug).

Tests (`tests/test_make_dwell_weights.py`): synthetic trajectories with
known segmentation (approach-only rest at start → weight 1; mid-traj
44-step settle → α = 44^−0.5; multi-block: block-1 resting while block-2
moves → weight 1); normalization E[α]=1; alignment length == N.

## 2. File coupling note

The .npy is a TRAINING-TIME input only (never needed at
checkpoint-restore). Record path + sha256 in the run config; store under
`results/dwell_weights/<env>_p05.npy` (durable, NOT /dev/shm).

## 3. Torch FB — `reweight_mode="dwell"` (P3; in `agents/fb/agent.py`)

The existing reweight framework (`coverage_weights`/`interaction_weights`,
`fb_loss(w=...)`) weights the GOAL/COLUMN axis (w_j on goals ~ ρ′). Dwell
weights are on the TRANSITION/ROW axis — a NEW, orthogonal path. Read
`fb_loss` and the update step first; then:

- New init args: `reweight_mode="dwell"` accepted alongside the existing
  modes; `dwell_weights_path` (.npy). At construction, load the [N]
  vector; the data loader must expose the flat transition index per batch
  row so the agent can gather α for the batch (verify how the torch
  training loop samples batches — if indices are not currently surfaced,
  plumb them; this is the one loader change, default-off inert).
- Application: multiply the PER-SAMPLE FB Bellman residual terms by α_i
  (row i = the (s,a,s′) transition) in BOTH the diagonal and
  off-diagonal terms' row axis. Columns (goals ~ ρ′) and z-sampling are
  UNTOUCHED — the measure stays defined w.r.t. ρ (the proposal's §3:
  change the projection, not the target). Batch-renormalize α to mean 1
  per batch (consistent with the framework's E[w]=1 convention).
- Ortho/covariance regularizer rows: apply α there too or not? PIN: the
  ortho/normalization term stays UNWEIGHTED (it normalizes B over ρ, not
  over the transition distribution; weighting it would change what B's
  normalization means). Document inline.
- Default-off bit-identity: mode unset/path empty ⇒ code path untouched.

Tests (`tests/test_fb_dwell_reweight.py`, torch venv `.venv`):
1. default-off bit-identical FB loss on a fixed batch/seed.
2. α=1 vector ⇒ bit-identical to off.
3. numpy reference: hand-α batch, weighted diag + off-diag rows match.
4. ortho term invariant to α.
5. gather correctness: batch indices → the right α rows (permutation
   test).

## 4. Wave (P3; only after the P2 report is reviewed)

Torch stack, same commit for arm and control, standing FB recipe (the
`fb_state_*` configs — ⚠ run GROUP NAMES LIE, verify from recorded
config; wandb entity amsks; tmux):

- `dw_fb_{cs,cd,sc}_p05` seeds 0/1/2 — dwell-balanced FB (p=0.5).
- `dw_fb_{cs,cd,sc}_ctrl` seed 0 — plain FB re-run at the same commit
  (drift control vs the historical ~0.42–0.48 cs anchor).

Read (pre-registered, spec 19 §3): improvement toward the density-arm
numbers = foundation-model result, motivates the P4 dwell-density knob;
≈ control = the dwell-capacity hypothesis is retired for OGBench play
and the estimator attribution stands. Also record reward-transfer /
occupancy-calibration probes if the existing FB eval harness exposes
them cheaply (the proposal's foundation-model-quality axis) — success
alone cannot separate "better representation" from noise.

## 5. Out of scope

- p sweeps beyond 0.5 (one dose, round 1 — the criticmix dose-cliff
  lesson).
- Dwell-weighted density CE (P4, gated on this wave winning).
- Grasp/contact-state detectors (obs[18] is ~redundant with state and
  the velocity+post-motion rule already protects pre-grasp rows; add
  contact logic only if the P2 report shows misclassification).
- Any change to the ORS arms (P1 consumes UNWEIGHTED occupancy — spec 19
  §2.4).
