# Handoff to luis: REBALANCE the cube-single play data toward its bottleneck rows (FB vs Density-FB)

Prepared and validated on evalbox 2026-09-18. Companion to the persist_lambda handoff (which made
the bottleneck rows RARER); this one runs the same sampler in reverse and makes them COMMON.
Code arrives as a git bundle on top of 22e1f16 (section "How to apply" at the end).

## 1. The question
The paper's failure story says FB loses on cube because the rows where the cube starts moving
(the grasp / first-move decision) are rare in the play data, so the successor representation
spends its rank elsewhere and the action gap at the bottleneck is lost; Density-FB's objective
keeps that gap. Sanity check: remove the rarity by RESAMPLING ONLY -- no expert data, no rows
dropped, state-action support unchanged, the dataset file untouched -- and ask whether
Density-FB still beats FB. Two pairs, two eval routes:
  * z pair     fb_tdj vs density_fb_z   reward-inference eval, Route B (eval_relabel.csv)
  * goal pair  fb_gc  vs density_fb     native goal-conditioned eval (eval.csv), no relabel flags
Each level runs in two modes (section 3): `sample` (whole-batch resampling) and `loss`
(critic-only per-row loss weights). The reading of the outcome is pre-registered in section 4.

### What the sampler does (impls/utils/datasets.py)
Dataset.enable_persist_oversample(block, thresh, lam=1.0, pool="motion", window=0,
transient_mass=None). Two buckets, persistent vs transient; a persistent row is worth `lam`
transient rows. lam > 1 is the persist_lambda lever; lam < 1 upweights the transient rows. New:
  * pool="onset": transient = rows within +-window of a MOTION ONSET (cube block obs[19:22]
    displacement < thresh at t-1->t and >= thresh at t->t+1, thresh 0.005, trajectory-boundary
    jumps masked); everything else persistent. pool="motion" = the old "cube is moving" split.
  * transient_mass=m: lam is solved so the transient bucket gets exactly sampling mass m
    (persist_mass = 1 - m). Every row keeps nonzero mass (support preserved).
Config keys: persist_pool, persist_window, persist_transient_mass, persist_block (19:22),
persist_vel_thresh (0.005); enabled when persist_oversample_lambda != 1 OR
persist_transient_mass is set. The sampler sits on the BASE dataset, so all four critics draw
the identical rebalanced stream.

Measured on cube-single-play-v0 at thresh 0.005 (evalbox):
  * the cube is moving in 47.8% of transitions  -> the motion:0.5 level is a near no-op CONTROL
  * motion ONSETS are 1.34% of transitions; the +-5-step onset window is 13.4%
    -> onset5:0.25 and onset5:0.5 raise the bottleneck-window sampling mass ~2x and ~4x
The lambda=1 (uniform) baselines ALREADY EXIST and are not re-run: fb_tdj / density_fb_z /
density_fb from tdjtab (exp/tdjtab), fb_gc from the goalcond handoff (exp/goalcond_gc).

## 2. What the onset windows contain (measured, docs/notes/2026-09-18-onset-window-contents.md)
Read-only audit of the sampler-exact +-5 windows on cube-single-play-v0 (in the bundle):
  * Onset rows are the GRASP-CLOSURE decision: the hand is at the cube (ee-cube distance
    median 0.4 cm), the closing command is active in 90% of onset rows, the cube is still
    within 1 cm of its rest position in 85% of window rows, and only 14% of window rows are
    post-onset rows where the cube has already been displaced.
  * Failed grasp attempts are mostly already inside the windows: 62% of their rows sit in an
    onset window (via the short "blip" onsets where the touch nudged the cube).
  * The carrying phase (36.5% of the data) is covered at 2.4% -- the sampler does not touch
    FB's post-acquisition failure. Genuine mid-transport drops do not exist in the play data
    (12 events per million rows), so no sampler can upweight them.
  * A union "decision" pool (onset + failed-attempt + drop windows) would be 13.7% of rows vs
    13.4% for the onset windows alone, so the onset rule stands. Describe the lever as
    "upweighting the grasp-closure decision", not "upweighting motion".

## 3. Two modes: resample the batch, or weight the critic loss
  * persist_mode=sample -- the whole batch is redrawn from the rebalanced sampler. This also
    changes which next-states appear as candidate future states in BOTH critics, so a win here
    reads as "this sampling change helps", not "giving these decisions more critic weight
    fixes the approximation".
  * persist_mode=loss -- sampling is UNCHANGED (uniform: the actor's training distribution and
    the reward-inference sample set stay as they are); the same transient/persistent split
    becomes a per-row CRITIC loss weight batch["row_weight"]: transient rows W, persistent 1,
    W chosen so the transient rows' share of TOTAL weight is transient_mass, then normalised to
    dataset-mean 1. Only the FB TD row terms (impls/critics/fb.py: fb_offdiag, fb_diag; ortho
    untouched, candidate columns untouched) and the density CE rows (impls/critics/density_fb.py
    stage_loss, and its sq ablation) consume it. Actor losses, the goal-branch actor readout,
    infer_cond and eval never read it. With the key absent or all ones the losses are exactly
    the old ones (tested), so no lambda=1 cell moves. This is the closest test of the paper's
    claim. On cube-single at onset5:0.5, W = 6.48 (13.4% of rows carry half the critic weight).
Save dir exp/rebalance/<mode>_<level>/ (sample_onset5_tm0.5, loss_onset5_tm0.5, ...).

## 4. PRE-REGISTERED READING (fixed before any run)
| outcome | reading |
|---|---|
| Reweighting improves FB until it matches Density-FB | this sampling rule can remove Density-FB's advantage on this task; an alternative remedy exists. |
| Both improve, but Density-FB remains better | giving these windows more training weight helps but does not explain the whole difference. |
| FB does not improve | this rule does not rescue FB; it does not prove motion is intrinsically harder or that other imbalances are irrelevant. |

Scope: this answers "does this targeted reweighting make Density-FB unnecessary on
cube-single?", not "would Density-FB still help after every relevant imbalance was removed?".

## 5. What to run
    MODE="sample loss" JOBS_ONLY=1 bash scripts/launch_rebalance.sh    # 72 commands for your scheduler/slurm
    # MODE=sample or MODE=loss alone -> 36 each; default MODE=sample
    # or: MODE="sample loss" GPUS="0 1 2 3 4 5 6 7" bash scripts/launch_rebalance.sh   (tmux, 1 job/GPU at a time)
72 commands = 3 levels (onset5:0.25, onset5:0.5, motion:0.5) x 4 critics (fb_tdj, density_fb_z,
fb_gc, density_fb) x 3 seeds x 2 modes; cube_single, 1M steps, flowbc actor, eval every 100k,
ckpt every 250k. Priority order if GPU-time forces a cut:
  1. loss mode, onset5:0.5   (12 runs: the cleanest test)
  2. loss mode, onset5:0.25  (12)
  3. sample mode, both onset levels (24)
  4. motion:0.5 controls, both modes (24; should sit on the uniform value)
The lambda=1 baselines already exist: the tdjtab cells for fb_tdj / density_fb_z / density_fb,
and fb_gc from the goalcond handoff. If fb_gc has NOT been run yet, its 3 unweighted seeds
(the goalcond handoff commands) are needed too.
Idempotent (--auto_resume; a run whose params_1000000.pkl exists is skipped). ~8 h per run at
1/GPU. Overrides per critic are exactly the tdjtab / goalcond ones (density_fb_z:
latent_centered actor reward, hard measure, hit spec 19:22:0.4, relabel eval; density_fb: hard
measure + hit spec; fb_tdj: relabel eval; fb_gc: nothing) plus the five persist_* keys. Each run
prints one `[persist-oversample] mode=... pool=onset window=5 ... persist_mass=0.5 ...` line at
startup (mode=loss also prints W) and records the persist_* keys in its config.json; if that
line is missing the lever is off.

## 6. What to report
Per level and mode (and the existing lambda=1 cells alongside), at 1M:
  * z pair:    Route-B success, mean of the last 3 eval passes of eval_relabel.csv, for fb_tdj
               and density_fb_z
  * goal pair: native goal-conditioned success from eval.csv for fb_gc and density_fb
  * action-ranking sign diagnostic on each z-pair checkpoint:
        scripts/probes/cube_bottleneck/sign_agreement.py --env cube-single-play-v0 --epoch 1000000
Headline: success vs bottleneck-window sampling mass (uniform 13.4% -> 25% -> 50%) for each
pair, with motion:0.5 as the control. Mailbox post from luis, or status/*.json per run;
checkpoints to this HF repo if you can.

## 7. WARNING for the persist_lambda runs you already have (fixed in this bundle)
scripts/launch_persist_lambda.sh at 22e1f16 passes --override 'persist_block=19:22'
UNQUOTED. The override parser (OmegaConf.from_dotlist) reads values as YAML, and YAML 1.1 turns
19:22 into the sexagesimal integer 1162, so every lever run dies at startup with
`ValueError: not enough values to unpack (expected 2, got 1)` in make_dataset (found on evalbox
2026-09-18 by the launcher smoke test; the earlier "validated" note was through the Python API,
not the launcher). If your lambda runs crashed at startup, this is why. The bundle quotes it
(--override 'persist_block="19:22"', the same rule as measure_hit_spec) in BOTH launchers and
makes make_dataset raise a clear message instead. Re-pull (apply the bundle below) before
re-launching the persist_lambda campaign.

## 8. How to apply
Git bundle on the hub: `handoff/rebalance/factored-fb-rebalance.bundle` (amsks/factored-fb-checkpoints,
repo_type model), built from 22e1f16..HEAD on evalbox; it contains the two commits on top of
22e1f16 (the base you already have from the persist_lambda bundle).
    huggingface-cli download amsks/factored-fb-checkpoints --include 'handoff/rebalance/*' --repo-type model --local-dir /tmp/dl
    cd <Factored-FB checkout>                      # at 22e1f16 or the branch carrying it
    git bundle verify /tmp/dl/handoff/rebalance/factored-fb-rebalance.bundle
    git fetch /tmp/dl/handoff/rebalance/factored-fb-rebalance.bundle density-fb:rebalance-handoff
    git checkout rebalance-handoff                 # or merge/cherry-pick onto your branch
    python -m pytest tests/impls/test_dataset_persist_rebalance.py -q     # 11 passed on evalbox
A plain patch of the code changes (impls/utils/datasets.py, impls/critics/{fb,density_fb}.py,
scripts/launch_persist_lambda.sh) is beside it as handoff/rebalance/rebalance.patch, for a
checkout that cannot fetch from a bundle; scripts/launch_rebalance.sh and the test file are
then under handoff/rebalance/files/ in the bundle's tree.
