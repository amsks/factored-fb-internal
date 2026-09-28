# Handoff to luis: OGBench density lever (persistence oversampling) + action-gap analysis

Prepared and validated on evalbox 2026-09-16. This is the cube analogue of the toy-MDP
"add ordinary-state mass" experiment: oversample quasi-static cube transitions by a factor
lambda and test whether FB degrades while Density-FB holds.

## Code delivery (evalbox has no GitHub push; comes as a git bundle on this HF repo)
File: `handoff/persist_lambda/factored-fb-density-lever.bundle` (387K), base 6cc2952 (shared
push point you have). Carries 6cc2952..22e1f16, including:
  * 22e1f16 -- the density lever: Dataset.get_random_idxs two-bucket weighted draw +
    enable_persist_oversample + make_dataset wiring (persist_oversample_lambda / persist_block /
    persist_vel_thresh) in impls/utils/datasets.py; scripts/launch_persist_lambda.sh; and the
    action-gap sign metric (scripts/probes/cube_bottleneck/sign_agreement.py + failure_plots.py).
  * 3de10b8 -- the earlier route-3 goalcond handoff (already on HF; harmless to have again).
Apply:
    huggingface-cli download amsks/factored-fb-checkpoints handoff/persist_lambda/factored-fb-density-lever.bundle --repo-type model --local-dir /tmp/dl
    cd <Factored-FB checkout>; git fetch /tmp/dl/handoff/persist_lambda/factored-fb-density-lever.bundle density-fb:lever && git merge lever   # or: git checkout 22e1f16

## What to run
    JOBS_ONLY=1 bash scripts/launch_persist_lambda.sh    # 24 commands for your scheduler/slurm
    # or: GPUS="0 1 2 3 4 5 6 7" bash scripts/launch_persist_lambda.sh
fb_tdj vs density_fb_z (the Figure-5 pair) x lambda in {1,2,4,8} x 3 seeds, cube_single, 1M,
Route-B relabel eval. ~8 h per run at 1/GPU. Save dir exp/persist_lambda/lam<K>/. Idempotent
(--auto_resume, final-ckpt skip). A transition is "persistent" when the cube block (obs 19:22)
moves < 0.005 between t and t+1; at lambda=8 persistent sampling mass goes 0.52 -> 0.90, support
preserved (validated on evalbox).

## The computation (either you, or push checkpoints to HF and it can be done anywhere)
1. Primary: Route-B success vs lambda, per method. Prediction: FB falls as lambda grows,
   Density-FB stays roughly flat.
2. Attribution (controls the actor/critic confound): run the action-gap sign metric on each
   trained arm --
       python scripts/probes/cube_bottleneck/failure_figs.py rollouts,banks,predict ... (as for Fig 5)
       python scripts/probes/cube_bottleneck/sign_agreement.py --env cube-single-play-v0 --epoch 1000000
   If FB's success drop is accompanied by its carrying sign-agreement collapsing toward 0.5
   while Density-FB's holds, the lever acted through the successor representation, as the
   rare-bottleneck lemma predicts -- not merely through fewer manipulation demos.
   If you push the trained checkpoints to this HF repo, the analysis can be rerun/verified elsewhere.

## Report back
Mailbox post from luis, or status/*.json per run. Headline plot: success (and carrying
sign-agreement) vs lambda for fb_tdj vs density_fb_z.
