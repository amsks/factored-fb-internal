# Corrected Density-RLDP/HILP training — September 11, 2026

Status: launched on September 11, 2026 at **02:59:55 UTC**. Eight training
processes started; four seed-2 runs are queued. These are fresh full-state,
latent-reward runs authorized after the
[categorical-objective audit](2026-09-11-rldp-hilp-ce-audit.md).

| Environment | Critic preset | Seeds | Updates per seed |
|---|---|---|---|
| `cube_single` | `rldp_density` | 0, 1, 2 | 1,000,000 |
| `cube_single` | `hilp_density_tdjc` | 0, 1, 2 | 1,000,000 |
| `scene_play` | `rldp_density` | 0, 1, 2 | 1,000,000 |
| `scene_play` | `hilp_density_tdjc` | 0, 1, 2 | 1,000,000 |

All 12 runs retain the main comparison's training settings: FlowBC, batch 256,
feature dimension 50, two forward heads, discount 0.99, temperature 1, full-state
backward input, and latent rewards. Exact-goal and object-only variants are outside
this matrix. Explicit `backward_input=state` and `goal_kernel=false` record this choice.

The later main campaign protocol is retained: evaluation at step 1 and every
100,000 updates, 20 goal-conditioned episodes per task, and a separate
reward-inference evaluation using 10,000 relabelled samples, reward shift 1.0,
and 10 episodes per task. Both environments have five evaluation tasks. Metrics
are logged every 5,000 updates; checkpoints every 250,000; no video episodes.
The last-three comparison uses 800,000/900,000/1,000,000 evaluations. Unlike the
oldest cube RLDP checkpoints, these runs enable reward inference during training
so the corrected coefficient solve is measured.

Outputs belong under
[`exp/ce_audit_retrain_20260911/`](../../exp/ce_audit_retrain_20260911).
Each run has its own `<env_name>_<critic>_flowbc_seed_<seed>` directory. Initial
launches must have no restore, warm-start, or automatic-resume flags.

The immutable source is
[`source/`](../../exp/ce_audit_retrain_20260911/source), copied from the working
tree, including all four audited production changes. It contains `impls/`,
`env/`, and the local `ogbench/` source/assets, plus dependency and audit records.
The [source manifest](../../exp/ce_audit_retrain_20260911/provenance/source_manifest.json)
hashes 287 files (39,204,820 bytes), with canonical record digest
`b5882084b682989b38752c2d3146a2283dd66a4876680e58bccfe8a1c6b19539`.
The [seed/config plan](../../exp/ce_audit_retrain_20260911/provenance/seed_config_plan.json)
records the matrix; source and provenance remain read-only.

The runtime uses the existing host interpreter `/dev/shm/.venv-jax/bin/python`
and existing cube-single/scene train and validation datasets. The compute check
found eight idle NVIDIA L4 GPUs. Host preparation recorded JAX/JAXlib 0.10.0,
Flax 0.12.7, Optax 0.2.8, NumPy 2.4.6, MuJoCo 3.3.0, and Gymnasium 1.3.0, and
confirmed imports come from the frozen source. No installation or download is
part of this launch. Offline W&B output and generated files stay outside the
source snapshot.

The reviewed [launcher](../../exp/ce_audit_retrain_20260911/launch.py) and
[active manifest](../../exp/ce_audit_retrain_20260911/launch_manifest.json)
record every argument, per-job environment, output path, and resolved-config
hash. The manifest SHA-256 is
`c5d925c0e1e4565ad5d3c744b1df3f1287b84b0e0f41b917160b11557277493b`;
the launcher SHA-256 is
`f92afa506ceea72ac9479ff369df6725a1767c3a1d91171d1389b164fbdae89b`.
All 12 [resolved configs](../../exp/ce_audit_retrain_20260911/resolved_configs)
passed independent review before submission. Earlier preparation artifacts
are archival only and did not launch jobs.

There is one worker per GPU in tmux session `ce-audit-retrain-20260911`.
GPUs 0–3 start seed 0 in this order: cube RLDP, cube HILP, scene RLDP, scene HILP.
GPUs 4–7 start the same four cells at seed 1. Each of GPUs 0–3 then runs the
corresponding seed 2 job, giving eight initial jobs and four queued jobs.
Each worker stops its queue if a process fails or finishes without the
1,000,000-step checkpoint. Each job explicitly pins both CUDA and EGL to its
assigned GPU. Source and launcher hashes and fresh output/log paths are checked
before starting processes.

The execution owner confirmed training PIDs 1100466–1100473 on GPUs 0–7 and
MuJoCo initialization for every initial run. An independent startup check found
that all eight written `config.json` records exactly matched their approved
resolved-config records, the runtime records matched the intended GPU/seed
assignment, and all four queued seed-2 run directories were still absent.
No checkpoint restore or resume flags were used. This confirms process startup
and configuration, not completed training or benchmark performance.

The launch command, run on the host from any directory, is:

```bash
/dev/shm/.venv-jax/bin/python /home/adityamohan/git/Factored-FB-seam/exp/ce_audit_retrain_20260911/launch.py --launch
```

Progress is recorded in
[`runtime/`](../../exp/ce_audit_retrain_20260911/runtime), including the launch
record, worker PIDs, per-run training PIDs, start/end times, and exit status.
The manifest's `log` and `run_dir` fields point to each run's console log and
output directory. For example, from the repository root:

```bash
tmux list-windows -t ce-audit-retrain-20260911
cat exp/ce_audit_retrain_20260911/runtime/cube-single-play-v0_rldp_density_flowbc_seed_0.json
tail -n 30 exp/ce_audit_retrain_20260911/logs/cube-single-play-v0_rldp_density_flowbc_seed_0.log
tail -n 3 exp/ce_audit_retrain_20260911/cube-single-play-v0_rldp_density_flowbc_seed_0/train.csv
```

Each output directory stores `config.json`, `train.csv`, `eval.csv`,
`eval_relabel.csv`, and periodic `params_<step>.pkl` checkpoints. Goal-conditioned
and reward-inference results remain separate. The runtime JSON describes launch
and exit state; the CSV step and console log establish actual training progress.

These runs inherit the verified implementation repairs. They do not yet provide
evidence that the repairs improve benchmark performance. The object-only HILP
label correction is included in the frozen code but is inactive in this
full-state matrix.
