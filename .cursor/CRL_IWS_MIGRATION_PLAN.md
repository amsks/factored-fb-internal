---
name: CRL IWS Migration
overview: "Port the confirmed `Unsupervised_Manipulation` SGCRL/IWS behavior into Factored-FB as a new side-by-side CRL-IWS implementation first. If the additive implementation works, merge the proven pieces back into the existing CRL path deliberately."
todos:
  - id: pin-source-behavior
    content: Document the exact source SGCRL/IWS behavior and config defaults to preserve from `../Unsupervised_Manipulation`.
    status: pending
  - id: add-crl-iws-agent
    content: Add a non-vendored, side-by-side `crl_iws` JAX agent/config wrapper under `tools/wandb_mode_shim` without changing existing CRL/CRL+FlowBC behavior.
    status: pending
  - id: port-iws-sampler
    content: Port the interaction-weighted goal sampler into a patched OGBench `GCDataset` implementation.
    status: pending
  - id: wire-configs
    content: Add Hydra config and launcher wiring for `cube_single_state_crl_iws`.
    status: pending
  - id: add-tests
    content: Add config, registration, and synthetic sampler tests, then run the focused test set.
    status: pending
  - id: compare-and-merge
    content: Compare CRL-IWS against existing CRL; if it works, merge reusable sampler/config pieces into the main CRL path behind explicit config flags.
    status: pending
isProject: false
---

# CRL IWS Migration Plan

## Recommendation
Use the existing JAX OGBench CRL track in [run_gciql.py](../run_gciql.py) as the runtime target, but create a new side-by-side implementation first. Factored-FB already runs CRL from [third_party/ogbench/impls/agents/crl.py](../third_party/ogbench/impls/agents/crl.py), and custom JAX CRL variants already live cleanly outside vendored OGBench in [tools/wandb_mode_shim/crl_flowbc.py](../tools/wandb_mode_shim/crl_flowbc.py). The initial port should not alter those existing CRL or CRL+FlowBC paths.

## Source Behavior To Preserve
The confirmed source is `../../Unsupervised_Manipulation`:

- CRL agent: [../../Unsupervised_Manipulation/src/unsupervised_manipulation/algorithms/pytorch/sgcrl.py](../../Unsupervised_Manipulation/src/unsupervised_manipulation/algorithms/pytorch/sgcrl.py)
- IWS identity wrapper: [../../Unsupervised_Manipulation/src/unsupervised_manipulation/algorithms/iws.py](../../Unsupervised_Manipulation/src/unsupervised_manipulation/algorithms/iws.py)
- IWS sampler: [../../Unsupervised_Manipulation/src/unsupervised_manipulation/data/episodes.py](../../Unsupervised_Manipulation/src/unsupervised_manipulation/data/episodes.py)
- Training wiring: [../../Unsupervised_Manipulation/src/unsupervised_manipulation/training/loops.py](../../Unsupervised_Manipulation/src/unsupervised_manipulation/training/loops.py)
- Benchmark defaults: [../../Unsupervised_Manipulation/benchmark/testbed/algorithms/iws.yaml](../../Unsupervised_Manipulation/benchmark/testbed/algorithms/iws.yaml)

The important IWS behavior is sampler-side: repeat a local interaction context, detect/contact-cache interaction events, sample anchors around event centers with temporal/spatial weights, optionally use full-episode future goal scope, and keep CRL losses otherwise unchanged.

## Target Shape In Factored-FB
Add a new non-vendored JAX variant, likely `crl_iws`, under [tools/wandb_mode_shim](../tools/wandb_mode_shim). This is an experimental implementation that can be deleted or merged cleanly after results:

- `crl_iws.py`: subclass/register the existing `CRLAgent` mainly for identity and config defaults, not for changing the core CRL loss.
- `iws_gcdataset.py`: subclass or monkey-patch OGBench `GCDataset.sample()` to implement the IWS index sampler while preserving OGBench batch keys: `observations`, `actions`, `value_goals`, `actor_goals`, `rewards`, `masks`.
- `sitecustomize.py`: register `crl_iws` and activate the IWS dataset patch only when the CRL-IWS config requests it. Existing `crl` and `crl_flowbc` runs must continue to use the current dataset behavior.
- `configs/gciql/cube_single_state_crl_iws.yaml`: mirror [configs/gciql/cube_single_state_crl.yaml](../configs/gciql/cube_single_state_crl.yaml), but point to `tools/wandb_mode_shim/crl_iws.py` and set IWS defaults from the source: context repetition `8`, context mode `interaction_weighted`, full-episode future `true`, and cube/contact radius parameters adapted to OGBench cube state geometry.

## Implementation Notes
Do not edit vendored OGBench. Do not change the current CRL implementation while validating CRL-IWS. `run_gciql.py` already puts [tools/wandb_mode_shim](../tools/wandb_mode_shim) first on `PYTHONPATH`, and [sitecustomize.py](../tools/wandb_mode_shim/sitecustomize.py) already patches OGBench safely for DrQ, wandb, resume, and `crl_flowbc`. Use that same mechanism for CRL-IWS, gated by the new config only.

For event detection, start with a small adapter layer rather than copying Air Hockey assumptions directly. Source IWS assumes `state[..., 0:2]` is source/effector and `state[..., 4:6]` is target/object. OGBench cube observations may need a domain-specific extractor, so add an explicit `iws_event_geometry` config, initially `cube_single`, and fail clearly for unknown layouts.

## Validation
Add focused tests before long runs:

- Config/argv test beside [tests/test_crl_flowbc_configs.py](../tests/test_crl_flowbc_configs.py) to prove `cube_single_state_crl_iws` resolves to the custom agent and passes IWS config.
- JAX subprocess test beside [tests/test_crl_flowbc_injection.py](../tests/test_crl_flowbc_injection.py) to prove `crl_iws` registers and an update runs.
- Dataset sampler unit test using a tiny synthetic OGBench-style trajectory to verify repeated interaction-local anchors, full-episode future scope, and fallback to vanilla sampling when no event exists.
- Smoke command: `make smoke-gciql` or a short `run_gciql.py --config-name cube_single_state_crl_iws train_steps=... wandb_mode=disabled` once the JAX env is available.

## Merge Path After It Works
After CRL-IWS is validated, merge only the stable pieces:

- Keep the sampler as a reusable module, but expose it through explicit config flags such as `dataset_class=IWSGCDataset` or `iws_enabled=true`.
- Move duplicated defaults from `cube_single_state_crl_iws.yaml` into shared config blocks only after the side-by-side runs match expectations.
- Preserve a plain `cube_single_state_crl.yaml` baseline config so future comparisons remain easy.
- Only consider making IWS the default CRL sampler after at least one short smoke and one meaningful training comparison show no regression.

## Optional Second Phase
If you need exact `Unsupervised_Manipulation` SGCRL loss parity, add a PyTorch `agents/sgcrl` later and wire it through [train.py](../train.py). That is a larger port because Factored-FB's PyTorch path is FB-oriented and samples flat offline `DictBuffer` batches from [data/ogbench.py](../data/ogbench.py), while the source SGCRL/IWS code expects online episode replay with future-goal relabeling. I would not make this the first step unless the JAX CRL+IWS results diverge materially from the known-good behavior.
