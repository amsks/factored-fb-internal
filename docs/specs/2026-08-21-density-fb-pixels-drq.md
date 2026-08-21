# Density-FB on image-based OGBench with the DrQ encoder

**Date:** 2026-08-21
**Status:** approved design, not implemented
**Base:** branch `density-fb`
**Related:** `docs/specs/2026-08-21-density-fb-agent-branch.md` (the state-based seam work)

---

## 1. Why this is not just "swap in an encoder"

Every other method in the comparison — GCIQL, bilinear GCIQL, GCIVL, HIQL, FB, CRL — is
goal-conditioned and ports to pixels by encoding the observation and the goal. Density-FB
does not, and the reason is the measure itself.

`impls/critics/density_fb.py` defines its goal-region hit kernel and its query support by
SLICING the observation vector:

```yaml
measure_hit_spec: "19:22:0.4"      # obs dims 19..22 = cube xyz, eps = 0.4 metres
```

`_raw_block_cols(nobs, blocks)` takes `nobs[..., 19:22]`. The image observation is
`(64, 64, 3) uint8`. **There is no dim 19:22.** The measure — the thing that makes this
Density-FB rather than FB — is undefined on images.

That asymmetry is itself a result worth reporting: the method needs privileged structure to
be defined at all, where the goal-conditioned baselines do not.

## 2. What the data and the env actually provide

Verified on this box, not assumed.

`~/.ogbench/data/visual-cube-single-play-v0.npz`:

```
observations : (1001000, 64, 64, 3) uint8
actions      : (1001000, 5)  float32
terminals    : (1001000,)    bool
qpos         : (1001000, 21) float32
qvel         : (1001000, 20) float32
```

The visual env at eval (`ogbench.make_env_and_datasets("visual-cube-single-play-v0")`):

```
obs                            (64, 64, 3) uint8
env.unwrapped.data.qpos        (21,)
info['privileged/block_0_pos'] the cube xyz, exposed directly
info['goal']                   the goal in STATE form
info['goal_rendered']          the goal as an IMAGE
```

So both channels the design needs exist natively, offline and online. No extra data
collection, and `privileged/block_0_pos` is cleaner than index arithmetic on `qpos`.

## 3. Decisions

| decision | choice |
|---|---|
| where the measure lives | **privileged qpos**: hit kernel + query support from object coords; encoder/actor/critic see pixels |
| encoder count | **two**, mirroring vanilla FB / TD-JEPA — one for the F side, one for the B side |
| encoder ownership | F-side encoder in the `density_f` group, B-side in `density_b`; **actor owns no encoder and detaches** |
| augmentation | host-side numpy random shifts (the bridge's existing convention) |
| scope | `visual-cube-single-play-v0`, one seed, feasibility only — not a table |

### 3.1 Why privileged, stated plainly

Running the measure on qpos keeps it IDENTICAL to the state-based runs, so a pixel-vs-state
comparison isolates "does the visual encoder hurt?" from "can the measure be defined at all?".
The cost is that this is privileged information **at training time**, so the result answers a
narrower question than a pixel-native method would. That belongs in the paper's claim, not in
a footnote.

## 4. How the reference trains its encoders

Read from the torch agent in this repo and from TD-JEPA upstream, because the convention
matters more than any guess.

`agents/fb/model.py` builds **two separate RGB encoders**:

```python
self._bw_encoder = rgb_encoder_cfg.build(obs_space)    # feeds the BACKWARD map
self._fw_encoder = rgb_encoder_cfg.build(obs_space)    # feeds the FORWARD map
self._left_encoder = left_encoder_cfg.build(self._fw_encoder.output_space, L_dim)
```

`agents/fb/agent.py:371-390` assigns them to optimizers:

| optimizer | holds | lr |
|---|---|---|
| `backward_optimizer` | `_backward_map` + `_bw_encoder` | `lr_b` |
| `forward_optimizer` | `_forward_map` + `_left_encoder` + `_fw_encoder` | `lr_f` |
| `actor_optimizer` | `_actor` **only — no encoder** | `lr_actor` |

and the actor additionally DETACHES: its update wraps the encode in `with torch.no_grad():`.
`_bw_encoder` is trained end-to-end by the backward loss (the source's own comment:
`grad -> trains B end-to-end`).

`metamotivo/agents/fb/agent.py:90-98` in TD-JEPA is structurally identical, so this repo's
torch FB is a faithful port of the upstream scheme rather than a local invention.

**TD-JEPA has no online mode.** Its `train.py` builds a replay buffer from offline data and
runs `for t in range(num_train_steps)` with no `env.step`; the only `env.step` in that tree is
in `envs/utils/rollout.py`, used for evaluation. There is no online encoder recipe upstream to
copy.

## 5. Architecture

### 5.1 The two-channel split

`u` (the query support) and `g` (the goal) each need TWO representations at once:

| consumer | gets | why |
|---|---|---|
| B tower | `u_img` -> `_bw_encoder` -> `B` | mirrors FB's `_bw_encoder(next_obs)` |
| hit kernel `_delta_phi` | `u_qpos` (raw block cols) | privileged measure; `eps=0.4` is METRES and only means anything in object-coordinate space |
| F tower conditioning | `g_img` -> `_fw_encoder` | the policy's view of the task |
| hit kernel `_delta_phi` | `g_qpos` | same reason as `u_qpos` |

There is no space mismatch in the bilinear form: `B(u) -> R^d` and `F(s,a,g) -> R^d` whatever
their input domains, so the inner product is unchanged.

`sample_cond` publishes both channels. Today it publishes `_density_u_raw` (raw cols, for the
kernel) and `_density_u_sc` (eps-scaled, for B). In pixel mode `_density_u_sc` is replaced by
the encoded image channel while `_density_u_raw` keeps its present meaning.

### 5.2 The silent-failure mode, and the guard

If a qpos channel is ever missing and an image tensor reaches `_delta_phi`, it will slice
`[..., 19:22]` of a `(B, 64, 64, 3)` array and return **numbers** — wrong ones, with no error.
Every kernel input therefore gets a hard rank/shape assertion at the boundary. This is a test,
not a comment.

### 5.3 Encoder groups

- `_fw_encoder` -> `density_f` group, trained by the density loss
- `_bw_encoder` -> `density_b` group, trained by the density loss
- actor: no encoder parameters, consumes encoded features under `stop_gradient`

No new seam group type is needed: `density_f` and `density_b` already exist and `_LR_KEY`
already maps both to `lr`. The seam's `stop_encoder_grad` keyword is already LIVE in
`impls/critics/fb.py` and `impls/critics/isfb.py` (both own a `left_encoder`), so the actor
path is existing machinery, not new.

## 6. Plumbing currently missing

| item | state |
|---|---|
| `impls/utils/encoders.py` | a 4-line STUB: `encoder_modules = {}` |
| DrQ encoder in JAX | exists on the bridge: `tools/wandb_mode_shim/ogbench_drq.py` (80 lines) |
| image augmentation | the seam's `GCDataset` DROPPED it on port |
| frame stacking | dropped in the datasets; `env/ogbench.py` already threads `frame_stack` |
| qpos channel in batches | not emitted; the npz has `qpos`, so no collection needed |

Augmentation stays **host-side numpy** (`random_shifts_batch`, pad=2, independent per key,
4-D arrays only). That is deliberate in the bridge: keeping it off-device lets augmentation
overlap the jitted device step. Do not "optimise" it onto the device.

## 7. Eval path

`cond` is built from `info`: `g_qpos <- info['goal']`, `g_img <- info['goal_rendered']`
(encoded), query qpos channel from `info['privileged/block_0_pos']`. `q_fn`'s
`_delta_phi(g_qpos, u_qpos, blocks)` then runs in metres exactly as in the state runs, so
`eps = 0.4` keeps its meaning.

**Train/eval channel agreement is a real risk.** Offline the qpos channel comes from the npz's
`qpos (N,21)`; online it comes from `info['privileged/block_0_pos']`. If those denote
different quantities, training and evaluation disagree silently. Pinned by a test that
compares both on the same states, not by trusting index arithmetic.

## 8. Testing

1. **Shape guard** — an image tensor where a qpos channel belongs must RAISE, not slice.
2. **Train/eval agreement** — npz-`qpos`-derived block xyz equals `info['privileged/block_0_pos']`.
3. **Encoder ownership** — `density_f`/`density_b` params move under the density loss; NO
   encoder parameter moves from the actor stage (the analogue of FB's `torch.no_grad()`).
4. **Augmentation** — per-key independent shifts; only 4-D arrays touched; others pass through.
5. **State path untouched** — the 60-pair bit-identity gate stays green. Pixels are additive.
6. **Smoke** — a few hundred steps on `visual-cube-single-play-v0` that actually trains.

## 9. Scope and risks

Scope is ONE domain, ONE seed, as a feasibility run. Pixel training is far slower than state,
and the state campaign is not finished.

- **Privileged at train time.** The comparison answers "does the encoder hurt, holding the
  measure fixed", NOT "is Density-FB pixel-native". State it in the claim.
- **`frame_stack` interacts with the qpos channel**: stacked image frames but a single-frame
  qpos. Decide and document which frame the measure refers to (the latest).
- **No result is implied.** Nothing here says Density-FB works on pixels; it says the method
  can be DEFINED there and run.

## 10. Out of scope

- The other critics on pixels. They need only an encoder swap and are not blocked by this.
- `visual-cube-double` / `visual-scene`, and any multi-seed pixel table.
- Making the measure pixel-native (latent-space queries, or regressing object coords from
  images). Both were considered and rejected for this pass: a latent measure moves its own
  coordinate system as the encoder trains and leaves `eps` with no natural scale, and a
  learned coord regressor injects its error straight into the measure.
