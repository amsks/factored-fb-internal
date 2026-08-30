# RLDP deep audit: the NETWORKS were never ported — only the losses were

Scope: line-by-line comparison of `impls/critics/rldp.py` + the modules it builds against
the OFFICIAL repo `~/git/baselines/rldp` (`rldp/nn_models.py`, `rldp/rldp/model.py`,
`rldp/rldp/agent.py`). Prompted by a measured defect the loss-level audit could not
explain: the trained critic is **action-blind**.

## 0. The measurement that forced this audit

On a finished 5M run-2 walker checkpoint, at the inferred reward z:

    Q at dataset (behavior) actions   193.51
    Q at uniform random actions       194.28
    Q at random CORNERS               193.96
    Q at the policy's own actions     195.73

Q varies ~1% between the behavior policy, uniform random actions, corners of the action
cube, and its own optimum. Quantified over 64 random actions per state:

| checkpoint | std across ACTIONS | std across STATES | ratio |
|---|---|---|---|
| run 2 walker | 3.15 | 51.7 | **0.061** |
| run 1 walker | 18.7 | 262.1 | **0.071** |
| run 2 cheetah | 3.17 | 124.8 | **0.025** |

The critic is ~95% a state-value function. Decomposing where the action signal goes:

    ||F|| 1485.6      F std across ACTIONS 3.35      F std across STATES 35.50
    F component ALONG z: std across actions   0.140
    F component PERP  z: std across actions  23.334
      => action signal surviving the projection onto z: 0.0016

The action moves F substantially, but almost entirely ORTHOGONAL to the reward direction.
And the factorization is miscalibrated: with ||B|| = 22.63 (= sqrt(512), correct),
M = F.B has diag 576 / offdiag 134 where the diagonal should be ~1/(1-gamma) = 50.
Both runs, so neither the ortho fix nor the actor fix touched it.

## 1. What MATCHES (verified line by line)

Forward map `F` architecture (embed_z over [obs,z], embed_sa over [obs,action], each
narrowing to h//2, concat in the reference's `[sa, ze]` order, trunk of ReLU layers,
output Dense(z_dim)); B's sqrt(d) normalization; `sample_mixed_z`; the FB loss form
(offdiag 0.5*(diff*off)^2/off_sum, diag -mean(diag(diff))*P); the SP pair alignment;
discount handling; `norm_obs` off. Plus the three divergences already fixed this campaign:
the TD3 actor objective, the online-B FB target, and the inert ortho.

## 2. WHAT DOES NOT MATCH — all four are NETWORKS

The earlier audit (`2026-08-28-rldp-audit.md`) compared the training LOOP and found the
seam to be a faithful td_jepa transcription. It never compared the MODULES. They are also
td_jepa's, and official's are different.

### 2.1 The actor — and the correct one was already in the tree, switched off

Official (`nn_models.Actor`, built by `build_actor`):

    embed_z = simple_embedding(obs_dim + z_dim, hidden, 2)   -> h//2
    embed_s = simple_embedding(obs_dim, hidden, 2)           -> h//2
    policy  = [Linear(h,h), ReLU] * hidden_layers + Linear(h, action_dim)
    mu      = torch.tanh(policy(cat([s_emb, z_emb])))        <-- SQUASHED

Ours ran `networks.Actor`: a single `MLP(hidden_dims)` over `concat([obs, cvec])`, with
`tanh_squash=False` and the comment "actions are clipped at the call sites instead".

**This is the saturation mechanism.** With `tanh` the squash is smooth and the gradient
shrinks toward the bounds, so the policy is self-limiting. With a hard clip the pre-clip
output drifts arbitrarily, and once every dimension is outside [-1,1] the gradient through
the clip is EXACTLY ZERO — an absorbing fixed point. Measured: policy saturation 1.000 on
walker AND cheetah, run 1 AND run 2, versus 0.26 for the behavior data; it appears within
the first 1M steps of phase 2 (0.000 at 2M -> 0.999 at 3M).

**`impls/actors/_motivo_nets.py::MotivoPolicy` is already a faithful port of official's
Actor** — dual embedding, `[e_s, e_z]` concat order, trunk, and `means = jnp.tanh(...)`
with a comment citing the reference's squashed mean. It is gated behind
`a_cfg.get("motivo_archi", False)`, which `configs/actor/ddpgbc.yaml` never sets. Both
campaigns ran the wrong actor while the right one sat behind an unset flag.

### 2.2 The backward map B

Official `.zs` is `BaseMLP(goal_dim, zs_dim, hdim, activ='elu', norm=True)`:

    y = ln_activ(l1(x), elu)     # ln_activ(x,a) = a(LayerNorm(x))
    y = ln_activ(l2(y), elu)
    y = l3(y);  sqrt(d) * normalize(y)

Ours is `PhiMap`: `Dense -> LayerNorm -> tanh`, `Dense -> relu` (NO LayerNorm), `Dense`,
sqrt(d)-normalize. Same three linears; **wrong activations (ELU vs tanh/relu) and a
missing LayerNorm on the second layer.** This is the module whose output IS the successor
measure's basis.

### 2.3 The predictor

Official is part of `EncoderMap` (`model.py:68` — note `build_backward` is COMMENTED OUT
and `build_encoder` is used, so B and the predictor are ONE module):

    za  = ELU(Linear(action_dim -> za_dim))
    zsa = BaseMLP(zs_dim + za_dim -> zsa_dim)
    out = sqrt(d) * normalize(Linear(zsa_dim -> zs_dim))

Ours is `PredictorMap`/`_PredictorTower`, whose own docstring says it is transcribed from
**td_jepa's `VForwardMap`** — a dual-embedding-plus-trunk module. Structurally different.
Official's is also SINGLE; ours is ensembled with `num_parallel=2`.

### 2.4 Phase-1 segment sampling

Official `sample_segments(..., H, B, L=1000)` draws starts uniformly from `[0, L-H]`, so a
window NEVER crosses or clamps at a trajectory end. Ours clamps
(`min(idx+k, final_idx)`), which repeats the terminal state. ~0.5% of rows at H=5, L=1000
— small, but it injects "nothing moved" targets that bias the predictor toward identity,
which is precisely the degenerate solution the predictiveness gate exists to catch.

## 3. Consequence for the campaign

Both campaigns compared a td_jepa-networks RLDP against published official-networks
numbers. The six official-parity knobs changed the losses and the schedule and left all
four networks as td_jepa's. So "the only difference is the density objective" is NOT yet
true, and no knob on the ladder (steps-3M, eval-50, pointmass gamma) could have made it
true.

## 4. Remediation, in dependency order

1. `actor.motivo_archi: true` — one config line, no new code, and it targets the measured
   saturation directly. Cheapest and highest-confidence.
2. Port `BaseMLP` as the backward tower (ELU + LayerNorm on both hidden layers), behind a
   default-off knob so every existing cell stays bit-identical.
3. Port `EncoderMap`'s predictor chain and make it single rather than ensembled.
4. Restrict phase-1 window starts instead of clamping.

Then re-run rung 3 with the networks matched. Only after that does a density-objective
comparison mean what the spec says it means.

Gate every step with `scripts/rldp_q_calibration.py` (is Q on the return's scale?) and the
action-sensitivity probe above — the loss curve was stable and the representation
measurably good while the critic was action-blind, so neither is sufficient evidence.
