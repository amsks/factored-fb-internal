"""DensityFBCritic — the continuous normalized successor density (bridge spec 18).

One-stage, from-scratch, two-tower softmax density f(s,a,g,u) = F(s,a,g)^T B(u) / tau over
a query support U resampled EVERY update from the batch's own next-state object
coordinates (so U == the batch's next_observations block columns; no prototype file, no
fixed support). Cube-single 0.929 on the bridge (origin/gciql-fb-bridge @ 542c058,
tools/wandb_mode_shim/gciql_flow.py — `density_loss` ~1975, the density branch of
`actor_loss` ~1089, the tower registration in `create` ~3147, the target soft-update in
`update` ~2587). Ported near-verbatim per docs/specs/2026-08-10-measure-objectives-on-mw-
play.md P1; see docs/successor-measure-objectives.md §1-5 for the code-reading guide.

Contract details preserved (each is pinned by a test in tests/impls/test_density_fb.py):
  * rewards/kernel counted at the NEXT state (x_{t+1});
  * `_delta_phi` eps-scales INTERNALLY — the kernel term gets RAW block cols
    (`_raw_block_cols`), the B tower gets PRE-SCALED queries (`_density_queries`); feeding
    the wrong one to the wrong callsite silently trains a different object;
  * softmax is over the QUERY axis (last axis), not the batch axis;
  * the bootstrap action a+ is drawn at the VALUE goal — `boot["next_action_vgoal"]`, i.e.
    a+ ~ pi(.|s', cond["value_goal"]) — NOT the seam's default `boot["next_action"]`, which
    `_cvec(cond)` conditions on the ACTOR goal. The bridge draws it as
    `actor(nobs, batch['value_goals'], noise)` (gciql_flow.py ~2030), and every other
    bootstrap in that file is at `vgoals` too. Reading the actor-goal action makes w_bar the
    occupancy of a policy heading to a different goal, i.e. a goal-averaged (behaviour-like)
    density: measured density_ess_median ~310 vs the bridge's healthy ~89 at M=B=1024;
  * queries are resampled every update from the CURRENT batch's own next-state object
    coords (stage_loss recomputes them from `batch['next_observations']` every call —
    there is no stored/cached support);
  * NO orthonormalization, NO mass anchor, NO scale normalization on this path — each was
    separately measured harmful on the bridge (ortho took cube-single 0.967 -> 0.225).

`actor_q_center` (bridge's "known gap", per the spec): the bridge's actor centers the
Q-LOSS NORMALIZER (sg[mean|q - mean(q)|] instead of sg[mean|q|]); our FlowBCActor has no
such knob. Per spec §2, implemented instead INSIDE `q_fn` — subtract the batch mean of Q
before it reaches the actor's own sg[|Q|.mean()] normalization. Whether this composition
reproduces the bridge's numbers is exactly what the cube anchor (P5) gates.

Self-contained by design (spec §2): the tower modules (MLP, the ensemble helper) are
copied here rather than imported from impls/utils/networks.py, so this file's isolation
from other critics is structural, not just a registry line. The only symbol shared with
td_fb.py is `impls.critics._measure_common.parse_hit_spec` (the hit-spec grammar).
"""
import copy
import dataclasses
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import flax.linen as nn

from impls.critics._measure_common import parse_hit_spec

# ── self-contained tower primitives (copied from the vendored OGBench networks.py the
# bridge itself imports MLP/ensemblize from — third_party/ogbench/impls/utils/networks.py
# in the bridge worktree — so the density towers' init/activation match the bridge's). ──
_DEFAULT_INIT = nn.initializers.variance_scaling(1.0, 'fan_avg', 'uniform')


def _ensemblize(cls, num_qs, out_axes=0, **kwargs):
    return nn.vmap(cls, variable_axes={'params': 0}, split_rngs={'params': True},
                   in_axes=None, out_axes=out_axes, axis_size=num_qs, **kwargs)


class _MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Any = nn.gelu
    activate_final: bool = False
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=_DEFAULT_INIT)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
        return x


# ── the density kernel — _delta_phi / _density_queries / _raw_block_cols (bridge
# gciql_flow.py ~1403-1449), NOT shared with td_fb (td_fb uses _hit_matrix, not this). ──
def _density_queries(nobs, blocks):
    """[M, d_c] eps-scaled block coords of next_observations — the resampled query
    support U AND the B-tower input rows."""
    parts = [nobs[:, a:b] / e for (a, b, e) in blocks]
    return jnp.concatenate(parts, axis=-1)


def _raw_block_cols(obs_rows, blocks):
    """[N, d_c] RAW (unscaled) concatenated block columns — the twin of
    `_density_queries` used as the `protos` argument to `_delta_phi`, which applies its
    own per-block eps scaling internally. Do NOT pre-scale this input."""
    parts = [obs_rows[:, a:b] for (a, b, e) in blocks]
    return jnp.concatenate(parts, axis=-1)


def _delta_phi(obs_rows, protos, blocks):
    """softmax_j( -0.5 * sum_blocks ||(x_blk - proto_blk)/eps_blk||^2 ), over the QUERY
    axis (last). `obs_rows` [N, obs_dim] RAW; `protos` [M, d_c] RAW block-concatenated
    columns (the `_raw_block_cols` convention) — eps scaling happens INTERNALLY here, so
    callers must feed both sides RAW (the `_density_queries` pre-scaled rows are the
    B-tower's own input, never this function's)."""
    protos = jnp.asarray(protos, jnp.float32)
    parts_x, parts_p, off = [], [], 0
    for (a, b, e) in blocks:
        w = b - a
        parts_x.append(obs_rows[:, a:b] / e)
        parts_p.append(protos[:, off:off + w] / e)
        off += w
    xc = jnp.concatenate(parts_x, axis=-1)            # [N, d_c]
    pc = jnp.concatenate(parts_p, axis=-1)             # [M, d_c]
    diffs = xc[:, None, :] - pc[None, :, :]            # [N, M, d_c]
    logits = -0.5 * jnp.sum(diffs ** 2, axis=-1)       # [N, M]
    return jax.nn.softmax(logits, axis=-1)             # query axis = last axis


@dataclasses.dataclass(frozen=True)
class DensityFBCritic:
    density_f: Any
    density_b: Any
    obs_dim: int
    blocks: tuple           # parsed measure_hit_spec, nonpytree (plain python tuple)
    discount: float
    temp: float
    tau: float
    actor_q_center: bool
    hard: bool
    # W1 (edge-attribution spec): 'actor' (default, bit-identical) bootstraps the TD target at
    # the actor's action, so the actor sits inside the critic's fixed point — mechanism M1,
    # the confirmed collapse cause. 'sarsa' bootstraps at the DATASET's next action instead,
    # giving GCIQL's in-sample-backup property while keeping the kernel readout, which is what
    # separates M1 from M2.
    bootstrap: str = "actor"
    # W2: temperature for the ACTOR READOUT ONLY (q_fn's softmax). None -> `temp`, i.e.
    # bit-identical. The CE training target in stage_loss always keeps `temp`; a knob that
    # moved both would confound readout smoothness (M2) with the measure being trained.
    actor_temp: Any = None

    GROUPS = ("density_f", "density_b")
    STAGES = ("density",)
    PROVIDES_V = False
    PROVIDES_COND = frozenset({"value_goal", "goal"})

    # F(s, a, g): the row is g-indexed, so the TD target must bootstrap at that same g.
    # Reading the actor-goal action instead makes w_bar a goal-averaged (behaviour-like)
    # density -- measured density_ess_median ~310 against a healthy ~89.
    BOOT_COND = "value_goal"

    @property
    def TARGETS(self):
        return {"density_f": self.tau, "density_b": self.tau}

    @property
    def cvec_dim(self):
        return self.obs_dim

    def stage_groups(self, stage):
        assert stage == "density"
        return self.GROUPS

    # --- module apply helpers ---
    def _F(self, p, x):
        return self.density_f.apply({"params": p}, x)     # [2, B, d]

    def _B(self, p, u):
        return self.density_b.apply({"params": p}, u)      # [M, d]

    def stage_loss(self, stage, params, target, batch, cond, boot):
        assert stage == "density"
        obs, nobs = batch["observations"], batch["next_observations"]
        acts = batch["actions"]
        g = cond["value_goal"]
        gamma, tau = self.discount, self.temp
        blocks = self.blocks

        u_raw = _raw_block_cols(nobs, blocks)                             # [M, d_c] RAW
        u_sc = _density_queries(nobs, blocks)                             # [M, d_c] eps-scaled

        b_emb = self._B(params["density_b"], u_sc)                        # [M, d]
        f_emb = self._F(params["density_f"], jnp.concatenate([obs, acts, g], -1))  # [2,B,d]
        logits = jnp.einsum('hbd,md->hbm', f_emb, b_emb) / tau            # [2, B, M]
        w = jax.nn.softmax(logits, -1)
        logp = jax.nn.log_softmax(logits, -1)

        # Bootstrap action from the actor (already computed outside the grad tape by
        # Agent.update -> actor.sample_bootstrap; stop_gradient is defense-in-depth).
        # It MUST be the action for the VALUE goal g (= cond["value_goal"], the same g the
        # F towers are conditioned on here) — the bridge draws it as
        # `a_plus = sg(actor(nobs, batch['value_goals'], noise))` (gciql_flow.py ~2030).
        # `boot["next_action"]` is conditioned on the ACTOR goal (`_cvec(cond)`), which is a
        # different goal distribution (p_trajgoal=1.0 vs the 0.2/0.5/0.3 value mix): using it
        # makes w_bar the occupancy of a policy heading elsewhere, i.e. a goal-averaged,
        # much more DIFFUSE density (measured: density_ess_median ~310 vs the bridge's ~89).
        # Indexed, not .get()-defaulted: a silent fallback is exactly how this was missed.
        if self.bootstrap == "sarsa":
            # W1 / KT-B: bootstrap at the action the BEHAVIOUR policy took at s'. The actor is
            # then outside the critic's fixed point entirely — it can no longer walk the
            # target around, which is GCIQL's M1 property, while the softmax-kernel readout
            # (M2) stays exactly as it is. That is what makes this arm a clean M1-vs-M2
            # discriminator rather than another mitigation.
            if "next_actions" not in batch:
                raise KeyError(
                    "measure_density_bootstrap='sarsa' needs batch['next_actions'], which "
                    "GCDataset emits. A dataset_class that does not serve it cannot run this "
                    "mode — do not fall back to the actor action, that silently restores M1.")
            a_plus = jax.lax.stop_gradient(batch["next_actions"])
        elif self.bootstrap == "actor":
            if "next_action_vgoal" not in boot:
                raise KeyError(
                    "density_fb needs boot['next_action_vgoal'] — the bootstrap action drawn "
                    "at cond['value_goal'] (FlowBCActor.sample_bootstrap emits it whenever "
                    "the critic's cond carries 'value_goal'). boot['next_action'] is the "
                    "ACTOR-goal action and is NOT a valid bootstrap for this target.")
            a_plus = jax.lax.stop_gradient(boot["next_action_vgoal"])
        else:
            raise ValueError(f"measure_density_bootstrap must be 'actor' or 'sarsa', "
                             f"got {self.bootstrap!r}")
        tb = self._B(target["density_b"], u_sc)                           # frozen target
        tf = self._F(target["density_f"], jnp.concatenate([nobs, a_plus, g], -1))
        w_bar = jax.nn.softmax(jnp.einsum('hbd,md->hbm', tf, tb) / tau, -1)  # [2, B, M]

        if self.hard:
            # spec 21: one-hot immediate target at the row's OWN successor (M == B because
            # the query support is this batch's next states). All (1-gamma) mass on the
            # observed next state; the soft eps-kernel was diluting the readout.
            k = jnp.eye(nobs.shape[0])
        else:
            k = _delta_phi(nobs, u_raw, blocks)                            # [B, M]
        t = jax.lax.stop_gradient((1.0 - gamma) * k[None] + gamma * w_bar)
        ce = -jnp.mean(jnp.sum(t * logp, axis=-1))

        ess_rows = 1.0 / jnp.sum(w ** 2, axis=-1)                          # [2, B]
        ess = jnp.mean(ess_rows)
        ess_median = jnp.median(ess_rows)                                  # MEDIAN, not mean — the gate stat
        ess_h0 = jnp.mean(ess_rows[0])
        ess_h1 = jnp.mean(ess_rows[1])
        k_self = jnp.mean(jnp.diagonal(k))
        info = {
            "density_ce": ce,
            "density_ess": ess,
            "density_ess_median": ess_median,
            "density_ess_h0": ess_h0,
            "density_ess_h1": ess_h1,
            "density_k_self": k_self,
        }
        # How far apart the two bootstrap choices are. Under `sarsa` this is the size of the
        # substitution being made; under `actor` it is the same R1 signal from the critic's
        # side — how far the policy has walked off the behaviour support at s'. Logged in both
        # modes so the two arms are directly comparable, and only when the actor's draw is
        # available (a critic-only harness may not supply it).
        if "next_action_vgoal" in boot and "next_actions" in batch:
            info["density_sarsa_subst"] = jnp.linalg.norm(
                jax.lax.stop_gradient(boot["next_action_vgoal"]) - batch["next_actions"],
                axis=-1).mean()
        return ce, info

    def q_fn(self, params, obs, cond, actions, *, stop_encoder_grad=False):
        """The actor's greedy readout: q_h = sum_j w_h(s,a,g,u_j) * kernel(goal, u_j), the
        density towers evaluated at FROZEN (stop-gradient) params — the measure trains
        only through stage_loss, never through the actor's -Q.mean() term. Requires
        `cond["_density_u_raw"]`/`cond["_density_u_sc"]` (stashed by `sample_cond` from the
        SAME batch's next_observations `q_fn` itself has no access to — see sample_cond)."""
        del stop_encoder_grad
        g = cond["goal"]
        u_raw = cond["_density_u_raw"]
        u_sc = cond["_density_u_sc"]
        kg = jax.lax.stop_gradient(_delta_phi(g, u_raw, self.blocks))      # [B, M]
        f_p = jax.lax.stop_gradient(params["density_f"])
        b_p = jax.lax.stop_gradient(params["density_b"])
        b_emb = self._B(b_p, u_sc)
        f_emb = self._F(f_p, jnp.concatenate([obs, actions, g], -1))
        # W2 / KT-C: the ACTOR readout may use its own temperature. Raising it flattens the
        # softmax the actor ascends (M2: a spiky readout is the residual suspect for the
        # precision gap, ESS ~13 under qc03) WITHOUT touching what the measure is trained to
        # be — stage_loss keeps self.temp. Default None => self.temp => bit-identical.
        a_tau = self.temp if self.actor_temp is None else float(self.actor_temp)
        w = jax.nn.softmax(jnp.einsum('hbd,md->hbm', f_emb, b_emb) / a_tau, -1)
        q1, q2 = (w * kg[None]).sum(-1)                                    # [2, B] -> [B],[B]
        q = jnp.minimum(q1, q2)
        if self.actor_q_center:
            # Bridge semantics (gciql_flow.py ~1168): actor_q_center touches ONLY the
            # normalizer — q_scale = |q - q.mean()|.mean(), with -q.mean()'s gradient
            # unchanged. FlowBC divides its q-term by sg[|Qs|.mean()], so the centering
            # goes into the SECOND return. Centering the first (the numerator) makes
            # -Q.mean() the batch-mean of a batch-centered quantity — identically ~0,
            # no greedy signal, pure-BC actor (the failed first anchor generation).
            return q, (q - q.mean())[None]
        return q, jnp.stack([q1, q2], 0)

    def cond_vec(self, cond):
        return cond["goal"]

    def sample_cond(self, net, batch, rng):
        # queries resampled every update from THIS batch's own next-state object coords.
        nobs = batch["next_observations"]
        u_raw = _raw_block_cols(nobs, self.blocks)
        u_sc = _density_queries(nobs, self.blocks)
        return {"value_goal": batch["value_goals"], "goal": batch["actor_goals"],
                "_density_u_raw": u_raw, "_density_u_sc": u_sc}

    def eval_cond0(self, ex_obs, config):
        return {"goal": jnp.zeros((self.obs_dim,), jnp.float64)}

    def infer_cond(self, net, obs, rewards):
        return {"goal": jnp.zeros((self.obs_dim,), jnp.float64)}

    @classmethod
    def build(cls, ex_obs, ex_actions, config):
        import jax.random as jr
        hd = tuple(config["value_hidden_dims"])
        ln = bool(config["layer_norm"])
        d = int(config["measure_density_dim"])
        blocks = parse_hit_spec(config.get("measure_hit_spec", "19:22:0.4"))
        obs_dim = int(ex_obs.shape[-1])

        density_f = _ensemblize(_MLP, 2)((*hd, d), activate_final=False, layer_norm=ln)
        density_b = _MLP((*hd, d), activate_final=False, layer_norm=ln)

        rf, rb = jr.split(jr.PRNGKey(config["seed"] + 6000), 2)
        ex_f = jnp.concatenate([ex_obs, ex_actions, ex_obs], -1)
        ex_u = _density_queries(ex_obs, blocks)
        params = {
            "density_f": density_f.init(rf, ex_f)["params"],
            "density_b": density_b.init(rb, ex_u)["params"],
        }
        targets = {"density_f": copy.deepcopy(params["density_f"]),
                   "density_b": copy.deepcopy(params["density_b"])}
        obj = cls(density_f=density_f, density_b=density_b, obs_dim=obs_dim, blocks=blocks,
                  discount=float(config["discount"]),
                  temp=float(config["measure_density_temp"]), tau=float(config["tau"]),
                  actor_q_center=bool(config.get("actor_q_center", True)),
                  # DEFAULT TRUE. The yaml has said true since the bridge-port spec's item
                  # A2, but this fallback said False -- so any path building the critic from
                  # a partial config silently got the losing variant. Now they agree.
                  hard=bool(config.get("measure_density_hard", True)),
                  bootstrap=str(config.get("measure_density_bootstrap", "actor")),
                  actor_temp=config.get("measure_density_actor_temp", None))
        return obj, params, targets
