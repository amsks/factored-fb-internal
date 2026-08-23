"""TDFBCritic — the two-phase ball-visitation + hindsight successor measure.

Two different objects, learned in sequence over the SAME bilinear critic
`M(s,a,g) = phi(s,a)^T psi(g) / sqrt(d)`, from the bridge worktree
(origin/gciql-fb-bridge @ 542c058, tools/wandb_mode_shim/gciql_flow.py: `_hit_matrix`
~1384, `_measure_terms` ~1488, `measure_loss` ~1574, the `tpd` hindsight branch ~1674).
Ported per docs/specs/2026-08-10-measure-objectives-on-mw-play.md P2; see
docs/successor-measure-objectives.md §6-10 for the code-reading guide.

**Phase 1 — first-passage (absorbing).** How soon do you FIRST enter the goal region:
    diff = M - hit - gamma*(1 - hit)*sg[tM]
`(1 - hit)` is the absorption — once you land in the region the recursion stops, so the
fixed point is bounded in [0,1] (~E[gamma^T] for the first-arrival time T). Trained to
provide the representation (psi) phase 2 starts from. `measure_first_passage=True,
measure_fp_absorbing=True`.

**Phase 2 — ball-visitation (non-absorbing).** Drop the absorption and you get genuine
occupancy, revisits counted:
    diff = M - hit - gamma*sg[tM]
Still driven by the SAME `hit` indicator (`measure_first_passage=True`), just with
`measure_fp_absorbing=False` — this is what actually turns the recursion into occupancy;
it is NOT `measure_first_passage=False` (that knob, off, drops `hit` entirely and falls
back to the plain FB TD residual `M - gamma*tM`, which is neither phase — see the
docstring on `_measure_terms` and docs/successor-measure-objectives.md §6). Phase 2 does
not train from scratch: `measure_stopgrad_psi=True` freezes the backward map psi at
whatever a phase-1 warm start supplied (`build()` refuses this without a warm start — the
bridge's "FrozenPsi trap").

**The columns.** The measure is fit as a [B,B] matrix: rows are batch transitions, columns
are test points g_j (`measure_testpoint_dist`). `hindsight` (`= batch['value_goals']`,
trajectory futures) is the only arm that works on multi-object environments — `batch`
(columns = the batch's own next states) collapses on conjunctive environments (dwell
constant) and is not exercised by any config in this program (ledger §9.16).

One module, both phases as loss FLAGS (not two classes): the param tree is identical
across phases, so a phase-2 warm start is a straight params restore
(impls/utils/flax_utils.py::warm_start_agent).

Self-contained by design: the bilinear tower (`_Bilinear`) and its MLP/ensemble
primitives are copied here rather than imported from impls/utils/networks.py or from
density_fb.py — the only symbol shared between the two measure critics is
`impls.critics._measure_common` (the hit-spec parser + hit matrix).
"""
import copy
import dataclasses
import math
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import flax.linen as nn

from impls.critics._measure_common import parse_hit_spec, hit_matrix

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


class _Bilinear(nn.Module):
    """M(s,a,g) = phi(s,a)^T psi(g) / sqrt(d). Faithful copy of the vendored
    GCBilinearValue the bridge itself imports (third_party/ogbench/impls/utils/
    networks.py) — same submodule names/init order/math as impls/utils/networks.py's
    GCBilinearValue (locked at 1e-10 there by tests/impls/test_gcbilinear.py)."""
    hidden_dims: Sequence[int]
    latent_dim: int
    layer_norm: bool = True
    ensemble: bool = True

    def setup(self):
        mlp_class = _ensemblize(_MLP, 2) if self.ensemble else _MLP
        self.phi = mlp_class((*self.hidden_dims, self.latent_dim),
                             activate_final=False, layer_norm=self.layer_norm)
        self.psi = mlp_class((*self.hidden_dims, self.latent_dim),
                             activate_final=False, layer_norm=self.layer_norm)

    def __call__(self, observations, goals, actions=None, info=False):
        phi_inputs = observations if actions is None else \
            jnp.concatenate([observations, actions], axis=-1)
        phi = self.phi(phi_inputs)
        psi = self.psi(goals)
        v = (phi * psi / jnp.sqrt(self.latent_dim)).sum(axis=-1)
        if info:
            return v, phi, psi
        return v


def _measure_terms(M, tM, gamma, scale_norm, eps=1e-6, hit=None, absorbing=True):
    """Reduce the online measure M and stop-gradded target tM into the FB (offdiag, diag)
    TD terms over the last two (BxB) axes (leading head axes preserved):
        diff = M - gamma*tM;  offdiag = 0.5*sum((diff*off)^2)/sum(off);  diag = -mean(diag(diff))
    With scale_norm, divide by a stop-gradded measure scale so the gradient is
    DIMENSIONLESS — the same trick the actor q_loss uses (q/sg[|q|]) — which stops the
    unbounded <F,z> inflation from blowing up the shared critic:
        offdiag /= sg[mean(M^2)+eps];  diag /= sg[sqrt(mean(M^2))+eps].
    Ortho term is handled by the caller (already scale-bounded by psi).

    measure_first_passage: when `hit` (a [B,B] 0/1 arrival indicator
    1[||cube(s'_i) - cube(s+_j)|| <= eps]) is supplied, the residual becomes the
    FIRST-PASSAGE (absorbing) successor-measure residual
        diff = M - hit - gamma*(1 - hit)*sg[tM]
    so a hit terminates the recursion (no bootstrap) and the per-column fixed point is
    bounded in [0,1]. Because `hit` is a REPRESENTABLE 0/1 reward (unlike the Dirac
    density), the diagonal is a plain TD residual and is REGRESSED (squared) — keeping
    FB's linear diagonal attraction here would reintroduce the unbounded diagonal pull and
    the fixed point would stop being E[gamma^T_hit].
    hit=None => plain FB TD residual (neither phase of this objective — see module
    docstring: dropping `hit` is not how phase 2 is reached; `absorbing=False` is)."""
    B = M.shape[-1]
    off = 1.0 - jnp.eye(B)
    if hit is None:
        diff = M - gamma * tM
        diagt = -jnp.mean(jnp.diagonal(diff, axis1=-2, axis2=-1), axis=-1)
    else:
        # absorbing=True: first-passage (hit terminates the recursion; fixed point
        # E[gamma^T_hit] in [0,1]). absorbing=False: discounted BALL-VISITATION measure
        # M = E[sum_t gamma^t 1[s_{t+1} in ball(s+)]] — genuine occupancy of the goal ball
        # (revisits counted), still a plain regressed residual against a representable 0/1
        # reward => gamma-contraction, fixed point bounded by 1/(1-gamma).
        cont = (1.0 - hit) if absorbing else 1.0
        diff = M - hit - gamma * cont * tM
        diagt = jnp.mean(jnp.diagonal(diff, axis1=-2, axis2=-1) ** 2, axis=-1)
    offdiag = 0.5 * jnp.sum((diff * off) ** 2, axis=(-2, -1)) / jnp.sum(off)
    if scale_norm:
        msq = jax.lax.stop_gradient(jnp.mean(M ** 2))
        offdiag = offdiag / (msq + eps)
        diagt = diagt / ((jnp.sqrt(msq) if hit is None else msq) + eps)
    return offdiag, diagt


@dataclasses.dataclass(frozen=True)
class TDFBCritic:
    critic: Any
    obs_dim: int
    latent_dim: int
    blocks: tuple
    discount: float
    tau: float
    scale_norm: bool
    first_passage: bool
    fp_absorbing: bool
    stopgrad_psi: bool
    testpoint_dist: str
    ortho_coef: float
    actor_q_center: bool

    GROUPS = ("critic",)
    STAGES = ("measure",)
    PROVIDES_V = False
    PROVIDES_COND = frozenset({"value_goal", "goal"})
    # infer_cond here is a fixed zero placeholder that the eval goal path overrides, so it
    # never reads `obs`. Declared rather than defaulted -- see base.py::INFER_COND_MARGINALS.
    INFER_COND_MARGINAL = "unused"

    @property
    def TARGETS(self):
        return {"critic": self.tau}

    @property
    def cvec_dim(self):
        return self.obs_dim

    def stage_groups(self, stage):
        assert stage == "measure"
        return self.GROUPS

    def _info(self, p, obs, goal, actions):
        return self.critic.apply({"params": p}, obs, goal, actions, info=True)

    def _call(self, p, obs, goal, actions):
        return self.critic.apply({"params": p}, obs, goal, actions)

    def stage_loss(self, stage, params, target, batch, cond, boot):
        assert stage == "measure"
        obs, nobs = batch["observations"], batch["next_observations"]
        acts = batch["actions"]
        vgoals = cond["value_goal"]
        gamma = self.discount
        sqd = math.sqrt(self.latent_dim)
        B = obs.shape[0]
        off = 1.0 - jnp.eye(B)

        col_states = vgoals if self.testpoint_dist == "hindsight" else nobs
        # a+ = actor(s', ACTOR goals) — `boot["next_action"]`, the _cvec draw — NOT the
        # value-goal draw the bridge writes (gciql_flow.py :1655-1668: `a_goals = vgoals;
        # a_next = actor(nobs, a_goals, noise)`). This is a DELIBERATE divergence from the
        # bridge's literal line, forced by measurement. The triple (cube-single, phase-1
        # first-passage recipe, 500k, seed 0, flowbc):
        #     boot["next_action"]        (a+ at actor_goals)  -> 0.920
        #     boot["next_action_vgoal"]  (a+ at value_goals)  -> 0.076   [GEN2 collapse]
        #     bridge, its own vgoal convention                -> 0.967
        # Every axis of measure_loss was diffed against the bridge line by line (M/tM tower
        # args, the hindsight psi/tpsi recompute at vgoals, sqrt(d), hit(nobs, col_states),
        # the (1-hit) absorption, the off/diag split and weights, scale_norm placement,
        # stop-gradients, one-shot unclipped actor draw): all MATCH. The vgoal draw is
        # nonetheless catastrophic HERE, and the failure is diagnostic, not noise —
        # train.csv for the GEN2 run: measure_orth (= psi psi^T reduction, logged even at
        # ortho_coef=0) 25 -> 397 -> 2.7e6, and actor q 1.0 -> 5.0 -> 1.6e4, while
        # measure_loss itself stays ~3e-3. i.e. M is pinned on the trained columns while the
        # Q the actor actually maximizes inflates by four orders of magnitude.
        #
        # Mechanism (why td_fb differs from density_fb, which genuinely wanted the vgoal
        # fix — density-hard 0.912/0.932/0.904 with it vs 0.828/0.740/0.576 without):
        # density_fb's forward tower takes the goal as an INPUT, F(s,a,g), so its row IS
        # g-indexed and the bootstrap must be taken at that same g. Here the forward map is
        # phi(s,a) with NO goal leg — one vector per row serving every column — so phi can
        # only ever be the successor measure of ONE policy, and the bootstrap action is what
        # picks it. The readout the actor maximizes is <phi(s,a), psi(actor_goal)>, so phi
        # must be the measure of pi(.|., actor_goal). Drawing a+ at value_goals fits phi to
        # a policy indexed by value_goals — 20% current-state + 30% random-state goals, i.e.
        # half the rows bootstrap a policy heading nowhere — and then nothing in the TD
        # constrains phi along the psi(actor_goal) directions the actor reads, which is
        # exactly the unbounded channel the orth/q traces above show it exploiting.
        # Explicit KeyError on a missing key is kept on purpose (a silent .get fallback is
        # how the density bug survived its first gate); `next_action_vgoal` is still emitted
        # by flowbc.sample_bootstrap for density_fb and is deliberately unread here.
        a_next = jax.lax.stop_gradient(boot["next_action"])

        _, phi, psi = self._info(params["critic"], obs, col_states, acts)
        if self.stopgrad_psi:
            psi = jax.lax.stop_gradient(psi)
        M = jnp.einsum('hid,hjd->hij', phi, psi) / sqd

        _, tphi, tpsi = self._info(target["critic"], nobs, col_states, a_next)
        tM = jax.lax.stop_gradient(jnp.einsum('hid,hjd->hij', tphi, tpsi) / sqd)

        if self.first_passage:
            hit = hit_matrix(nobs, col_states, self.blocks, M.dtype)
            offdiag, diagt = _measure_terms(M, tM, gamma, self.scale_norm,
                                            hit=hit, absorbing=self.fp_absorbing)
        else:
            offdiag, diagt = _measure_terms(M, tM, gamma, self.scale_norm)

        Cov = jnp.einsum('hid,hjd->hij', psi, psi)
        o_off = 0.5 * jnp.sum((Cov * off[None]) ** 2, axis=(1, 2)) / jnp.sum(off)
        o_diag = -jnp.mean(jnp.diagonal(Cov, axis1=1, axis2=2), axis=1)
        orth = o_off + o_diag
        loss = jnp.mean(offdiag + diagt + self.ortho_coef * orth)
        info = {"measure_loss": loss, "measure_offdiag": jnp.mean(offdiag),
                "measure_diag": jnp.mean(diagt), "measure_orth": jnp.mean(orth)}
        return loss, info

    def q_fn(self, params, obs, cond, actions, *, stop_encoder_grad=False):
        del stop_encoder_grad
        g = cond["goal"]
        p = jax.lax.stop_gradient(params["critic"])
        q1, q2 = self._call(p, obs, g, actions)
        q = jnp.minimum(q1, q2)
        if self.actor_q_center:
            # Bridge semantics (gciql_flow.py ~1168): actor_q_center touches ONLY the
            # q-loss NORMALIZER — q_scale = |q - q.mean()|.mean() — while -q.mean()'s
            # gradient stays raw. FlowBC divides by sg[|Qs|.mean()], so the centering goes
            # into the SECOND return. Centering the numerator makes the greedy term the
            # batch-mean of a batch-centered quantity (~0, no gradient) — the phase-2
            # anchor collapsed to the BC floor (0.12) exactly this way (2026-08-11),
            # mirroring density_fb's failed first generation.
            return q, (q - q.mean())[None]
        return q, jnp.stack([q1, q2], 0)

    def cond_vec(self, cond):
        return cond["goal"]

    def sample_cond(self, net, batch, rng):
        return {"value_goal": batch["value_goals"], "goal": batch["actor_goals"]}

    def eval_cond0(self, ex_obs, config):
        return {"goal": jnp.zeros((self.obs_dim,), jnp.float64)}

    def infer_cond(self, net, obs, rewards):
        return {"goal": jnp.zeros((self.obs_dim,), jnp.float64)}

    @classmethod
    def build(cls, ex_obs, ex_actions, config):
        import jax.random as jr
        hd = tuple(config["value_hidden_dims"])
        ln = bool(config["layer_norm"])
        d = int(config["value_latent_dim"])
        blocks = parse_hit_spec(config.get("measure_hit_spec", "19:22:0.4"))
        obs_dim = int(ex_obs.shape[-1])

        stopgrad_psi = bool(config.get("measure_stopgrad_psi", False))
        warm_start = bool(config.get("warm_start", False))
        # The bridge's FrozenPsi trap: measure_stopgrad_psi=True is the phase-2 recipe and
        # is correct ONLY with a warm-started (phase-1-trained) psi. A from-scratch run
        # that sets it freezes psi at random init and floors the cell for a reason that has
        # nothing to do with whatever hypothesis the run is testing. `warm_start` here is a
        # plain boolean config knob the launcher must set alongside --warm_start_path/
        # --warm_start_epoch (impls/main.py) — build() has no visibility into those CLI
        # flags, which are consumed after Agent.create() (impls/utils/flax_utils.py::
        # warm_start_agent), so this checks the operator's declared intent, not the restore
        # itself.
        assert not (stopgrad_psi and not warm_start), (
            "measure_stopgrad_psi=True requires warm_start=True (set together with "
            "--warm_start_path/--warm_start_epoch at launch) — a from-scratch run would "
            "freeze psi at random init and floor the cell for a reason unrelated to any "
            "hypothesis (the bridge's FrozenPsi trap; docs/successor-measure-objectives.md §10)."
        )

        critic = _Bilinear(hidden_dims=hd, latent_dim=d, layer_norm=ln, ensemble=True)
        ex_goal = jnp.zeros_like(ex_obs)
        rc = jr.PRNGKey(config["seed"] + 7000)
        params = {"critic": critic.init(rc, ex_obs, ex_goal, ex_actions)["params"]}
        targets = {"critic": copy.deepcopy(params["critic"])}

        obj = cls(critic=critic, obs_dim=obs_dim, latent_dim=d, blocks=blocks,
                  discount=float(config["discount"]), tau=float(config["tau"]),
                  scale_norm=bool(config.get("measure_scale_norm", True)),
                  first_passage=bool(config.get("measure_first_passage", False)),
                  fp_absorbing=bool(config.get("measure_fp_absorbing", True)),
                  stopgrad_psi=stopgrad_psi,
                  testpoint_dist=str(config.get("measure_testpoint_dist", "hindsight")),
                  ortho_coef=float(config.get("measure_ortho_coef", 0.0)),
                  actor_q_center=bool(config.get("actor_q_center", False)))
        return obj, params, targets
