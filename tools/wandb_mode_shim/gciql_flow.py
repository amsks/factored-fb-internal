"""GCIQL + FB FlowBC actor (JAX). Subclasses the vendored GCIQLAgent: keeps the
IQL expectile value + double critic UNCHANGED, and replaces the Gaussian DDPG+BC
actor with FQL's flow actor -- a flow-matching velocity field + a one-shot
noise-conditioned policy distilled from it, with Q-guidance from GCIQL's IQL
min(q1, q2). Registered into agents.agents['gciql_flow'] by sitecustomize.
Never edits OGBench.

This is the actor-only ablation of GCIQL (everything else identical), so it is
directly comparable to the GCIQL DDPG+BC runs. It differs from canonical FQL only
in the critic: GCIQL's IQL (expectile V + advantage) instead of FQL's plain TD.
"""
import copy
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from agents.gciql import GCIQLAgent, get_config as _gciql_get_config
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import GCValue, GCBilinearValue, GCActor, MLP, ensemblize

from ogbench_flow import GCNoiseActor, GCVectorField


def _sphere(x, d):
    """Project x onto the sphere of radius sqrt(d) along the last axis -- FB pins
    ||B||=||z||=sqrt(d) via a Norm layer (agents/fb/model.py)."""
    return x * jnp.sqrt(d * 1.0) / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-6)


def _parse_hit_spec(spec):
    """Parse `measure_hit_spec` -> tuple of (start, stop, eps) observation blocks.

    Spec 08 E7 (second environment). The first-passage / ball-visitation hit indicator
    h(s', g) was hardcoded to cube-single's cube xyz (obs[19:22], eps=0.4 obs units =
    4cm). Other manipspace envs need a CONJUNCTION over several blocks, each with its
    own radius, because their success criterion is a conjunction:

        cube-single  '19:22:0.4'                                     (the default)
        cube-double  '19:22:0.4,28:31:0.4'                           both cubes within 4cm
        scene        '19:22:0.4,28:30:0.5,32:34:0.5,36:37:0.72,38:39:0.6'
                     cube 4cm | button-0 one-hot | button-1 one-hot | drawer | window
                     (one-hot equality <=> L2 <= 0.5, since ||e_i - e_j|| in {0, sqrt2};
                      drawer/window use the env's 0.04m tolerance at their own scalers,
                      18 and 15 -- see ogbench scene_env.py:498-499, 671-673)

    A STRING (not a list): ml_collections rejects list-valued agent overrides on the
    Hydra command line (spec06 launch gotcha).
    """
    blocks = []
    for part in str(spec).split(','):
        part = part.strip()
        if not part:
            continue
        fields = part.split(':')
        assert len(fields) == 3, f"hit-spec block must be 'start:stop:eps', got {part!r}"
        a, b, e = int(fields[0]), int(fields[1]), float(fields[2])
        assert 0 <= a < b, f"hit-spec block {part!r}: need 0 <= start < stop"
        assert e >= 0.0, f"hit-spec block {part!r}: radius must be >= 0"
        blocks.append((a, b, e))
    assert blocks, f"measure_hit_spec parsed to no blocks: {spec!r}"
    return tuple(blocks)


def _parse_eps_levels(spec):
    """Parse `measure_eps_multiscale` -> tuple of hit-ball radii (obs units).

    Campaign v2 knob 1. Comma-separated positive floats, e.g. '0.2,0.4,0.8'.
    Each training column j draws one level; the measure_hit_spec block radii are
    scaled PROPORTIONALLY by level/0.4 (so multi-block specs keep their block
    ratios and scale together). A STRING for the same ml_collections reason as
    measure_hit_spec."""
    levels = tuple(float(x) for x in str(spec).split(',') if x.strip())
    assert levels, f"measure_eps_multiscale parsed to no levels: {spec!r}"
    assert all(lv > 0.0 for lv in levels), \
        f"measure_eps_multiscale radii must be > 0, got {levels}"
    return levels


class EpsGainTable(nn.Module):
    """measure_eps_multiscale: learned per-level psi gain table [n_levels, d],
    initialized to ONES (identity at init => M bit-identical to the base for any
    level assignment; only the hit radii move until the table trains). Registered
    ONLY when the knob is on, so default param trees stay byte-identical and
    warm_start splices every other module and fresh-inits exactly this one."""
    n_levels: int
    latent_dim: int

    @nn.compact
    def __call__(self, level_idx):
        table = self.param('table', nn.initializers.ones,
                           (self.n_levels, self.latent_dim))
        return table[level_idx]


class NhatHead(nn.Module):
    """measure_nhat_head (phase 2b): the TRAINED kernel-mass denominator
    N-hat(g) ~= rho(B_eps(g)) -- softplus(MLP_2x256(sg[psi(g)])) > 0. Trained by
    plain regression on the hit-matrix COLUMN MEANS (an unbiased per-batch MC
    estimate of the goal ball's data mass; with measure_hit_spec conjunctions it
    is exactly the conjunctive region's mass). Input is the STOP-GRADDED
    head-mean critic psi, so gradients reach only this head. Registered ONLY
    when the knob is on (default param trees stay byte-identical; warm_start
    fresh-inits this module alone, the eps_gain pattern)."""
    hidden: int = 256

    @nn.compact
    def __call__(self, psi_sg):
        h = MLP((self.hidden, self.hidden, 1), activate_final=False)(psi_sg)
        return jax.nn.softplus(h[..., 0])


class GCStatePsiValue(GCBilinearValue):
    """Bilinear critic with a STATE-CONDITIONED backward leg: Q=phi(s,a).psi(s,g)/sqrt(d).

    GCBilinearValue factorizes Q(s,a,g)=phi(s,a).psi(g): psi sees ONLY the goal, which is
    what makes it an FB-style rank-d goal embedding and what makes z=psi(g) a well-defined,
    state-independent task vector. This module feeds psi the pair (s,g) instead.

    It is the expressivity ablation of the bilinear bridge cell: does the bilinear win come
    from the low-rank GOAL bottleneck specifically, or merely from the inner-product head?
    A state-conditioned psi keeps the head but destroys the bottleneck.

    NOTE: this deliberately breaks the zero-shot task-inference API -- there is no longer a
    state-independent z=psi(g) to infer. That is the point of the ablation. Do not pair it
    with actor_cond='z', fb_forward_z, or any measure/SF cell that reads psi as a task vector.
    """

    def __call__(self, observations, goals, actions=None, info=False):
        if self.state_encoder is not None:
            observations = self.state_encoder(observations)
        if self.goal_encoder is not None:
            goals = self.goal_encoder(goals)

        phi_inputs = observations if actions is None else jnp.concatenate(
            [observations, actions], axis=-1)
        phi = self.phi(phi_inputs)
        psi = self.psi(jnp.concatenate([observations, goals], axis=-1))

        v = (phi * psi / jnp.sqrt(self.latent_dim)).sum(axis=-1)
        if self.value_exp:
            v = jnp.exp(v)
        return (v, phi, psi) if info else v


class GCStatePsiBoundedValue(GCStatePsiValue):
    """bilinear_sg_bounded (ogbench-sota item 5): GCStatePsiValue with a tanh OUTPUT BOUND
    on the state-conditioned backward leg: psi = scale*tanh(psi_raw/scale), bounding every
    component to (-scale, scale). Targets the T2 BVN diagnosis (unbounded psi + no V-target
    + local pockets). scale = p99 of trained psi(g) norms (config psi_bound_scale)."""
    psi_bound_scale: float = 1.0

    def __call__(self, observations, goals, actions=None, info=False):
        if self.state_encoder is not None:
            observations = self.state_encoder(observations)
        if self.goal_encoder is not None:
            goals = self.goal_encoder(goals)
        phi_inputs = observations if actions is None else jnp.concatenate(
            [observations, actions], axis=-1)
        phi = self.phi(phi_inputs)
        psi_raw = self.psi(jnp.concatenate([observations, goals], axis=-1))
        s = self.psi_bound_scale
        psi = s * jnp.tanh(psi_raw / s)
        v = (phi * psi / jnp.sqrt(self.latent_dim)).sum(axis=-1)
        if self.value_exp:
            v = jnp.exp(v)
        return (v, phi, psi) if info else v


class GCNormCapValue(GCBilinearValue):
    """phi_norm_cap (ogbench-sota item 4b): plain bilinear critic with a hard cap on the
    forward-leg norm: phi -> phi * min(1, c/||phi||), c = sg[p99 of ||phi|| over the batch].
    Bounds the off-support peak's phi-norm to the on-support bulk (residual-relocation
    surgery), leaving direction and the psi leg untouched. c=0 never happens (p99>0)."""

    def __call__(self, observations, goals, actions=None, info=False):
        if self.state_encoder is not None:
            observations = self.state_encoder(observations)
        if self.goal_encoder is not None:
            goals = self.goal_encoder(goals)
        phi_inputs = observations if actions is None else jnp.concatenate(
            [observations, actions], axis=-1)
        phi = self.phi(phi_inputs)
        psi = self.psi(goals)
        norm = jnp.linalg.norm(phi, axis=-1, keepdims=True)
        c = jax.lax.stop_gradient(jnp.quantile(norm, 0.99))
        phi = phi * jnp.minimum(1.0, c / (norm + 1e-8))
        v = (phi * psi / jnp.sqrt(self.latent_dim)).sum(axis=-1)
        if self.value_exp:
            v = jnp.exp(v)
        return (v, phi, psi) if info else v


class _LateFusionCore(nn.Module):
    """One critic head: towers phi(s,a), psi(g) in R^d, then a 2-layer MLP readout on
    concat[phi;psi] instead of the inner product. Ensemblized as a whole (params split
    per head), so each head owns its phi/psi/fusion -- the fusion MLP has parameters, so
    it must live INSIDE the vmap (ensemblize broadcasts a single [B,.] input to both heads)."""
    hidden_dims: tuple
    latent_dim: int
    fusion_hidden: int
    layer_norm: bool

    @nn.compact
    def __call__(self, phi_inputs, goals):
        phi = MLP((*self.hidden_dims, self.latent_dim), activate_final=False,
                  layer_norm=self.layer_norm, name='phi')(phi_inputs)
        psi = MLP((*self.hidden_dims, self.latent_dim), activate_final=False,
                  layer_norm=self.layer_norm, name='psi')(goals)
        fused = MLP((self.fusion_hidden, self.fusion_hidden, 1), activate_final=False,
                    layer_norm=self.layer_norm, name='fusion')(jnp.concatenate([phi, psi], axis=-1))
        return fused.squeeze(-1), phi, psi


class GCLateFusionValue(GCBilinearValue):
    """T5a late-fusion control (ogbench-sota implementations item 1). Same twin towers as
    GCBilinearValue (phi(s,a), psi(g) in R^d) but the readout is a 2-layer MLP on
    concat[phi;psi] instead of phi.psi/sqrt(d). Isolates Layer 1 (gradient pooling through
    the LINEAR readout): if this reranks in the bilinear band, Layer 1 is falsified.

    Q is NOT bilinear here, so there is no state-independent z=psi(g) task readout -- do not
    pair with actor_cond='z', fb_forward_z, or any measure/SF cell (enforced in create())."""
    fusion_hidden: int = 512

    def setup(self):
        core = _LateFusionCore
        if self.ensemble:
            core = ensemblize(core, 2)
        self.core = core(tuple(self.hidden_dims), self.latent_dim, self.fusion_hidden, self.layer_norm)

    def __call__(self, observations, goals, actions=None, info=False):
        if self.state_encoder is not None:
            observations = self.state_encoder(observations)
        if self.goal_encoder is not None:
            goals = self.goal_encoder(goals)
        phi_inputs = observations if actions is None else jnp.concatenate([observations, actions], axis=-1)
        v, phi, psi = self.core(phi_inputs, goals)
        if self.value_exp:
            v = jnp.exp(v)
        return (v, phi, psi) if info else v


class _SplitBilinearCore(nn.Module):
    """One head of the factored bilinear critic: Q = sum_k phi_k(s,a).psi_k(g_k)/sqrt(d_k),
    a separate goal tower psi_k per factor (goal slice), phi emitting K sub-vectors. All
    inside one ensemblized core (psi_k towers have params)."""
    hidden_dims: tuple
    d_k: int
    factor_bounds: tuple    # cumulative goal-dim boundaries, length K+1
    layer_norm: bool

    @nn.compact
    def __call__(self, phi_inputs, goals):
        K = len(self.factor_bounds) - 1
        phi = MLP((*self.hidden_dims, K * self.d_k), activate_final=False,
                  layer_norm=self.layer_norm, name='phi')(phi_inputs)
        phi_k = phi.reshape(phi.shape[:-1] + (K, self.d_k))            # [..., K, d_k]
        q = 0.0
        psis = []
        for k in range(K):
            g_k = goals[..., self.factor_bounds[k]:self.factor_bounds[k + 1]]
            psi_k = MLP((*self.hidden_dims, self.d_k), activate_final=False,
                        layer_norm=self.layer_norm, name=f'psi_{k}')(g_k)
            q = q + (phi_k[..., k, :] * psi_k / jnp.sqrt(self.d_k)).sum(axis=-1)
            psis.append(psi_k)
        psi = jnp.concatenate(psis, axis=-1)                          # [..., K*d_k]
        return q, phi, psi


class GCSplitBilinearValue(GCBilinearValue):
    """T5 scene split-head (ogbench-sota item 7): Q=sum_k phi_k(s,a).psi_k(g_k)/sqrt(d_k)
    with per-factor goal towers (k = the env's goal factors). Tests whether a factored task
    family (scene's objects/doors) needs a factored head. No single z=psi(g) -> refuse the
    task-vector paths (enforced in create())."""
    factor_bounds: tuple = (0,)

    def setup(self):
        core = _SplitBilinearCore
        if self.ensemble:
            core = ensemblize(core, 2)
        self.core = core(tuple(self.hidden_dims), self.latent_dim, tuple(self.factor_bounds),
                         self.layer_norm)

    def __call__(self, observations, goals, actions=None, info=False):
        if self.state_encoder is not None:
            observations = self.state_encoder(observations)
        if self.goal_encoder is not None:
            goals = self.goal_encoder(goals)
        phi_inputs = observations if actions is None else jnp.concatenate(
            [observations, actions], axis=-1)
        v, phi, psi = self.core(phi_inputs, goals)
        if self.value_exp:
            v = jnp.exp(v)
        return (v, phi, psi) if info else v


class GCOccValue(GCBilinearValue):
    """Occupancy-IQL shared-backward value/critic (occupancy.tex sec 3). ONE backward map
    B(g) is shared by V and both Q heads -- the shared successor-measure factorization the
    plain bilinear cell lacks (there psi_V, psi_Q1, psi_Q2 are separate).

        actions=None  -> V(s,g)   = F_V(s).B(g)/sqrt(d)      (single head)
        actions given -> Q_k(s,a,g)= F_k(s,a).B(g)/sqrt(d)   (2 heads)

    info=True returns (val, F, Bg) so the loss can read the shared B(goals) directly (for the
    occupancy reward + whitening). Registered as the 'critic'/'target_critic' modules; there
    is NO separate 'value' module -- value_loss calls this with actions=None."""

    def setup(self):
        d = self.latent_dim
        self.F_V = MLP((*self.hidden_dims, d), activate_final=False, layer_norm=self.layer_norm)
        self.F_crit = ensemblize(MLP, 2)((*self.hidden_dims, d), activate_final=False,
                                         layer_norm=self.layer_norm)
        self.B = MLP((*self.hidden_dims, d), activate_final=False, layer_norm=self.layer_norm)

    def backward(self, goals):
        """Shared backward map B(g) (for the occupancy reward + ortho)."""
        if self.goal_encoder is not None:
            goals = self.goal_encoder(goals)
        return self.B(goals)

    def __call__(self, observations, goals, actions=None, info=False):
        if self.state_encoder is not None:
            observations = self.state_encoder(observations)
        Bg = self.backward(goals)
        sqd = jnp.sqrt(self.latent_dim)
        F_v = self.F_V(observations)          # ALWAYS called so F_V is initialized (flax lazy init:
                                              # create() inits with actions, so the V path must run too)
        if actions is None:
            F = F_v
            val = (F * Bg).sum(axis=-1) / sqd                       # [.]
        else:
            F = self.F_crit(jnp.concatenate([observations, actions], axis=-1))   # [2,.,d]
            val = (F * Bg[None]).sum(axis=-1) / sqd                 # [2,.]
        if self.value_exp:
            val = jnp.exp(val)
        return (val, F, Bg) if info else val


class GCFBValue(GCBilinearValue):
    """FB-style bilinear critic with a TASK-CONDITIONED forward map.

    GCIQL's GCBilinearValue factorizes Q(s,a,g)=phi(s,a).psi(g)/sqrt(d): the forward
    leg phi(s,a) has NO task index, so a grafted FB successor measure collapses to the
    measure of a single (goal-randomized) policy. FB's defining structure is
    M(s,a,g)=F(s,a,z).B(g) -- a z-INDEXED family. This module restores that by feeding
    a stop-gradded task vector z=psi(z_goals) into the forward map:

        phi = phi([s, a, z]),  psi = B = psi(b_goals),  M = phi.psi/sqrt(d)

    z is detached (B and z share the psi MLP, but z is a frozen task input, per FB).
    When ensembled, z is the head-mean of psi (a single task vector per sample).
    `project_z` hard-projects z and B onto the sqrt(d) sphere (FB's normalization).
    """
    project_z: bool = False

    def setup(self):
        mlp_module = MLP
        if self.ensemble:
            mlp_module = ensemblize(mlp_module, 2)
        self.phi = mlp_module((*self.hidden_dims, self.latent_dim), activate_final=False, layer_norm=self.layer_norm)
        self.psi = mlp_module((*self.hidden_dims, self.latent_dim), activate_final=False, layer_norm=self.layer_norm)

    def _psi(self, goals):
        if self.goal_encoder is not None:
            goals = self.goal_encoder(goals)
        return self.psi(goals)

    def __call__(self, observations, b_goals, actions, z_goals, info=False, z_vec=None):
        if self.state_encoder is not None:
            observations = self.state_encoder(observations)
        if z_vec is not None:
            # explicit task vector (measure_z_random_frac mixed-z path); caller owns
            # any sphere projection -- z is a frozen task input either way.
            z = jax.lax.stop_gradient(z_vec)
        else:
            z = self._psi(z_goals)                  # task embedding (ensembled => [2,B,d])
            if self.ensemble:
                z = z.mean(axis=0)                  # single task vector per sample [B,d]
            z = jax.lax.stop_gradient(z)
            if self.project_z:
                z = _sphere(z, self.latent_dim)
        phi = self.phi(jnp.concatenate([observations, actions, z], axis=-1))
        psi = self._psi(b_goals)                    # backward map B = psi(b_goals)
        if self.project_z:
            psi = _sphere(psi, self.latent_dim)
        v = (phi * psi / jnp.sqrt(self.latent_dim)).sum(axis=-1)
        if info:
            return v, phi, psi
        return v


class GCDuelingValue(GCBilinearValue):
    """Behavior-centered dueling critic: Q(s,a,g) = U(s,g) + A_bil(s,a,g).

    A_bil(s,a,g) = (phi(s,a) - b(s)).psi(g)/sqrt(d) is GCBilinearValue's bilinear
    score with the forward leg phi RE-CENTERED against a learned baseline feature
    b(s), an MLP head (same output dim + ensemble structure as phi) trained to
    approximate the behavior-policy mean E_{a~mu}[phi(s,a)] (gciql_flow.py's
    critic_loss adds a regression loss L_b = ||b(s) - sg(phi(s,a_data))||^2). U(s,g)
    is a plain state-goal MLP scalar head (the GCValue pattern), ensembled to 2 like
    the critic.

    No separate loss trains U or the Q-level split: Q=U+A_bil is trained end-to-end
    by the SAME IQL expectile critic TD as the plain bilinear critic. Because
    E_{a~mu}[A_bil] ~= 0 once b tracks phi's behavior mean, U absorbs the
    state-goal cardinal level and A_bil is left carrying the action-relative part --
    the flow actor's q-term then reads A_bil alone (see actor_loss), so maximizing
    it maximizes the part of Q the actor can actually influence through a, instead of
    U(s,g), which the actor cannot move at all.

    __call__ returns Q=U+A_bil (ensembled [2,B]) -- a drop-in for GCBilinearValue-
    style critics, so eval/atlas code that calls network.select('critic')(obs,g,a)
    is unaffected. info=True returns (Q, phi, psi), matching GCBilinearValue's
    (v, phi, psi) convention. full_info=True additionally returns a dict with
    A_bil, U, and b(s) split out, for the b-regression loss (critic_loss) and the
    actor's A_bil-only q-term (actor_loss)."""

    def setup(self):
        super().setup()   # self.phi, self.psi (GCBilinearValue's bilinear legs)
        mlp_module = MLP
        if self.ensemble:
            mlp_module = ensemblize(mlp_module, 2)
        self.b = mlp_module((*self.hidden_dims, self.latent_dim), activate_final=False, layer_norm=self.layer_norm)
        self.U_net = mlp_module((*self.hidden_dims, 1), activate_final=False, layer_norm=self.layer_norm)

    def __call__(self, observations, goals, actions=None, info=False, full_info=False):
        assert actions is not None, \
            "GCDuelingValue is critic-only (Q needs actions); use GCValue for the expectile V(s,g)"
        if self.state_encoder is not None:
            observations = self.state_encoder(observations)
        if self.goal_encoder is not None:
            goals = self.goal_encoder(goals)

        phi = self.phi(jnp.concatenate([observations, actions], axis=-1))   # [2,B,d]
        psi = self.psi(goals)                                               # [2,B,d]
        bs = self.b(observations)                                           # [2,B,d]
        A_bil = ((phi - bs) * psi / jnp.sqrt(self.latent_dim)).sum(axis=-1)  # [2,B]
        U = self.U_net(jnp.concatenate([observations, goals], axis=-1)).squeeze(-1)  # [2,B]
        Q = U + A_bil

        if full_info:
            return {'Q': Q, 'A_bil': A_bil, 'U': U, 'phi': phi, 'bs': bs, 'psi': psi}
        if info:
            return Q, phi, psi
        return Q


class _TwoTowerPhi(nn.Module):
    """FB-style two-tower forward map: F = trunk( MLP[s,a] , MLP[s,z] ).

    Mirrors agents/fb/model.py's ForwardMap (embed_sa over [s,a], embed_z over [s,z],
    combined by a trunk) -- the structural alternative to GCBilinearValue's flat
    phi=MLP([s,a]). The two embedding towers end activated (ReLU-like, activate_final);
    the trunk is linear to latent_dim. Submodules are named so the wiring is testable.
    """
    hidden_dims: tuple
    latent_dim: int
    layer_norm: bool = True

    @nn.compact
    def __call__(self, obs, actions, z):
        h = self.hidden_dims
        sa = obs if actions is None else jnp.concatenate([obs, actions], axis=-1)
        e_sa = MLP((*h, h[-1]), activate_final=True, layer_norm=self.layer_norm, name='embed_sa')(sa)
        e_sz = MLP((*h, h[-1]), activate_final=True, layer_norm=self.layer_norm, name='embed_sz')(
            jnp.concatenate([obs, z], axis=-1))
        return MLP((*h, self.latent_dim), activate_final=False, layer_norm=self.layer_norm, name='trunk')(
            jnp.concatenate([e_sa, e_sz], axis=-1))


class GCTwoTowerCritic(GCBilinearValue):
    """Bilinear critic whose forward leg is FB's two-tower F(s,a,z), with z=sg(psi(g))
    the goal embedding (#4). Q(s,a,g)=<F(s,a,z), psi(g)>/sqrt(d). Drop-in for
    GCBilinearValue -- same (observations, goals, actions) signature, z computed
    INTERNALLY from the goal -- so the IQL value/critic losses are unchanged. This is
    the architecture-only ablation of the ~90 bilinear-GCIQL (NOT the measure objective):
    does porting FB's two-tower forward map cost the IQL bilinear anything?

    psi doubles as the task source (z = stop_grad(head-mean psi(g))) exactly as FB's
    forward map embeds (s,a) and (s,z) in separate towers off the same backward map.
    `project_z` hard-projects z and B onto the sqrt(d) sphere (FB's normalization)."""
    project_z: bool = False

    def setup(self):
        two_tower, psi_mlp = _TwoTowerPhi, MLP
        if self.ensemble:
            two_tower = ensemblize(two_tower, 2)
            psi_mlp = ensemblize(psi_mlp, 2)
        self.phi = two_tower(hidden_dims=tuple(self.hidden_dims), latent_dim=self.latent_dim,
                             layer_norm=self.layer_norm)
        self.psi = psi_mlp((*self.hidden_dims, self.latent_dim), activate_final=False, layer_norm=self.layer_norm)

    def _psi(self, goals):
        if self.goal_encoder is not None:
            goals = self.goal_encoder(goals)
        return self.psi(goals)

    def __call__(self, observations, goals, actions=None, info=False):
        if self.state_encoder is not None:
            observations = self.state_encoder(observations)
        psi = self._psi(goals)
        if self.project_z:
            psi = _sphere(psi, self.latent_dim)
        z = psi.mean(axis=0) if self.ensemble else psi   # single task vector per sample
        z = jax.lax.stop_gradient(z)
        if self.project_z:
            z = _sphere(z, self.latent_dim)              # re-project after head-mean
        phi = self.phi(observations, actions, z)
        v = (phi * psi / jnp.sqrt(self.latent_dim)).sum(axis=-1)
        if self.value_exp:
            v = jnp.exp(v)
        if info:
            return v, phi, psi
        return v


class GCISFBValue(GCBilinearValue):
    """FB critic with a SECOND forward leg for the value successor measure (SG-ISFB,
    PAPER/fb-modified/isfb.tex). Shares the backward map B=psi between:
        M(s,a,s') = <F(s,a,z),   B(s')>/sqrt(d)   (action measure; z=sg(psi(z_goals)))
        N(s,s')   = <F_V(s,z),   B(s')>/sqrt(d)   (value measure; state/z only)
    so the IQL (Q,V) decoupling is done at the measure level over ONE shared B. Same
    (obs, b_goals, actions, z_goals) signature as GCFBValue; `return_fv=True` with
    `info=True` additionally exposes (fv, z) for the ISFB loss. `project_z` pins
    ||z||=||B||=sqrt(d)."""
    project_z: bool = False

    def setup(self):
        m = MLP
        if self.ensemble:
            m = ensemblize(m, 2)
        self.phi = m((*self.hidden_dims, self.latent_dim), activate_final=False, layer_norm=self.layer_norm)
        self.fv = m((*self.hidden_dims, self.latent_dim), activate_final=False, layer_norm=self.layer_norm)
        self.psi = m((*self.hidden_dims, self.latent_dim), activate_final=False, layer_norm=self.layer_norm)

    def _psi(self, goals):
        if self.goal_encoder is not None:
            goals = self.goal_encoder(goals)
        return self.psi(goals)

    def __call__(self, observations, b_goals, actions, z_goals, info=False, return_fv=False):
        if self.state_encoder is not None:
            observations = self.state_encoder(observations)
        z = self._psi(z_goals)
        if self.ensemble:
            z = z.mean(axis=0)                       # single task vector per sample [B,d]
        z = jax.lax.stop_gradient(z)
        if self.project_z:
            z = _sphere(z, self.latent_dim)
        phi = self.phi(jnp.concatenate([observations, actions, z], axis=-1))   # F(s,a,z)
        fv = self.fv(jnp.concatenate([observations, z], axis=-1))               # F_V(s,z)
        psi = self._psi(b_goals)                                               # B(s')
        if self.project_z:
            psi = _sphere(psi, self.latent_dim)
        v = (phi * psi / jnp.sqrt(self.latent_dim)).sum(axis=-1)
        if info and return_fv:
            return v, phi, psi, fv, z
        if info:
            return v, phi, psi
        return v


class GCIdentityBValue(GCBilinearValue):
    """Bilinear critic with an IDENTITY base/goal leg, for successor-FEATURE learning:

        Q(s,a,g) = <phi(s,a), g>/sqrt(d),   psi(g) = b(g) = g  (param-free)

    phi(s,a) is the successor-feature vector -- the ONLY learned leg; the base map b=psi is the
    identity (raw obs), so latent_dim MUST equal the goal/obs dim. The SF vector TD that trains
    phi (phi(s,a) <- b(s)+gamma*phi(s',a')) lives in the agent's `sf_loss`, NOT here -- this
    module only defines the bilinear readout. Mirrors agents/fb/fixed_backward.py's
    FixedIdentityBackward (B(obs)=obs). With `fb_forward_z` the forward map is TASK-CONDITIONED
    phi([s,a,z=g]) (the reference SF net input cat([obs, task]))."""
    fb_forward_z: bool = False

    def setup(self):
        m = MLP
        if self.ensemble:
            m = ensemblize(m, 2)
        self.phi = m((*self.hidden_dims, self.latent_dim), activate_final=False, layer_norm=self.layer_norm)

    def _psi(self, goals):
        return goals   # identity base, param-free (raw; no goal_encoder)

    def __call__(self, observations, b_goals, actions=None, z_goals=None, info=False):
        if self.state_encoder is not None:
            observations = self.state_encoder(observations)
        inp = observations if actions is None else jnp.concatenate([observations, actions], axis=-1)
        if self.fb_forward_z:                                  # task-condition phi([s,a,z=psi(g)])
            z = self._psi(z_goals)
            if z.ndim == 3:                                    # head-mean if ensembled
                z = z.mean(axis=0)
            inp = jnp.concatenate([inp, jax.lax.stop_gradient(z)], axis=-1)
        phi = self.phi(inp)                                    # [2,B,d] (ensembled) or [B,d]
        psi = jnp.broadcast_to(self._psi(b_goals), phi.shape)  # b(g)=g, matched to phi's shape
        v = (phi * psi / jnp.sqrt(self.latent_dim)).sum(axis=-1)
        if info:
            return v, phi, psi
        return v


class GCFrozenBValue(GCBilinearValue):
    """Bilinear critic with a FROZEN RANDOM base/goal leg psi, for successor-FEATURE learning
    at a FREE rank d (e.g. 50, matching FB / bilinear-GCIQL):

        Q(s,a,g) = <phi(s,a), psi(g)>/sqrt(d),   psi = stop_grad(MLP)  (random init, never trained)

    phi(s,a) is the trained successor-feature vector; the base feature is b(s)=psi(s), the frozen
    random features of the state. Freezing psi (stop_grad on every use -> zero gradient -> params
    stay at init) is the standard SF anti-collapse fix (a random encoder) and DECOUPLES the
    representation rank d from the obs dim, so d can match the FB / bilinear-GCIQL 50. With
    `fb_forward_z` the forward map is TASK-CONDITIONED phi([s,a,z=psi(g)]) (frozen features).
    `project_z` hard-projects psi (and hence the z fed to phi) onto the sqrt(d) sphere -- FB's
    ||B||=||z||=sqrt(d) Norm layer."""
    fb_forward_z: bool = False
    project_z: bool = False

    def setup(self):
        m = MLP
        if self.ensemble:
            m = ensemblize(m, 2)
        self.phi = m((*self.hidden_dims, self.latent_dim), activate_final=False, layer_norm=self.layer_norm)
        self.psi = m((*self.hidden_dims, self.latent_dim), activate_final=False, layer_norm=self.layer_norm)

    def _psi(self, goals):
        if self.goal_encoder is not None:
            goals = self.goal_encoder(goals)
        psi = jax.lax.stop_gradient(self.psi(goals))    # FROZEN: zero grad -> stays at random init
        return _sphere(psi, self.latent_dim) if self.project_z else psi

    def __call__(self, observations, b_goals, actions=None, z_goals=None, info=False):
        if self.state_encoder is not None:
            observations = self.state_encoder(observations)
        inp = observations if actions is None else jnp.concatenate([observations, actions], axis=-1)
        if self.fb_forward_z:                           # task-condition phi([s,a,z=psi(g)]) (frozen z)
            z = self._psi(z_goals)                      # frozen features of the goal
            if z.ndim == 3:
                z = z.mean(axis=0)                      # head-mean if ensembled
            inp = jnp.concatenate([inp, z], axis=-1)    # already stop-gradded in _psi
        phi = self.phi(inp)                             # [2,B,d] (ensembled) or [B,d]
        psi = self._psi(b_goals)                        # frozen random features, same shape as phi
        v = (phi * psi / jnp.sqrt(self.latent_dim)).sum(axis=-1)
        if info:
            return v, phi, psi
        return v


class GCLearnedBValue(GCFrozenBValue):
    """Bilinear critic with a LEARNED base/goal leg psi, for the CORRECTED successor-feature
    cell (E3, audit-20260706): psi is trained ONLY by reward regression (`sf_reward_regress`)

        r_hat(s,g) = <psi(s), psi(g)>/sqrt(d)  ->  regress on batch rewards

    so the goal indicator becomes linearly READABLE in the base (the identity/frozen cells'
    semantic confound: r=<s,g> is linear in s, argmax != goal). The SF vector TD still trains
    ONLY phi (sf_loss stop-grads b(s)=psi(s) in its target): psi makes the reward expressible,
    phi accumulates it. Division of labor, one shared psi network.

    NOTE (2026-07-10, re-read of the reference): sf_simple.py:322 regresses the reference's TASK
    VECTOR, not its basis -- the basis there is computed under torch.no_grad() and is trained by
    the critic TD instead. So the reference's reward model is ASYMMETRIC, <phi_basis(s',a'), z>,
    between a network and a FREE nn.Parameter vector. Our symmetric <psi(s), psi(g)> ties both
    legs to one net, which is exactly why it has a psi=0 stationary point (the saddle that killed
    E3 v1 and forced sf_reward_shift). An asymmetric two-embedding reward model would remove that
    degeneracy structurally rather than by shifting the targets."""

    def _psi(self, goals):
        if self.goal_encoder is not None:
            goals = self.goal_encoder(goals)
        # LEARNED (vs GCFrozenBValue): grads arrive via sf_reward_regress and/or sf_joint_psi.
        psi = self.psi(goals)
        return _sphere(psi, self.latent_dim) if self.project_z else psi

    def __call__(self, observations, b_goals, actions=None, z_goals=None, info=False):
        if self.state_encoder is not None:
            observations = self.state_encoder(observations)
        inp = observations if actions is None else jnp.concatenate([observations, actions], axis=-1)
        if self.fb_forward_z:                           # task-condition phi([s,a,z=psi(g)])
            z = self._psi(z_goals)
            if z.ndim == 3:
                z = z.mean(axis=0)                      # head-mean if ensembled
            inp = jnp.concatenate([inp, jax.lax.stop_gradient(z)], axis=-1)  # no grad into psi via phi's input
        phi = self.phi(inp)                             # [2,B,d] (ensembled) or [B,d]
        psi = self._psi(b_goals)                        # learned base features
        v = (phi * psi / jnp.sqrt(self.latent_dim)).sum(axis=-1)
        if info:
            return v, phi, psi
        return v


class GCSimpleSFValue(GCBilinearValue):
    """Chua et al.'s Simple SF head, ported (reference: simple_successor_features/agent/sf_simple.py).

        h(s,a,z)  = trunk([s, a, sg[z]])          # ONE trunk, not ensembled
        b(s,a,z)  = h / ||h||                     # reward BASIS -- zero params of its own
        Psi_i     = head_i(h)                     # twin heads (CriticSF.Q1/Q2)
        z         = psi(g)                        # the task vector (sqrt(d)-sphere if project_z)
        Q_i       = <Psi_i, z_used>/sqrt(d)

    b and Psi are DIFFERENT vectors off a shared trunk, and b carries no parameters: its only
    gradient arrives through the trunk, from the TD. That is what makes the reward model
    <b, psi(g)> ASYMMETRIC (a network against a task vector) and hence free of the psi=0
    stationary point that killed E3 v1 (whose <psi(s), psi(g)> is symmetric, so its gradient
    vanishes at zero and needed sf_reward_shift to escape).

    z is stop-gradded in the TRUNK INPUT in both rungs -- the reference passes task.detach() to
    update_critic (sf_simple.py:283). Otherwise the trunk's TD gradient would reach psi through
    the input concatenation and ground_psi would not isolate psi's signal. With ground_psi the
    readout also stop-grads z, leaving the reward regression as psi's ONLY gradient path."""
    ground_psi: bool = False
    project_z: bool = False

    def setup(self):
        head_m = ensemblize(MLP, 2) if self.ensemble else MLP
        self.trunk = MLP((*self.hidden_dims, self.latent_dim), activate_final=False,
                         layer_norm=self.layer_norm)
        self.head = head_m((self.hidden_dims[-1], self.latent_dim), activate_final=False,
                           layer_norm=self.layer_norm)
        self.psi = MLP((*self.hidden_dims, self.latent_dim), activate_final=False,
                       layer_norm=self.layer_norm)

    def __call__(self, observations, goals, actions=None, info=False):
        if self.state_encoder is not None:
            observations = self.state_encoder(observations)
        g = self.goal_encoder(goals) if self.goal_encoder is not None else goals
        z = self.psi(g)                                          # [B,d], grad ON
        if self.project_z:
            z = _sphere(z, self.latent_dim)
        z_in = jax.lax.stop_gradient(z)                          # trunk input NEVER trains psi
        parts = [observations] if actions is None else [observations, actions]
        h = self.trunk(jnp.concatenate([*parts, z_in], axis=-1))  # [B,d]
        b = h / (jnp.linalg.norm(h, axis=-1, keepdims=True) + 1e-6)
        Psi = self.head(h)                                       # [2,B,d] or [B,d]
        z_used = jax.lax.stop_gradient(z) if self.ground_psi else z
        v = (Psi * z_used).sum(axis=-1) / jnp.sqrt(self.latent_dim)
        if info:
            return v, Psi, z, b
        return v


class GCIQLFlowAgent(GCIQLAgent):
    """GCIQL with FB's FlowBC actor. Inherits value_loss/critic_loss/update."""

    # spec 17b, fix round 1 (CRITICAL 2): prototypes as a STATIC (nonpytree) field --
    # a hashable nested tuple-of-tuples ([K, d_c], row-major), None when off. Must NOT
    # be a pytree leaf: restore_agent (flax_utils.py) calls
    # flax.serialization.from_state_dict on the WHOLE agent, and flax's dataclass
    # to_state_dict/from_state_dict only walk pytree_node=True fields (data_fields) --
    # see flax.struct's generated to_state_dict/from_state_dict. A pytree-leaf array
    # here would make every pre-17b checkpoint's agent dict (only {'rng','network'})
    # fail restore with "Missing field delta_protos". As a static/meta field it is
    # never serialized and never required on restore -- self.replace(...) in update()
    # still preserves it automatically (dataclasses.replace only overrides named
    # kwargs). Static fields must be hashable for jax.jit's cache key, which a raw
    # numpy/jnp array is not (and its equality is elementwise/ambiguous) -- a tuple of
    # Python floats is. _delta_phi converts it back to a jnp array on every call;
    # inside jit that array becomes a folded constant baked into the trace, and the
    # per-call hash of a K x d_c (e.g. 256x3) nested tuple is negligible.
    delta_protos: Any = nonpytree_field(default=None)

    def _flow_rollout(self, obs, noise, params):
        """Euler-integrate the BC velocity field from noise -> action (flow_steps).
        `obs` is the vf's obs input -- under actor_vf_cond != 'none' the caller
        passes the already-conditioned input (_vf_obs), not the raw obs."""
        fs = self.config['flow_steps']
        actions = noise
        for i in range(fs):
            t = jnp.ones((noise.shape[0], 1)) * (i / fs)
            vels = self.network.select('actor_vf')(obs, actions, t, params=params)
            actions = actions + vels / fs
        return jnp.clip(actions, -1.0, 1.0)

    def _value_psi(self, goals, params=None):
        """B := the value net's psi (action-free GCBilinearValue). psi depends
        only on the goal arg; the obs arg is a placeholder for phi(s)."""
        kw = {'info': True}
        if params is not None:
            kw['params'] = params
        _v, _phi, psi = self.network.select('value')(goals, goals, **kw)
        return psi

    def _critic_psi(self, goals, params=None):
        """B := the CRITIC net's psi (audit #2 / D1). Under measure_only the value net
        is never trained, so the actor's z must come from the critic psi (=B, trained by
        the measure loss). Actions are a placeholder (psi is action-free); z_goals is
        passed when fb_forward_z. Ensembled critic psi -> head-mean [B,d]."""
        # placeholder actions match goals' leading (batch) dims -- goals may be a single
        # unbatched observation at eval, so key off shape[:-1], not shape[0].
        ex_a = jnp.zeros((*goals.shape[:-1], self.config['action_dim']))
        zarg = (goals,) if self.config.get('fb_forward_z', False) else ()
        kw = {'info': True}
        if params is not None:
            kw['params'] = params
        _v, _phi, psi = self.network.select('critic')(goals, goals, ex_a, *zarg, **kw)
        return psi.mean(axis=0)   # critic is ensembled: head-mean over axis 0 -> [..., d]

    def _actor_z(self, goals, params=None):
        """Task vector the flow actor conditions on: critic psi if actor_z_from_critic,
        else the value net psi (the G4 zcond default)."""
        if self.config.get('actor_z_from_critic', False):
            return self._critic_psi(goals, params)
        return self._value_psi(goals, params)

    def _vf_obs(self, obs, goals, params=None):
        """RG-FB v0 (actor_vf_cond): the obs input the BC vector field AND the
        distillation rollout consume. 'none' -> obs unchanged (bit-identical stock
        behavior: goal-agnostic behavioral anchor). 'goal' -> [obs, raw goal]
        (parity with the awr_flow vf, which concatenates the goal into its obs
        input). 'z' -> [obs, sg(psi(goal))] -- the SAME task vector the one-shot
        z-actor consumes (_actor_z), stop-gradded so the vf's flow-matching loss
        never trains psi (the representation stays purely IQL-trained)."""
        mode = self.config.get('actor_vf_cond', 'none')
        if mode == 'none':
            return obs
        cond = goals if mode == 'goal' else jax.lax.stop_gradient(self._actor_z(goals, params))
        return jnp.concatenate([obs, cond], axis=-1)

    def _infer_actor_z(self, psi, rng):
        """exp4: the INFERRED task vector z=E[r*psi] per sample. Draw a random sparse
        reward over the batch's goal embeddings psi [B,d] (each row keeps its own goal
        plus a random ~10% subset of the others), read out z=sum_j r_ij psi_j, and
        sphere-Norm to sqrt(d) -- the noisy inferred task distribution the actor is
        evaluated on (z=E[r psi(s')]), as opposed to the exact embedding psi(g)."""
        B = psi.shape[0]
        mask = (jax.random.uniform(rng, (B, B)) < 0.1).astype(psi.dtype) + jnp.eye(B, dtype=psi.dtype)
        z = mask @ psi
        return _sphere(z, self.config['value_latent_dim'])

    def value_loss(self, batch, grad_params):
        """IQL expectile value loss; ablate clipped double-Q (double_q=False => q1)."""
        occ = self.config.get('value_arch', 'mlp') == 'occupancy'
        zarg = (batch['value_goals'],) if self.config.get('fb_forward_z', False) else ()
        q1, q2 = self.network.select('target_critic')(
            batch['observations'], batch['value_goals'], batch['actions'], *zarg)
        q = jnp.minimum(q1, q2) if self.config['double_q'] else q1
        if occ:
            # V = F_V(s).B(g)/sqrt(d): the SAME shared-B critic module, actions=None branch.
            v = self.network.select('critic')(batch['observations'], batch['value_goals'],
                                              params=grad_params)
        else:
            v = self.network.select('value')(batch['observations'], batch['value_goals'], params=grad_params)
        value_loss = self.expectile_loss(q - v, q - v, self.config['expectile']).mean()
        return value_loss, {'value_loss': value_loss, 'v_mean': v.mean(),
                            'v_max': v.max(), 'v_min': v.min()}

    def critic_loss(self, batch, grad_params):
        """IQL critic loss; thread z_goals when fb_forward_z (B1): the z-conditioned
        forward map takes an extra task arg, so Q(s,a,g)=<F(s,a,psi(g)),psi(g)>/sqrt(d).

        value_arch='dueling': Q=U+A_bil is trained by the SAME TD (no change in form),
        but the critic call additionally exposes phi/bs (full_info=True) so the
        baseline regression L_b = mean(||b(s) - sg(phi(s,a_data))||^2) can be added
        here, weighted by dueling_b_coef. Uses the BATCH's own (behavior) actions --
        NOT an actor sample -- so b(s) is trained to track E_{a~mu}[phi(s,a)]."""
        if self.config.get('value_arch', 'mlp') == 'occupancy':
            # Occupancy-IQL critic backup (occupancy.tex sec 4/6): scalar reward replaced by the
            # CRL occupancy reward from the TARGET (EMA) shared-B map, and the goal-achievement
            # mask replaced by the env-continuation mask. B whitened by the ortho penalty.
            obs, nobs = batch['observations'], batch['next_observations']
            vgoals, acts = batch['value_goals'], batch['actions']
            d = self.config['value_latent_dim']; sqd = jnp.sqrt(d); gamma = self.config['discount']
            Bt_sprime = self.network.select('target_critic')(nobs, nobs, info=True)[2]   # B_t(s')
            Bt_g = self.network.select('target_critic')(nobs, vgoals, info=True)[2]       # B_t(g)
            r_B = jax.lax.stop_gradient((1.0 - gamma) * (Bt_sprime * Bt_g).sum(-1) / sqd)  # [B]
            m_env = self.config.get('occ_env_mask', 1.0)
            next_v = self.network.select('critic')(nobs, vgoals)                          # V(s',g) online, actions=None
            y = r_B + gamma * m_env * next_v
            q1, q2 = self.network.select('critic')(obs, vgoals, acts, params=grad_params)  # [2,B]
            crit = ((q1 - y) ** 2 + (q2 - y) ** 2).mean()
            Bo = self.network.select('critic')(obs, vgoals, info=True, params=grad_params)[2]  # online B(g) [B,d]
            C = (Bo.T @ Bo) / Bo.shape[0]                                                 # [d,d]
            orth = jnp.mean((C - jnp.eye(d)) ** 2)
            loss = crit + self.config.get('occ_ortho_coef', 1.0) * orth
            return loss, {'critic_loss': crit, 'occ_reward_mean': r_B.mean(), 'occ_ortho': orth,
                          'q_mean': (0.5 * (q1 + q2)).mean(), 'q_max': jnp.maximum(q1, q2).max(),
                          'occ_B_normsq': jnp.mean(jnp.sum(Bo ** 2, axis=-1))}
        next_v = self.network.select('value')(batch['next_observations'], batch['value_goals'])
        q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v
        if self.config.get('value_arch', 'mlp') == 'dueling':
            out = self.network.select('critic')(
                batch['observations'], batch['value_goals'], batch['actions'],
                full_info=True, params=grad_params)
            q1, q2 = out['Q']
            critic_loss = ((q1 - q) ** 2 + (q2 - q) ** 2).mean()
            # SCALE-NORM the b-regression: phi inflates during training (bilinear feature
            # growth), so raw ||b-phi||^2 blows up and dominates the TD (v1 hit critic_loss
            # ~1.25e4 == b_loss, starving the Q backup -> policy collapsed to floor). Dividing
            # by sg[mean(phi^2)] makes it a dimensionless relative error ~O(1), balanced against
            # the TD -- mirrors the repo's measure_scale_norm fix for raw measure grafts.
            b_scale = jax.lax.stop_gradient(jnp.mean(out['phi'] ** 2) + 1e-6)
            b_loss = jnp.mean((out['bs'] - jax.lax.stop_gradient(out['phi'])) ** 2) / b_scale
            b_coef = self.config.get('dueling_b_coef', 1.0)
            critic_loss = critic_loss + b_coef * b_loss
            return critic_loss, {'critic_loss': critic_loss, 'q_mean': q.mean(),
                                 'q_max': q.max(), 'q_min': q.min(),
                                 'dueling_b_loss': b_loss, 'dueling_b_scale': b_scale,
                                 'dueling_A_mean': out['A_bil'].mean(),
                                 'dueling_U_mean': out['U'].mean()}
        zarg = (batch['value_goals'],) if self.config.get('fb_forward_z', False) else ()
        q1, q2 = self.network.select('critic')(
            batch['observations'], batch['value_goals'], batch['actions'], *zarg, params=grad_params)
        critic_loss = ((q1 - q) ** 2 + (q2 - q) ** 2).mean()
        return critic_loss, {'critic_loss': critic_loss, 'q_mean': q.mean(),
                             'q_max': q.max(), 'q_min': q.min()}

    def actor_loss(self, batch, grad_params, rng=None, task_z=None):
        # task_z (RG-FB v1, spec06 item4): when supplied, the flow actor conditions on the
        # SAMPLED task vector z (per-sample [B,d], stop-gradded) instead of psi(actor_goals) --
        # the vf input, the one-shot actor's z-leg, AND the Q-max readout all use it.
        # actor_kind='awr_flow' => IQL's canonical AWR extraction on a flow policy:
        # goal-conditioned flow-matching BC weighted per-sample by
        # exp(awr_alpha * (min_i Q_i(s,a,g) - V(s,g))) -- the gcivl_flow/hiql_flow
        # readout, but with GCIQL's sharper Q-V advantage. No Q-max, no distillation,
        # no one-step actor: actions are sampled by Euler-integrating actor_vf.
        if self.config['actor_kind'] == 'awr_flow':
            obs, goals, actions = batch['observations'], batch['actor_goals'], batch['actions']
            q1, q2 = self.network.select('critic')(obs, goals, actions)
            q = jnp.minimum(q1, q2)
            v = self.network.select('value')(obs, goals)
            adv = q - v
            exp_a = jnp.minimum(jnp.exp(adv * self.config['awr_alpha']), 100.0)

            rng = rng if rng is not None else self.rng
            rng, x0_rng, t_rng = jax.random.split(rng, 3)
            x_0 = jax.random.normal(x0_rng, actions.shape)
            t = jax.random.uniform(t_rng, (actions.shape[0], 1))
            x_t = (1 - t) * x_0 + t * actions
            vel = actions - x_0
            obs_goal = jnp.concatenate([obs, goals], axis=-1)
            pred = self.network.select('actor_vf')(obs_goal, x_t, t, params=grad_params)
            fm = jnp.mean((pred - vel) ** 2, axis=-1)
            actor_loss = jnp.mean(exp_a * fm)
            return actor_loss, {'actor_loss': actor_loss, 'adv': adv.mean(),
                                'flow_bc': fm.mean(), 'exp_a': exp_a.mean()}
        # actor_kind='ddpgbc' => fall back to GCIQL's stock gaussian DDPG+BC actor
        # (bilinear critic kept); only the flow-actor path is custom below.
        if self.config['actor_kind'] == 'ddpgbc':
            if not self.config.get('fb_forward_z', False):
                return GCIQLAgent.actor_loss(self, batch, grad_params, rng)
            # fb_forward_z: the SF critic is task-conditioned and takes the z=goal leg
            # (obs, goals, actions, z_goals). Stock GCIQL calls it with 3 args, so mirror
            # its DDPG+BC loss here but pass the z arg (à la the flow path's zarg).
            obs, goals, actions = batch['observations'], batch['actor_goals'], batch['actions']
            rng = rng if rng is not None else self.rng
            dist = self.network.select('actor')(obs, goals, params=grad_params)
            q_actions = dist.mode() if self.config['const_std'] else dist.sample(seed=rng)
            q_actions = jnp.clip(q_actions, -1, 1)
            q1, q2 = self.network.select('critic')(obs, goals, q_actions, goals)
            q = jnp.minimum(q1, q2)
            q_loss = -q.mean() / jax.lax.stop_gradient(jnp.abs(q).mean() + 1e-6)
            log_prob = dist.log_prob(actions)
            bc_loss = -(self.config['alpha'] * log_prob).mean()
            actor_loss = q_loss + bc_loss
            return actor_loss, {
                'actor_loss': actor_loss, 'q_loss': q_loss, 'bc_loss': bc_loss,
                'q_mean': q.mean(), 'bc_log_prob': log_prob.mean(),
                'mse': jnp.mean((dist.mode() - actions) ** 2),
            }
        obs, goals, actions = batch['observations'], batch['actor_goals'], batch['actions']
        rng = rng if rng is not None else self.rng
        rng, x0_rng, t_rng, n_rng, z_rng = jax.random.split(rng, 5)

        # 1) flow-matching loss trains actor_vf (the BC flow policy).
        # RG-FB (actor_vf_cond != 'none'): the vf is goal/z-conditioned -- vf_obs
        # carries the condition; identical to obs when actor_vf_cond='none'.
        vf_obs = self._vf_obs(obs, goals, grad_params)
        if task_z is not None and self.config.get('actor_vf_cond', 'none') == 'z':
            # v1: the vf conditions on the sampled task z (actor_vf_cond='z' is enforced
            # there, so this gate is bit-identical for v1). Spec 07's task_prior_actor
            # keeps the FP cell's goal-agnostic vf (actor_vf_cond='none'): the BC flow --
            # and the distillation target it rolls -- sees NO z; the sampled z enters only
            # through the one-shot actor and its q readout.
            vf_obs = jnp.concatenate([obs, jax.lax.stop_gradient(task_z)], axis=-1)
        x_1 = actions
        x_0 = jax.random.normal(x0_rng, actions.shape)
        t = jax.random.uniform(t_rng, (actions.shape[0], 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0
        pred = self.network.select('actor_vf')(vf_obs, x_t, t, params=grad_params)
        bc_flow_loss = jnp.mean((pred - vel) ** 2)

        # 2) Q-max on the one-shot actor's actions, using GCIQL's IQL min(q1,q2).
        # The critic is evaluated at FROZEN params (no params=grad_params): otherwise
        # -q.mean() backprops into the critic and the single optimizer inflates Q
        # instead of improving the actor (see crl-actor-loss-critic-gradient).
        noise = jax.random.normal(n_rng, actions.shape)
        # z-conditioning (G4): the actor conditions on z=stop_grad(psi(goal))
        # (B := value net's psi). stop_grad keeps the representation purely
        # IQL-trained. The critic Q-term keeps the raw goal obs -- it computes
        # its own psi internally.
        z_infer = None
        if task_z is not None:
            # v1: greedy readout on the SAMPLED task vector (reuses the z_infer Q branch below).
            z_infer = jax.lax.stop_gradient(task_z)
            actor_cond_input = z_infer
        elif self.config.get('actor_z_infer', False):
            # exp4: condition on the INFERRED task distribution z=E[r*psi] (sparse-reward
            # readout over the batch), not the exact goal embedding psi(g).
            psi_b = jax.lax.stop_gradient(self._actor_z(goals, grad_params))
            z_infer = self._infer_actor_z(psi_b, z_rng)
            actor_cond_input = z_infer
        elif self.config['actor_cond'] == 'z':
            actor_cond_input = jax.lax.stop_gradient(self._actor_z(goals, grad_params))
        else:
            actor_cond_input = goals
        # actor_z_greedy_frac (campaign v2 knob 2): per-sample, with probability frac,
        # replace the actor's conditioning z by a random sqrt(d)-sphere direction
        # (measure_z_random_frac's z_rand construction) -- those samples' q term is
        # read as FB's NATIVE greedy objective q = F(s,a,z).z/sqrt(d) below. The
        # assert-lattice pins actor_cond='z' + fb_forward_z + flow actor, and z_rng
        # is otherwise consumed only by actor_z_infer (refused), so splitting it here
        # leaves every other stream untouched. frac=0.0 (static) = path unchanged.
        azg = self.config.get('actor_z_greedy_frac', 0.0)
        greedy_pick = None
        if azg > 0:
            gz_rng, gp_rng = jax.random.split(z_rng)
            z_rand = _sphere(jax.random.normal(gz_rng, actor_cond_input.shape),
                             self.config['value_latent_dim'])
            greedy_pick = jax.random.bernoulli(gp_rng, azg, (obs.shape[0], 1))
            actor_cond_input = jnp.where(greedy_pick, z_rand, actor_cond_input)
        actor_actions = self.network.select('actor')(obs, actor_cond_input, noise, params=grad_params)
        if self.config.get('value_arch', 'mlp') == 'dueling':
            # dueling: the actor's q-term is A_bil ALONE (the behavior-relative
            # bilinear advantage), min over the 2-head ensemble -- NOT the full
            # Q=U+A_bil and NOT (Q-V). U(s,g) is a state-goal cardinal the actor
            # cannot move through a; maximizing U would just be chasing a bias term.
            # Frozen critic params (no params=grad_params), same as every other cell.
            out = self.network.select('critic')(obs, goals, actor_actions, full_info=True)
            q1, q2 = out['A_bil']
        elif z_infer is not None:
            # Q_z(s,a)=<phi(s,a), z>/sqrt(d): greedy readout on the inferred task vector.
            _, phi, _ = self.network.select('critic')(obs, goals, actor_actions, info=True)
            q = (phi * z_infer[None]).sum(-1) / jnp.sqrt(self.config['value_latent_dim'])
            q1, q2 = q[0], q[1]
        elif self.config.get('measure_delta_partition', False):
            # spec 17b: q_h = sum_k p_h(s,a,g)_k * phi(goal)_k, delta_head at FROZEN
            # params (no params=grad_params -- crl f7b623d grad-hygiene lesson). z_g is
            # the goal's cell mass, stop-gradded (the partition phi is fixed, not
            # trained through the actor).
            blocks17b = _parse_hit_spec(self.config.get('measure_hit_spec', '19:22:0.4'))
            z_g = jax.lax.stop_gradient(self._delta_phi(goals, self.delta_protos, blocks17b))
            logits_a = self.network.select('delta_head')(
                jnp.concatenate([obs, actor_actions, goals], -1))              # frozen params
            q1, q2 = (jax.nn.softmax(logits_a, -1) * z_g[None]).sum(-1)
        else:
            # fb_forward_z: q = F(s,a,z).B(g)/sqrt(d) with z=sg(psi(g)) -> the FB greedy readout.
            zarg = (goals,) if self.config.get('fb_forward_z', False) else ()
            if greedy_pick is not None:
                # actor_z_greedy_frac: mixed per-sample readout. Greedy-picked rows use
                # their own (random) conditioning z through the z_vec path with FB's
                # native q = F(s,a,z).z/sqrt(d); the rest keep the diagonal F.B(g)
                # readout EXACTLY (same call as the stock branch below). The BC flow /
                # distillation terms stay on the same actor_actions; the q-scale
                # normalization downstream sees the mixed q vector.
                q1d, q2d = self.network.select('critic')(obs, goals, actor_actions, *zarg)
                _, phi_m, _ = self.network.select('critic')(
                    obs, goals, actor_actions, *zarg, info=True, z_vec=actor_cond_input)
                qg = jnp.einsum('hid,id->hi', phi_m, actor_cond_input) \
                    / jnp.sqrt(self.config['value_latent_dim'])
                pick1 = greedy_pick[:, 0]
                q1 = jnp.where(pick1, qg[0], q1d)
                q2 = jnp.where(pick1, qg[1], q2d)
            else:
                q1, q2 = self.network.select('critic')(obs, goals, actor_actions, *zarg)
        q = jnp.minimum(q1, q2) if self.config['double_q'] else q1
        actor_loss = -q.mean()

        # 3) BC-distillation: one-shot actor -> frozen multi-step flow rollout (FQL
        # one-step distillation). Q-loss is normalized by |q| mean to be scale
        # invariant, exactly as in GCIQL's DDPG+BC actor loss.
        distill_error = 0.0
        if self.config['bc_coeff'] > 0:
            # RG-FB: the distillation target rolls the SAME conditioned vf (vf_obs).
            target = self._flow_rollout(vf_obs, noise, jax.lax.stop_gradient(grad_params))
            distill_error = jnp.mean((actor_actions - target) ** 2)
            bc_loss = self.config['bc_coeff'] * distill_error
            # actor_q_center: normalize by the CENTERED magnitude sg[mean|q - mean(q)|] so the
            # q/bc balance is invariant to an affine offset in Q. Cell W's M = 1+(1-gamma)Q puts
            # |q|~1 while the controllable variation is (1-gamma)-scaled, diluting the greedy
            # signal vs bc_loss; -q.mean()'s gradient is unchanged by the centering itself.
            if self.config.get('actor_q_center', False):
                q_scale = jnp.abs(q - q.mean()).mean()
            else:
                q_scale = jnp.abs(q).mean()
            actor_loss = actor_loss / jax.lax.stop_gradient(q_scale + 1e-6) + bc_loss
        actor_loss = actor_loss + bc_flow_loss

        return actor_loss, {
            'actor_loss': actor_loss,
            'bc_flow_loss': bc_flow_loss,
            'distill_loss': distill_error,
            'q_mean': q.mean(),
            'q_abs_mean': jnp.abs(q).mean(),
        }

    def _sample_tasks(self, batch, grad_params, rng):
        """RG-FB v1 (spec06 item4): draw a per-sample task -> (z, z_actor, reward, mask, type).
        Three arms (categorical task_prior_probs):
          0 Dirac    z=psi(value_goal); reward=batch reward; mask=batch mask (the v0 task).
          1 Mixture  K goals from the batch next-states (hindsight), Dirichlet(1) weights w;
                     z=sum_k w_k psi(g_k), renormed to the match-data norm (mean||psi(s')||)
                     with a STOP-GRADDED scale; reward = max_k 1[||cube(s')-cube(g_k)|| <=
                     0.4 obs units (=4cm)] - 1, i.e. gc_negative {-1,0} MATCHING the Dirac
                     arm (a 1-goal mixture IS the Dirac task; a {0,1} convention here would
                     give near-identical z's conflicting targets -- spec06 review finding 2);
                     mask = 1[no hit] (absorbing inside the ball).
          2 Linear   z~ uniform on the sphere * mean||psi|| (constant); reward =
                     sg[psi(s')].z~/sqrt(d) (dense, signed); mask = 1 (no absorbing set).
        GRADIENT DISCIPLINE (spec06 item 4, corrected -- review finding 1): z CARRIES
        psi-gradients for arms 0/1, so the scalar TD trains psi through z exactly as v0's
        L_Q trains psi(g) (fully sg'd z leaves psi at random init = the frozen-psi 0.061
        row). Stop-grad applies ONLY to (i) the actor's copy z_actor, (ii) the linear arm's
        reward features sg[psi(s')] (else the reward is gameable), (iii) the mixture renorm
        SCALE factor, and (iv) TD targets (inside _taskprior_vc_loss). z_actor additionally
        rebuilds the Dirac rows from psi(actor_goals) -- trajectory-future only -- preserving
        v0's value/actor goal split (D8; review finding 4)."""
        obs, nobs = batch['observations'], batch['next_observations']
        vgoals = batch['value_goals']
        d = self.config['value_latent_dim']; sqd = jnp.sqrt(d)
        B = obs.shape[0]
        K = int(self.config['task_prior_mix_k'])
        eps = 0.4  # 4cm in obs units (cube xyz at obs[19:22], 1 obs unit = 10cm)
        rng, t_rng, w_rng, g_rng, zt_rng = jax.random.split(rng, 5)

        psi_g = self._actor_z(vgoals, grad_params)                                 # [B,d], grads ON
        psi_next = self._actor_z(nobs, grad_params)                                # [B,d], grads ON
        mean_norm = jax.lax.stop_gradient(jnp.mean(jnp.linalg.norm(psi_next, axis=-1))) + 1e-8
        cube_next = nobs[:, 19:22]                                                 # [B,3]

        # Dirac arm.
        z_dirac, r_dirac, m_dirac = psi_g, batch['rewards'], batch['masks']

        # Mixture arm.
        idxK = jax.random.randint(g_rng, (B, K), 0, B)                            # goals = batch next-states
        psi_K = psi_next[idxK]                                                    # [B,K,d], grads ON
        cube_K = cube_next[idxK]                                                  # [B,K,3]
        w = jax.random.gamma(w_rng, 1.0, (B, K)); w = w / (w.sum(-1, keepdims=True) + 1e-8)
        z_mix = (w[..., None] * psi_K).sum(1)                                     # [B,d]
        z_mix = z_mix * jax.lax.stop_gradient(
            mean_norm / (jnp.linalg.norm(z_mix, axis=-1, keepdims=True) + 1e-8))  # sg SCALE, live direction
        dmin = jnp.linalg.norm(cube_next[:, None, :] - cube_K, axis=-1)           # [B,K]
        hit_mix = (dmin <= eps).any(-1).astype(obs.dtype)
        r_mix = hit_mix - 1.0                                                     # gc_negative {-1,0}
        m_mix = 1.0 - hit_mix

        # Linear arm.
        zt = jax.random.normal(zt_rng, (B, d))
        zt = zt * mean_norm / (jnp.linalg.norm(zt, axis=-1, keepdims=True) + 1e-8)
        r_lin = (jax.lax.stop_gradient(psi_next) * zt).sum(-1) / sqd              # reward NOT gameable
        z_lin, m_lin = zt, jnp.ones((B,), obs.dtype)

        probs = jnp.asarray(self.config['task_prior_probs'], obs.dtype)
        ttype = jax.random.categorical(t_rng, jnp.log(probs + 1e-12), shape=(B,))  # [B] in {0,1,2}
        d0 = (ttype == 0)[:, None]; d1 = (ttype == 1)[:, None]
        z = jnp.where(d0, z_dirac, jnp.where(d1, z_mix, z_lin))                    # grads ON (arms 0/1)
        r = jnp.where(ttype == 0, r_dirac, jnp.where(ttype == 1, r_mix, r_lin))
        mask = jnp.where(ttype == 0, m_dirac, jnp.where(ttype == 1, m_mix, m_lin))
        # Actor copy: fully sg; Dirac rows from psi(actor_goals) (traj-future, the D8 split).
        z_actor = jnp.where(d0, self._actor_z(batch['actor_goals'], grad_params), z)
        return z, jax.lax.stop_gradient(z_actor), r, mask, ttype

    def _taskprior_vc_loss(self, batch, grad_params, z, r, mask, ttype):
        """RG-FB v1 value + critic TD evaluated AT THE SAMPLED z (a raw [B,d] task vector):
        V(s,z)=phi_V(s).z/sqrt(d), Q(s,a,z)=phi(s,a).z/sqrt(d). The ONLINE predictions use
        the live z (psi-gradients flow for the Dirac/mixture arms -- v0's L_Q trains both
        legs); the TD TARGETS use sg[z] (v0's targets are fully stop-gradded; a live-z
        target would let the loss descend the target into the prediction). Each sample's
        TD loss is normalized by the stop-gradded mean |target| of its OWN task type, so
        the linear arm's dense rewards cannot dominate the sparse (Dirac/mixture) arms."""
        obs, nobs, acts = batch['observations'], batch['next_observations'], batch['actions']
        d = self.config['value_latent_dim']; sqd = jnp.sqrt(d); gamma = self.config['discount']
        z_t = jax.lax.stop_gradient(z)                                                        # target-side z

        _, phi_v, _ = self.network.select('value')(obs, obs, info=True, params=grad_params)  # [B,d]
        v = (phi_v * z).sum(-1) / sqd
        _, tphi, _ = self.network.select('target_critic')(obs, obs, acts, info=True)         # [2,B,d]
        q_tc = jnp.minimum((tphi[0] * z_t).sum(-1), (tphi[1] * z_t).sum(-1)) / sqd
        value_loss_i = self.expectile_loss(q_tc - v, q_tc - v, self.config['expectile'])      # [B]

        _, phi_vn, _ = self.network.select('value')(nobs, nobs, info=True)                    # frozen V(s')
        next_v = (phi_vn * z_t).sum(-1) / sqd
        q_target = r + gamma * mask * next_v
        _, phi, _ = self.network.select('critic')(obs, obs, acts, info=True, params=grad_params)  # [2,B,d]
        q1 = (phi[0] * z).sum(-1) / sqd; q2 = (phi[1] * z).sum(-1) / sqd
        critic_loss_i = (q1 - q_target) ** 2 + (q2 - q_target) ** 2                           # [B]

        # per-task-type scale = sg[ mean_{i in type} |target| ]; divide each row by its type's scale.
        absT = jnp.abs(q_target)
        def type_scale(t):
            msk = (ttype == t).astype(absT.dtype)
            return jax.lax.stop_gradient((absT * msk).sum() / (msk.sum() + 1e-6)) + 1e-6
        scales = jnp.stack([type_scale(0), type_scale(1), type_scale(2)])                     # [3]
        scale_i = scales[ttype]                                                               # [B]
        critic_loss = jnp.mean(critic_loss_i / scale_i)
        value_loss = jnp.mean(value_loss_i / scale_i)

        def per_type(x):  # mean of x over each type (raw diagnostic)
            return jnp.stack([jnp.sum(x * (ttype == t)) / (jnp.sum(ttype == t) + 1e-6) for t in range(3)])
        raw_ct = per_type(critic_loss_i)
        norm_ct = per_type(critic_loss_i / scale_i)
        info = {
            'taskprior_value_loss': value_loss, 'taskprior_critic_loss': critic_loss,
            'taskprior_scale_dirac': scales[0], 'taskprior_scale_mix': scales[1],
            'taskprior_scale_lin': scales[2],
            'taskprior_raw_ct_dirac': raw_ct[0], 'taskprior_raw_ct_lin': raw_ct[2],
            'taskprior_norm_ct_dirac': norm_ct[0], 'taskprior_norm_ct_lin': norm_ct[2],
            'taskprior_frac_mix': jnp.mean((ttype == 1).astype(jnp.float32)),
            'taskprior_r_mix_pos': jnp.mean(jnp.where(ttype == 1, r, 0.0)),
        }
        return value_loss + critic_loss, info

    def _sample_actor_tasks(self, batch, grad_params, rng):
        """Spec 07 item 1 (task_prior_actor='mix'): per-sample z-prior for the ACTOR ONLY,
        in the FP corner. Three arms (categorical task_prior_actor_probs), all purely a
        z-distribution -- no reward, no mask, none of _sample_tasks's vc-loss machinery:
          0 Dirac    z = psi(actor_goals) -- the existing actor-z path (D8 trajectory-future
                     split preserved; psi = the measure-trained critic B, head-mean).
          1 Mixture  K goals from the batch next-states, w ~ Dirichlet(1);
                     z = sum_k w_k psi(g_k), renormed to the match-data norm mean||psi(s')||.
          2 Random   z uniform on the sphere * mean||psi(s')||.
        GRADIENT DISCIPLINE (the decisive simplification vs _sample_tasks): the actor z is
        FULLY stop-gradded in EVERY arm -- psi trains ONLY through the FP matrix loss,
        exactly as in the winning FP-NoOrtho cell (no vc loss needs live psi-grads here).
        Returns (z sg [B,d], ttype [B], z_mix sg [B,d], cube_K [B,K,3]): z_mix/cube_K are
        the FULL-batch mixture arm (computed for every row before the arm select), consumed
        by the optional task_prior_critic_mix grounding (spec 07 item 4)."""
        nobs = batch['next_observations']
        d = self.config['value_latent_dim']
        B = nobs.shape[0]
        K = int(self.config['task_prior_actor_mix_k'])
        t_rng, g_rng, w_rng, zt_rng = jax.random.split(rng, 4)

        psi_next = self._actor_z(nobs, grad_params)                               # [B,d] critic psi (B)
        mean_norm = jnp.mean(jnp.linalg.norm(psi_next, axis=-1)) + 1e-8           # match-data norm

        # Dirac arm: the existing actor-z path (psi of the ACTOR goal -- the D8 split).
        z_dirac = self._actor_z(batch['actor_goals'], grad_params)                # [B,d]

        # Mixture arm (the _sample_tasks arm-1 z recipe; the final sg covers the scale too).
        idxK = jax.random.randint(g_rng, (B, K), 0, B)                            # goals = batch next-states
        psi_K = psi_next[idxK]                                                    # [B,K,d]
        cube_K = nobs[:, 19:22][idxK]                                             # [B,K,3] mixture-goal cubes
        w = jax.random.gamma(w_rng, 1.0, (B, K)); w = w / (w.sum(-1, keepdims=True) + 1e-8)
        z_mix = (w[..., None] * psi_K).sum(1)                                     # [B,d]
        z_mix = z_mix * mean_norm / (jnp.linalg.norm(z_mix, axis=-1, keepdims=True) + 1e-8)

        # Random arm: uniform sphere direction * match-data norm.
        zt = jax.random.normal(zt_rng, (B, d))
        zt = zt * mean_norm / (jnp.linalg.norm(zt, axis=-1, keepdims=True) + 1e-8)

        probs = jnp.asarray(self.config['task_prior_actor_probs'], nobs.dtype)
        ttype = jax.random.categorical(t_rng, jnp.log(probs + 1e-12), shape=(B,))  # [B] in {0,1,2}
        d0 = (ttype == 0)[:, None]; d1 = (ttype == 1)[:, None]
        z = jnp.where(d0, z_dirac, jnp.where(d1, z_mix, zt))
        return (jax.lax.stop_gradient(z), ttype,
                jax.lax.stop_gradient(z_mix), cube_K)

    def _tpactor_critic_mix_loss(self, batch, grad_params, z_mix, cube_K, rng):
        """Spec 07 item 4 (task_prior_critic_mix) -- the linearity-conflict probe. ADD a
        scalar first-passage TD grounding at the actor's mixture z ON TOP of the untouched
        matrix loss:
            r = max_k 1[||cube(s') - cube(g_k)|| <= 0.4]      ([0,1], NOT gc_negative --
                                                               matching the FP cell)
            y = r + gamma*(1-r)*sg[min_h Q_tgt(s', a'~pi(s',z_mix), z_mix)]   (absorbing)
            Q(s,a,z) = phi(s,a).z/sqrt(d)  through the target critic for y, the online
            critic (grad_params) for the prediction.
        z_mix arrives FULLY sg from _sample_actor_tasks (psi trains only through the FP
        matrix loss), so this loss trains phi alone. Pre-registered prediction (spec 07
        item 0 fact 1): this HURTS the Dirac anchor -- E[gamma^{T_{A u B}}] is not any
        fixed linear combination of the per-goal columns, so grounding at z_mix fights the
        Dirac columns through the z-linearity."""
        obs, nobs, acts = batch['observations'], batch['next_observations'], batch['actions']
        d = self.config['value_latent_dim']; sqd = jnp.sqrt(d); gamma = self.config['discount']
        eps = 0.4  # 4cm in obs units (cube xyz at obs[19:22], 1 obs unit = 10cm)
        dist = jnp.linalg.norm(nobs[:, 19:22][:, None, :] - cube_K, axis=-1)      # [B,K]
        r = (dist <= eps).any(-1).astype(obs.dtype)                               # [B] in {0,1}
        # a' ~ pi(s', z_mix): the one-shot z-actor (actor_cond='z' is forced by the lattice).
        noise = jax.random.normal(rng, acts.shape)
        a_next = jax.lax.stop_gradient(self.network.select('actor')(nobs, z_mix, noise))
        _, tphi, _ = self.network.select('target_critic')(nobs, nobs, a_next, info=True)  # [2,B,d]
        q_t = jnp.min((tphi * z_mix[None]).sum(-1), axis=0) / sqd                 # min over heads [B]
        y = jax.lax.stop_gradient(r + gamma * (1.0 - r) * q_t)
        _, phi, _ = self.network.select('critic')(obs, obs, acts, info=True,
                                                  params=grad_params)             # [2,B,d]
        q1 = (phi[0] * z_mix).sum(-1) / sqd
        q2 = (phi[1] * z_mix).sum(-1) / sqd
        loss = jnp.mean((q1 - y) ** 2 + (q2 - y) ** 2)
        return loss, {'criticmix_loss': loss, 'criticmix_r_mean': r.mean(),
                      'criticmix_q_mean': 0.5 * (q1 + q2).mean(),
                      'criticmix_target_mean': y.mean()}

    @staticmethod
    def _hit_matrix(row_obs, col_obs, blocks, dtype, eps_scale=None):
        """[R,C] goal-region indicator: AND over blocks of ||row[a:b] - col[a:b]|| <= eps.

        `blocks` comes from _parse_hit_spec. With the default single cube-single block
        this is exactly the old `(norm(nobs[:,19:22] - col[:,19:22]) <= 0.4)` expression,
        so cube-single training is bit-identical.

        eps_scale (measure_eps_multiscale): optional [C] per-COLUMN radius multiplier
        eps_j/0.4 -- every block's radius scales proportionally, broadcast as
        (dist <= e * eps_scale[None, :]). None (default) = the scalar path, unchanged."""
        hit = None
        for (a, b, e) in blocks:
            dist = jnp.linalg.norm(row_obs[:, None, a:b] - col_obs[None, :, a:b], axis=-1)
            thr = e if eps_scale is None else e * eps_scale[None, :]
            h = (dist <= thr).astype(dtype)
            hit = h if hit is None else hit * h
        return hit

    @staticmethod
    def _delta_phi(obs_rows, protos, blocks):
        """phi(x) in Delta^K (spec 17b): softmax over K of -0.5 * sum_blocks
        ||(x_blk - proto_blk)/eps_blk||^2. obs_rows [B, obs_dim]; protos [K, d_c] raw
        obs units, columns = the blocks concatenated in _parse_hit_spec order (accepts
        an array-like OR the agent's stored hashable nested-tuple form -- delta_protos
        is a static/nonpytree field, so this cast is what turns it back into an array
        on every call; inside jit it folds to a constant); per-block eps scaling makes
        heterogeneous tolerances (scene) one knob.

        Implementation: slice each block from obs_rows, divide by that block's eps,
        concat -> xc [B, d_c]; scale protos columns the same way (block column
        offsets = cumulative widths) -> pc [K, d_c]; logits =
        -0.5*||xc[:,None]-pc[None]||^2 summed over the last axis; softmax(-1) -> [B, K]."""
        protos = jnp.asarray(protos, jnp.float32)
        parts_x, parts_p, off = [], [], 0
        for (a, b, e) in blocks:
            w = b - a
            parts_x.append(obs_rows[:, a:b] / e)
            parts_p.append(protos[:, off:off + w] / e)
            off += w
        xc = jnp.concatenate(parts_x, axis=-1)          # [B, d_c]
        pc = jnp.concatenate(parts_p, axis=-1)           # [K, d_c]
        diffs = xc[:, None, :] - pc[None, :, :]           # [B, K, d_c]
        logits = -0.5 * jnp.sum(diffs ** 2, axis=-1)      # [B, K]
        return jax.nn.softmax(logits, axis=-1)

    @staticmethod
    def _measure_terms(M, tM, gamma, scale_norm, eps=1e-6, hit=None, absorbing=True):
        """Reduce the online measure M and stop-gradded target tM into the FB (offdiag, diag)
        TD terms over the last two (BxB) axes (leading head axes preserved):
            diff = M - gamma*tM;  offdiag = 0.5*sum((diff*off)^2)/sum(off);  diag = -mean(diag(diff))
        With measure_scale_norm, divide by a stop-gradded measure scale so the gradient is
        DIMENSIONLESS -- the same trick the actor q_loss uses (q/sg[|q|]) -- which stops the
        unbounded <F,z> inflation from blowing up the shared critic (audit-20260706 §1):
            offdiag /= sg[mean(M^2)+eps];  diag /= sg[sqrt(mean(M^2))+eps].
        Ortho term is handled by the caller (already scale-bounded by psi).

        measure_first_passage (spec 06 item 5): when `hit` (a [B,B] 0/1 arrival indicator
        1[||cube(s'_i) - cube(s+_j)|| <= eps]) is supplied, the residual becomes the
        FIRST-PASSAGE (absorbing) successor-measure residual
            diff = M - hit - gamma*(1 - hit)*sg[tM]
        so a hit terminates the recursion (no bootstrap) and the per-column fixed point is
        bounded in [0,1]. Because `hit` is a REPRESENTABLE 0/1 reward (unlike the Dirac
        density), the diagonal is a plain TD residual and is REGRESSED (squared) -- keeping
        FB's linear diagonal attraction here would reintroduce the unbounded diagonal pull
        (T10-b) and the fixed point would stop being E[gamma^T_hit] (spec06 review finding 3).
        hit=None => bit-identical stock behaviour."""
        B = M.shape[-1]
        off = 1.0 - jnp.eye(B)
        if hit is None:
            diff = M - gamma * tM
            diagt = -jnp.mean(jnp.diagonal(diff, axis1=-2, axis2=-1), axis=-1)
        else:
            # absorbing=True: first-passage (hit terminates the recursion; fixed point
            # E[gamma^T_hit] in [0,1]). absorbing=False (2026-07-23): discounted BALL-
            # VISITATION measure M = E[sum_t gamma^t 1[s_{t+1} in ball(s+)]] -- genuine
            # occupancy of the goal ball (revisits counted), still a plain regressed
            # residual against a representable 0/1 reward => gamma-contraction, fixed
            # point bounded by 1/(1-gamma). No linear diagonal attraction anywhere --
            # the FB occupancy divergence channel (q_abs 154->5e4 on frozen psi) is
            # structurally absent.
            cont = (1.0 - hit) if absorbing else 1.0
            diff = M - hit - gamma * cont * tM
            diagt = jnp.mean(jnp.diagonal(diff, axis1=-2, axis2=-1) ** 2, axis=-1)
        offdiag = 0.5 * jnp.sum((diff * off) ** 2, axis=(-2, -1)) / jnp.sum(off)
        if scale_norm:
            msq = jax.lax.stop_gradient(jnp.mean(M ** 2))
            offdiag = offdiag / (msq + eps)
            # linear diag scales like M (sqrt); the squared first-passage diag scales like M^2.
            diagt = diagt / ((jnp.sqrt(msq) if hit is None else msq) + eps)
        return offdiag, diagt

    @staticmethod
    def _nstep_window_terms(hits_w, umask, gamma, absorbing):
        """Spec 16 (measure_nstep): reduce the per-step window hit matrices into the
        k-step OBSERVED mass O and the not-yet-hit continuation cont.

        hits_w [K,B,B]: hit(s_{t+u}, col_j) at window step u=1..K (index u-1);
        umask [K,B]: 1[u <= k_i]. Steps past the realized horizon are zeroed BEFORE
        the cumprod, so they neither add mass nor touch the continuation product.
            absorbing:  O = sum_u gamma^{u-1} h_u prod_{v<u}(1-h_v);  cont = prod_{u<=k_i}(1-h_u)
            visitation: O = sum_u gamma^{u-1} h_u;                    cont = 1
        k_i=1 reduces to (hit, 1-hit) / (hit, 1) -- the one-step targets."""
        K = hits_w.shape[0]
        hw = hits_w * umask[:, :, None]
        gpow = (gamma ** jnp.arange(K, dtype=hits_w.dtype))[:, None, None]
        if absorbing:
            notyet = jnp.cumprod(1.0 - hw, axis=0)                                # prod_{v<=u}
            before = jnp.concatenate([jnp.ones_like(hw[:1]), notyet[:-1]], axis=0)
            O = jnp.sum(gpow * hw * before, axis=0)
            cont = notyet[-1]
        else:
            O = jnp.sum(gpow * hw, axis=0)
            cont = jnp.ones_like(O)
        return O, cont

    @staticmethod
    def _measure_terms_nstep(M, tM, O, cont, boot_w, scale_norm, eps=1e-6):
        """Spec 16 k-step region residual: diff = M - O - gamma^{k_i} * cont * sg[tM],
        per-ROW bootstrap weight boot_w [B,1] (the realized horizon k_i varies by
        sample). Same squared-diagonal + offdiag reduction and scale_norm semantics as
        _measure_terms' hit path (the observed window is a representable reward)."""
        B = M.shape[-1]
        off = 1.0 - jnp.eye(B)
        diff = M - O - boot_w * cont * tM
        diagt = jnp.mean(jnp.diagonal(diff, axis1=-2, axis2=-1) ** 2, axis=-1)
        offdiag = 0.5 * jnp.sum((diff * off) ** 2, axis=(-2, -1)) / jnp.sum(off)
        if scale_norm:
            msq = jax.lax.stop_gradient(jnp.mean(M ** 2))
            offdiag = offdiag / (msq + eps)
            diagt = diagt / (msq + eps)
        return offdiag, diagt

    def measure_loss(self, batch, grad_params, rng):
        """FB successor-measure loss grafted onto GCIQL's bilinear features. FB's update_fb
        split: off-diagonal TD over random goals + diagonal attraction to the observed
        successor. measure_coef=0 reproduces stock GCIQL exactly.

        measure_legs='critic' (default, FB-FAITHFUL): the state-ACTION measure on the CRITIC
        legs -- M(s,a,s')=<phi(s,a),psi(s')>/sqrt(d) (phi=F, psi=B), bootstrap a+ ~ pi(s+,g)
        (stop-grad) via the target_critic. Per-head loss then averaged. Shares phi/psi with the
        IQL critic the actor maximizes. 'value': the state-only measure on the value legs.

        Two default-OFF scale-control knobs (audit-20260706 §1 / E2 spec):
          measure_scale_norm  -> dimensionless TD terms (see _measure_terms).
          measure_stopgrad_phi-> stop-grad phi so the measure gradient shapes only psi (the
                                 readout/backward leg), quarantined from the forward leg."""
        obs, nobs = batch['observations'], batch['next_observations']
        d = self.config['value_latent_dim']; sqd = jnp.sqrt(d); gamma = self.config['discount']
        B = obs.shape[0]; off = 1.0 - jnp.eye(B)
        scale_norm = self.config.get('measure_scale_norm', False)
        sg_phi = self.config.get('measure_stopgrad_phi', False)
        # phase-2 decomposition: stop-grad EVERY online psi column so the measure TD trains
        # only phi against a frozen backward map (warm_start_ckpt supplies the trained psi).
        # The ortho term also sees sg'd psi, consistent with "psi is frozen".
        sg_psi = self.config.get('measure_stopgrad_psi', False)
        qz_mean = None   # set only by the rank-1 ('task') test set, in the critic branch
        nhat_loss = nhat_pred_mean = None   # set only by measure_nhat_head (phase 2b)
        nstep_k_mean = None   # set only by measure_nstep (spec 16)
        delta_event_rate = delta_hit_level = delta_msq = None   # set only by measure_delta (spec 17a)
        if self.config.get('measure_legs', 'critic') == 'value':
            _, phi, psi = self.network.select('value')(obs, nobs, info=True, params=grad_params)
            if sg_phi: phi = jax.lax.stop_gradient(phi)
            if sg_psi: psi = jax.lax.stop_gradient(psi)
            M = (phi @ psi.T) / sqd
            _, tphi, tpsi = self.network.select('value')(nobs, nobs, info=True, params=grad_params)
            tM = jax.lax.stop_gradient((tphi @ tpsi.T) / sqd)
            offdiag, diagt = self._measure_terms(M, tM, gamma, scale_norm)
            Cov = psi @ psi.T                                            # FB ortho on B:=psi (grads on)
            orth = 0.5 * jnp.sum((Cov * off) ** 2) / jnp.sum(off) - jnp.mean(jnp.diag(Cov))
            loss = offdiag + diagt + self.config.get('measure_ortho_coef', 0.0) * orth
        else:  # critic legs: state-action successor measure (the FB-faithful intervention)
            acts, vgoals = batch['actions'], batch['value_goals']
            # fb_forward_z: forward map is task-conditioned phi(s,a,z), z=sg(psi(vgoals)),
            # so M[i,j]=F(s_i,a_i,z_i).B(s'_j) is a genuine z-indexed family. Else phi(s,a).
            zarg = (vgoals,) if self.config.get('fb_forward_z', False) else ()
            # measure_z_random_frac: FB's mixed z-sampling (metamotivo sample_mixed_z) --
            # a per-sample fraction of the task vectors is drawn as sqrt(d)-sphere gaussian
            # DIRECTIONS instead of psi(vgoals), so the z-indexed family is trained across
            # z-space (where inference-time moment-z lives), not only on the psi(goal) manifold.
            # Same z on both TD sides (FB backup is at fixed z). frac=0 => bit-exact stock path.
            zfrac = self.config.get('measure_z_random_frac', 0.0)   # static config MAX
            zkw = {}
            z_mix = None
            if zarg and zfrac > 0:
                rng, zrng, prng = jax.random.split(rng, 3)
                # measure_z_random_frac_anneal (spec 08 E6): linearly ramp the random-z
                # PROBABILITY 0 -> zfrac over anneal_ramp steps, then hold. Only the bernoulli
                # probability is annealed (a traced scalar from network.step); the branch stays
                # static on the config max so jit shapes and the knob-off path are unchanged.
                if self.config.get('measure_z_random_frac_anneal', False):
                    t = jnp.asarray(self.network.step, jnp.float32)
                    ramp = float(self.config.get('anneal_ramp', 300_000))
                    zprob = zfrac * jnp.clip(t / ramp, 0.0, 1.0)
                else:
                    zprob = zfrac
                _, _, psi_vg = self.network.select('critic')(vgoals, vgoals, acts, vgoals,
                                                             info=True, params=grad_params)
                z_goal = jax.lax.stop_gradient(psi_vg.mean(axis=0))       # head-mean [B,d]
                if self.config.get('project_z', False):
                    z_goal = _sphere(z_goal, d)
                z_rand = _sphere(jax.random.normal(zrng, (B, d)), d)
                pick = jax.random.bernoulli(prng, zprob, (B, 1))
                z_mix = jnp.where(pick, z_rand, z_goal)
                zkw = {'z_vec': z_mix}
            _, phi, psi = self.network.select('critic')(obs, nobs, acts, *zarg, info=True,
                                                        params=grad_params, **zkw)
            if sg_phi: phi = jax.lax.stop_gradient(phi)
            if sg_psi: psi = jax.lax.stop_gradient(psi)
            M = jnp.einsum('hid,hjd->hij', phi, psi) / sqd                # [2, B, B]
            # a+ ~ pi(s+,g); under actor_cond='z' the actor's goal-leg is the d-dim task
            # vector, so map the raw vgoals through _actor_z (critic/value psi) first.
            # With the mixed z active, the bootstrap policy must be z-CONSISTENT: a+ ~ pi(s+, z_i)
            # at the SAME z the TD is evaluated at (FB: the measure of the z-greedy policy).
            if self.config['actor_cond'] == 'z':
                a_goals = z_mix if z_mix is not None else self._actor_z(vgoals, grad_params)
            else:
                a_goals = vgoals
            if self.config['actor_kind'] == 'ddpgbc':
                # stock gaussian actor returns a distribution: a+ = mode (or a sample if learned std)
                dist = self.network.select('actor')(nobs, a_goals)
                a_next = dist.mode() if self.config['const_std'] else dist.sample(seed=rng)
                a_next = jnp.clip(a_next, -1, 1)
            else:
                noise = jax.random.normal(rng, acts.shape)
                a_next = self.network.select('actor')(nobs, a_goals, noise)
            a_next = jax.lax.stop_gradient(a_next)
            _, tphi, tpsi = self.network.select('target_critic')(nobs, nobs, a_next, *zarg, info=True, **zkw)
            tM = jax.lax.stop_gradient(jnp.einsum('hid,hjd->hij', tphi, tpsi) / sqd)            # [2, B, B]
            test_states = self.config.get('measure_test_states', 'rho')
            test_rank = self.config.get('measure_test_rank', 'full')
            fp = self.config.get('measure_first_passage', False)
            exm = self.config.get('measure_expectile', False)
            tpd = self.config.get('measure_testpoint_dist', 'batch')
            if fp or exm or tpd != 'batch':
                # Spec 06 single-matrix measure axes (asserts guarantee rho/full + bilinear).
                # item7: swap the test-point columns to psi(value_goals) (the OGBench hindsight
                # mix). NO importance correction -- the object is no longer the rho-referenced
                # measure (intentional; documented in the run notes).
                if tpd == 'hindsight':
                    _, _, psi = self.network.select('critic')(obs, vgoals, acts, *zarg, info=True,
                                                              params=grad_params, **zkw)
                    if sg_psi: psi = jax.lax.stop_gradient(psi)
                    _, _, tpsi = self.network.select('target_critic')(nobs, vgoals, a_next, *zarg,
                                                                      info=True, **zkw)
                    col_states = vgoals
                    M = jnp.einsum('hid,hjd->hij', phi, psi) / sqd
                    tM = jax.lax.stop_gradient(jnp.einsum('hid,hjd->hij', tphi, tpsi) / sqd)
                else:
                    col_states = nobs
                # item5 first-passage hit: 1[s'_i lands in the goal region of s+_j], the region
                # being the conjunction of the measure_hit_spec blocks. Default '19:22:0.4' =
                # cube-single's cube xyz within 4cm (xyz_scaler=10 => 1 obs unit = 10cm, verified
                # vs ogbench cube_env.compute_observation); spec 08 E7 ports it to cube-double
                # (two cubes) and scene (cube + buttons + drawer + window) by conjunction.
                if fp:
                    blocks = _parse_hit_spec(self.config.get('measure_hit_spec', '19:22:0.4'))
                    ms_spec = self.config.get('measure_eps_multiscale', '')
                    if ms_spec:
                        # measure_eps_multiscale (campaign v2 knob 1): per-COLUMN hit
                        # radius drawn from the level set each gradient step, plus a
                        # learned per-level psi gain (init ones -> M starts at the base).
                        # The assert-lattice pins this to the plain critic-legs
                        # first-passage path (tpd='batch', no expectile, no task_prior),
                        # so the M/tM recompute below overrides the canonical single-eps
                        # matrices computed above. rng was consumed raw by the a_next
                        # sampling, so a fold_in derives a fresh independent stream
                        # without touching any existing consumer.
                        levels = _parse_eps_levels(ms_spec)
                        lrng = jax.random.fold_in(rng, 17)
                        lvl = jax.random.randint(lrng, (B,), 0, len(levels))            # [B]
                        eps_scale = jnp.asarray(levels, M.dtype)[lvl] / 0.4             # [B]
                        hit = self._hit_matrix(nobs, col_states, blocks, M.dtype,      # [B,B]
                                               eps_scale=eps_scale)
                        gain = self.network.select('eps_gain')(lvl, params=grad_params)  # [B,d]
                        # per-column gained psi on BOTH TD sides: shared table,
                        # stop-gradded on the target side (the target must track the
                        # online readout's scale, not train it twice).
                        M = jnp.einsum('hid,hjd->hij', phi, psi * gain[None]) / sqd
                        tM = jax.lax.stop_gradient(
                            jnp.einsum('hid,hjd->hij', tphi,
                                       tpsi * jax.lax.stop_gradient(gain)[None]) / sqd)
                    else:
                        hit = self._hit_matrix(nobs, col_states, blocks, M.dtype)       # [B,B]
                    if self.config.get('measure_delta', False):
                        # spec 17a: signed event target hit(x')-hit(x). hit_n stays for the info
                        # keys; the residual sees only the temporal difference, so the dwell
                        # plateau (inside->inside) contributes zero and the level channel is
                        # deleted from the supervision (ledger 9.16 / 9.15c motivation).
                        hit_n = hit
                        hit_c = self._hit_matrix(obs, col_states, blocks, M.dtype)      # [B,B]
                        hit = hit_n - hit_c
                        delta_event_rate = jnp.mean(jnp.abs(hit))
                        delta_hit_level = jnp.mean(hit_n)
                        # gate G2 instrument (spec 17 §6): "no zero-flux collapse (msq
                        # bounded away from 0)" has no readout otherwise -- the scale-normed
                        # loss terms are insensitive to msq by construction (they divide it
                        # out), so it must be exposed separately. 17a-gated only.
                        delta_msq = jnp.mean(M ** 2)
                    if self.config.get('measure_nhat_head', False):
                        # measure_nhat_head (phase 2b): regress the trained kernel-mass
                        # denominator N-hat(g_j) onto the hit-matrix COLUMN MEAN
                        # mean_i h_ij -- an unbiased MC estimate of rho(B_eps(g_j))
                        # (noisy at B=1024 per step; the regression averages over
                        # steps). Plain regression, NO TD; both the psi input and the
                        # target are stop-gradded, so the gradient reaches ONLY the
                        # head and every existing loss term is bit-identical.
                        psi_sg = jax.lax.stop_gradient(
                            psi.mean(axis=0) if psi.ndim == 3 else psi)                 # [B,d]
                        nhat_pred = self.network.select('nhat_head')(
                            psi_sg, params=grad_params)                                 # [B]
                        nhat_tgt = jax.lax.stop_gradient(hit.mean(axis=0))              # [B]
                        nhat_loss = jnp.mean((nhat_pred - nhat_tgt) ** 2)
                        nhat_pred_mean = nhat_pred.mean()
                else:
                    hit = None
                if exm:
                    # item6 expectile-measure: state-only value V_M(s,s+)=phi_V(s).psi~(s+)/sqrt(d),
                    # per-column expectile fit to the online action-measure, and an IN-SAMPLE backup
                    # (bootstrap on V_M(s'), NO a'). Unmasked (no first-passage): FB sphere anchor on
                    # psi (the ortho anchor is enforced by the assert-lattice).
                    psi_col = psi if fp else _sphere(psi, d)
                    M_act = jnp.einsum('hid,hjd->hij', phi, psi_col) / sqd                      # [2,B,B]
                    phiV = self.network.select('measure_v_phi')(obs, params=grad_params)        # [2,B,d]
                    VM = jnp.einsum('hid,hjd->hij', phiV, psi_col) / sqd                        # [2,B,B]
                    tau = self.config.get('measure_expectile_tau', 0.9)
                    if self.config.get('measure_expectile_source', 'online') == 'target_min':
                        # Cell W: the anchor's double-DQN split -- expectile source = min over
                        # TARGET-critic heads at (s, a_data), exactly GC-IQL's L_V input.
                        _, ttphi, _ = self.network.select('target_critic')(obs, col_states, acts,
                                                                           info=True)
                        tpsi_col = tpsi if fp else _sphere(tpsi, d)
                        M_src = jnp.min(jnp.einsum('hid,hjd->hij', ttphi, tpsi_col), axis=0)[None] / sqd
                        resV = jax.lax.stop_gradient(M_src) - VM
                    else:
                        resV = jax.lax.stop_gradient(M_act) - VM
                    ellV = self.expectile_loss(resV, resV, tau)                                  # [.,B,B]
                    phiV_next = self.network.select('measure_v_phi')(nobs)                      # online net at s'
                    VM_next = jax.lax.stop_gradient(
                        jnp.einsum('hid,hjd->hij', phiV_next, jax.lax.stop_gradient(psi_col)) / sqd)
                    if fp:
                        r_delta = hit[None]; mask_m = (1.0 - hit)[None]                         # absorbing
                    else:
                        r_delta = jnp.eye(B)[None]; mask_m = jnp.ones((1, B, B))                # visitation
                    targM = r_delta + gamma * mask_m * VM_next
                    resM = M_act - jax.lax.stop_gradient(targM)
                    # Cell W off-diagonal weighting: the diagonal entries are the PAIRED
                    # hindsight columns (with tpd='hindsight' the diagonal-only limit w=0 is
                    # EXACTLY bilinear GC-IQL under M = 1 + (1-gamma)Q, since
                    # delta_M = (1-gamma)*delta_Q identically); the off-diagonal block adds
                    # (B-1) UNPAIRED columns per row, so at w=1 the per-row task weighting is
                    # ~99.6% unpaired-random -- axis A re-entering through the estimator.
                    # w ~ 0.3/(B-1)-ish restores a GC-IQL-like effective mix.
                    w_off = self.config.get('measure_offdiag_weight', 1.0)
                    if w_off == 1.0:
                        Lv = ellV.mean()
                        Lm = jnp.mean(resM ** 2)
                    else:
                        wmat = jnp.eye(B) + w_off * (1.0 - jnp.eye(B))                          # [B,B]
                        wsum = wmat.sum()
                        Lv = (ellV * wmat[None]).sum() / (ellV.shape[0] * wsum)
                        Lm = ((resM ** 2) * wmat[None]).sum() / (resM.shape[0] * wsum)
                    if self.config.get('measure_scale_norm', False):
                        # This branch bypasses _measure_terms, so measure_scale_norm was
                        # silently a no-op here; with no ortho anchor nothing pins the
                        # bilinear scale and the unnormalized TD runs away (q_abs 1e4-1e5,
                        # 2/3 seeds). Same dimensionless-gradient trick as _measure_terms.
                        sM = jax.lax.stop_gradient(jnp.mean(M_act ** 2)) + 1e-6
                        Lv, Lm = Lv / sM, Lm / sM
                    offdiag, diagt = Lm, Lv          # reuse the ortho + loss assembly below
                    psi = psi_col                    # ortho (Cov below) bounds the leg V_M/M use
                elif tpd == 'hindsight' and hit is None:
                    # item7 (review finding 5): the off-diagonal residual moves to the hindsight
                    # columns, but the DIAGONAL (positive/attraction) term must come from
                    # TRAJECTORY-FUTURE states only -- actor_goals (uniform future on the same
                    # trajectory). value_goals include 30% random states, which are samples
                    # from NO policy's future; attracting mass to them corrupts the object.
                    # (With first-passage, hit itself supplies the positives and every entry is
                    # a plain residual, so this special case applies only to hit=None.)
                    offdiag, _ = self._measure_terms(M, tM, gamma, scale_norm)                  # [2]
                    agoals = batch['actor_goals']
                    _, _, psi_dg = self.network.select('critic')(obs, agoals, acts, *zarg, info=True,
                                                                 params=grad_params, **zkw)
                    if sg_psi: psi_dg = jax.lax.stop_gradient(psi_dg)
                    _, _, tpsi_dg = self.network.select('target_critic')(nobs, agoals, a_next, *zarg,
                                                                         info=True, **zkw)
                    Md = jnp.einsum('hid,hid->hi', phi, psi_dg) / sqd                           # [2,B]
                    tMd = jax.lax.stop_gradient(jnp.einsum('hid,hid->hi', tphi, tpsi_dg) / sqd)
                    diagt = -jnp.mean(Md - gamma * tMd, axis=-1)                                # [2]
                    if scale_norm:
                        msq = jax.lax.stop_gradient(jnp.mean(M ** 2))
                        diagt = diagt / (jnp.sqrt(msq) + 1e-6)
                elif fp and int(self.config.get('measure_nstep', 0) or 0) > 0:
                    # Spec 16 k-step region target (PAPER/Multistep-FB): observed window
                    # mass + gamma^{k_i} bootstrap at s_{t+k_i}. The bootstrap noise is
                    # derived from the SAME rng as the one-step a_next above, so K=1 with
                    # a window built from nobs is bit-identical to the one-step path.
                    win = batch['mstep_next_obs']                                  # [B,K,obs]
                    ks = batch['mstep_k'].astype(jnp.int32)                        # [B]
                    K = win.shape[1]
                    hits_w = jax.vmap(
                        lambda w: self._hit_matrix(w, col_states, blocks, M.dtype),
                        in_axes=1, out_axes=0)(win)                                # [K,B,B]
                    umask = (jnp.arange(K)[:, None] < ks[None, :]).astype(M.dtype)  # [K,B]
                    O, cont = self._nstep_window_terms(
                        hits_w, umask, gamma, self.config.get('measure_fp_absorbing', True))
                    sb = jnp.take_along_axis(win, (ks - 1)[:, None, None], axis=1)[:, 0]  # [B,obs]
                    if self.config['actor_kind'] == 'ddpgbc':
                        dist_b = self.network.select('actor')(sb, a_goals)
                        a_b = dist_b.mode() if self.config['const_std'] else dist_b.sample(seed=rng)
                        a_b = jnp.clip(a_b, -1, 1)
                    else:
                        a_b = self.network.select('actor')(sb, a_goals,
                                                           jax.random.normal(rng, acts.shape))
                    a_b = jax.lax.stop_gradient(a_b)
                    _, tphi_b, tpsi_b = self.network.select('target_critic')(
                        sb, col_states, a_b, *zarg, info=True, **zkw)
                    tM_b = jax.lax.stop_gradient(
                        jnp.einsum('hid,hjd->hij', tphi_b, tpsi_b) / sqd)          # [2,B,B]
                    boot_w = (gamma ** ks.astype(M.dtype))[:, None]                # [B,1]
                    offdiag, diagt = self._measure_terms_nstep(
                        M, tM_b, O, cont, boot_w, scale_norm)                      # [2]
                    nstep_k_mean = jnp.mean(ks.astype(M.dtype))
                else:
                    offdiag, diagt = self._measure_terms(
                        M, tM, gamma, scale_norm, hit=hit,
                        absorbing=self.config.get('measure_fp_absorbing', True))                # [2]
            elif test_states == 'rho' and test_rank == 'full':
                # Legacy FB path, untouched: offdiag and diag are the two halves of ONE matrix,
                # which only works because the test states ARE the next-states.
                offdiag, diagt = self._measure_terms(M, tM, gamma, scale_norm)                  # [2]
            else:
                # Task norm. The test set moves to nu=psi(value_goals); the DIRAC term (the reward,
                # -E_i M(s_i,a_i,s'_i)) stays on psi(s'_i), and the ortho below stays on psi(nobs).
                # Letting the Dirac follow the test set would make the diagonal Q_i - gamma*Qbar_i
                # and delete the reward. phi depends on (obs,acts,z) only -- b_goals feeds psi
                # alone -- so we reuse phi/tphi and take only the psi legs from the extra calls.
                _, _, psi_t = self.network.select('critic')(obs, vgoals, acts, *zarg, info=True,
                                                            params=grad_params, **zkw)
                if sg_psi: psi_t = jax.lax.stop_gradient(psi_t)
                _, _, tpsi_t = self.network.select('target_critic')(nobs, vgoals, a_next, *zarg,
                                                                    info=True, **zkw)
                M_t = jnp.einsum('hid,hjd->hij', phi, psi_t) / sqd                              # [2,B,B]
                tM_t = jax.lax.stop_gradient(jnp.einsum('hid,hjd->hij', tphi, tpsi_t) / sqd)
                diff_t = M_t - gamma * tM_t
                if test_rank == 'full':
                    offdiag = 0.5 * jnp.sum((diff_t * off[None]) ** 2, axis=(1, 2)) / jnp.sum(off)
                else:  # 'task': rank-1 projection of each residual onto its OWN z_i = psi(g_i)
                    qz = jnp.diagonal(diff_t, axis1=1, axis2=2)                                 # [2,B]
                    offdiag = 0.5 * jnp.mean(qz ** 2, axis=1)                                   # [2]
                    qz_mean = jnp.mean(qz)
                # Dirac term: the diagonal of the RHO matrix (M, tM), which is already computed
                # above. Taking it from that matrix rather than recomputing it with an einsum
                # keeps this term BIT-IDENTICAL to the legacy _measure_terms diag -- the reduction
                # order matters in float32, and the term is a small number obtained by
                # cancellation, so a different order shifts it by ~1e-3 relative.
                dir_i = jnp.diagonal(M - gamma * tM, axis1=1, axis2=2)                          # [2,B]
                diagt = -jnp.mean(dir_i, axis=1)                                                # [2]
                if scale_norm:
                    # Each term is normalized by ITS OWN matrix's scale: offdiag lives on the task
                    # matrix M_t, the Dirac on the rho matrix M. Using M_t for both (the naive
                    # port of _measure_terms) would make the Dirac differ from the anchor's and
                    # silently break the "only the test set moved" invariant.
                    msq_t = jax.lax.stop_gradient(jnp.mean(M_t ** 2))
                    msq_o = jax.lax.stop_gradient(jnp.mean(M ** 2))
                    offdiag = offdiag / (msq_t + 1e-6)
                    diagt = diagt / (jnp.sqrt(msq_o) + 1e-6)
            Cov = jnp.einsum('hid,hjd->hij', psi, psi)                    # FB ortho on B:=psi(s') (grads on)
            o_off = 0.5 * jnp.sum((Cov * off[None]) ** 2, axis=(1, 2)) / jnp.sum(off)
            o_diag = -jnp.mean(jnp.diagonal(Cov, axis1=1, axis2=2), axis=1)
            orth = o_off + o_diag                                          # [2]
            lam = self.config.get('measure_ortho_coef', 0.0)
            loss = jnp.mean(offdiag + diagt + lam * orth)                  # per-head, then average
            if nhat_loss is not None:
                # measure_nhat_head: coefficient 1.0; the head is gradient-isolated,
                # so this addition changes NO other term's gradient or value.
                loss = loss + nhat_loss
            offdiag, diagt, orth = jnp.mean(offdiag), jnp.mean(diagt), jnp.mean(orth)
        info = {'measure_loss': loss, 'measure_offdiag': offdiag, 'measure_diag': diagt,
                'measure_orth': orth}
        if qz_mean is not None:
            info['measure_qz_mean'] = qz_mean
        if nhat_loss is not None:
            info['measure_nhat_loss'] = nhat_loss
            info['measure_nhat_pred_mean'] = nhat_pred_mean
        if nstep_k_mean is not None:
            info['measure_nstep_k_mean'] = nstep_k_mean
        if delta_event_rate is not None:
            info['measure_delta_event_rate'] = delta_event_rate
            info['measure_delta_hit_level'] = delta_hit_level
            info['measure_delta_msq'] = delta_msq
        return loss, info

    def delta_partition_loss(self, batch, grad_params, rng):
        """Spec 17b: one-stage simplex successor measure CE loss over a fixed
        K-prototype partition. p(s,a,g)=softmax(delta_head([s,a,g])) in Delta^K; TD
        target p_tgt = sg[(1-gamma)*phi(x') + gamma*p_bar(s',a+~pi,g)] (a probability
        vector, counted from the NEXT state -- spec section 3); L = CE(p_tgt, p). The
        bootstrap action a+ is a ONE-SHOT flow-actor draw at (s', g), stop-gradded,
        drawing its noise directly off the rng key passed in (no split -- matches the
        spec's exact reproducibility contract)."""
        obs, nobs = batch['observations'], batch['next_observations']
        acts, g = batch['actions'], batch['value_goals']
        blocks = _parse_hit_spec(self.config.get('measure_hit_spec', '19:22:0.4'))
        gamma = self.config['discount']
        logits = self.network.select('delta_head')(
            jnp.concatenate([obs, acts, g], -1), params=grad_params)          # [2,B,K]
        p = jax.nn.softmax(logits, -1)
        # bootstrap action: one-shot flow actor at (s', g), stop-gradded -- the same
        # machinery measure_loss uses (actor_kind='flow' is pinned by the lattice).
        noise = jax.random.normal(rng, acts.shape)
        a_plus = jax.lax.stop_gradient(self.network.select('actor')(nobs, g, noise))
        t_logits = self.network.select('target_delta_head')(
            jnp.concatenate([nobs, a_plus, g], -1))                            # [2,B,K]
        p_bar = jax.nn.softmax(t_logits, -1)
        phi_n = self._delta_phi(nobs, self.delta_protos, blocks)               # [B,K]
        p_tgt = jax.lax.stop_gradient((1.0 - gamma) * phi_n[None] + gamma * p_bar)
        ce = -jnp.mean(jnp.sum(p_tgt * jnp.log(p + 1e-8), axis=-1))
        loss = self.config.get('measure_delta_coef', 1.0) * ce

        phi_c = self._delta_phi(obs, self.delta_protos, blocks)                # metrics only
        p_entropy = -jnp.mean(jnp.sum(p * jnp.log(p + 1e-8), axis=-1))
        tgt_entropy = -jnp.mean(jnp.sum(p_tgt * jnp.log(p_tgt + 1e-8), axis=-1))
        p_max = jnp.mean(jnp.max(p, axis=-1))
        cell_change = 0.5 * jnp.mean(jnp.sum(jnp.abs(phi_n - phi_c), axis=-1))
        info = {
            'delta_ce': ce,
            'delta_p_entropy': p_entropy,
            'delta_tgt_entropy': tgt_entropy,
            'delta_p_max': p_max,
            'delta_cell_change': cell_change,
        }
        return loss, info

    def centered_measure_loss(self, batch, grad_params, rng):
        """T10 -- centered (advantage) successor measure (PAPER/rescuing-fb/
        04-centered-measure-spec.md). Every failed measure cell trains raw M, whose
        loss is dominated by the shared occupancy baseline; the action-contingent
        component (what control needs) is ~0.2% of Q-variance and gets ~no gradient.
        Centering gives it its own loss and scale:

            b(s)      ~= E_{a~mu} phi(s,a)   (mu = the frozen goal-agnostic BC flow;
                          K=center_b_samples flow samples; regression to sg[mean phi],
                          scale-normed by sg[mean(phi_mu^2)] -- the dueling-v1 lesson)
            DeltaM    = (phi(s,a) - sg[b(s)]) . psi(s') / sqrt(d)
            DeltaM TD = the usual BxB measure residual (offdiag + diag attraction) on
                        DeltaM - gamma*sg[DeltaM_target(s',a'~pi,.)], own scale-norm.

        centered_measure='aux' (T10-a): this is an AUXILIARY on top of the stock scalar
        IQL losses -- U stays grounded by expectile TD; the actor consumes the full
        critic Q = phi.psi = (b + Delta-phi).psi = U + DeltaM, normalized by sg|Q_full|
        (spec §4 risk 1: NEVER DeltaM alone -- the dueling Arch-11 floor).

        centered_measure='only' (T10-b): NO scalar TD anywhere. A second, BASE leg
        grounds the raw measure M = phi.psi with a BEHAVIOR-ANCHORED measure TD:
        bootstrap a' = the DATASET next action (SARSA, batch['next_actions'] -- the
        fixed-policy measure operator, a genuine gamma-contraction), while the pi
        bootstrap lives ONLY on the DeltaM leg (the split-bootstrap design). The actor
        again consumes the full Q = phi.psi.

        b(s) is trained ONLY by the regression (stop-gradded inside DeltaM, online net
        used for the target side -- there is no target b). Logged: b-fidelity
        ||E_mu phi - b||/||phi|| (spec risk 4) + the T2 early-warning set (E||psi||^2,
        E||phi||^2, q_mean/q_max on Q_full, per-leg scales)."""
        obs, nobs = batch['observations'], batch['next_observations']
        acts, vgoals = batch['actions'], batch['value_goals']
        d = self.config['value_latent_dim']; sqd = jnp.sqrt(d)
        gamma = self.config['discount']
        B = obs.shape[0]; off = 1.0 - jnp.eye(B)
        mode = self.config['centered_measure']
        rng, mu_rng, pi_rng = jax.random.split(rng, 3)

        # Online legs: phi(s,a) and psi(s') -- test states = rho (next-observations),
        # exactly as measure_loss's critic branch.
        _, phi, psi = self.network.select('critic')(obs, nobs, acts, info=True,
                                                    params=grad_params)          # [2,B,d]

        # centered_scale_anchors (T10-b2, 2026-07-16): with NO scalar TD, scale-norming
        # BOTH measure legs makes the whole objective scale-invariant -- nothing pins
        # |phi.psi| and the diag-attraction inflates it without bound (observed: q_mean
        # 1e8->3e13, norms 1e14 by 160k; the T2 early-warning metrics caught it). FB's
        # own anchors, restored: hard sphere-norm on psi (||psi||=sqrt(d)) + a RAW
        # (un-normalized) regression on the SARSA-anchored base leg to pin phi's scale.
        anchors = bool(self.config.get('centered_scale_anchors', False))
        if anchors:
            psi = _sphere(psi, d)

        # --- b(s): behavior-mean forward feature, regression to sg[mean_{a~mu} phi]. ---
        # K flow samples per state, batched as one rollout of (B*K); rows (i,k)=i*K+k.
        # Frozen params everywhere on the target side (values only, no grads).
        K = int(self.config['center_b_samples'])
        obs_rep = jnp.repeat(obs, K, axis=0)                                     # [B*K, obs]
        mu_noise = jax.random.normal(mu_rng, (B * K, self.config['action_dim']))
        a_mu = self._flow_rollout(obs_rep, mu_noise, jax.lax.stop_gradient(grad_params))
        _, phi_mu, _ = self.network.select('critic')(obs_rep, obs_rep, a_mu, info=True)
        phi_mu = jax.lax.stop_gradient(phi_mu).reshape(2, B, K, d).mean(axis=2)  # sg[E_mu phi] [2,B,d]
        bs = self.network.select('center_b')(obs, params=grad_params)            # [2,B,d]
        b_scale = jax.lax.stop_gradient(jnp.mean(phi_mu ** 2)) + 1e-6
        b_loss = jnp.mean((bs - phi_mu) ** 2) / b_scale
        # b-fidelity (spec risk 4): a lazy b silently un-centers DeltaM.
        b_fid = (jnp.mean(jnp.linalg.norm(phi_mu - jax.lax.stop_gradient(bs), axis=-1))
                 / (jnp.mean(jnp.linalg.norm(phi_mu, axis=-1)) + 1e-6))

        # --- DeltaM leg: centered measure TD, bootstrap a' ~ pi (one-shot actor). ---
        pi_noise = jax.random.normal(pi_rng, acts.shape)
        a_pi = jax.lax.stop_gradient(self.network.select('actor')(nobs, vgoals, pi_noise))
        _, tphi, tpsi = self.network.select('target_critic')(nobs, nobs, a_pi, info=True)
        if anchors:
            tpsi = _sphere(tpsi, d)
        tb = jax.lax.stop_gradient(self.network.select('center_b')(nobs))        # online b at s'
        dphi = phi - jax.lax.stop_gradient(bs)          # b trained ONLY by its regression
        if self.config.get('measure_stopgrad_phi', False):
            dphi = jax.lax.stop_gradient(dphi)
        dM = jnp.einsum('hid,hjd->hij', dphi, psi) / sqd                          # [2,B,B]
        tdM = jax.lax.stop_gradient(jnp.einsum('hid,hjd->hij', tphi - tb, tpsi) / sqd)
        dm_off, dm_diag = self._measure_terms(dM, tdM, gamma, scale_norm=True)   # own scale-norm
        dm_loss = jnp.mean(dm_off + dm_diag)

        info = {
            'b_loss': b_loss, 'b_scale': b_scale, 'b_fidelity': b_fid,
            'dm_loss': dm_loss, 'dm_offdiag': jnp.mean(dm_off), 'dm_diag': jnp.mean(dm_diag),
            'dm_msq': jnp.mean(dM ** 2),
            # T2 early-warning set: the psi-leg norm runaway + q_max are the divergence trace.
            'phi_norm_sq': jnp.mean(jnp.sum(phi ** 2, axis=-1)),
            'psi_norm_sq': jnp.mean(jnp.sum(psi ** 2, axis=-1)),
        }

        # --- Base/U leg (T10-b only): behavior-anchored raw-measure TD (SARSA). ---
        base_loss = 0.0
        if mode == 'only':
            if self.config.get('centered_split_bootstrap', True):
                assert 'next_actions' in batch, (
                    "centered_measure='only' + centered_split_bootstrap needs "
                    "batch['next_actions'] (dataset SARSA bootstrap on the base leg); "
                    "run with OGBENCH_NEXT_ACTIONS=1 so sitecustomize patches "
                    "Dataset.get_subset (see scripts/rescuing_fb/launch_t10.sh)")
                a_base = batch['next_actions']
            else:   # non-contraction control: pi bootstrap on the base leg too
                a_base = a_pi
            _, tphi_b, tpsi_b = self.network.select('target_critic')(nobs, nobs, a_base, info=True)
            if anchors:
                tpsi_b = _sphere(tpsi_b, d)
            M = jnp.einsum('hid,hjd->hij', phi, psi) / sqd                        # [2,B,B]
            tM = jax.lax.stop_gradient(jnp.einsum('hid,hjd->hij', tphi_b, tpsi_b) / sqd)
            # anchors: RAW regression here (scale_norm off) -- the contractive SARSA leg
            # is the one place an absolute scale can be pinned without an E2-style fight.
            base_off, base_diag = self._measure_terms(M, tM, gamma, scale_norm=not anchors)
            base_loss = jnp.mean(base_off + base_diag)
            info.update({'base_loss': base_loss, 'base_offdiag': jnp.mean(base_off),
                         'base_diag': jnp.mean(base_diag), 'base_msq': jnp.mean(M ** 2)})

        # Optional FB orthonormality on psi(s') (reuses measure_ortho_coef; default 0).
        lam_o = self.config.get('measure_ortho_coef', 0.0)
        orth = 0.0
        if lam_o > 0:
            Cov = jnp.einsum('hid,hjd->hij', psi, psi)
            o_off = 0.5 * jnp.sum((Cov * off[None]) ** 2, axis=(1, 2)) / jnp.sum(off)
            o_diag = -jnp.mean(jnp.diagonal(Cov, axis1=1, axis2=2), axis=1)
            orth = jnp.mean(o_off + o_diag)
            info['orth'] = orth

        # Q_full watch (spec risk 3: q_mean must stay <= 0 and bounded). Frozen params,
        # metrics only -- the actor's own q-normalizer already sees sg|Q_full|.
        q1, q2 = self.network.select('critic')(obs, vgoals, acts)
        q_full = jnp.minimum(q1, q2)
        info.update({'q_mean': q_full.mean(), 'q_max': q_full.max(), 'q_min': q_full.min()})

        lam = self.config.get('centered_measure_coef', 1.0)
        bcoef = self.config.get('center_b_coef', 1.0)
        loss = base_loss + lam * dm_loss + bcoef * b_loss + lam_o * orth
        info['centered_loss'] = loss
        return loss, info

    def sf_loss(self, batch, grad_params, rng):
        """Successor-FEATURE vector TD on phi (SF-style, à la simple_successor_features), with
        an IDENTITY base b(s)=s. Trains ONLY phi (the goal/base leg psi=b is param-free):

            phi(s,a) <- b(s) + gamma * mask * phi_target(s',a'),   a' ~ pi(s',g)
            L_SF = mean_i || phi(s_i,a_i) - sg[ b(s_i) + gamma m_i phi_target(s'_i,a'_i) ] ||^2

        A PER-SAMPLE vector regression -- NOT the B x B successor-measure matrix. The readout
        Q(s,a,g)=<phi(s,a), psi(g)> drives the greedy flow actor. Base b(s): 'identity' -> b(s)=s
        (d==obs dim); 'frozen' -> b(s)=psi(s), the critic's frozen random features (d free)."""
        obs, nobs, acts = batch['observations'], batch['next_observations'], batch['actions']
        vgoals = batch['value_goals']
        gamma = self.config['discount']
        # sf_mask: 'goal' -> batch['masks'] = 1[s != g], an ABSORBING goal (Q = gamma^T, a
        # first-hitting value). 'none' -> m == 1, no absorption (Q = sum_t gamma^t 1[s_t=g], a
        # visitation DENSITY -- FB's object; measure_loss has no mask at all).
        mask = batch['masks'] if self.config.get('sf_mask', 'goal') == 'goal' else jnp.ones_like(batch['masks'])
        # sf_base_index: 'current' -> b(s)=psi(s), forced by GCDataset tying rewards AND masks to
        # the current index. 'next' -> b(s)=psi(s'), FB's arriving-state convention (its measure
        # counts arrival at s'). Never mix the two (guard 4).
        base_next = self.config.get('sf_base_index', 'current') == 'next'
        joint = self.config.get('sf_joint_psi', False)
        # fb_forward_z: task-condition phi([s,a,z=psi(g)]) -> thread z_goals=vgoals through the
        # phi, base-feature, and target-critic calls (same task g on both sides of the SF backup).
        zarg = (vgoals,) if self.config.get('fb_forward_z', False) else ()
        _, phi, _psi = self.network.select('critic')(obs, vgoals, acts, *zarg, info=True, params=grad_params)  # [2,B,d]
        if self.config.get('sf_base', 'identity') in ('frozen', 'learned'):
            # b = psi(s) or psi(s'): call psi via b_goals. Stop-gradded UNLESS sf_joint_psi, in
            # which case the TD residual trains psi through the base (held non-degenerate by
            # sf_psi_ortho_coef -- without it phi=0, psi=0 is a global minimum).
            b_goals = nobs if base_next else obs
            _, _, b = self.network.select('critic')(obs, b_goals, acts, *zarg, info=True, params=grad_params)  # [2,B,d]
            if not joint:
                b = jax.lax.stop_gradient(b)
        else:
            b = (nobs if base_next else obs)[None]                    # identity base b(s)=s -> [1,B,d]
        if self.config['actor_kind'] == 'ddpgbc':
            # stock gaussian actor: a' = mode of pi(.|s',g) (or a sample if learned std).
            dist = self.network.select('actor')(nobs, vgoals)
            a_next = dist.mode() if self.config['const_std'] else dist.sample(seed=rng)
        else:
            noise = jax.random.normal(rng, acts.shape)
            a_next = self.network.select('actor')(nobs, vgoals, noise)          # a' ~ pi(s',g)
        a_next = jax.lax.stop_gradient(jnp.clip(a_next, -1, 1))
        _, tphi, _ = self.network.select('target_critic')(nobs, vgoals, a_next, *zarg, info=True)  # [2,B,d]
        boot = gamma * mask[None, :, None] * jax.lax.stop_gradient(tphi)                      # [2,B,d]
        # b carries gradient iff sf_joint_psi; the bootstrap is always semi-gradient (FB does the
        # same). With joint=False this is identical to the old sg[b + gamma*m*tphi] target.
        target = b + boot
        td = jnp.mean((phi - target) ** 2)
        loss = td
        info = {'sf_loss': td, 'phi_absmax': jnp.max(jnp.abs(phi)),
                'target_absmax': jnp.max(jnp.abs(jax.lax.stop_gradient(target)))}
        lam = self.config.get('sf_ortho_coef', 0.0)
        if lam > 0:
            # anti-collapse aux reg: orthonormality ||E[phi phi^T]-I|| on the SF vector phi
            # (sf_laplacian-style), per-head then averaged. Same form as measure_loss's ortho.
            Bn = obs.shape[0]; off = 1.0 - jnp.eye(Bn)
            Cov = jnp.einsum('hid,hjd->hij', phi, phi)                    # [2,B,B]
            o_off = 0.5 * jnp.sum((Cov * off[None]) ** 2, axis=(1, 2)) / jnp.sum(off)
            o_diag = -jnp.mean(jnp.diagonal(Cov, axis1=1, axis2=2), axis=1)
            orth = jnp.mean(o_off + o_diag)
            loss = td + lam * orth
            info['orth'] = orth
        lam_psi = self.config.get('sf_psi_ortho_coef', 0.0)
        if lam_psi > 0:
            # FB orthonormality ||E_rho[psi psi^T] - I|| on the BASE/goal leg psi -- the same
            # penalty and the same distribution measure_loss uses (B := psi(s'), the rho marginal,
            # NOT the HER value_goals that ortho_loss uses). This is what keeps a jointly-trained
            # psi non-degenerate: un-stop-gradding the base makes phi=0, psi=0 a GLOBAL minimum of
            # the TD, and the -tr(C) term below turns the origin into a strict local maximum
            # (repelling). Distinct from sf_ortho_coef, which penalizes phi.
            _, _, psi_o = self.network.select('critic')(obs, nobs, acts, *zarg, info=True, params=grad_params)
            Bn = obs.shape[0]; off_p = 1.0 - jnp.eye(Bn)
            CovP = jnp.einsum('hid,hjd->hij', psi_o, psi_o)                 # [2,B,B]
            p_off = 0.5 * jnp.sum((CovP * off_p[None]) ** 2, axis=(1, 2)) / jnp.sum(off_p)
            p_diag = -jnp.mean(jnp.diagonal(CovP, axis1=1, axis2=2), axis=1)
            orth_psi = jnp.mean(p_off + p_diag)
            loss = loss + lam_psi * orth_psi
            info['psi_ortho'] = orth_psi
            info['psi_ortho_off'] = jnp.mean(p_off)
            info['psi_ortho_diag'] = jnp.mean(p_diag)
            info['psi_norm_mean'] = jnp.mean(jnp.linalg.norm(psi_o, axis=-1))
        if self.config.get('sf_reward_regress', False):
            # E3 (corrected SF cell): ground the base in the ACTUAL reward -- bilinear reward
            # model r_hat(s,g) = <psi(s), psi(g)>/sqrt(d) regressed on batch rewards. This is
            # psi's ONLY grad path (the TD target stop-grads b; the actor Q-term takes no critic
            # grads), and it is what makes the goal indicator linearly readable in the base.
            # CURRENT-state convention, forced by GCDataset: successes = (idxs == value_goal_idxs)
            # ties BOTH rewards (=successes-1) and masks (=1-successes) to the CURRENT index, and
            # the SF target's immediate feature is b(s). Regressing on psi(s') instead would be
            # off-by-one against reward, mask, and base alike (an arriving-state convention would
            # additionally require retargeting the TD to b(s') and relabelling on idxs+1).
            # At the goal s=g this pins <psi(g),psi(g)> = ||psi(g)||^2 = r_at_goal + shift.
            _, _, psi_s = self.network.select('critic')(obs, obs, acts, *zarg, info=True, params=grad_params)  # psi(s)
            _, _, psi_g = self.network.select('critic')(obs, vgoals, acts, *zarg, info=True, params=grad_params)  # psi(g)
            r_hat = (psi_s * psi_g).sum(axis=-1) / jnp.sqrt(psi_s.shape[-1])   # [2,B]
            # sf_reward_shift: regress on r+shift. Raw OGBench rewards {-1 off-goal, 0 at-goal}
            # collapse the SYMMETRIC bilinear model to the psi=0 saddle (grad of a bilinear form
            # vanishes at psi=0, where the majority class is already fit); shift=1 gives {0, 1}
            # targets -- majority class 0 is saddle-consistent, goal pairs demand <psi,psi>=1
            # (a peaked kernel). Note r+1 == successes == 1-masks, i.e. the goal INDICATOR.
            # Argmax-invariance is NOT "a constant offset telescopes away" -- it does not, since
            # the mask makes the goal absorbing so the horizon is the hitting time T. It holds
            # because both readouts are strictly increasing in gamma^T:
            #   r_hat = 1[s=g]      -> Q = gamma^T
            #   r_hat = 1[s=g] - 1  -> Q = -(1 - gamma^T)/(1 - gamma)
            # so they induce the same ordering over actions and the same greedy policy.
            r_tgt = batch['rewards'] + self.config.get('sf_reward_shift', 0.0)
            rew = jnp.mean((r_hat - r_tgt[None]) ** 2)
            loss = loss + self.config.get('sf_reward_coef', 1.0) * rew
            info['sf_reward_loss'] = rew
            info['sf_rhat_mean'] = jnp.mean(r_hat)
        return loss, info

    def simplesf_reward_loss(self, batch, grad_params):
        """Simple SF's reward-prediction loss (sf_simple.py:302-325), the task vector's ONLY
        gradient path when simplesf_ground_psi:

            r_hat(s,a,g) = <sg[b(s,a,z)], psi(g)>/sqrt(d)   ->  regress on batch['rewards']

        b is stop-gradded exactly as the reference computes next_rep under torch.no_grad() and
        steps only task_opt. CURRENT index: GCDataset ties rewards AND masks to
        idxs == value_goal_idxs (datasets.py:242-244), so b is evaluated at (s,a), not (s',a').

        Applied to BOTH nets: grounding only the critic's psi would leak goal information back
        through V(s',g) into the critic target. No reward shift is needed -- the model is
        ASYMMETRIC (b is a separate object from psi), so it has no psi=0 saddle, and with
        ||b||=1, ||z||=sqrt(d) its range [-1,1] already covers the reward range {-1,0}."""
        obs, vgoals, acts = batch['observations'], batch['value_goals'], batch['actions']
        r = batch['rewards']
        d = self.config['value_latent_dim']
        _, _, z_q, b_q = self.network.select('critic')(obs, vgoals, acts, info=True, params=grad_params)
        _, _, z_v, b_v = self.network.select('value')(obs, vgoals, info=True, params=grad_params)
        rhat_q = (jax.lax.stop_gradient(b_q) * z_q).sum(axis=-1) / jnp.sqrt(d)      # [B]
        rhat_v = (jax.lax.stop_gradient(b_v) * z_v).sum(axis=-1) / jnp.sqrt(d)      # [B]
        loss = jnp.mean((rhat_q - r) ** 2) + jnp.mean((rhat_v - r) ** 2)
        return loss, {
            'reward_loss': loss,
            'rhat_mean': jnp.mean(rhat_q), 'rhat_std': jnp.std(rhat_q),
            'b_norm_mean': jnp.mean(jnp.linalg.norm(b_q, axis=-1)),
            'z_norm_mean': jnp.mean(jnp.linalg.norm(z_q, axis=-1)),
        }

    def ortho_loss(self, batch, grad_params):
        """FB orthonormality penalty ||E[psi psi^T]-I|| on the IQL critic's backward
        map psi(value_goals) -- with NO measure bootstrap. Isolates the ortho constraint
        on the ~90 IQL bilinear ('bilinear GCIQL + ortho only, nothing else from FB').
        Same per-head formula as measure_loss's critic-leg ortho."""
        obs, vgoals, acts = batch['observations'], batch['value_goals'], batch['actions']
        _, _, psi = self.network.select('critic')(obs, vgoals, acts, info=True, params=grad_params)
        B = obs.shape[0]; off = 1.0 - jnp.eye(B)
        Cov = jnp.einsum('hid,hjd->hij', psi, psi)                    # [2, B, B]
        o_off = 0.5 * jnp.sum((Cov * off[None]) ** 2, axis=(1, 2)) / jnp.sum(off)
        o_diag = -jnp.mean(jnp.diagonal(Cov, axis1=1, axis2=2), axis=1)
        orth = jnp.mean(o_off + o_diag)
        return orth, {'ortho_loss': orth}

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """IQL value + critic + flow actor (+ optional FB measure). measure_only=True drops the
        IQL value/critic losses entirely: the critic phi/psi are trained ONLY by the successor
        measure (so Q=<phi(s,a),psi(g)> becomes the FB measure M(s,a,g)), and the flow actor goes
        greedy on it -- i.e. FB's objective on GCIQL's bilinear architecture."""
        info = {}
        rng = rng if rng is not None else self.rng

        if self.config.get('task_prior', 'dirac') == 'mix':
            # RG-FB v1 (spec06 item4): per-sample task prior. The scalar IQL value+critic TD
            # trains at the sampled z (psi-grads ON, targets sg); the flow actor consumes the
            # sg'd z_actor (Dirac rows rebuilt from actor_goals = the D8 split). No other
            # objective.
            rng, s_rng, a_rng = jax.random.split(rng, 3)
            z, z_actor, r, mask, ttype = self._sample_tasks(batch, grad_params, s_rng)
            vc_loss, vc_info = self._taskprior_vc_loss(batch, grad_params, z, r, mask, ttype)
            a_loss, a_info = self.actor_loss(batch, grad_params, a_rng, task_z=z_actor)
            for k, v in vc_info.items():
                info[f'taskprior/{k}'] = v
            for k, v in a_info.items():
                info[f'actor/{k}'] = v
            return vc_loss + a_loss, info

        rng, actor_rng = jax.random.split(rng)
        tpa_z = tpa_zmix = tpa_cubeK = None
        if self.config.get('task_prior_actor', 'dirac') == 'mix':
            # Spec 07 item 1: per-sample actor z-prior in the FP corner. ONLY the actor's
            # conditioning z changes; all new randomness (arm draw, mixture indices/weights,
            # sphere z, and item 4's bootstrap noise below) comes from splits consumed
            # ONLY inside these knob-on branches -- the measure rng stream (split off `rng`
            # below) never sees them, so knob-off stays bitwise identical.
            actor_rng, tpa_rng = jax.random.split(actor_rng)
            tpa_z, tpa_ttype, tpa_zmix, tpa_cubeK = self._sample_actor_tasks(
                batch, grad_params, tpa_rng)
            info['tpactor/frac_dirac'] = jnp.mean((tpa_ttype == 0).astype(jnp.float32))
            info['tpactor/frac_mix'] = jnp.mean((tpa_ttype == 1).astype(jnp.float32))
            info['tpactor/frac_rand'] = jnp.mean((tpa_ttype == 2).astype(jnp.float32))
            info['tpactor/z_norm'] = jnp.mean(jnp.linalg.norm(tpa_z, axis=-1))
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng, task_z=tpa_z)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        if self.config.get('sf_feature_td', False):    # SF-ONLY critic (no IQL): successor-feature vector TD on phi
            rng, sf_rng = jax.random.split(rng)
            sf_l, sf_info = self.sf_loss(batch, grad_params, sf_rng)
            for k, v in sf_info.items():
                info[f'sf/{k}'] = v
            return self.config.get('sf_coef', 1.0) * sf_l + actor_loss, info

        if self.config.get('centered_measure', 'off') == 'only':
            # T10-b: NO scalar TD anywhere. Base leg = behavior-anchored (SARSA) measure
            # TD grounds M=phi.psi; DeltaM leg = centered measure TD with a pi bootstrap;
            # b(s) regression. The IQL value/critic losses are dropped (value net inits
            # but never trains, exactly like measure_only).
            rng, cm_rng = jax.random.split(rng)
            cm_loss, cm_info = self.centered_measure_loss(batch, grad_params, cm_rng)
            for k, v in cm_info.items():
                info[f'centered/{k}'] = v
            return cm_loss + actor_loss, info

        if self.config.get('measure_only', False):     # SM-ONLY critic (no IQL): FB objective
            rng, m_rng = jax.random.split(rng)
            if self.config.get('measure_delta_partition', False):
                # spec 17b: one-stage simplex successor measure CE INSTEAD of the
                # bilinear measure TD -- same measure_only wiring/prefix, different loss.
                m_loss, m_info = self.delta_partition_loss(batch, grad_params, m_rng)
            else:
                m_loss, m_info = self.measure_loss(batch, grad_params, m_rng)
            for k, v in m_info.items():
                info[f'measure/{k}'] = v
            if self.config.get('measure_delta_partition', False):
                # CRITICAL 1 fix (round 1 review): delta_partition_loss already applies
                # measure_delta_coef internally. measure_coef defaults to 0.0 and the
                # 17b launch config deliberately DROPS it (spec 17b §6) -- gating this
                # path by measure_coef too would silently zero the CE's gradient
                # (delta_head never trains, target frozen forever) while total_loss
                # still returns a finite, plausible-looking loss and info dict.
                loss = m_loss + actor_loss
            else:
                loss = self.config.get('measure_coef', 1.0) * m_loss + actor_loss
            if self.config.get('task_prior_critic_mix', False):
                # Spec 07 item 4 (linearity-conflict probe): scalar FP TD at the actor's
                # mixture z ON TOP of the untouched matrix loss. The bootstrap noise is a
                # further split of `rng` consumed only here (knob-off bitwise identical).
                rng, cmx_rng = jax.random.split(rng)
                cmx_loss, cmx_info = self._tpactor_critic_mix_loss(
                    batch, grad_params, tpa_zmix, tpa_cubeK, cmx_rng)
                loss = loss + cmx_loss
                for k, v in cmx_info.items():
                    info[f'tpactor/{k}'] = v
            return loss, info

        value_loss, value_info = self.value_loss(batch, grad_params)
        for k, v in value_info.items():
            info[f'value/{k}'] = v
        critic_loss, critic_info = self.critic_loss(batch, grad_params)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v
        # anneal_iql_fade: crossfade the IQL value+critic losses OUT (weight 1 -> 0) as the measure
        # ramps IN, so the endpoint is measure-only, warm-started. Tests whether the good basin the
        # mixture holds is a genuine fixed point or was propped up by L_IQL. The actor loss is NOT
        # faded -- the flow policy must keep tracking whatever Q the fading critic produces.
        iql_w = 1.0
        if self.config.get('anneal_iql_fade', False):
            t = jnp.asarray(self.network.step, jnp.float32)
            ramp = float(self.config.get('anneal_ramp', 300_000))
            iql_w = 1.0 - jnp.clip(t / ramp, 0.0, 1.0)
            info['anneal/iql_w'] = iql_w
        loss = iql_w * (value_loss + critic_loss) + actor_loss
        if self.config.get('centered_measure', 'off') == 'aux':
            # T10-a: centered-measure ADVANTAGE as an auxiliary on top of the stock
            # scalar IQL grounding (which keeps training the full Q = phi.psi). The
            # actor path is untouched -- it consumes Q_full normalized by sg|Q_full|.
            rng, cm_rng = jax.random.split(rng)
            cm_loss, cm_info = self.centered_measure_loss(batch, grad_params, cm_rng)
            for k, v in cm_info.items():
                info[f'centered/{k}'] = v
            loss = loss + cm_loss
        if self.config.get('simplesf_ground_psi', False):
            # rung 3 (the paper's cell): psi(g) is detached inside value/critic (GCSimpleSFValue),
            # so this regression is its ONLY gradient. The trunk still learns from the TD, which
            # is what trains the basis b.
            ss_loss, ss_info = self.simplesf_reward_loss(batch, grad_params)
            for k, v in ss_info.items():
                info[f'simplesf/{k}'] = v
            loss = loss + self.config.get('simplesf_reward_coef', 1.0) * ss_loss
        if self.config.get('anneal_simplesf_ground', False):
            # SimpleSF curriculum: the reward-regression grounding fades IN (weight 0 -> reward_coef
            # over anneal_ramp) as the true-reward TD fades OUT via iql_w above. psi keeps the rung-2
            # architecture (simplesf_ground_psi off, z NOT detached), so the fade -- not a
            # stop-gradient -- hands psi's training from the TD to the reward regression. Endpoint =
            # reward-regressed psi on the (TD-faded, hence frozen) rung-2 trunk.
            t = jnp.asarray(self.network.step, jnp.float32)
            ramp = float(self.config.get('anneal_ramp', 300_000))
            frac = jnp.clip(t / ramp, 0.0, 1.0)
            ss_loss, ss_info = self.simplesf_reward_loss(batch, grad_params)
            for k, v in ss_info.items():
                info[f'simplesf/{k}'] = v
            info['anneal/ground_frac'] = frac
            loss = loss + frac * self.config.get('simplesf_reward_coef', 1.0) * ss_loss
        if self.config.get('measure_coef', 0.0) > 0:   # graft FB measure onto GCIQL's rep
            rng, m_rng = jax.random.split(rng)
            m_loss, m_info = self.measure_loss(batch, grad_params, m_rng)
            lam = self.config['measure_coef']
            if self.config.get('anneal_measure', False):
                # Warm-start dose-response: grow the SM weight from 0, then HOLD at lam_max.
                # t is network.step, which prepare_anneal_ckpt.py resets to 0 -- restore_agent
                # otherwise brings back the anchor's step (1_000_001) and the ramp would start
                # fully saturated. The measure loss is still COMPUTED at lam=0 (static jit shapes),
                # just scaled to zero, so the first finetune step is a no-op on the restored geometry.
                # jnp.asarray, not .astype: TrainState.step is a plain Python int leaf until jit
                # traces it, and `int.astype` does not exist.
                t = jnp.asarray(self.network.step, jnp.float32)
                ramp = float(self.config.get('anneal_ramp', 300_000))
                lam = lam * jnp.clip(t / ramp, 0.0, 1.0)
                info['anneal/lambda'] = lam
                info['anneal/t'] = t
            loss = loss + lam * m_loss
            for k, v in m_info.items():
                info[f'measure/{k}'] = v
        if self.config.get('iql_ortho_coef', 0.0) > 0:  # FB orthonormality ONLY (no measure)
            o_loss, o_info = self.ortho_loss(batch, grad_params)
            loss = loss + self.config['iql_ortho_coef'] * o_loss
            for k, v in o_info.items():
                info[f'ortho/{k}'] = v
        return loss, info

    @jax.jit
    def update(self, batch):
        """Identical to GCIQLAgent.update (agents/gciql.py) plus a conditional extra
        target soft-update for spec 17b's delta_head. GCIQLFlowAgent otherwise inherits
        update from GCIQLAgent, which only soft-updates 'critic' -- without this override
        target_delta_head would stay frozen at its init values forever. self.config is a
        static (nonpytree) field, so the branch resolves at trace time: bit-identical to
        the inherited update when the knob is off."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'critic')
        if self.config.get('measure_delta_partition', False):
            self.target_update(new_network, 'delta_head')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        # actor_kind='ddpgbc' => stock GCIQL gaussian actor sampling.
        if self.config['actor_kind'] == 'ddpgbc':
            dist = self.network.select('actor')(observations, goals, temperature=temperature)
            return jnp.clip(dist.sample(seed=seed), -1.0, 1.0)
        if self.config['actor_kind'] == 'awr_flow':
            # Euler-rollout of the goal-conditioned velocity field (no one-step actor).
            del temperature
            batch_shape = observations.shape[:-1]
            noise = jax.random.normal(seed, (*batch_shape, self.config['action_dim']))
            obs_goal = jnp.concatenate([observations, goals], axis=-1)
            fs = self.config['flow_steps']
            actions = noise
            for i in range(fs):
                t = jnp.full((*batch_shape, 1), i / fs)
                actions = actions + self.network.select('actor_vf')(obs_goal, actions, t) / fs
            return jnp.clip(actions, -1.0, 1.0)
        del temperature  # flow actor is noise-driven
        # Noise must share the obs batch shape. Eval passes a single, unbatched
        # observation; training/tests pass batches. Event rank is 1 for state and
        # 3 (H,W,C) for pixels (encoder present), so strip the event dims.
        event_rank = 1 if self.config['encoder'] is None else 3
        batch_shape = observations.shape[:-event_rank]
        noise = jax.random.normal(seed, (*batch_shape, self.config['action_dim']))
        # z-conditioning (G4): the eval harness (and the in-training eval loop)
        # hands the RAW goal obs; map it through B := value.psi to the task vector
        # the z-actor expects. This makes in-training eval report exact-z success.
        # For an arbitrary precomputed z, use sample_actions_with_z (no mapping).
        if self.config['actor_cond'] == 'z':
            goals = self._actor_z(goals)
        actions = self.network.select('actor')(observations, goals, noise)
        return jnp.clip(actions, -1.0, 1.0)

    @jax.jit
    def sample_actions_with_z(self, observations, z, seed=None, temperature=1.0):
        """Roll the z-actor from a PRECOMPUTED task vector z (no psi mapping).
        Used by the G4 eval script's inferred-z arm (z = E[r.psi(s')])."""
        del temperature
        event_rank = 1 if self.config['encoder'] is None else 3
        batch_shape = observations.shape[:-event_rank]
        noise = jax.random.normal(seed, (*batch_shape, self.config['action_dim']))
        actions = self.network.select('actor')(observations, z, noise)
        return jnp.clip(actions, -1.0, 1.0)

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)
        ex_goals = ex_observations
        action_dim = ex_actions.shape[-1]
        config = dict(config)
        config['action_dim'] = action_dim

        if config.get('anneal_iql_fade', False):
            # the crossfade only makes sense paired with something ramping IN as IQL fades OUT,
            # else only the actor loss survives. Two pairings: the SM measure ramp (bilinear
            # warm-start) or the SimpleSF reward-regression grounding (simplesf, cold curriculum).
            assert config.get('anneal_measure', False) or config.get('anneal_simplesf_ground', False), \
                "anneal_iql_fade requires anneal_measure or anneal_simplesf_ground (something must ramp in as IQL fades out)"
        if config.get('anneal_simplesf_ground', False):
            # SimpleSF curriculum: fade the true-reward TD out (anneal_iql_fade) while the
            # reward-regression grounding fades in over the same ramp. psi keeps the rung-2
            # architecture (NO stop-gradient), so the fade -- not a detach -- hands psi's training
            # from the TD to the reward regression. Endpoint = reward-regressed psi on a frozen
            # rung-2 trunk (the frozen-representation readout probe).
            assert config.get('anneal_iql_fade', False), \
                "anneal_simplesf_ground requires anneal_iql_fade (the true-reward TD must fade as the grounding grows)"
            assert config.get('value_arch', 'mlp') == 'simplesf', \
                "anneal_simplesf_ground requires value_arch='simplesf'"
            assert not config.get('simplesf_ground_psi', False), \
                "anneal_simplesf_ground grounds psi via the fade, NOT the stop-gradient; keep simplesf_ground_psi=False"
            assert int(config.get('anneal_ramp', 300_000)) > 0, "anneal_ramp must be > 0"
        if config.get('anneal_measure', False):
            # (1) we graft the SM loss onto the IQL losses; measure_only would replace them.
            assert not config.get('measure_only', False), \
                "anneal_measure grafts onto the IQL losses; measure_only=True replaces them"
            # (2) raw (non-scale-normed) measure grafts blew the Q scale and crashed 18/18 runs
            # (audit-20260706 section 1). Every surviving E2 cell used scale_norm.
            assert config.get('measure_scale_norm', False), \
                "anneal_measure requires measure_scale_norm=True (raw measure grafts blow the Q scale)"
            # (3) the value-legs measure is a different object; the dose-response is on the critic.
            assert config.get('measure_legs', 'critic') == 'critic', \
                "anneal_measure requires measure_legs='critic'"
            # (4) a zero-length ramp would divide by zero; lam_max may be 0 (the control cell).
            assert int(config.get('anneal_ramp', 300_000)) > 0, "anneal_ramp must be > 0"
            assert config.get('measure_coef', 0.0) >= 0.0, "measure_coef (lam_max) must be >= 0"
            # (5) PARAM-TREE IDENTITY with the warm-start checkpoint. Any of these changes the
            # parameter tree, so restore_agent would fail (or, worse, the prep script's splice
            # would). Refuse at construction.
            assert config.get('value_arch', 'mlp') == 'bilinear', \
                "anneal_measure requires value_arch='bilinear' (the checkpoint's architecture)"
            assert config.get('psi_base', 'learned') == 'learned', \
                "anneal_measure requires psi_base='learned' (psi_base changes the critic class)"
            for knob in ('fb_forward_z', 'fwd_two_tower', 'isfb', 'sf_feature_td'):
                assert not config.get(knob, False), \
                    f"anneal_measure is incompatible with {knob} (it changes the param tree)"
            assert config.get('measure_test_states', 'rho') == 'rho' \
                and config.get('measure_test_rank', 'full') == 'full', \
                "anneal_measure is incompatible with the task-norm knobs"

        if (config.get('measure_test_states', 'rho') != 'rho'
                or config.get('measure_test_rank', 'full') != 'full'):
            # (1) the rho diagonal IS the Dirac term; projecting there would double-count the reward.
            if config.get('measure_test_rank', 'full') == 'task':
                assert config.get('measure_test_states', 'rho') == 'goals', \
                    "measure_test_rank='task' requires measure_test_states='goals'"
            # (2) both knobs only affect measure_loss.
            assert config.get('measure_only', False) or config.get('measure_coef', 0.0) > 0, \
                "measure_test_states/measure_test_rank require measure_only=True or measure_coef>0"
            # (3) the value-legs branch has no goal leg to test against.
            assert config.get('measure_legs', 'critic') == 'critic', \
                "measure_test_states/measure_test_rank require measure_legs='critic'"
            # (4) these replace the loss entirely and would silently override the measure objective.
            assert not config.get('sf_feature_td', False), \
                "measure_test_states/measure_test_rank are incompatible with sf_feature_td"
            assert config.get('value_arch', 'mlp') != 'simplesf', \
                "measure_test_states/measure_test_rank are incompatible with value_arch='simplesf'"

        if config.get('simplesf_ground_psi', False) or config.get('value_arch', 'mlp') == 'simplesf':
            # (1) the grounding knob is meaningless without the architecture.
            assert config.get('value_arch', 'mlp') == 'simplesf', \
                "simplesf_ground_psi requires value_arch='simplesf'"
            # (2) every knob below selects its own critic class or replaces the loss entirely and
            # would silently override the SimpleSF architecture.
            for knob in ('sf_feature_td', 'measure_only', 'fb_forward_z', 'fwd_two_tower',
                         'isfb', 'sf_joint_psi'):
                assert not config.get(knob, False), f"value_arch='simplesf' is incompatible with {knob}"
            assert config.get('measure_coef', 0.0) == 0.0, \
                "value_arch='simplesf' is incompatible with measure_coef>0"
            # ortho_loss unpacks the critic's 3-tuple (v, phi, psi); GCSimpleSFValue returns a
            # 4-tuple (v, Psi, z, b), so this would raise. Refuse it at construction instead.
            assert config.get('iql_ortho_coef', 0.0) == 0.0, \
                "value_arch='simplesf' is incompatible with iql_ortho_coef>0 (ortho_loss expects a 3-tuple critic)"
            # (3) psi_base dispatches GCBilinearValue subclasses; the two must not compose silently.
            assert config.get('psi_base', 'learned') == 'learned', \
                "value_arch='simplesf' requires psi_base='learned'"
            # (4) the flow actor conditions on the raw goal in every cell.
            assert config.get('actor_cond', 'goal') == 'goal' and not config.get('actor_z_infer', False), \
                "value_arch='simplesf' requires actor_cond='goal'"

        if config.get('sf_feature_td', False):
            # phi = the ONLY learned leg (trained by sf_loss); goal enters via the readout, so
            # force actor_cond='goal' (avoids the z-dim guard below). Base leg:
            #   'identity' -> psi(g)=g, so latent_dim MUST == goal dim (d pinned to obs dim).
            #   'frozen'   -> psi = frozen random MLP, d is FREE (e.g. 50 to match FB/GCIQL).
            assert config.get('value_arch', 'mlp') == 'bilinear', "sf_feature_td requires value_arch='bilinear'"
            if config.get('sf_base', 'identity') == 'identity':
                goal_dim = ex_goals.shape[-1]
                assert config['value_latent_dim'] == goal_dim, \
                    f"sf_feature_td sf_base='identity' requires value_latent_dim==goal_dim ({goal_dim}), got {config['value_latent_dim']}"
            # E3 guard: reward regression on a FIXED base is the audited confound (the linear
            # proxy r=<s,g> cannot express the goal indicator) -> refuse.
            if config.get('sf_reward_regress', False):
                assert config.get('sf_base', 'identity') == 'learned', \
                    "sf_reward_regress requires sf_base='learned' (fixed bases can't express the goal indicator)"
            joint = config.get('sf_joint_psi', False)
            if joint:
                # (1) exactly ONE gradient path into psi.
                assert config.get('sf_base', 'identity') == 'learned', \
                    "sf_joint_psi requires sf_base='learned' (a fixed psi has no gradient to receive)"
                assert not config.get('sf_reward_regress', False), \
                    "sf_joint_psi and sf_reward_regress are mutually exclusive (two grad paths into psi)"
                # (2) without the ortho penalty, phi=0 and psi=0 is a GLOBAL minimum of the TD.
                assert config.get('sf_psi_ortho_coef', 0.0) > 0, \
                    "sf_joint_psi requires sf_psi_ortho_coef>0 (else phi=0,psi=0 is a global min of the SF TD)"
                # (3) a goal-dependent absorption coefficient cannot sit in a goal-agnostic
                # recursion. Scoped to joint cells: the legacy SF grid is agnostic AND masked,
                # and is left untouched (the objection is documented, not retro-enforced).
                if config.get('sf_mask', 'goal') == 'goal':
                    assert config.get('fb_forward_z', False), \
                        "sf_joint_psi with sf_mask='goal' requires fb_forward_z (goal-dependent mask needs a goal-conditioned phi)"
            # (4) never mix the arriving-state base with the current-index mask: GCDataset ties
            # rewards AND masks to the current index, so b=psi(s') + m_g(s) is off-by-one against
            # the mask (the class of bug fixed in E3, 2026-07-09).
            assert not (config.get('sf_base_index', 'current') == 'next'
                        and config.get('sf_mask', 'goal') == 'goal'), \
                "sf_base_index='next' requires sf_mask='none' (arriving-state base + current-index mask mixes conventions)"
            # (5) a learned psi with no training signal at all.
            if config.get('sf_base', 'identity') == 'learned':
                assert config.get('sf_reward_regress', False) or joint, \
                    "sf_base='learned' requires sf_reward_regress=True or sf_joint_psi=True (else psi has no training signal)"
            config['actor_cond'] = 'goal'

        if config.get('psi_base', 'learned') != 'learned':
            # Readout-vs-TD crux (sf-design-space 1b): cripple the goal leg EXACTLY as the SF
            # cells do, but keep GCIQL's scalar expectile TD. Only meaningful for the plain
            # bilinear critic -- every other branch below picks its own critic class and would
            # silently override psi_base, and sf_base already owns the base under sf_feature_td.
            assert config.get('value_arch', 'mlp') == 'bilinear', \
                "psi_base requires value_arch='bilinear' (an mlp critic has no goal leg to freeze)"
            assert not config.get('sf_feature_td', False), \
                "psi_base is for the SCALAR-TD control; under sf_feature_td use sf_base instead"
            for knob in ('fb_forward_z', 'fwd_two_tower', 'isfb'):
                assert not config.get(knob, False), f"psi_base is incompatible with {knob} (it selects its own critic class)"
            if config['psi_base'] == 'identity':
                goal_dim = ex_goals.shape[-1]
                assert config['value_latent_dim'] == goal_dim, \
                    f"psi_base='identity' requires value_latent_dim==goal_dim ({goal_dim}), got {config['value_latent_dim']}"

        # Spec 07 item 1 (task_prior_actor): actor trained across a z-prior in the FP corner.
        # 'mix' changes ONLY the actor's conditioning z (per-sample Dirac/mixture/random arm,
        # fully sg -- psi trains through the FP matrix loss alone, exactly as in the winning
        # cell); the FP matrix critic loss, its bootstrap and its rng stream are untouched.
        # This block sits BEFORE the ex_actor_goals sizing below because it forces the
        # z-conditioned actor form: under measure_only the value net never trains, so the
        # actor's z must read the measure-trained critic psi = B (the D1 lesson) -- forcing
        # actor_cond='z' + actor_z_from_critic HERE (the sf_feature_td config-mutation
        # pattern) keeps the launch yaml a single-knob move on the FP-NoOrtho cell.
        tpa = config.get('task_prior_actor', 'dirac')
        assert tpa in ('dirac', 'mix'), f"unknown task_prior_actor={tpa!r}"
        if tpa == 'mix':
            assert config.get('measure_only', False), \
                "task_prior_actor='mix' requires measure_only=True (the actor knob rides the FP matrix critic)"
            assert config.get('value_arch', 'mlp') == 'bilinear', \
                "task_prior_actor='mix' requires value_arch='bilinear' (z readout Q=phi.z/sqrt(d))"
            assert config.get('measure_first_passage', False), \
                "task_prior_actor='mix' requires measure_first_passage=True (occupancy floors -- there is nothing to extend)"
            assert config.get('actor_kind', 'flow') == 'flow', \
                "task_prior_actor='mix' requires actor_kind='flow'"
            assert config['encoder'] is None, \
                "task_prior_actor='mix' supports state obs only (mixture goals read cube xyz at obs[19:22])"
            assert config.get('measure_hit_spec', '19:22:0.4') == '19:22:0.4', \
                "task_prior_actor='mix' still hardcodes cube-single's obs[19:22]/eps=0.4 mixture reward (spec 08 E7 generalized only the measure_loss region); it would train a mis-specified reward under a non-cube-single measure_hit_spec"
            assert config.get('task_prior', 'dirac') == 'dirac', \
                "task_prior_actor='mix' is mutually exclusive with task_prior='mix' (the scalar-corner knob -- different grounding, disjoint namespaces)"
            assert not config.get('actor_z_infer', False), \
                "task_prior_actor='mix' owns the actor's z (the sampled arm); actor_z_infer would conflict"
            _tpa_probs = list(config.get('task_prior_actor_probs', [0.6, 0.2, 0.2]))
            assert len(_tpa_probs) == 3 and all(p >= 0 for p in _tpa_probs) \
                and abs(sum(_tpa_probs) - 1.0) < 1e-5, \
                f"task_prior_actor_probs must be 3 non-negative probs summing to 1, got {_tpa_probs}"
            assert int(config.get('task_prior_actor_mix_k', 2)) >= 1, \
                "task_prior_actor_mix_k must be >= 1"
            config['actor_cond'] = 'z'
            config['actor_z_from_critic'] = True
        if config.get('task_prior_critic_mix', False):
            assert tpa == 'mix', \
                "task_prior_critic_mix requires task_prior_actor='mix' (it grounds the critic at the actor's mixture z)"

        # --- campaign v2 lattice (both knobs default-off, bit-identical) ---
        _ms = config.get('measure_eps_multiscale', '')
        if _ms:
            _parse_eps_levels(_ms)                       # validate the level string loudly
            assert config.get('measure_first_passage', False), \
                "measure_eps_multiscale requires measure_first_passage=True (it scales the hit-ball radii)"
            assert config.get('measure_legs', 'critic') == 'critic', \
                "measure_eps_multiscale lives on the critic-legs measure path"
            assert config.get('value_arch', 'mlp') == 'bilinear', \
                "measure_eps_multiscale requires value_arch='bilinear' (per-column psi gains)"
            assert config.get('measure_only', False) or config.get('measure_coef', 0.0) > 0, \
                "measure_eps_multiscale needs an active measure loss (measure_only or measure_coef>0)"
            assert not config.get('measure_expectile', False), \
                "measure_eps_multiscale composes only with the single-matrix FP/ball-visit path (the expectile tower reads unguained psi columns)"
            assert config.get('measure_testpoint_dist', 'batch') == 'batch', \
                "measure_eps_multiscale requires measure_testpoint_dist='batch' (hindsight columns re-route M before the gain is applied)"
            assert config.get('task_prior', 'dirac') == 'dirac' \
                and config.get('task_prior_actor', 'dirac') == 'dirac' \
                and not config.get('task_prior_critic_mix', False), \
                "measure_eps_multiscale refuses the task_prior arms (they hardcode the 0.4 hit ball; same discipline as measure_hit_spec)"
        if config.get('measure_nhat_head', False):
            # phase 2b: trained kernel-mass denominator. Same corner as the other
            # v2 knobs; composes freely with actor_z_greedy_frac (disjoint losses).
            assert config.get('measure_first_passage', False), \
                "measure_nhat_head requires measure_first_passage=True (its target is the hit-matrix column mean)"
            assert config.get('measure_legs', 'critic') == 'critic', \
                "measure_nhat_head lives on the critic-legs measure path"
            assert config.get('value_arch', 'mlp') == 'bilinear', \
                "measure_nhat_head requires value_arch='bilinear' (input = the critic psi)"
            assert config.get('measure_only', False) or config.get('measure_coef', 0.0) > 0, \
                "measure_nhat_head needs an active measure loss (measure_only or measure_coef>0)"
            assert not config.get('measure_expectile', False), \
                "measure_nhat_head composes only with the single-matrix FP/ball-visit path"
            assert config.get('measure_testpoint_dist', 'batch') == 'batch', \
                "measure_nhat_head requires measure_testpoint_dist='batch' (hindsight columns are not rho samples -- the column mean would estimate the wrong mass)"
            assert not config.get('measure_eps_multiscale', ''), \
                "measure_nhat_head targets the SINGLE-radius column mean; the multiscale radii would mix kernel scales in one regression"
            assert config.get('task_prior', 'dirac') == 'dirac' \
                and config.get('task_prior_actor', 'dirac') == 'dirac' \
                and not config.get('task_prior_critic_mix', False), \
                "measure_nhat_head refuses the task_prior arms (same discipline as the other measure knobs)"
        _azg = float(config.get('actor_z_greedy_frac', 0.0))
        if _azg > 0:
            assert _azg <= 1.0, f"actor_z_greedy_frac must be in (0, 1], got {_azg}"
            assert config.get('actor_cond', 'goal') == 'z' and config.get('fb_forward_z', False), \
                "actor_z_greedy_frac requires actor_cond='z' + fb_forward_z (FB's greedy readout F(s,a,z).z)"
            assert config.get('actor_kind', 'flow') == 'flow', \
                "actor_z_greedy_frac is wired into the flow-actor branch of actor_loss"
            assert not config.get('actor_z_infer', False), \
                "actor_z_greedy_frac owns the actor's z mixing (and its z_rng split); actor_z_infer would conflict"
            assert config.get('task_prior', 'dirac') == 'dirac' \
                and config.get('task_prior_actor', 'dirac') == 'dirac', \
                "actor_z_greedy_frac is mutually exclusive with the task_prior arms (both own the actor's z)"

        # Actor goal-input example. In z-conditioning (G4) the actor consumes a
        # d-dim task vector z=psi(goal) (B := value net's psi), not the raw goal
        # observation, so its goal-leg params must be sized to value_latent_dim.
        if config.get('actor_cond', 'goal') == 'z' or config.get('actor_z_infer', False):
            assert config.get('value_arch', 'mlp') == 'bilinear', \
                "actor_cond='z'/actor_z_infer requires value_arch='bilinear' (B := value net's psi)"
            ex_actor_goals = jnp.zeros((ex_observations.shape[0], config['value_latent_dim']))
        else:
            ex_actor_goals = ex_goals

        # RG-FB v0 (actor_vf_cond): condition the BC vector field + distillation
        # rollout on the goal ('goal': raw goal concatenated, awr_flow parity) or
        # on the task vector ('z': sg(psi(goal)) -- the SAME z the one-shot actor
        # consumes). 'none' (default) keeps the stock goal-agnostic vf and a
        # byte-identical param tree, so every existing checkpoint keeps loading.
        vf_cond = config.get('actor_vf_cond', 'none')
        assert vf_cond in ('none', 'goal', 'z'), f"unknown actor_vf_cond={vf_cond!r}"
        if vf_cond != 'none':
            assert config.get('actor_kind', 'flow') == 'flow', \
                "actor_vf_cond requires actor_kind='flow' (ddpgbc has no vf; awr_flow's vf is already goal-conditioned)"
            assert config['encoder'] is None, \
                "actor_vf_cond supports state obs only (a pixel vf would need an encoded-goal concat path)"
            if vf_cond == 'z':
                # value_arch='bilinear' (an arch WITH a psi) is enforced transitively
                # by the actor_cond='z' guard above; 'goal' needs no psi and works on
                # any arch with a flow actor.
                assert config.get('actor_cond', 'goal') == 'z', \
                    "actor_vf_cond='z' requires actor_cond='z' (the vf consumes the same z=sg(psi(goal)) as the one-shot actor)"
                assert not config.get('actor_z_infer', False), \
                    "actor_vf_cond='z' conditions the vf on the exact psi(goal); actor_z_infer trains the actor on inferred z and would mismatch"

        if config.get('value_arch', 'mlp') == 'dueling':
            # GCDuelingValue is a NEW critic class (not a GCBilinearValue subclass swap
            # over an existing param tree) that reads phi/psi/b/U through its own
            # full_info=True path -- every knob below either reads the critic's info=True
            # 3-tuple with FB-specific semantics, picks its OWN critic class, or wires an
            # extra arg the dueling critic's __call__ doesn't accept. Refuse them at
            # construction rather than silently mis-composing.
            assert config.get('actor_kind', 'flow') == 'flow', \
                "value_arch='dueling' requires actor_kind='flow' (its A_bil-only q-term is wired into the flow-actor branch of actor_loss)"
            assert config.get('measure_coef', 0.0) == 0.0 and not config.get('measure_only', False), \
                "value_arch='dueling' is incompatible with the FB measure knobs (measure_coef/measure_only)"
            assert not config.get('sf_feature_td', False), \
                "value_arch='dueling' is incompatible with sf_feature_td (a different critic class owns phi)"
            assert config.get('psi_base', 'learned') == 'learned', \
                "value_arch='dueling' is incompatible with psi_base != 'learned' (dueling defines its own critic class)"
            assert not config.get('fb_forward_z', False), \
                "value_arch='dueling' is incompatible with fb_forward_z (Q=U+A_bil has no z-conditioned forward-map arg)"
            assert not config.get('fwd_two_tower', False) and not config.get('isfb', False), \
                "value_arch='dueling' is incompatible with fwd_two_tower/isfb (they select their own critic class)"
            assert config.get('actor_cond', 'goal') != 'z' and not config.get('actor_z_infer', False), \
                "value_arch='dueling' has no state-independent z=psi(g) task vector; actor_cond='z'/actor_z_infer are incompatible"
            assert config.get('iql_ortho_coef', 0.0) == 0.0, \
                "value_arch='dueling' is incompatible with iql_ortho_coef (ortho_loss's info=True psi has different semantics under dueling)"

        # T10 (centered measure) lattice: the centered loss reads the critic's info=True
        # 3-tuple with plain-bilinear semantics (phi(s,a), psi(g)), samples mu from the
        # goal-agnostic BC flow, and bootstraps DeltaM on the one-shot flow actor -- so
        # it composes with exactly ONE cell: plain bilinear + flow + raw-goal actor.
        cm = config.get('centered_measure', 'off')
        assert cm in ('off', 'aux', 'only'), f"unknown centered_measure={cm!r}"
        if cm != 'off':
            assert config.get('value_arch', 'mlp') == 'bilinear', \
                "centered_measure requires value_arch='bilinear' (DeltaM centers the plain bilinear phi)"
            assert config.get('psi_base', 'learned') == 'learned', \
                "centered_measure requires psi_base='learned' (a frozen/identity psi has different info semantics)"
            assert config.get('actor_kind', 'flow') == 'flow', \
                "centered_measure requires actor_kind='flow' (mu = the BC flow via _flow_rollout; a'~pi = the one-shot actor)"
            assert config.get('actor_vf_cond', 'none') == 'none', \
                "centered_measure requires actor_vf_cond='none' (b(s) regresses on the GOAL-AGNOSTIC behavior flow mu)"
            assert config.get('actor_cond', 'goal') == 'goal' and not config.get('actor_z_infer', False), \
                "centered_measure requires actor_cond='goal' (under 'only' the value net -- the z source -- never trains)"
            assert config['encoder'] is None, "centered_measure supports state obs only"
            for knob in ('fb_forward_z', 'fwd_two_tower', 'isfb', 'sf_feature_td', 'measure_only',
                         'anneal_measure', 'anneal_iql_fade', 'anneal_simplesf_ground'):
                assert not config.get(knob, False), f"centered_measure is incompatible with {knob}"
            assert config.get('measure_coef', 0.0) == 0.0, \
                "centered_measure is incompatible with the raw measure graft (measure_coef>0); it IS the measure term"
            assert config.get('iql_ortho_coef', 0.0) == 0.0, \
                "centered_measure owns the psi ortho term (measure_ortho_coef); iql_ortho_coef would double it"
            assert not config.get('freeze_critic_ckpt', ''), \
                "centered_measure adds a center_b module the freeze-splice list doesn't cover"
            assert int(config.get('center_b_samples', 8)) >= 1, "center_b_samples must be >= 1"
            assert config.get('centered_measure_coef', 1.0) >= 0.0, "centered_measure_coef must be >= 0"

        # Spec 06 measure-axis lattice (items 5/6/7). Each is a SINGLE-AXIS move on the plain
        # measure-only cell (bilinear critic, state-action SM on the critic legs, rho/full test
        # set). They compose with each other (6 masks on 5's hit; 7 swaps the test-point states)
        # but with nothing exotic -- refuse anything that changes the matrix's meaning.
        fp = config.get('measure_first_passage', False)
        exm = config.get('measure_expectile', False)
        tpd = config.get('measure_testpoint_dist', 'batch')
        assert tpd in ('batch', 'hindsight'), f"unknown measure_testpoint_dist={tpd!r}"
        if fp or exm or tpd != 'batch':
            assert config.get('value_arch', 'mlp') == 'bilinear', \
                "measure_first_passage/expectile/testpoint_dist require value_arch='bilinear'"
            assert config.get('measure_legs', 'critic') == 'critic', \
                "spec06 measure axes require measure_legs='critic' (the state-action SM)"
            assert config.get('measure_only', False) or config.get('measure_coef', 0.0) > 0, \
                "spec06 measure axes require measure_only=True or measure_coef>0"
            assert config.get('measure_test_states', 'rho') == 'rho' \
                and config.get('measure_test_rank', 'full') == 'full', \
                "spec06 measure axes require the single-matrix rho/full test set (not the task-norm knobs)"
            assert config.get('centered_measure', 'off') == 'off', \
                "spec06 measure axes are incompatible with centered_measure (a different critic object)"
            for knob in ('fwd_two_tower', 'isfb', 'sf_feature_td'):
                assert not config.get(knob, False), f"spec06 measure axes are incompatible with {knob} (single-axis only)"
            # (z,x)-separation (2026-07-23): fb_forward_z + measure_z_random_frac ARE allowed
            # with measure_first_passage -- rows then carry an independent policy index z_i
            # (goal embeddings mixed with random directions, z-consistent bootstrap) while the
            # hit columns carry the query x_j: the sampled two-argument object
            # M(s,a,z,x)=F(s,a,z).B(x). The exm/tpd-only cells keep the original guard.
            if not fp:
                for knob in ('fb_forward_z',):
                    assert not config.get(knob, False), f"spec06 measure axes are incompatible with {knob} (single-axis only)"
                assert config.get('measure_z_random_frac', 0.0) == 0.0, \
                    "spec06 measure axes are incompatible with measure_z_random_frac (single-axis only)"
            else:
                assert not (config.get('fb_forward_z', False) and exm), \
                    "fb_forward_z composes with the plain first-passage/visitation residual only (not the expectile-measure branch)"
            if exm and not fp:
                assert config.get('measure_ortho_coef', 0.0) > 0.0, \
                    "unmasked measure_expectile (no first-passage bound) needs the FB ortho anchor: set measure_ortho_coef>0 (T10-b2)"
                assert tpd == 'batch', \
                    "unmasked measure_expectile with hindsight test points has no coherent reward " \
                    "(the eye-diagonal visitation reward on random goals is a sample from no policy's " \
                    "future); compose with measure_first_passage or use measure_testpoint_dist='batch'"
            assert config.get('measure_expectile_tau', 0.9) is not None
        # Cell W knobs live inside the expectile-measure branch only.
        w_off = config.get('measure_offdiag_weight', 1.0)
        exm_src = config.get('measure_expectile_source', 'online')
        assert w_off >= 0.0, f"measure_offdiag_weight must be >= 0, got {w_off}"
        assert exm_src in ('online', 'target_min'), f"unknown measure_expectile_source={exm_src!r}"
        if w_off != 1.0 or exm_src != 'online':
            assert exm, "measure_offdiag_weight/measure_expectile_source require measure_expectile=True"

        # Spec 16 multi-step measure lattice: the k-step observed window composes with the
        # plain region (hit-spec) residual only -- refuse anything that changes the target's
        # meaning or the bootstrap policy's index.
        nstep = int(config.get('measure_nstep', 0) or 0)
        nlam = float(config.get('measure_nstep_lambda', 0.0) or 0.0)
        assert nstep >= 0, f"measure_nstep must be >= 0, got {nstep}"
        assert 0.0 <= nlam < 1.0, f"measure_nstep_lambda must be in [0,1), got {nlam}"
        if nlam > 0.0:
            assert nstep > 0, "measure_nstep_lambda requires measure_nstep>0 (K is the truncation cap)"
        if nstep > 0:
            assert config.get('measure_first_passage', False), \
                "measure_nstep requires measure_first_passage=True (the hit-spec region target; " \
                "ball-visitation via measure_fp_absorbing=False)"
            for knob in ('measure_expectile', 'measure_nhat_head', 'fb_forward_z'):
                assert not config.get(knob, False), \
                    f"measure_nstep is incompatible with {knob} (spec 16 round-1 lattice)"
            assert not config.get('measure_eps_multiscale', ''), \
                "measure_nstep is incompatible with measure_eps_multiscale (spec 16 round-1 lattice)"
            assert config.get('measure_z_random_frac', 0.0) == 0.0, \
                "measure_nstep is incompatible with measure_z_random_frac (z-indexed rows change the bootstrap policy)"
            assert config['encoder'] is None, "measure_nstep supports state observations only (window gather)"

        # Spec 17a delta (temporally-centered) measure lattice: the target reward becomes the
        # signed event indicator hit(x')-hit(x) on the existing hit-region path. Refuse anything
        # that changes the target's meaning, the bootstrap policy, or composes with 17b.
        if config.get('measure_delta', False):
            assert config.get('measure_first_passage', False), \
                "measure_delta rides the hit-region path: set measure_first_passage=True"
            assert not config.get('measure_fp_absorbing', True), \
                "measure_delta is non-absorbing by construction: set measure_fp_absorbing=False"
            assert config.get('measure_legs', 'critic') == 'critic', \
                "measure_delta lives on the critic-legs measure path"
            assert config.get('value_arch', 'mlp') == 'bilinear', \
                "measure_delta requires value_arch='bilinear'"
            assert config.get('measure_only', False) or config.get('measure_coef', 0.0) > 0, \
                "measure_delta needs an active measure loss"
            assert int(config.get('measure_nstep', 0) or 0) == 0, \
                "measure_delta x measure_nstep composition is refused in round 1 (spec 17 §7)"
            for bad in ('measure_expectile', 'measure_nhat_head', 'fb_forward_z'):
                assert not config.get(bad, False), f"measure_delta is incompatible with {bad}"
            assert not config.get('measure_eps_multiscale', ''), \
                "measure_delta x multiscale is untested; refused in round 1"
            assert config.get('task_prior', 'dirac') == 'dirac' \
                and config.get('task_prior_actor', 'dirac') == 'dirac', \
                "measure_delta refuses the task_prior arms (same discipline as the other measure knobs)"
            assert not config.get('measure_delta_partition', False), \
                "measure_delta and measure_delta_partition are mutually exclusive (17a vs 17b)"

        # Spec 17b (variant B): one-stage simplex successor measure over a fixed
        # K-prototype partition. Refuse anything that changes the target's meaning,
        # the bootstrap policy, or composes with the region-matrix / 17a paths.
        if config.get('measure_delta_partition', False):
            assert config.get('measure_only', False), \
                "measure_delta_partition (17b) replaces the measure loss: set measure_only=True"
            assert not config.get('measure_delta', False), \
                "measure_delta (17a) and measure_delta_partition (17b) are mutually exclusive"
            assert not config.get('measure_first_passage', False), \
                "measure_delta_partition does not use the region-matrix path: set measure_first_passage=False"
            assert config.get('actor_kind', 'flow') == 'flow', \
                "measure_delta_partition requires actor_kind='flow'"
            assert config.get('actor_cond', 'goal') == 'goal', \
                "measure_delta_partition requires actor_cond='goal'"
            assert config.get('actor_q_center', False), \
                "measure_delta_partition requires actor_q_center=True (p.z_g carries the " \
                "action-independent level phi(x0).z_g)"
            assert config['encoder'] is None, "measure_delta_partition supports state observations only"
            assert config.get('measure_delta_protos', ''), \
                "measure_delta_partition needs measure_delta_protos (.npy path)"
            assert config.get('measure_ortho_coef', 0.0) == 0.0, \
                "no ortho anywhere near measure_delta_partition (fp-noortho lesson)"
            for bad in ('fb_forward_z', 'sf_feature_td', 'measure_expectile', 'measure_nhat_head',
                        'actor_z_infer', 'fwd_two_tower', 'isfb', 'freeze_critic_ckpt', 'warm_start_ckpt'):
                assert not config.get(bad, False), \
                    f"measure_delta_partition is incompatible with {bad}"
            assert int(config.get('measure_nstep', 0) or 0) == 0, \
                "measure_delta_partition is incompatible with measure_nstep"
            assert not config.get('measure_eps_multiscale', ''), \
                "measure_delta_partition is incompatible with measure_eps_multiscale"
            assert config.get('centered_measure', 'off') == 'off', \
                "measure_delta_partition is incompatible with centered_measure"
            assert config.get('task_prior', 'dirac') == 'dirac', \
                "measure_delta_partition refuses the task_prior arms"
            assert config.get('task_prior_actor', 'dirac') == 'dirac', \
                "measure_delta_partition refuses the task_prior_actor arms"

        # RG-FB v1 task-prior lattice (spec 06 item 4). 'mix' evaluates the scalar IQL TD and
        # the flow actor AT THE SAMPLED z (a raw d-dim task vector), so it needs the bilinear
        # inner-product head + the z-conditioned actor/vf, on the plain scalar-TD cell (no
        # measure/SF/centered objective competing for the same critic).
        tp = config.get('task_prior', 'dirac')
        assert tp in ('dirac', 'mix'), f"unknown task_prior={tp!r}"
        if tp == 'mix':
            assert config.get('value_arch', 'mlp') == 'bilinear', \
                "task_prior='mix' requires value_arch='bilinear' (Q(s,a,z)=phi.z/sqrt(d))"
            assert config.get('actor_cond', 'goal') == 'z' and config.get('actor_vf_cond', 'none') == 'z', \
                "task_prior='mix' requires actor_cond='z' and actor_vf_cond='z' (actor conditions on the sampled z)"
            assert config.get('actor_kind', 'flow') == 'flow', "task_prior='mix' requires actor_kind='flow'"
            assert config['encoder'] is None, "task_prior='mix' supports state obs only (cube-xyz reward on obs[19:22])"
            assert config.get('measure_hit_spec', '19:22:0.4') == '19:22:0.4', \
                "task_prior='mix' still hardcodes cube-single's obs[19:22]/eps=0.4 mixture reward (spec 08 E7 generalized only the measure_loss region); it would train a mis-specified reward under a non-cube-single measure_hit_spec"
            assert not config.get('actor_z_infer', False), \
                "task_prior='mix' owns the actor's z (the sampled task); actor_z_infer would conflict"
            for knob in ('measure_only', 'sf_feature_td', 'fb_forward_z', 'fwd_two_tower', 'isfb'):
                assert not config.get(knob, False), f"task_prior='mix' is incompatible with {knob} (it owns the scalar critic)"
            assert config.get('measure_coef', 0.0) == 0.0 and config.get('centered_measure', 'off') == 'off', \
                "task_prior='mix' is incompatible with the measure/centered objectives (v1 grounds the scalar TD)"
            probs = list(config.get('task_prior_probs', [0.6, 0.2, 0.2]))
            assert len(probs) == 3 and all(p >= 0 for p in probs) and abs(sum(probs) - 1.0) < 1e-5, \
                f"task_prior_probs must be 3 non-negative probs summing to 1, got {probs}"
            assert int(config.get('task_prior_mix_k', 2)) >= 1, "task_prior_mix_k must be >= 1"

        # Encoders: IQL value/critic use the GCEncoder wrapper (as in gciql.py);
        # the flow nets take the raw encoder module (as in crl_flowbc.py).
        encoders = {}
        if config['encoder'] is not None:
            em = encoder_modules[config['encoder']]
            encoders['value'] = GCEncoder(concat_encoder=em())
            encoders['critic'] = GCEncoder(concat_encoder=em())
            encoders['actor_obs'] = em()
            encoders['actor_goal'] = em()

        # value_arch: 'mlp' = canonical GCIQL GCValue (concat MLP over [s,g,a]);
        # 'bilinear' = inner-product value V=phi(s).psi(g)/sqrt(d), Q=phi(s,a).psi(g)/sqrt(d)
        # (Direction A: GCIQL's IQL training poured through an FB-style rank-d head).
        if config.get('value_arch', 'mlp') == 'simplesf':
            # Simple SF (Chua et al.): one trunk -> (b, Psi); Q = <Psi, psi(g)>/sqrt(d).
            # Applied to BOTH the value net and the critic: grounding only the critic's psi would
            # leak goal information back through V(s',g) into the critic target.
            d = config['value_latent_dim']
            ss_kwargs = dict(
                hidden_dims=config['value_hidden_dims'], latent_dim=d,
                layer_norm=config['layer_norm'],
                ground_psi=config.get('simplesf_ground_psi', False),
                project_z=config.get('project_z', False),
            )
            value_def = GCSimpleSFValue(
                ensemble=False, state_encoder=encoders.get('value'),
                goal_encoder=encoders.get('value'), **ss_kwargs)
            critic_def = GCSimpleSFValue(
                ensemble=True, state_encoder=encoders.get('critic'),
                goal_encoder=encoders.get('critic'), **ss_kwargs)
        elif config.get('value_arch', 'mlp') == 'bilinear_sg':
            # State-conditioned backward leg: Q=phi(s,a).psi(s,g)/sqrt(d). Keeps the
            # inner-product head, destroys the low-rank GOAL bottleneck. Separate branch on
            # purpose: the 'bilinear' path below stays byte-identical, so every bridge
            # checkpoint on disk keeps loading.
            d = config['value_latent_dim']
            for bad in ('fb_forward_z', 'sf_feature_td', 'fwd_two_tower', 'isfb',
                        'measure_only', 'actor_z_infer'):
                assert not config.get(bad, False), \
                    f"value_arch='bilinear_sg' has no state-independent z=psi(g); '{bad}' " \
                    "reads psi as a task vector and is incompatible."
            assert config.get('actor_cond', 'goal') != 'z', \
                "value_arch='bilinear_sg' has no state-independent z=psi(g); actor_cond='z' " \
                "is incompatible."
            sg_kwargs = dict(
                hidden_dims=config['value_hidden_dims'], latent_dim=d,
                layer_norm=config['layer_norm'],
            )
            value_def = GCStatePsiValue(
                ensemble=False, state_encoder=encoders.get('value'),
                goal_encoder=encoders.get('value'), **sg_kwargs)
            critic_def = GCStatePsiValue(
                ensemble=True, state_encoder=encoders.get('critic'),
                goal_encoder=encoders.get('critic'), **sg_kwargs)
        elif config.get('value_arch', 'mlp') == 'latefusion':
            # T5a (ogbench-sota item 1): same twin towers as bilinear, but an MLP readout on
            # concat[phi;psi] instead of the inner product. Q is NOT bilinear -> no z=psi(g)
            # task readout; refuse every task-vector path.
            d = config['value_latent_dim']
            for bad in ('fb_forward_z', 'sf_feature_td', 'fwd_two_tower', 'isfb',
                        'measure_only', 'actor_z_infer'):
                assert not config.get(bad, False), \
                    f"value_arch='latefusion' has no bilinear z=psi(g); '{bad}' is incompatible."
            assert config.get('actor_cond', 'goal') != 'z', \
                "value_arch='latefusion' has no z=psi(g); actor_cond='z' is incompatible."
            assert config.get('measure_coef', 0.0) == 0.0 and config.get('centered_measure', 'off') == 'off', \
                "value_arch='latefusion' is incompatible with the measure objectives (Q is not bilinear)"
            lf_kwargs = dict(hidden_dims=config['value_hidden_dims'], latent_dim=d,
                             layer_norm=config['layer_norm'],
                             fusion_hidden=config.get('value_fusion_hidden', 512))
            value_def = GCLateFusionValue(ensemble=False, state_encoder=encoders.get('value'),
                                          goal_encoder=encoders.get('value'), **lf_kwargs)
            critic_def = GCLateFusionValue(ensemble=True, state_encoder=encoders.get('critic'),
                                           goal_encoder=encoders.get('critic'), **lf_kwargs)
        elif config.get('value_arch', 'mlp') == 'bilinear_sg_bounded':
            # T5 bounded-psi (ogbench-sota item 5): state-conditioned backward leg psi(s,g)
            # with a tanh OUTPUT BOUND -- targets the T2 diagnosis (no output bound + no
            # V-target + local pockets drove the online/offline BVN inversion).
            d = config['value_latent_dim']
            for bad in ('fb_forward_z', 'sf_feature_td', 'fwd_two_tower', 'isfb',
                        'measure_only', 'actor_z_infer'):
                assert not config.get(bad, False), \
                    f"value_arch='bilinear_sg_bounded' has no z=psi(g); '{bad}' is incompatible."
            assert config.get('actor_cond', 'goal') != 'z', \
                "value_arch='bilinear_sg_bounded' has no z=psi(g); actor_cond='z' is incompatible."
            assert float(config.get('psi_bound_scale', 0.0)) > 0.0, \
                "value_arch='bilinear_sg_bounded' requires psi_bound_scale>0 (the tanh bound = p99 of trained psi norms)"
            b_kwargs = dict(hidden_dims=config['value_hidden_dims'], latent_dim=d,
                            layer_norm=config['layer_norm'],
                            psi_bound_scale=float(config['psi_bound_scale']))
            value_def = GCStatePsiBoundedValue(ensemble=False, state_encoder=encoders.get('value'),
                                               goal_encoder=encoders.get('value'), **b_kwargs)
            critic_def = GCStatePsiBoundedValue(ensemble=True, state_encoder=encoders.get('critic'),
                                                goal_encoder=encoders.get('critic'), **b_kwargs)
        elif config.get('value_arch', 'mlp') == 'bilinear_split':
            # T5 scene split-head (ogbench-sota item 7): factored Q=sum_k phi_k.psi_k/sqrt(d_k).
            d = config['value_latent_dim']
            for bad in ('fb_forward_z', 'sf_feature_td', 'fwd_two_tower', 'isfb',
                        'measure_only', 'actor_z_infer'):
                assert not config.get(bad, False), \
                    f"value_arch='bilinear_split' has no single z=psi(g); '{bad}' is incompatible."
            assert config.get('actor_cond', 'goal') != 'z', \
                "value_arch='bilinear_split' has no single z=psi(g); actor_cond='z' is incompatible."
            goal_dim = ex_goals.shape[-1]
            sd_raw = config.get('value_split_dims', '')
            if isinstance(sd_raw, str):
                split_dims = [int(x) for x in sd_raw.replace('(', '').replace(')', '').split(',')
                              if x.strip()] or None
            else:
                split_dims = list(sd_raw) if sd_raw else None    # in-code list still accepted
            if split_dims:
                assert sum(split_dims) == goal_dim, \
                    f"value_split_dims {split_dims} must sum to goal_dim {goal_dim}"
                dims = list(split_dims)
            else:
                K = int(config.get('value_split_factors', 1))
                assert K >= 1, "value_split_factors must be >= 1"
                base, rem = divmod(goal_dim, K)
                dims = [base + (1 if i < rem else 0) for i in range(K)]   # equal-ish chunks
            bounds = [0]
            for dd in dims:
                bounds.append(bounds[-1] + dd)
            sp_kwargs = dict(hidden_dims=config['value_hidden_dims'], latent_dim=d,
                             layer_norm=config['layer_norm'], factor_bounds=tuple(bounds))
            value_def = GCSplitBilinearValue(ensemble=False, state_encoder=encoders.get('value'),
                                             goal_encoder=encoders.get('value'), **sp_kwargs)
            critic_def = GCSplitBilinearValue(ensemble=True, state_encoder=encoders.get('critic'),
                                              goal_encoder=encoders.get('critic'), **sp_kwargs)
        elif config.get('value_arch', 'mlp') == 'bilinear':
            d = config['value_latent_dim']
            # phi_norm_cap (item4b): swap in the norm-capped plain-bilinear class. Only valid on
            # the plain learned-psi bilinear critic (it caps the forward leg inside the inner
            # product; no z-task/measure/SF path, which own their own critic classes).
            cap = config.get('phi_norm_cap', False)
            if cap:
                assert config.get('psi_base', 'learned') == 'learned', \
                    "phi_norm_cap requires psi_base='learned'"
                for bad in ('fb_forward_z', 'sf_feature_td', 'fwd_two_tower', 'isfb'):
                    assert not config.get(bad, False), f"phi_norm_cap is incompatible with {bad}"
            learned_cls = GCNormCapValue if cap else GCBilinearValue
            # psi_base: cripple the goal leg under the NORMAL scalar IQL TD (1b control).
            # Applied SYMMETRICALLY to the value net and the critic -- freezing only the critic
            # would leave V(s,g) a fully-trained bilinear head and confound the readout.
            value_cls = {'learned': learned_cls, 'frozen': GCFrozenBValue,
                         'identity': GCIdentityBValue}[config.get('psi_base', 'learned')]
            value_def = value_cls(
                hidden_dims=config['value_hidden_dims'], latent_dim=d,
                layer_norm=config['layer_norm'], ensemble=False,
                state_encoder=encoders.get('value'), goal_encoder=encoders.get('value'),
            )
            # fb_forward_z: task-condition the critic's forward map -> phi(s,a,z), the
            # FB-faithful F(s,a,z).B(s'). Only valid as a measure-only critic (its
            # extra z arg would break the IQL value/critic Bellman losses).
            critic_cls = value_cls
            if config.get('sf_feature_td', False):
                # phi = successor-feature vector trained by sf_loss. Base leg: 'frozen' -> frozen
                # random psi (d free); 'identity' -> psi(g)=g (d==obs dim); 'learned' -> psi
                # trained by reward regression (sf_reward_regress, E3).
                critic_cls = {'identity': GCIdentityBValue, 'frozen': GCFrozenBValue,
                              'learned': GCLearnedBValue}[config.get('sf_base', 'identity')]
            elif config.get('fb_forward_z', False):
                # z-conditioned forward map F(s,a,z), z=sg(psi(g)). Runs under the measure
                # loss (FB objective) OR the IQL backup (B1): value_loss/critic_loss below
                # thread z_goals=value_goals so Q(s,a,g)=<F(s,a,psi(g)),psi(g)>/sqrt(d).
                # isfb: GCISFBValue adds the value-measure leg F_V(s,z) for SG-ISFB.
                critic_cls = GCISFBValue if config.get('isfb', False) else GCFBValue
            elif config.get('fwd_two_tower', False):
                # FB-style two-tower forward map F(s,a,z), z=sg(psi(g)), computed
                # internally -> drop-in for the IQL critic (architecture-only ablation).
                critic_cls = GCTwoTowerCritic
            crit_kwargs = dict(
                hidden_dims=config['value_hidden_dims'], latent_dim=d,
                layer_norm=config['layer_norm'], ensemble=True,
                state_encoder=encoders.get('critic'), goal_encoder=encoders.get('critic'),
            )
            if critic_cls in (GCFBValue, GCTwoTowerCritic, GCISFBValue):
                crit_kwargs['project_z'] = config.get('project_z', False)
            if critic_cls in (GCIdentityBValue, GCFrozenBValue, GCLearnedBValue):
                crit_kwargs['fb_forward_z'] = config.get('fb_forward_z', False)
            if critic_cls in (GCFrozenBValue, GCLearnedBValue):
                # NOT GCIdentityBValue: its _psi returns the raw goal, and sphering it would
                # destroy the identity base b(g)=g.
                crit_kwargs['project_z'] = config.get('project_z', False)
            critic_def = critic_cls(**crit_kwargs)
        elif config.get('value_arch', 'mlp') == 'dueling':
            # Behavior-centered dueling critic Q=U(s,g)+A_bil(s,a,g) (GCDuelingValue).
            # The critic is a NEW class (not a GCBilinearValue drop-in swap): only the
            # critic net gets it -- the value net stays the plain GCValue expectile V,
            # a DISTINCT object from the in-Q baseline U.
            d = config['value_latent_dim']
            value_def = GCValue(
                hidden_dims=config['value_hidden_dims'],
                layer_norm=config['layer_norm'],
                ensemble=False,
                gc_encoder=encoders.get('value'),
            )
            critic_def = GCDuelingValue(
                hidden_dims=config['value_hidden_dims'], latent_dim=d,
                layer_norm=config['layer_norm'], ensemble=True,
                state_encoder=encoders.get('critic'), goal_encoder=encoders.get('critic'),
            )
        elif config.get('value_arch', 'mlp') == 'bilinear_mlpv':
            # Ablation (ogbench-sota): bilinear Q + PLAIN-MLP V. Isolates whether V's
            # bilinearity is load-bearing, or only Q's low-rank goal bottleneck drives the
            # win. The IQL value/critic TD is unchanged in FORM; only the V head's arch
            # differs. MLP V has no psi -> no z=psi(g), so refuse every task-vector path.
            d = config['value_latent_dim']
            for bad in ('fb_forward_z', 'sf_feature_td', 'fwd_two_tower', 'isfb',
                        'measure_only', 'actor_z_infer'):
                assert not config.get(bad, False), \
                    f"value_arch='bilinear_mlpv': MLP V has no psi; '{bad}' is incompatible."
            assert config.get('actor_cond', 'goal') != 'z', \
                "value_arch='bilinear_mlpv': MLP V has no psi=z; actor_cond='z' is incompatible."
            assert config.get('measure_coef', 0.0) == 0.0 and config.get('centered_measure', 'off') == 'off', \
                "value_arch='bilinear_mlpv' is incompatible with the measure objectives (V has no bilinear legs)"
            assert config.get('psi_base', 'learned') == 'learned', \
                "value_arch='bilinear_mlpv' is incompatible with psi_base != 'learned'"
            value_def = GCValue(                       # PLAIN-MLP V(s,g)
                hidden_dims=config['value_hidden_dims'],
                layer_norm=config['layer_norm'], ensemble=False,
                gc_encoder=encoders.get('value'),
            )
            critic_def = GCBilinearValue(              # BILINEAR Q(s,a,g)=phi(s,a).psi(g)/sqrt(d)
                hidden_dims=config['value_hidden_dims'], latent_dim=d,
                layer_norm=config['layer_norm'], ensemble=True,
                state_encoder=encoders.get('critic'), goal_encoder=encoders.get('critic'),
            )
        elif config.get('value_arch', 'mlp') == 'occupancy':
            # Occupancy-IQL (occupancy.tex sec 6): ONE shared backward map B across V, Q1, Q2;
            # the scalar {-1,0} reward replaced by the CRL occupancy reward r_B(s',g). The
            # shared-B module IS the critic (V = actions=None branch); there is NO separate
            # value module, so B is genuinely shared. Everything else = standard GCIQL (expectile
            # V + flow actor). Refuse the task-vector/measure paths (occupancy owns the critic).
            d = config['value_latent_dim']
            for bad in ('fb_forward_z', 'sf_feature_td', 'fwd_two_tower', 'isfb',
                        'measure_only', 'measure_coef', 'actor_z_infer'):
                assert not config.get(bad, False), f"value_arch='occupancy' is incompatible with {bad}"
            assert config.get('centered_measure', 'off') == 'off', \
                "value_arch='occupancy' is incompatible with centered_measure"
            assert config.get('actor_kind', 'flow') == 'flow', "value_arch='occupancy' requires actor_kind='flow'"
            assert config.get('actor_cond', 'goal') == 'goal', \
                "value_arch='occupancy' requires actor_cond='goal' (no z=psi(g); the actor reads Q)"
            assert config.get('psi_base', 'learned') == 'learned', "value_arch='occupancy' needs psi_base='learned'"
            assert config['encoder'] is None, "value_arch='occupancy' supports state obs only"
            occ_kwargs = dict(hidden_dims=config['value_hidden_dims'], latent_dim=d,
                              layer_norm=config['layer_norm'])
            value_def = None                           # shared B lives inside the critic
            critic_def = GCOccValue(ensemble=True, state_encoder=encoders.get('critic'),
                                    goal_encoder=encoders.get('critic'), **occ_kwargs)
        else:
            value_def = GCValue(
                hidden_dims=config['value_hidden_dims'],
                layer_norm=config['layer_norm'],
                ensemble=False,
                gc_encoder=encoders.get('value'),
            )
            critic_def = GCValue(
                hidden_dims=config['value_hidden_dims'],
                layer_norm=config['layer_norm'],
                ensemble=True,
                gc_encoder=encoders.get('critic'),
            )
        # z-conditioned critic takes an extra z_goals arg (raw goal obs -> task vector z).
        critic_ex = ((ex_observations, ex_goals, ex_actions, ex_goals)
                     if config.get('fb_forward_z', False)
                     else (ex_observations, ex_goals, ex_actions))
        network_info = dict(
            critic=(critic_def, critic_ex),
            target_critic=(copy.deepcopy(critic_def), critic_ex),
        )
        if value_def is not None:                      # occupancy shares B inside the critic -> no 'value' module
            network_info['value'] = (value_def, (ex_observations, ex_goals))
        if cm != 'off':
            # T10 b(s): behavior-mean forward feature, an MLP head with phi's output
            # dim + ensemble structure (2 heads, matching the critic), trained ONLY by
            # the scale-normed regression in centered_measure_loss. Registered only
            # when the knob is on, so the default param tree stays byte-identical and
            # every existing checkpoint keeps loading.
            center_b_def = ensemblize(MLP, 2)(
                (*config['value_hidden_dims'], config['value_latent_dim']),
                activate_final=False, layer_norm=config['layer_norm'])
            network_info['center_b'] = (center_b_def, (ex_observations,))
        if config.get('measure_expectile', False):
            # spec06 item6: the state-only measure value tower phi_V(s) -> d, ensembled to
            # 2 heads to match the critic's [2,B,d] phi/psi. V_M(s,s+)=phi_V(s).psi~(s+)/sqrt(d).
            # Registered only when on, so the default param tree stays byte-identical.
            mv_def = ensemblize(MLP, 2)(
                (*config['value_hidden_dims'], config['value_latent_dim']),
                activate_final=False, layer_norm=config['layer_norm'])
            network_info['measure_v_phi'] = (mv_def, (ex_observations,))
        if config.get('measure_eps_multiscale', ''):
            # campaign v2 knob 1: the per-level psi gain table (init ONES). Registered
            # only when the knob is on, so the default param tree stays byte-identical;
            # warm_start splices every matching module and fresh-inits exactly this one.
            _n_lv = len(_parse_eps_levels(config['measure_eps_multiscale']))
            eps_gain_def = EpsGainTable(n_levels=_n_lv,
                                        latent_dim=config['value_latent_dim'])
            network_info['eps_gain'] = (eps_gain_def, (jnp.zeros((1,), jnp.int32),))
        if config.get('measure_nhat_head', False):
            # phase 2b: the trained N-hat denominator head. Registered only when on
            # (byte-identical default tree; warm_start fresh-inits it alone).
            network_info['nhat_head'] = (
                NhatHead(),
                (jnp.zeros((1, config['value_latent_dim']), jnp.float32),))
        delta_protos_np = None   # spec 17b: set below only when the knob is on
        if config.get('measure_delta_partition', False):
            # spec 17b: fixed K-prototype partition -- delta_head/target_delta_head learn
            # p(s,a,g) in Delta^K by CE TD against phi(x') (the partition, computed from
            # the loaded prototypes; no gradient). Registered only when the knob is on,
            # so the default param tree stays byte-identical.
            delta_protos_np = np.load(config['measure_delta_protos'])          # [K, d_c]
            _blocks17b_create = _parse_hit_spec(config.get('measure_hit_spec', '19:22:0.4'))
            _d_c17b = sum(b - a for (a, b, e) in _blocks17b_create)
            assert delta_protos_np.ndim == 2 and delta_protos_np.shape[1] == _d_c17b \
                and delta_protos_np.shape[0] >= 2, \
                f"measure_delta_protos must be [K>=2, {_d_c17b}], got {delta_protos_np.shape}"
            _K17b = int(delta_protos_np.shape[0])
            dh_def = ensemblize(MLP, 2)((*config['value_hidden_dims'], _K17b),
                                        activate_final=False, layer_norm=config['layer_norm'])
            ex_dh = (jnp.concatenate([ex_observations, ex_actions, ex_goals], -1),)
            network_info['delta_head'] = (dh_def, ex_dh)
            network_info['target_delta_head'] = (copy.deepcopy(dh_def), ex_dh)
        if config.get('actor_kind', 'flow') == 'ddpgbc':
            # Stock GCIQL gaussian DDPG+BC actor on the (possibly bilinear) critic.
            actor_def = GCActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                state_dependent_std=False,
                const_std=config['const_std'],
                gc_encoder=encoders.get('actor_obs'),
            )
            network_info['actor'] = (actor_def, (ex_observations, ex_goals))
        elif config.get('actor_kind', 'flow') == 'awr_flow':
            # AWR flow readout: a single GOAL-CONDITIONED velocity field (goal
            # concatenated into the vf's obs input, as in gcivl_flow). State-only;
            # no one-step actor, no distillation.
            assert config['encoder'] is None, "awr_flow supports state obs only"
            assert config.get('actor_cond', 'goal') == 'goal' and not config.get('fb_forward_z', False), \
                "awr_flow is incompatible with z-conditioning paths"
            actor_vf_def = GCVectorField(
                hidden_dim=config['actor_vf_hidden_dim'],
                hidden_layers=config['actor_vf_hidden_layers'],
                obs_encoder=None,
            )
            ex_t = jnp.zeros((ex_actions.shape[0], 1))
            ex_obs_goal = jnp.concatenate([ex_observations, ex_goals], axis=-1)
            network_info['actor_vf'] = (actor_vf_def, (ex_obs_goal, ex_actions, ex_t))
        else:
            actor_vf_def = GCVectorField(
                hidden_dim=config['actor_vf_hidden_dim'],
                hidden_layers=config['actor_vf_hidden_layers'],
                obs_encoder=encoders.get('actor_obs'),
            )
            actor_def = GCNoiseActor(
                hidden_dim=config['actor_hidden_dim'],
                hidden_layers=config['actor_hidden_layers'],
                embedding_layers=config['actor_embedding_layers'],
                obs_encoder=encoders.get('actor_obs'),
                goal_encoder=encoders.get('actor_goal'),
            )
            ex_noise = jnp.zeros_like(ex_actions)
            ex_t = jnp.zeros((ex_actions.shape[0], 1))
            # RG-FB (actor_vf_cond): size the vf's obs example to the conditioned
            # input. 'goal' appends the raw goal obs; 'z' appends the d-dim task
            # vector (ex_actor_goals is zeros(B, value_latent_dim) under
            # actor_cond='z', which actor_vf_cond='z' requires).
            if vf_cond == 'goal':
                ex_vf_obs = jnp.concatenate([ex_observations, ex_goals], axis=-1)
            elif vf_cond == 'z':
                ex_vf_obs = jnp.concatenate([ex_observations, ex_actor_goals], axis=-1)
            else:
                ex_vf_obs = ex_observations
            network_info['actor_vf'] = (actor_vf_def, (ex_vf_obs, ex_actions, ex_t))
            network_info['actor'] = (actor_def, (ex_observations, ex_actor_goals, ex_noise))
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}
        network_def = ModuleDict(networks)
        # anneal_lr_warmup: linear LR warmup, then constant. Protects the restored weights from a
        # large first step, and lets Adam's fresh moments adapt before the SM gradient grows.
        # It CANNOT be faked by scaling the loss (Adam is ~scale-invariant per parameter), so it
        # must be a real LR schedule -- which changes the opt_state tree and is why such a run can
        # only restore from a checkpoint written by scripts/prepare_anneal_ckpt.py.
        warmup = int(config.get('anneal_lr_warmup', 0))
        if warmup > 0:
            lr_sched = optax.linear_schedule(init_value=0.0, end_value=config['lr'],
                                             transition_steps=warmup)
            network_tx = optax.adam(learning_rate=lr_sched)
        else:
            network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']

        assert not (config.get('freeze_critic_ckpt', '') and config.get('warm_start_ckpt', '')), \
            "warm_start_ckpt and freeze_critic_ckpt are mutually exclusive (both splice modules)"
        # Exp 2/3: splice a frozen trained critic into the fresh-actor tree + mask the optimizer.
        if config.get('freeze_critic_ckpt', ''):
            import glob as _glob
            import pickle as _pickle
            from flax.core import unfreeze as _unfreeze
            assert not config.get('anneal_measure', False) and not config.get('anneal_iql_fade', False), \
                "freeze_critic_ckpt is incompatible with the anneal warm-start path"
            _cand = _glob.glob(config['freeze_critic_ckpt'])
            assert len(_cand) == 1, f"freeze_critic_ckpt matched {len(_cand)}: {_cand}"
            _pkl = f"{_cand[0]}/params_{int(config['freeze_critic_epoch'])}.pkl"
            with open(_pkl, 'rb') as _f:
                _ck = _pickle.load(_f)['agent']['network']['params']
            _FREEZE = ('modules_value', 'modules_critic', 'modules_target_critic')
            network_params = _unfreeze(network_params)
            for _k in _FREEZE:
                assert _k in _ck, f"checkpoint missing {_k}; freeze_critic config must match its architecture"
                network_params[_k] = _ck[_k]          # frozen trained critic/value; actor stays fresh-init
            print(f"[freeze_critic] spliced {_FREEZE} from {_pkl}; actor re-initialised, "
                  f"optimizer masked to actor-only", flush=True)

            def _freeze_labels(params):
                return {k: jax.tree_util.tree_map(lambda _l: ('freeze' if k in _FREEZE else 'train'), v)
                        for k, v in params.items()}
            network_tx = optax.multi_transform(
                {'train': network_tx, 'freeze': optax.set_to_zero()}, _freeze_labels)

        # Warm-start: splice a TRAINED checkpoint's modules in, ALL trainable (fresh optimizer).
        if config.get('warm_start_ckpt', ''):
            import glob as _glob
            import os as _os
            import pickle as _pickle
            from flax.core import unfreeze as _unfreeze
            assert not config.get('freeze_critic_ckpt', ''), \
                "warm_start_ckpt and freeze_critic_ckpt are mutually exclusive (both splice modules)"
            assert not config.get('anneal_measure', False) and not config.get('anneal_iql_fade', False), \
                "warm_start_ckpt is a fresh-optimizer warm-start, not the anneal ramp path"
            _cand = _glob.glob(_os.path.expanduser(config['warm_start_ckpt']))
            assert len(_cand) == 1, f"warm_start_ckpt matched {len(_cand)}: {_cand}"
            _pkl = f"{_cand[0]}/params_{int(config['warm_start_epoch'])}.pkl"
            with open(_pkl, 'rb') as _f:
                _ck = _pickle.load(_f)['agent']['network']['params']
            network_params = _unfreeze(network_params)
            _copied, _fresh = [], []
            for _k in list(network_params.keys()):
                if _k not in _ck:
                    _fresh.append(_k); continue
                _fl = jax.tree_util.tree_leaves(network_params[_k])
                _cl = jax.tree_util.tree_leaves(_ck[_k])
                if len(_fl) == len(_cl) and all(a.shape == b.shape for a, b in zip(_fl, _cl)):
                    network_params[_k] = _ck[_k]; _copied.append(_k)
                else:
                    _fresh.append(f"{_k}(shape-mismatch)")
            assert 'modules_critic' in _copied and 'modules_value' in _copied, \
                f"warm_start spliced neither critic nor value (copied={_copied}); arch must match the checkpoint"
            print(f"[warm_start] spliced {_copied} from {_pkl}; fresh-init {_fresh}; ALL trainable", flush=True)

        # freeze_actor (spec 08 E2): mask the optimizer to zero-out ALL actor modules
        # (modules_actor + modules_actor_vf) so phase 2 is fixed-policy visitation
        # evaluation -- pi stays at the warm-started phase-1 policy while phi trains against
        # the frozen backward map. Mirror of the freeze_critic_ckpt masking above.
        if config.get('freeze_actor', False):
            assert not config.get('freeze_critic_ckpt', ''), \
                "freeze_actor and freeze_critic_ckpt both mask the optimizer (opposite submodules); combining them freezes everything"
            _FREEZE_ACTOR = tuple(k for k in network_params if 'actor' in k)
            assert _FREEZE_ACTOR, "freeze_actor set but no actor modules in the param tree"
            _actor_base_tx = network_tx

            def _actor_freeze_labels(params):
                return {k: jax.tree_util.tree_map(lambda _l: ('freeze' if k in _FREEZE_ACTOR else 'train'), v)
                        for k, v in params.items()}
            network_tx = optax.multi_transform(
                {'train': _actor_base_tx, 'freeze': optax.set_to_zero()}, _actor_freeze_labels)
            print(f"[freeze_actor] optimizer masked to freeze {_FREEZE_ACTOR}; phi/psi trainable", flush=True)

        # actor_freeze_steps (§9.16j): DELAYED actor -- zero the actor modules' gradients
        # BEFORE adam for the first N optimizer steps (m=v=0 => update exactly 0, adam state
        # effectively fresh at unfreeze), then train normally. Shields the actor through the
        # phase-1 -> phase-2 objective migration; the critic trains from step 0 throughout.
        if config.get('actor_freeze_steps', 0):
            assert not config.get('freeze_actor', False), \
                "actor_freeze_steps is the delayed version of freeze_actor; combining is ambiguous"
            assert not config.get('freeze_critic_ckpt', ''), \
                "actor_freeze_steps + freeze_critic_ckpt would freeze everything for N steps"
            _DELAY_ACTOR = tuple(k for k in network_params if 'actor' in k)
            assert _DELAY_ACTOR, "actor_freeze_steps set but no actor modules in the param tree"
            _N_FRZ = int(config['actor_freeze_steps'])
            _delay_base_tx = network_tx

            def _actor_delay_labels(params):
                return {k: jax.tree_util.tree_map(lambda _l: ('delayed' if k in _DELAY_ACTOR else 'train'), v)
                        for k, v in params.items()}
            _gate = optax.scale_by_schedule(lambda count: jnp.where(count < _N_FRZ, 0.0, 1.0))
            network_tx = optax.multi_transform(
                {'train': _delay_base_tx, 'delayed': optax.chain(_gate, _delay_base_tx)},
                _actor_delay_labels)
            print(f"[actor_freeze_steps] {_DELAY_ACTOR} gradient-gated for the first {_N_FRZ} steps", flush=True)

        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network_params
        params['modules_target_critic'] = params['modules_critic']
        if config.get('measure_delta_partition', False):
            params['modules_target_delta_head'] = params['modules_delta_head']

        # nonpytree/static field (CRITICAL 2 fix): a hashable nested tuple of Python
        # floats, NOT a jnp/np array -- see the field's docstring for why.
        delta_protos = (tuple(tuple(float(v) for v in row) for row in delta_protos_np)
                        if delta_protos_np is not None else None)
        return cls(rng, network=network, config=flax.core.FrozenDict(**config),
                  delta_protos=delta_protos)


def get_config():
    config = _gciql_get_config()
    with config.unlocked():
        config.agent_name = 'gciql_flow'
        # Value architecture: 'mlp' (canonical GCIQL) | 'bilinear' (inner-product, Direction A)
        # | 'simplesf' (Chua et al.: trunk -> b=h/||h|| and Psi=head(h); Q=<Psi,psi(g)>/sqrt(d))
        # | 'dueling' (behavior-centered dueling critic: Q=U(s,g)+A_bil(s,a,g), A_bil=
        #   (phi(s,a)-b(s)).psi(g)/sqrt(d); see GCDuelingValue).
        config.value_arch = 'mlp'
        config.dueling_b_coef = 1.0  # weight on the dueling baseline regression L_b=||b(s)-sg(phi(s,a_data))||^2 (critic_loss); value_arch='dueling' only
        config.simplesf_ground_psi = False  # rung 3 (the paper's cell): stop-grad z in the value/critic losses so psi(g) is trained ONLY by the reward-prediction loss L_rew
        config.simplesf_reward_coef = 1.0   # weight on L_rew, summed over the value net and the critic
        config.value_latent_dim = 512
        # --- ogbench-sota implementations (mechanism cells; new value_archs are opt-in) ---
        # value_arch='latefusion' (item1): twin towers phi(s,a),psi(g) but an MLP readout on
        #   concat[phi;psi] instead of the inner product (Layer-1 control). value_fusion_hidden
        #   sizes that readout MLP.
        # value_arch='bilinear_sg_bounded' (item5): psi(s,g) with a tanh output bound
        #   scale*tanh(psi/scale); psi_bound_scale = the bound (p99 of trained psi norms).
        # value_arch='bilinear_split' (item7, scene): Q=sum_k phi_k(s,a).psi_k(g_k)/sqrt(d)
        #   with per-factor goal towers; value_split_factors=K, value_latent_dim=d_k per factor.
        # phi_norm_cap (item4b): on plain bilinear, hard-cap ||phi||<=sg[p99 batch ||phi||].
        # value_arch='bilinear_mlpv' (ablation): bilinear Q + PLAIN-MLP V -- isolates whether
        # V's bilinearity is load-bearing or only Q's low-rank goal bottleneck drives the win.
        config.value_fusion_hidden = 512   # latefusion readout MLP width
        config.psi_bound_scale = 0.0       # bilinear_sg_bounded tanh bound (>0 required for that arch)
        config.phi_norm_cap = False        # item4b: cap ||phi|| to sg[p99 over batch] (plain bilinear only)
        config.value_split_factors = 1     # bilinear_split: number of goal factors K (equal goal chunks; d_k = value_latent_dim each)
        config.value_split_dims = ''       # bilinear_split: explicit per-factor goal dims as a comma-string "19,9,4,4,2,2" (summing to goal_dim); '' -> equal split by value_split_factors. String (not list) so it overrides via the ml_collections flag
        # value_arch='occupancy' (occupancy.tex): shared-B occupancy-IQL. Reward r_B(s',g)=
        # (1-gamma)*sg[B_tgt(s').B_tgt(g)]/sqrt(d); target y=r_B+gamma*occ_env_mask*V(s',g);
        # B whitened by occ_ortho_coef*||E[BB^T]-I||. Keeps the IQL expectile V + flow actor.
        config.occ_ortho_coef = 1.0        # whitening penalty on the shared backward map B (C_B ~ I)
        config.occ_env_mask = 1.0          # environment-continuation mask (cube: no early termination => 1.0)
        # Actor conditioning identity (G4 objective-isolation arm):
        # 'goal' = raw goal obs (stock); 'z' = z=stop_grad(value.psi(goal)),
        # FB-style single backward map. 'z' requires value_arch='bilinear'.
        config.actor_cond = 'goal'
        # RG-FB v0 (PAPER/rescuing-fb/02-rgfb-spec.md §2.1): condition the BC flow
        # vector field AND the distillation rollout. 'none' = stock goal-agnostic
        # behavioral anchor (bit-identical, default) | 'goal' = raw goal concat
        # (awr_flow parity) | 'z' = sg(psi(goal)), the SAME task vector the one-shot
        # z-actor consumes (requires actor_cond='z'; flow actor + state obs only).
        config.actor_vf_cond = 'none'
        # Clipped double-Q: True = min(q1,q2) (stock); False = q1 only (pessimism ablation).
        config.double_q = True
        # Actor family: 'flow' (FlowBC + Q-max distillation, the custom actor),
        # 'ddpgbc' (stock GCIQL gaussian actor, no-flow-actor control; needs
        # actor_loss=ddpgbc), or 'awr_flow' (goal-conditioned flow-BC weighted by
        # exp(awr_alpha*(minQ - V)) -- IQL's AWR extraction on a flow policy).
        config.actor_kind = 'flow'
        # Flow actor hyperparameters.
        config.flow_steps = 10  # Euler steps for the BC flow rollout / distillation.
        config.bc_coeff = 3.0  # Distillation (BC) coefficient; the key actor knob.
        config.actor_q_center = False  # flow actor q-loss normalizer: sg[mean|q - mean(q)|] (CENTERED scale, affine-offset invariant) instead of sg[mean|q|]. Cell W's M=1+(1-gamma)Q offset makes |q|~1 while the controllable variation is (1-gamma)-scaled, diluting the greedy signal ~3.8x vs bc_loss (wave-2 forensics 2026-07-22). Default off = bit-identical
        config.awr_alpha = 10.0  # AWR temperature for actor_kind='awr_flow' (OGBench GCIVL value).
        config.actor_vf_hidden_dim = 512
        config.actor_vf_hidden_layers = 4
        config.actor_hidden_dim = 512
        config.actor_hidden_layers = 2
        config.actor_embedding_layers = 2
        config.measure_coef = 0.0  # FB measure loss weight grafted onto GCIQL's value rep (0 = stock GCIQL)
        config.measure_legs = 'critic'  # 'critic' = state-action SM on Q legs (FB-faithful) | 'value' = state SM on V legs
        config.measure_only = False  # True = drop IQL value/critic losses; critic trained ONLY by the SM (FB objective)
        config.measure_ortho_coef = 0.0  # FB orthonormality coef lambda on psi (||E[psi psi^T]-I||); FB uses 1000
        config.measure_scale_norm = False  # E2: divide the measure TD terms by sg[mean(M^2)] so the gradient is dimensionless (stops the <F,z> inflation that crashed 18/18 mc* runs; audit §1)
        config.measure_stopgrad_phi = False  # E2: stop-grad phi inside measure_loss so the measure gradient shapes ONLY psi (readout/backward leg), quarantined from the value-critical forward leg
        config.measure_stopgrad_psi = False  # phase-2 decomposition (2026-07-23): stop-grad EVERY online psi column in measure_loss so the measure TD trains ONLY phi against a FROZEN backward map. Use with warm_start_ckpt (psi from the FP-NoOrtho 0.967 cell) + measure_only (else the IQL losses still train psi) + measure_ortho_coef=0. Tests whether occupancy TD fails because it CORRUPTS the representation (frozen-psi occupancy works) or because the target is unlearnable on play data (still floors)
        config.measure_z_random_frac = 0.0  # FB-fidelity: fraction of measure-loss task vectors z drawn as sqrt(d)-sphere gaussian directions (metamotivo sample_mixed_z) instead of psi(vgoals); needs fb_forward_z
        # --- Spec 06 measure-axis knobs (single-axis moves on the plain measure-only cell) ---
        config.measure_first_passage = False  # item5: first-passage (absorbing) residual delta=M-hit-gamma(1-hit)sg[Mbar], hit=1[||cube(s')-cube(s+)||<=0.4 obs units (=4cm)]; diagonal REGRESSED (squared, hit is a representable reward -- no linear attraction); per-column fixed point bounded in [0,1]
        config.measure_hit_spec = '19:22:0.4'  # spec 08 E7: the goal REGION of the first-passage/ball-visitation hit, as comma-separated 'start:stop:eps' obs blocks ANDed together. Default = cube-single's cube xyz within 4cm (bit-identical to the hardcoded literal it replaces). cube-double: '19:22:0.4,28:31:0.4'; scene: '19:22:0.4,28:30:0.5,32:34:0.5,36:37:0.72,38:39:0.6'. String, not a list -- ml_collections rejects list overrides on the Hydra CLI
        config.measure_fp_absorbing = True    # False (needs measure_first_passage): drop the (1-hit) absorption mask => discounted BALL-VISITATION measure M=E[sum gamma^t 1[ball(s+)]] -- genuine occupancy semantics (revisits counted) with the FP cell's estimator hygiene (regressed 0/1 reward, gamma-contraction, fixed point <= 1/(1-gamma)). The occupancy-on-FP repair after the FB-form linear-diagonal divergence (occfz 2026-07-23: q_abs 154->5e4 on frozen psi). Default True = bit-identical first-passage
        config.measure_nstep = 0   # spec 16 (PAPER/Multistep-FB): k-step window cap K for the region-measure TD target -- diff = M - sum_{u<=k_i} gamma^{u-1} h(s_{t+u}) - gamma^{k_i} cont * sg[tM(s_{t+k_i})]. Moves the dwell mass into the OBSERVED term (one-step TD rebuilds it by bootstrap iteration and collapses to the dwell constant on sparse regions, ledger 9.16); off-policy bias of the window is gated by gamma^{T_int} (dwell-invariance lemma). Requires measure_first_passage (absorbing or ball-visitation); GCDataset supplies mstep_next_obs [B,K,obs] + mstep_k [B]. 0 = stock one-step path, bit-identical
        config.measure_nstep_lambda = 0.0  # spec 16: 0 = fixed k_i=min(K, steps-to-final); in (0,1) = k_i ~ Geometric(1-lambda) truncated at K, i.e. the sampled lambda-return (contraction modulus gamma(1-lambda)/(1-lambda*gamma)); mean horizon 1/(1-lambda). Needs measure_nstep>0
        config.measure_delta = False  # spec 17a: delta (temporally-centered) region measure -- the
          # target reward becomes hit(x')-hit(x) (enter +1 / dwell 0 / exit -1), non-absorbing,
          # fixed point J = (1-gamma)*M_visitation - h(x0), bounded in [-1,1], same action argmax as
          # the visitation measure. Requires measure_first_passage=True (the hit-region path) +
          # measure_fp_absorbing=False; refuses nstep/expectile/multiscale/nhat/task_prior.
          # False = off, bit-identical.
        config.measure_delta_partition = False  # spec 17b: one-stage simplex successor measure over a
          # fixed K-prototype partition of the hit-spec object blocks. delta_head([s,a,g]) -> K logits,
          # p=softmax; CE fit to sg[(1-gamma)*phi(x') + gamma*p_target(s',a'~pi,g)]; actor q = p . phi(goal)
          # at frozen head params. Requires measure_only + flow actor + actor_q_center + state obs;
          # mutually exclusive with measure_delta (17a). False = off, bit-identical.
        config.measure_delta_protos = ''   # 17b: path to the [K, d_c] .npy prototype file (raw obs units,
          # rows = dataset draws of the concatenated measure_hit_spec blocks). Required when the knob is on.
        config.measure_delta_coef = 1.0    # 17b: weight on the CE loss (keep 1.0; no dose sweeps round 1)
        config.measure_expectile = False      # item6: state-only measure value V_M=phiV(s).psi~(s+)/sqrt(d); per-column expectile fit ell2_tau(sg[M]-V_M) + backup M<-r_delta+gamma*mask*sg[V_M(s')] (NO a'). Composes with measure_first_passage (supplies r_delta=hit and (1-hit) mask). Adds the measure_v_phi tower ONLY when on
        config.measure_expectile_tau = 0.9    # expectile for the measure-value fit (item6)
        config.measure_testpoint_dist = 'batch'  # item7: off-diagonal test points s+. 'batch' -> psi(next_observations), the rho-referenced measure (bit-identical) | 'hindsight' -> psi(value_goals), the OGBench scalar-side mix (0.5 traj-future/0.3 random/0.2 current); NO importance correction (intentional -- object is no longer the rho-referenced measure)
        # --- Cell W (the theorem-backed win cell; needs measure_expectile) ---
        config.measure_offdiag_weight = 1.0  # weight on OFF-DIAGONAL (unpaired-column) entries of the expectile-measure losses. 1.0 = uniform matrix (corner cell); ~0.3 = GC-IQL-like effective task mix; 0.0 = diagonal-only, which with tpd='hindsight'+fp is EXACTLY bilinear GC-IQL under M=1+(1-gamma)Q (delta_M=(1-gamma)*delta_Q identically) -- the guaranteed-win successor measure
        config.measure_expectile_source = 'online'  # 'online' (sg online M_act; the corner cell) | 'target_min' (min over TARGET-critic heads at (s,a_data) -- the anchor's double-DQN split; Cell W)
        # --- T10 centered (advantage) successor measure (PAPER/rescuing-fb/04-centered-measure-spec.md) ---
        config.centered_measure = 'off'  # 'off' (stock, byte-identical) | 'aux' (T10-a: centered-measure DeltaM TD as an auxiliary on the scalar-IQL grounding) | 'only' (T10-b: no scalar TD; SARSA-anchored base measure TD + pi-bootstrapped centered DeltaM TD)
        config.center_b_samples = 8      # K flow samples per state for the b(s) regression target sg[mean_{a~mu} phi(s,a)] (mu = the frozen goal-agnostic BC flow)
        config.center_b_coef = 1.0       # weight on the scale-normed b(s) regression loss
        config.centered_measure_coef = 1.0  # lambda_DeltaM: weight on the centered-measure TD leg (spec doses {0.3, 1})
        config.centered_split_bootstrap = True  # T10-b base leg bootstrap: True -> a' = DATASET next action (SARSA, the contractive fixed-policy operator; needs batch['next_actions'] via OGBENCH_NEXT_ACTIONS=1) | False -> a' ~ pi (the non-contraction control)
        config.centered_scale_anchors = False  # T10-b2: with NO scalar TD, scale-norming both measure legs is scale-INVARIANT and the representation inflates without bound (observed q_mean +3e13). True restores FB's anchors: sphere-norm psi (||psi||=sqrt(d)) everywhere + RAW (un-normalized) regression on the SARSA base leg (+ set measure_ortho_coef>0). No effect on 'aux'/'off' paths beyond the DeltaM psi-sphere when explicitly enabled.
        config.anneal_measure = False    # ramp the SM loss weight instead of holding it constant: lam(t) = measure_coef * clip(step/anneal_ramp, 0, 1), then HOLD. measure_coef is reinterpreted as lam_max. Warm-start finetuning only (see scripts/prepare_anneal_ckpt.py, which resets network.step to 0)
        config.anneal_ramp = 300_000     # linear ramp length in updates; lam holds at lam_max afterwards
        config.anneal_lr_warmup = 0      # linear LR warmup length in updates (0 = constant LR, today's behaviour). NOTE: a nonzero value changes the optax opt_state tree, so such a run can only restore from a checkpoint written by prepare_anneal_ckpt.py
        config.anneal_iql_fade = False   # crossfade the IQL value+critic losses OUT (weight 1->0 over anneal_ramp) as the measure ramps IN -> endpoint is measure-only, warm-started. Tests fixed-point vs IQL-propping. Requires anneal_measure or anneal_simplesf_ground
        config.anneal_simplesf_ground = False  # SimpleSF curriculum: pair with anneal_iql_fade to fade the true-reward TD OUT while the reward-regression grounding (L_rew) fades IN over anneal_ramp. Cold-start SimpleSF (value_arch='simplesf', simplesf_ground_psi=False). Endpoint = reward-regressed psi on a frozen rung-2 trunk. Tests optimization-path vs representability of rung 3
        config.measure_test_states = 'rho'   # distribution the measure residual is TESTED against: 'rho' -> psi(next_observations), FB-faithful | 'goals' -> psi(value_goals), the task distribution nu (the research note's task norm). The ortho and the Dirac term ALWAYS stay on psi(next_observations)
        config.measure_test_rank = 'full'    # test set rank: 'full' -> B x B cross-pairs (i != j) | 'task' -> diagonal, project each residual onto its OWN task vector z_i = psi(g_i) (rank 1). 'task' requires measure_test_states='goals'
        config.fb_forward_z = False  # task-condition the critic's forward map: phi(s,a,z), z=sg(psi(goal)) -> true F(s,a,z).B(s') (needs bilinear; runs under measure OR the IQL backup, B1)
        config.psi_base = 'learned'   # goal leg psi(g) under the SCALAR IQL TD (readout-vs-TD control, sf-design-space 1b): 'learned' -> trained (stock bilinear GCIQL) | 'frozen' -> frozen random MLP (d free) | 'identity' -> psi(g)=g (d==goal dim). Applied to value net AND critic; needs bilinear, excludes sf_feature_td/fb_forward_z/fwd_two_tower
        config.sf_feature_td = False  # SUCCESSOR-FEATURE vector TD on phi: phi(s,a)<-b(s)+gamma*phi(s',a'), Q=<phi,psi(g)> (train ONLY phi; needs bilinear)
        config.sf_coef = 1.0          # weight on the SF vector-TD loss (sf_feature_td)
        config.sf_base = 'identity'   # SF base feature b: 'identity' -> b(s)=s (d==obs dim); 'frozen' -> b(s)=psi(s) frozen random MLP (d free, e.g. 50); 'learned' -> psi trained by reward regression (needs sf_reward_regress)
        config.sf_reward_regress = False  # E3 corrected SF: learn the base by reward regression r_hat(s,g)=<psi(s),psi(g)>/sqrt(d) on batch rewards (current-state convention: GCDataset ties reward+mask to idxs==value_goal_idxs); requires sf_base='learned'
        config.sf_reward_coef = 1.0   # weight on the reward-regression loss (sf_reward_regress)
        config.sf_reward_shift = 0.0  # regress on r+shift; use 1.0 with OGBench {-1,0} rewards to escape the psi=0 saddle of the symmetric bilinear reward model (argmax-invariant)
        config.sf_ortho_coef = 0.0    # aux orthonormality reg ||E[phi phi^T]-I|| on the SF vector phi (anti-collapse; sf_feature_td only)
        config.sf_psi_ortho_coef = 0.0  # FB orthonormality ||E_rho[psi psi^T]-I|| on the BASE leg psi(s') (FB uses 1000). DISTINCT from sf_ortho_coef, which penalizes phi. Required by sf_joint_psi (else phi=0,psi=0 is a global min of the SF TD)
        config.sf_joint_psi = False   # train psi by the SF vector TD itself (un-stop-grad the base b). Requires sf_base='learned' + sf_psi_ortho_coef>0; mutually exclusive with sf_reward_regress
        config.sf_base_index = 'current'  # immediate feature: 'current' -> b(s)=psi(s) (GCDataset ties rewards+masks to the current index) | 'next' -> b(s)=psi(s') (FB's arriving-state measure convention). Never mix 'next' with sf_mask='goal'
        config.sf_mask = 'goal'       # bootstrap gate: 'goal' -> batch['masks'], absorbing goal (Q=gamma^T, hitting time) | 'none' -> m==1 (Q=sum gamma^t 1[s_t=g], visitation density; FB's measure_loss has no mask)
        config.fwd_two_tower = False  # FB-style two-tower forward map F=trunk(MLP[s,a],MLP[s,z]) on the IQL bilinear critic (z=sg(psi(g)); architecture-only ablation, no measure loss)
        config.iql_ortho_coef = 0.0  # FB orthonormality ||E[psi psi^T]-I|| on the IQL bilinear critic's psi, with NO measure loss (isolates the ortho constraint on the ~90 IQL bilinear)
        config.project_z = False  # hard-project z and B=psi onto the sqrt(d) sphere (FB's ||B||=||z||=sqrt(d) Norm); applies to GCFBValue/GCTwoTowerCritic
        config.actor_z_from_critic = False  # source the flow actor's z from the CRITIC psi (=B, measure-trained), not the value net (audit #2 / D1); needs actor_cond='z'
        config.actor_z_infer = False  # exp4: train the flow actor on the INFERRED task dist z=E[r*psi] (sparse-reward readout over the batch), not exact psi(g); needs bilinear
        # --- Exp 2/3 (bilinear-mechanism-program): train a FRESH actor against a RESTORED,
        # FROZEN critic. Splice the checkpoint's value/critic/target_critic into a fresh-init
        # param tree (actor stays random-init) and mask the optimizer so ONLY actor params move
        # (set_to_zero on critic/value). Isolates the critic's actor-training FIELD from
        # actor-critic co-adaptation. The config MUST match the checkpoint's critic architecture
        # (value_arch, value_latent_dim, ...) or the splice tree mismatches. Empty = off.
        config.freeze_critic_ckpt = ''     # glob path to the run dir holding params_<epoch>.pkl
        config.freeze_critic_epoch = 0     # checkpoint step to load the frozen critic from
        # --- warm-start (2026-07-21): splice a TRAINED scalar-Q checkpoint in, keep it TRAINABLE ---
        # Like freeze_critic but (1) splices EVERY module the checkpoint shares with the fresh tree
        # (so the measure cell's extra towers -- measure_v_phi/center_b -- are handled: spliced if
        # present, fresh-init otherwise), and (2) leaves the optimizer UNMASKED. Tests whether the
        # good Q basin, once carved, SURVIVES the measure objective (cold-start never reaches it;
        # the whole offdiag sweep incl. w=0 sat at the floor). Glob must match exactly one run dir.
        config.warm_start_ckpt = ''
        config.warm_start_epoch = 0
        config.freeze_actor = False        # spec 08 E2: mask the optimizer to freeze ALL actor modules (modules_actor + modules_actor_vf) -> phase 2 becomes fixed-policy visitation evaluation (pi held at the warm-started phase-1 policy while phi trains). Default-off is bit-identical.
        config.actor_freeze_steps = 0      # §9.16j: delayed actor -- gradient-gate ALL actor modules to zero for the first N optimizer steps (adam sees zero grads => update exactly 0, fresh-adam at unfreeze), then train normally. Shields the actor through the phase-1->phase-2 objective migration. 0 = off, bit-identical. Refuses freeze_actor/freeze_critic_ckpt.
        config.measure_z_random_frac_anneal = False   # spec 08 E6: linearly ramp the random-z probability 0 -> measure_z_random_frac over anneal_ramp steps, then hold (ZX-form z-coverage schedule). Only the bernoulli prob is annealed; knob-off is bit-identical.
        # --- campaign v2 (2026-07-27): two default-off knobs, both bit-identical when off ---
        config.measure_eps_multiscale = ''  # knob 1: comma-string of hit-ball radii (obs units), e.g. '0.2,0.4,0.8'. Per gradient step each COLUMN draws one level: the FP/ball-visit hit uses that radius (measure_hit_spec blocks scaled by level/0.4) and the psi column is multiplied by a LEARNED per-level gain (EpsGainTable, init ones -> M starts at the base). Needs critic-legs + measure_first_passage + bilinear; refuses expectile/hindsight/task_prior. '' = off, bit-identical (no eps_gain module registered)
        config.actor_z_greedy_frac = 0.0    # knob 2: per-sample prob of replacing the flow actor's conditioning z by a random sqrt(d)-sphere direction (measure_z_random_frac's z_rand construction) AND reading those samples' q as FB's native greedy q=F(s,a,z).z/sqrt(d) (z_vec path); non-selected samples keep the diagonal F.B(g) readout exactly. Needs actor_cond='z' + fb_forward_z + flow actor. 0.0 = off, bit-identical
        config.measure_nhat_head = False    # phase 2b (2026-07-27): train the kernel-mass denominator N-hat(g)=softplus(MLP_2x256(sg[psi(g)])) by plain regression onto sg[hit-matrix column means] (unbiased MC of rho(B_eps(g)); coefficient 1.0, NO TD). Gradient reaches ONLY the head (psi input + target both sg) -- every existing loss term stays bit-identical. Needs critic-legs + measure_first_passage + bilinear; refuses expectile/hindsight/multiscale/task_prior; composes with actor_z_greedy_frac. False = off, bit-identical (no nhat_head module registered)
        # --- RG-FB v1 (spec 06 item 4): task-prior scalar grounding. Default 'dirac' => v0 ---
        config.task_prior = 'dirac'  # 'dirac' (default, bit-identical v0: the single OGBench goal task) | 'mix' (per-sample task drawn from [Dirac goal, K-goal max-mixture (gc_negative reward), random-linear reward]); the scalar IQL TD trains AT THE SAMPLED z with psi-grads ON (targets sg); the actor consumes the sg'd z_actor (Dirac rows from actor_goals, D8 split). Needs value_arch='bilinear', actor_cond='z', actor_vf_cond='z', flow actor, state obs
        config.task_prior_probs = [0.6, 0.2, 0.2]  # categorical [p_dirac, p_mixture, p_linear] over the per-sample task type
        config.task_prior_mix_k = 2   # K goals per mixture task (from the batch next-states, Dirichlet(1) weights)
        # --- Spec 07 item 1 (task_prior_actor): actor z-prior in the FP corner. Default 'dirac' => bit-identical FP cell ---
        config.task_prior_actor = 'dirac'  # 'dirac' (default, bit-identical FP-NoOrtho cell) | 'mix' (per-sample ACTOR-ONLY z drawn from [Dirac sg[psi(actor_goal)], K-goal Dirichlet(1) mixture of batch next-state psi's renormed to the match-data norm, uniform-sphere random * mean||psi(s')||]); the FP matrix critic loss, its bootstrap and its rng stream are UNTOUCHED; the actor z is FULLY sg in every arm (psi trains only through the FP matrix loss). Forces actor_cond='z' + actor_z_from_critic (the D1 z-actor form of the FP cell). Needs measure_only + measure_first_passage + bilinear + flow actor + state obs; mutually exclusive with task_prior='mix'
        config.task_prior_actor_probs = [0.6, 0.2, 0.2]  # categorical [p_dirac, p_mixture, p_random] over the actor's per-sample z arm (kept equal to v1's task_prior_probs for comparability)
        config.task_prior_actor_mix_k = 2  # K goals per mixture z (from the batch next-states, Dirichlet(1) weights)
        # --- Spec 07 item 4 (task_prior_critic_mix): the linearity-conflict probe, 1 seed ---
        config.task_prior_critic_mix = False  # ADD a scalar first-passage TD grounding at the mixture z on top of the UNTOUCHED matrix loss: r=max_k hit over the K mixture goals ([0,1] convention, NOT gc_negative), absorbing target r+gamma*(1-r)*sg[min_h Q_tgt(s',a'~pi(s',z_mix),z_mix)], Q=phi.z/sqrt(d); z_mix fully sg (trains phi only). Pre-registered prediction: HURTS the Dirac anchor (E[gamma^T_{A u B}] is not linear in the Dirac columns). Requires task_prior_actor='mix'
    return config
