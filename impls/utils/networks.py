"""FB networks — transcribed verbatim from PSMFlows utils/{networks,psm_networks,fb_networks}.py
for bit-exact parity with the PyTorch reference (nn_models.py). Combined into one module.

Reference weight_init: nn.Linear -> orthogonal(gain=1); DenseParallel -> orthogonal(gain=relu≈√2).
flax LayerNorm uses epsilon=1e-5 to match torch's default. GELU is exact (erf), not the tanh approx.
"""
import math
from typing import Any, Optional, Sequence

import distrax
import jax
import flax.linen as nn
import jax.numpy as jnp


# ── ensemble primitive (from utils/networks.py) ──────────────────────────────
def ensemblize(cls, num_qs, in_axes=None, out_axes=0, **kwargs):
    """Ensemblize a module via nn.vmap."""
    return nn.vmap(
        cls,
        variable_axes={'params': 0, 'intermediates': 0},
        split_rngs={'params': True},
        in_axes=in_axes,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs,
    )


# ── inits (from utils/psm_networks.py) ───────────────────────────────────────
_ORTH1 = nn.initializers.orthogonal()                          # gain 1 (plain Linear)
_ORTH_RELU = nn.initializers.orthogonal(scale=math.sqrt(2.0))  # ReLU gain (ensembled towers)


def truncated_clamp(x, low=-1.0, high=1.0, eps=1e-6):
    """Straight-through clamp from the reference TruncatedNormal._clamp."""
    clamped = jnp.clip(x, low + eps, high - eps)
    return x - jax.lax.stop_gradient(x) + jax.lax.stop_gradient(clamped)


def truncated_sample(loc, scale, noise, clip=None, low=-1.0, high=1.0, eps=1e-6):
    """Reference TruncatedNormal.sample with externally supplied standard-normal noise."""
    e = noise * scale
    if clip is not None:
        e = jnp.clip(e, -clip, clip)
    return truncated_clamp(loc + e, low, high, eps)


def psm_norm(x):
    """Reference `Norm`/`_L2`: sqrt(dim) * x / ||x||_2 (torch F.normalize eps=1e-12)."""
    d = x.shape[-1]
    denom = jnp.maximum(jnp.linalg.norm(x, axis=-1, keepdims=True), 1e-12)
    return jnp.sqrt(d) * x / denom


class PhiMap(nn.Module):
    """phi(goal) -> R^z_dim. Sequence: Dense, ntanh, [Dense, relu]*(L-1), Dense, [norm]."""

    z_dim: int
    hidden_dim: int
    hidden_layers: int = 2
    norm: bool = True

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim, kernel_init=_ORTH1)(x)
        x = nn.LayerNorm(epsilon=1e-5)(x)
        x = jnp.tanh(x)
        for _ in range(self.hidden_layers - 1):
            x = nn.Dense(self.hidden_dim, kernel_init=_ORTH1)(x)
            x = nn.relu(x)
        x = nn.Dense(self.z_dim, kernel_init=_ORTH1)(x)
        if self.norm:
            x = psm_norm(x)
        return x


def _simple_embedding(x, hidden_dim, embedding_layers):
    """Reference nn_models.simple_embedding (num_parallel=1): Linear->LayerNorm->Tanh,
    (embedding_layers-2)x[Linear->ReLU], Linear(hidden//2)->ReLU. Returns hidden//2."""
    assert embedding_layers >= 2, "must have at least 2 embedding layers"
    x = jnp.tanh(nn.LayerNorm(epsilon=1e-5)(nn.Dense(hidden_dim, kernel_init=_ORTH1)(x)))
    for _ in range(embedding_layers - 2):
        x = nn.relu(nn.Dense(hidden_dim, kernel_init=_ORTH1)(x))
    return nn.relu(nn.Dense(hidden_dim // 2, kernel_init=_ORTH1)(x))


class NoiseConditionedActor(nn.Module):
    """Port of nn_models.NoiseConditionedActor (the FlowBC one-step actor head).

    a = tanh(policy(concat[ embed_s([obs,noise]), embed_z([obs,z,noise]) ])).
    """

    action_dim: int
    hidden_dim: int = 512
    hidden_layers: int = 2
    embedding_layers: int = 2

    @nn.compact
    def __call__(self, obs, z, noise):
        z_embedding = _simple_embedding(jnp.concatenate([obs, z, noise], -1),
                                        self.hidden_dim, self.embedding_layers)
        s_embedding = _simple_embedding(jnp.concatenate([obs, noise], -1),
                                        self.hidden_dim, self.embedding_layers)
        h = jnp.concatenate([s_embedding, z_embedding], -1)
        for _ in range(self.hidden_layers):
            h = nn.relu(nn.Dense(self.hidden_dim, kernel_init=_ORTH1)(h))
        return jnp.tanh(nn.Dense(self.action_dim, kernel_init=_ORTH1)(h))


class FlowVectorField(nn.Module):
    """Port of nn_models.VectorField (SimpleVectorField): unconditional flow velocity
    v(obs, x_t, t). Linear->GELU, (hidden_layers-1)x[Linear->GELU], Linear(adim)."""

    action_dim: int
    hidden_dim: int = 512
    hidden_layers: int = 4

    @nn.compact
    def __call__(self, obs, action, t):
        x = jnp.concatenate([obs, action, t], -1)
        # torch nn.GELU() is the exact (erf) gelu; flax nn.gelu defaults to the tanh approx.
        x = nn.gelu(nn.Dense(self.hidden_dim, kernel_init=_ORTH1)(x), approximate=False)
        for _ in range(self.hidden_layers - 1):
            x = nn.gelu(nn.Dense(self.hidden_dim, kernel_init=_ORTH1)(x), approximate=False)
        return nn.Dense(self.action_dim, kernel_init=_ORTH1)(x)


# ── FB forward/backward maps (from utils/fb_networks.py) ──────────────────────
# BackwardMap and left_encoder are PhiMaps.
BackwardMap = PhiMap


class _ForwardTower(nn.Module):
    """One (non-ensembled) forward-map tower. embed_z over [obs_feat, z], embed_sa over
    [obs_feat, action]; concat -> `hidden_layers` x [Dense->relu] -> Dense(z_dim)."""

    z_dim: int
    hidden_dim: int
    hidden_layers: int = 2
    embedding_layers: int = 2

    def setup(self):
        assert self.embedding_layers == 2, "only embedding_layers=2 supported (reference default)"
        h = self.hidden_dim
        self.embed_z_0 = nn.Dense(h, kernel_init=_ORTH_RELU)
        self.embed_z_ln = nn.LayerNorm(epsilon=1e-5)
        self.embed_z_3 = nn.Dense(h // 2, kernel_init=_ORTH_RELU)
        self.embed_sa_0 = nn.Dense(h, kernel_init=_ORTH_RELU)
        self.embed_sa_ln = nn.LayerNorm(epsilon=1e-5)
        self.embed_sa_3 = nn.Dense(h // 2, kernel_init=_ORTH_RELU)
        # `hidden_layers` trunk linears (each relu'd) + a final output linear (no relu).
        self.fs = [nn.Dense(h, kernel_init=_ORTH_RELU) for _ in range(self.hidden_layers)] \
            + [nn.Dense(self.z_dim, kernel_init=_ORTH_RELU)]

    def __call__(self, obs_feat, z, action):
        ze = nn.relu(self.embed_z_3(jnp.tanh(self.embed_z_ln(
            self.embed_z_0(jnp.concatenate([obs_feat, z], -1))))))
        sa = nn.relu(self.embed_sa_3(jnp.tanh(self.embed_sa_ln(
            self.embed_sa_0(jnp.concatenate([obs_feat, action], -1))))))
        x = jnp.concatenate([sa, ze], -1)
        for layer in self.fs[:-1]:
            x = nn.relu(layer(x))
        return self.fs[-1](x)


class ForwardMap(nn.Module):
    """Ensembled forward map F(left_enc(obs), z, action) -> [num_parallel, B, z_dim]."""

    z_dim: int
    hidden_dim: int
    num_parallel: int = 2
    hidden_layers: int = 2
    embedding_layers: int = 2

    @nn.compact
    def __call__(self, obs_feat, z, action):
        tower = ensemblize(_ForwardTower, self.num_parallel, in_axes=None)(
            z_dim=self.z_dim, hidden_dim=self.hidden_dim,
            hidden_layers=self.hidden_layers, embedding_layers=self.embedding_layers,
            name="tower",
        )
        return tower(obs_feat, z, action)


class FBTd3Actor(nn.Module):
    """TD3 actor (reference nn_models.Actor): mu = tanh(policy(concat[embed_s(obs),
    embed_z([obs,z])])). Carried for the td3 path; unused on the flowbc cube path."""

    action_dim: int
    hidden_dim: int
    hidden_layers: int = 2
    embedding_layers: int = 2

    @nn.compact
    def __call__(self, obs, z):
        ze = _simple_embedding(jnp.concatenate([obs, z], -1), self.hidden_dim, self.embedding_layers)
        se = _simple_embedding(obs, self.hidden_dim, self.embedding_layers)
        h = jnp.concatenate([se, ze], -1)
        for _ in range(self.hidden_layers):
            h = nn.relu(nn.Dense(self.hidden_dim, kernel_init=_ORTH1)(h))
        return jnp.tanh(nn.Dense(self.action_dim, kernel_init=_ORTH1)(h))


# ── Gaussian actor / value / flow-VF nets (from utils/networks.py, for PR2 actors) ──
def default_init(scale=1.0):
    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Any = nn.gelu
    activate_final: bool = False
    kernel_init: Any = default_init()
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
        return x


class TransformedWithMode(distrax.Transformed):
    def mode(self):
        return self.bijector.forward(self.distribution.mode())


class Actor(nn.Module):
    """Gaussian actor network (distrax MultivariateNormalDiag)."""

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    log_std_min: Optional[float] = -5
    log_std_max: Optional[float] = 2
    tanh_squash: bool = False
    state_dependent_std: bool = False
    const_std: bool = True
    final_fc_init_scale: float = 1e-2

    def setup(self):
        self.actor_net = MLP(self.hidden_dims, activate_final=True, layer_norm=self.layer_norm)
        self.mean_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        if self.state_dependent_std:
            self.log_std_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        else:
            if not self.const_std:
                self.log_stds = self.param('log_stds', nn.initializers.zeros, (self.action_dim,))

    def __call__(self, observations, temperature=1.0):
        outputs = self.actor_net(observations)
        means = self.mean_net(outputs)
        if self.state_dependent_std:
            log_stds = self.log_std_net(outputs)
        else:
            log_stds = jnp.zeros_like(means) if self.const_std else self.log_stds
        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)
        distribution = distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds) * temperature)
        if self.tanh_squash:
            distribution = TransformedWithMode(distribution, distrax.Block(distrax.Tanh(), ndims=1))
        return distribution


class Value(nn.Module):
    """Value/critic network. Ensembled MLP -> scalar (V(s,g) or Q(s,a,g))."""

    hidden_dims: Sequence[int]
    layer_norm: bool = True
    num_ensembles: int = 2

    def setup(self):
        mlp_class = MLP
        if self.num_ensembles > 1:
            mlp_class = ensemblize(mlp_class, self.num_ensembles)
        self.value_net = mlp_class((*self.hidden_dims, 1), activate_final=False, layer_norm=self.layer_norm)

    def __call__(self, observations, actions=None):
        inputs = [observations]
        if actions is not None:
            inputs.append(actions)
        inputs = jnp.concatenate(inputs, axis=-1)
        return self.value_net(inputs).squeeze(-1)


class GCValue(nn.Module):
    """Goal-conditioned value/critic, faithful to OGBench GCValue (state-based, no encoder).

    ensemble=False -> V(s, g) shape (B,);  ensemble=True -> Q(s, g, a) shape (2, B).
    Uses the shared MLP (default nn.gelu + default nn.LayerNorm) so it is numerically
    identical to the vendored GCValue. Submodule name `value_net` matches vendored.
    """
    hidden_dims: Sequence[int]
    layer_norm: bool = True
    ensemble: bool = True

    def setup(self):
        mlp_class = MLP
        if self.ensemble:
            mlp_class = ensemblize(mlp_class, 2)
        self.value_net = mlp_class((*self.hidden_dims, 1), activate_final=False, layer_norm=self.layer_norm)

    def __call__(self, observations, goals=None, actions=None):
        inputs = [observations]
        if goals is not None:
            inputs.append(goals)
        if actions is not None:
            inputs.append(actions)
        inputs = jnp.concatenate(inputs, axis=-1)
        return self.value_net(inputs).squeeze(-1)


class ActorVectorField(nn.Module):
    """Actor vector field for flow matching. Input [obs, action(, time)] -> velocity."""

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False

    def setup(self):
        self.mlp = MLP((*self.hidden_dims, self.action_dim), activate_final=False, layer_norm=self.layer_norm)

    @nn.compact
    def __call__(self, observations, actions, times=None):
        if times is None:
            inputs = jnp.concatenate([observations, actions], axis=-1)
        else:
            inputs = jnp.concatenate([observations, actions, times], axis=-1)
        return self.mlp(inputs)


class GCBilinearValue(nn.Module):
    """Goal-conditioned bilinear value/critic, faithful to OGBench GCBilinearValue.

    V(s, g) = phi(s)·psi(g) / sqrt(d)   (actions=None)
    Q(s, a, g) = phi(s, a)·psi(g) / sqrt(d)   (actions given)

    phi and psi are both MLP((*hidden_dims, latent_dim), activate_final=False), ensembled
    to 2 heads when ensemble=True. Submodule names `phi`/`psi` match vendored. The encoder
    arguments of the vendored class are omitted (state-based only, as with GCValue here).

    The /sqrt(latent_dim) scale is load-bearing: it is what keeps the head's output scale
    comparable across a d-sweep. Locked at 1e-10 by tests/impls/test_gcbilinear.py.
    """
    hidden_dims: Sequence[int]
    latent_dim: int
    layer_norm: bool = True
    ensemble: bool = True
    value_exp: bool = False

    def setup(self):
        mlp_class = MLP
        if self.ensemble:
            mlp_class = ensemblize(mlp_class, 2)
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

        if self.value_exp:
            v = jnp.exp(v)

        if info:
            return v, phi, psi
        return v


def sphere_project(x, d):
    """Project x onto the sqrt(d) sphere along the last axis — FB's ||B||=||z||=sqrt(d) Norm.

    eps is 1e-6, matching tools/wandb_mode_shim/gciql_flow.py::_sphere, NOT fb_math.project_z
    (which uses 1e-12). Keeping the bridge's eps means an IqlBilinear run with project_z=True
    is comparable to the bridge sweeps it replaces; mixing the two would shift the sphered
    vectors at the 1e-6 level for near-zero rows.
    """
    return x * jnp.sqrt(d * 1.0) / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-6)


class GCBilinearPsiValue(GCBilinearValue):
    """GCBilinearValue with a `psi_base` knob on the goal leg and a `project_z` sphere.

    psi_base:
      'learned'  — psi is a trained MLP. IDENTICAL to GCBilinearValue (same submodule names,
                   same init order, same math) when project_z=False, so the vendored 1e-10
                   lock on the base class transitively covers this path.
      'frozen'   — psi is stop_gradient(MLP): zero gradient, so its params stay at random
                   init forever. Decouples the rank d from the obs dim (the standard SF
                   anti-collapse random encoder).
      'identity' — psi(g) = g, param-free. There is no `psi` submodule at all, so d is
                   PINNED to the goal dim (audit S12); the caller must enforce that.

    project_z hard-projects psi onto the sqrt(d) sphere. Unlike the bridge reference, this
    module is used SYMMETRICALLY for the value net and the critic net by IqlBilinearCritic
    (audit S3), so the actor never conditions on an unsphered psi while the critic trains a
    sphered one.
    """
    psi_base: str = "learned"
    project_z: bool = False

    def setup(self):
        mlp_class = MLP
        if self.ensemble:
            mlp_class = ensemblize(mlp_class, 2)
        self.phi = mlp_class((*self.hidden_dims, self.latent_dim),
                             activate_final=False, layer_norm=self.layer_norm)
        if self.psi_base != "identity":
            # same name/shape/init order as GCBilinearValue.psi => param trees interchange.
            self.psi = mlp_class((*self.hidden_dims, self.latent_dim),
                                 activate_final=False, layer_norm=self.layer_norm)

    def psi_of(self, goals):
        """The goal-leg embedding, i.e. the task vector z the actor may condition on.
        Exposed (rather than private) because IqlBilinearCritic reads it directly for
        actor_cond='z' — the actor must see EXACTLY the psi the critic trains."""
        if self.psi_base == "identity":
            return goals
        psi = self.psi(goals)
        if self.psi_base == "frozen":
            psi = jax.lax.stop_gradient(psi)   # zero grad -> params never leave init
        return sphere_project(psi, self.latent_dim) if self.project_z else psi

    def __call__(self, observations, goals, actions=None, info=False):
        phi_inputs = observations if actions is None else \
            jnp.concatenate([observations, actions], axis=-1)

        phi = self.phi(phi_inputs)
        psi = self.psi_of(goals)
        if self.psi_base == "identity":
            # param-free psi is unensembled; match phi's [2,B,d] when the head is ensembled.
            psi = jnp.broadcast_to(psi, phi.shape)

        v = (phi * psi / jnp.sqrt(self.latent_dim)).sum(axis=-1)

        if self.value_exp:
            v = jnp.exp(v)

        if info:
            return v, phi, psi
        return v




class GCSimpleSFValue(nn.Module):
    """Chua et al.'s Simple SF head (reference: simple_successor_features/agent/sf_simple.py,
    ported through tools/wandb_mode_shim/gciql_flow.py::GCSimpleSFValue).

        z         = psi(g)                        # the task vector (sqrt(d)-sphere if project_z)
        h(s,a,z)  = trunk([s, a, sg[z]])          # ONE trunk, not ensembled
        b(s,a,z)  = h / (||h|| + eps)             # reward BASIS -- zero params of its own
        Psi_i     = head_i(h)                     # twin heads (CriticSF.Q1/Q2) when ensemble
        Q_i       = <Psi_i, z_used> / sqrt(d)

    b and Psi are DIFFERENT vectors off a shared trunk, and b carries no parameters of its
    own: its only gradient arrives through the trunk, from the TD. That asymmetry (a network
    b against a task vector z=psi(g)) is what frees the reward model <b, psi(g)> from the
    psi=0 stationary point that killed the symmetric <psi(s),psi(g)> SF variant.

    z is stop_gradded in the TRUNK INPUT in BOTH rungs (the reference passes task.detach() to
    update_critic): otherwise the trunk's TD gradient would reach psi through the input
    concatenation and `ground_psi` would not isolate psi's signal.

    ground_psi is the ONLY stop_gradient-psi in the whole factored design: with ground_psi the
    READOUT also stop-grads z, so the IQL TD gives psi EXACTLY ZERO gradient and the reward
    regression (SimpleSfCritic._reward_loss) becomes psi's only training signal. Forward
    output is identical either way -- ground_psi is a gradient property, tested by the
    exact-zero-gradient seam test, not the forward invariants.
    """
    hidden_dims: Sequence[int]
    latent_dim: int
    layer_norm: bool = True
    ensemble: bool = True
    ground_psi: bool = False
    project_z: bool = False

    def setup(self):
        head_m = ensemblize(MLP, 2) if self.ensemble else MLP
        self.trunk = MLP((*self.hidden_dims, self.latent_dim),
                         activate_final=False, layer_norm=self.layer_norm)
        self.head = head_m((self.hidden_dims[-1], self.latent_dim),
                           activate_final=False, layer_norm=self.layer_norm)
        self.psi = MLP((*self.hidden_dims, self.latent_dim),
                       activate_final=False, layer_norm=self.layer_norm)

    def psi_of(self, goals):
        """The task vector z = psi(g), sphered onto radius sqrt(d) when project_z. Exposed so
        the reward-regression loss can read exactly the z the readout uses."""
        z = self.psi(goals)
        if self.project_z:
            z = sphere_project(z, self.latent_dim)
        return z

    def __call__(self, observations, goals, actions=None, info=False):
        z = self.psi_of(goals)                                   # [B,d], grad ON
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




class GCSfBilinearValue(GCBilinearPsiValue):
    """Successor-FEATURE bilinear head. Same bilinear readout Q(s,a,g)=<phi(s,a), psi(g)>/√d
    as GCBilinearPsiValue, but here phi(s,a) is the LEARNED successor-feature vector and the
    goal leg psi is the SF *base* b (the per-sample vector TD that trains phi lives in the
    critic's stage_loss, NOT in this module). `psi_base` names the base map, mirroring the
    bridge's three SF critic classes:

      'identity' — b(g)=g, param-free (GCIdentityBValue); latent_dim PINNED to the goal dim.
      'frozen'   — b=stop_gradient(MLP), a frozen random encoder (GCFrozenBValue); d is free.
      'learned'  — b=MLP, trained ONLY by the agent's reward regression / joint-ψ path
                   (GCLearnedBValue).

    Adds `fb_forward_z`: task-condition phi on z=sg(psi(g)) by concatenating it (head-meaned
    when ψ is ensembled, always stop-gradded) onto phi's input — the bridge's
    phi([s,a,z=psi(g)]). The stop_gradient is unconditional here, matching the bridge across
    all three bases (identity/learned stop-grad z explicitly; frozen's psi_of already froze
    it, and a second stop_gradient is a no-op). Unverified — no external reference exists for
    the z-indexed SF forward map (parity scout item 1); locked structurally only.
    """
    fb_forward_z: bool = False

    def __call__(self, observations, b_goals, actions=None, z_goals=None, info=False):
        phi_inputs = observations if actions is None else \
            jnp.concatenate([observations, actions], axis=-1)
        if self.fb_forward_z:
            z = self.psi_of(z_goals)
            if z.ndim == 3:                       # head-mean an ensembled ψ down to [B,d]
                z = z.mean(axis=0)
            phi_inputs = jnp.concatenate([phi_inputs, jax.lax.stop_gradient(z)], axis=-1)

        phi = self.phi(phi_inputs)
        psi = self.psi_of(b_goals)
        if self.psi_base == "identity":
            psi = jnp.broadcast_to(psi, phi.shape)   # param-free ψ is unensembled

        v = (phi * psi / jnp.sqrt(self.latent_dim)).sum(axis=-1)
        if self.value_exp:
            v = jnp.exp(v)
        if info:
            return v, phi, psi
        return v


class GCDuelingValue(GCBilinearValue):
    """Behavior-centered dueling critic: Q(s,a,g) = U(s,g) + A_bil(s,a,g).

    A_bil(s,a,g) = (phi(s,a) - b(s))·psi(g)/sqrt(d) is GCBilinearValue's bilinear score
    with the forward leg phi RE-CENTERED against a learned baseline feature b(s) — an MLP
    head (same output dim + ensemble structure as phi) trained to approximate the
    behavior-policy mean E_{a~mu}[phi(s,a)]. U(s,g) is a plain state-goal scalar head
    (the GCValue pattern), ensembled to 2 like the critic. No separate loss trains U or
    the Q-level split: Q=U+A_bil is trained end-to-end by the SAME expectile-TD as the
    plain bilinear critic. Once b tracks phi's behavior mean, E_{a~mu}[A_bil] ~= 0, so U
    absorbs the state-goal cardinal level and A_bil carries the action-relative part; the
    actor's q-term reads A_bil alone (the part it can move through a).

    Transcribed from tools/wandb_mode_shim/gciql_flow.py::GCDuelingValue (state-based, no
    encoders). full_info=True returns the dict the b-regression and the actor's A_bil-only
    q-term consume; info=True returns (Q, phi, psi) matching GCBilinearValue's convention.
    """

    def setup(self):
        super().setup()   # self.phi, self.psi (GCBilinearValue's bilinear legs)
        mlp_class = MLP
        if self.ensemble:
            mlp_class = ensemblize(mlp_class, 2)
        self.b = mlp_class((*self.hidden_dims, self.latent_dim),
                           activate_final=False, layer_norm=self.layer_norm)
        self.U_net = mlp_class((*self.hidden_dims, 1),
                               activate_final=False, layer_norm=self.layer_norm)

    def __call__(self, observations, goals, actions=None, info=False, full_info=False):
        assert actions is not None, \
            "GCDuelingValue is critic-only (Q needs actions); use GCValue for the expectile V(s,g)"
        phi = self.phi(jnp.concatenate([observations, actions], axis=-1))   # [2,B,d]
        psi = self.psi(goals)                                               # [2,B,d]
        bs = self.b(observations)                                           # [2,B,d]
        A_bil = ((phi - bs) * psi / jnp.sqrt(self.latent_dim)).sum(axis=-1)  # [2,B]
        U = self.U_net(jnp.concatenate([observations, goals], axis=-1)).squeeze(-1)  # [2,B]
        Q = U + A_bil
        if full_info:
            return {"Q": Q, "A_bil": A_bil, "U": U, "phi": phi, "bs": bs, "psi": psi}
        if info:
            return Q, phi, psi
        return Q






class _TwoTowerPhi(nn.Module):
    """FB-style two-tower forward leg: F = trunk( MLP[s,a] , MLP[s,z] ).

    Verbatim port of tools/wandb_mode_shim/gciql_flow.py::_TwoTowerPhi (which mirrors torch
    agents/fb/model.py ForwardMap): the (s,a) and (s,z) embeddings each end activated
    (activate_final), the trunk is linear to latent_dim. Submodule names (`embed_sa`,
    `embed_sz`, `trunk`) are load-bearing so the wiring is testable. z is a REQUIRED input
    here (the task index enters through embed_sz) — which is exactly what lets the two-tower
    COMPOSE with the z-indexed family instead of being an either/or sibling (audit S1)."""
    hidden_dims: tuple
    latent_dim: int
    layer_norm: bool = True

    @nn.compact
    def __call__(self, obs, actions, z):
        h = tuple(self.hidden_dims)
        sa = obs if actions is None else jnp.concatenate([obs, actions], axis=-1)
        e_sa = MLP((*h, h[-1]), activate_final=True, layer_norm=self.layer_norm, name="embed_sa")(sa)
        e_sz = MLP((*h, h[-1]), activate_final=True, layer_norm=self.layer_norm, name="embed_sz")(
            jnp.concatenate([obs, z], axis=-1))
        return MLP((*h, self.latent_dim), activate_final=False, layer_norm=self.layer_norm,
                   name="trunk")(jnp.concatenate([e_sa, e_sz], axis=-1))


PHI_MODULES = ("flat", "two_tower", "isfb_twin_leg")


class GCFbForwardValue(GCBilinearValue):
    """z-INDEXED bilinear critic: Q(s,a,g) = <F(s,a,z), psi(g)>/sqrt(d), z = sg(psi(g)).

    The forward-map analogue of GCBilinearPsiValue: where that head's forward leg is z-BLIND
    (phi(s,a)), this one feeds the head-mean, stop-gradded task vector z=psi(goals) INTO the
    forward map, restoring FB's defining z-indexed family M(s,a,z)=<F(s,a,z),B(g)>. It
    unifies the reference's GCFBValue / GCTwoTowerCritic / GCISFBValue (three sibling classes
    selected by a mutually-exclusive if/elif in gciql_flow.py — audit S1) into ONE module
    whose forward architecture is a `phi_module` CHOICE. Because z always enters the SAME way
    (internal, from psi(goals)), the two-tower composes with the z-index by construction —
    which is what makes B2/B4 (two_tower + z-forward + sphere + ortho) constructible.

    phi_module:
      'flat'          — phi = MLP([s, a, z]).                    (GCFBValue)
      'two_tower'     — phi = trunk(MLP[s,a], MLP[s,z]).         (GCTwoTowerCritic)
      'isfb_twin_leg' — flat phi PLUS a second leg F_V = MLP([s, z]) (action-free) for the
                        value successor measure N=<F_V,B>.       (GCISFBValue)

    psi is ALWAYS learned (freezing/identity-pinning the goal leg is IqlBilinear's axis, not
    this one). `project_z` hard-projects BOTH z and psi onto the sqrt(d) sphere, applied
    SYMMETRICALLY (audit S3). Signature is (observations, goals, actions) — the GCBilinearValue
    3-arg drop-in — with z computed internally, so every phi_module shares it.
    """
    phi_module: str = "flat"
    project_z: bool = False

    def setup(self):
        assert self.phi_module in PHI_MODULES, \
            f"phi_module must be one of {PHI_MODULES}, got {self.phi_module!r}"
        mlp = MLP
        two_tower = _TwoTowerPhi
        if self.ensemble:
            mlp = ensemblize(mlp, 2)
            two_tower = ensemblize(two_tower, 2)
        self.psi = mlp((*self.hidden_dims, self.latent_dim),
                       activate_final=False, layer_norm=self.layer_norm)
        if self.phi_module == "two_tower":
            self.phi = two_tower(hidden_dims=tuple(self.hidden_dims), latent_dim=self.latent_dim,
                                 layer_norm=self.layer_norm)
        else:
            self.phi = mlp((*self.hidden_dims, self.latent_dim),
                           activate_final=False, layer_norm=self.layer_norm)
        if self.phi_module == "isfb_twin_leg":
            # F_V(s, z): action-free forward leg giving the value successor measure N. The
            # ISFB (Q,V) decoupling is done at the measure level over the ONE shared psi.
            self.fv = mlp((*self.hidden_dims, self.latent_dim),
                          activate_final=False, layer_norm=self.layer_norm)

    def psi_of(self, goals):
        """Backward map B = psi(goals), sphered if project_z. Exposed like
        GCBilinearPsiValue.psi_of so the actor's z (actor_cond='z') is EXACTLY the psi the
        critic trains."""
        psi = self.psi(goals)
        return sphere_project(psi, self.latent_dim) if self.project_z else psi

    def _z(self, z_goals, z_vec):
        """The task vector fed to the forward map. z_vec overrides it (measure mixed-z path);
        otherwise z = sg(head-mean psi(z_goals)), sphered if project_z. Detached either way — z
        is a frozen task INPUT, per FB (psi is trained as the backward map B, not as z)."""
        if z_vec is not None:
            return jax.lax.stop_gradient(z_vec)
        z = self.psi(z_goals)
        if self.ensemble:
            z = z.mean(axis=0)                      # single task vector per sample [B,d]
        z = jax.lax.stop_gradient(z)
        return sphere_project(z, self.latent_dim) if self.project_z else z

    def _phi(self, observations, actions, z):
        if self.phi_module == "two_tower":
            return self.phi(observations, actions, z)
        inp = observations if actions is None else jnp.concatenate([observations, actions], -1)
        return self.phi(jnp.concatenate([inp, z], -1))

    def _fv(self, observations, z):
        return self.fv(jnp.concatenate([observations, z], -1))

    def __call__(self, observations, goals, actions, z_goals=None, info=False,
                 return_fv=False, z_vec=None):
        # goals feeds the backward map B=psi(goals); z_goals feeds the task index z (defaults
        # to goals). They DIFFER in the measure loss: B is psi(next_obs), z is psi(value_goals).
        z = self._z(goals if z_goals is None else z_goals, z_vec)
        phi = self._phi(observations, actions, z)
        psi = self.psi_of(goals)
        v = (phi * psi / jnp.sqrt(self.latent_dim)).sum(axis=-1)
        if self.value_exp:
            v = jnp.exp(v)
        # Materialize the F_V leg on EVERY call under isfb_twin_leg (not only the return_fv
        # path). Flax creates a setup-declared submodule's params lazily on first invocation;
        # if fv is only ever called under return_fv=True, a plain 3-arg .init(obs,goals,actions)
        # omits it and a later apply(...,return_fv=True) raises ScopeCollectionNotFound. Calling
        # it here makes the advertised 3-arg drop-in create fv, so return_fv is always safe. The
        # extra leg is cheap (isfb_twin_leg only exists under objective='isfb', which uses it).
        fv = self._fv(observations, z) if self.phi_module == "isfb_twin_leg" else None
        if info and return_fv:
            assert self.phi_module == "isfb_twin_leg", \
                "return_fv requires phi_module='isfb_twin_leg' (no F_V leg otherwise)"
            return v, phi, psi, fv, z
        if info:
            return v, phi, psi
        return v
