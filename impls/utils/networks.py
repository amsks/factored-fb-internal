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
