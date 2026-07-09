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

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from agents.gciql import GCIQLAgent, get_config as _gciql_get_config
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState
from utils.networks import GCValue, GCBilinearValue, GCActor, MLP, ensemblize

from ogbench_flow import GCNoiseActor, GCVectorField


def _sphere(x, d):
    """Project x onto the sphere of radius sqrt(d) along the last axis -- FB pins
    ||B||=||z||=sqrt(d) via a Norm layer (agents/fb/model.py)."""
    return x * jnp.sqrt(d * 1.0) / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-6)


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
    `fb_forward_z` the forward map is TASK-CONDITIONED phi([s,a,z=psi(g)]) (frozen features)."""
    fb_forward_z: bool = False

    def setup(self):
        m = MLP
        if self.ensemble:
            m = ensemblize(m, 2)
        self.phi = m((*self.hidden_dims, self.latent_dim), activate_final=False, layer_norm=self.layer_norm)
        self.psi = m((*self.hidden_dims, self.latent_dim), activate_final=False, layer_norm=self.layer_norm)

    def _psi(self, goals):
        if self.goal_encoder is not None:
            goals = self.goal_encoder(goals)
        return jax.lax.stop_gradient(self.psi(goals))   # FROZEN: zero grad -> stays at random init

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
    cell (E3, audit-20260706): psi is trained ONLY by reward regression (`sf_reward_regress`,
    the reference's reward-prediction basis, sf_simple.py:322)

        r_hat(s',g) = <psi(s'), psi(g)>/sqrt(d)  ->  regress on batch rewards

    so the goal indicator becomes linearly READABLE in the base (the identity/frozen cells'
    semantic confound: r=<s,g> is linear in s, argmax != goal). The SF vector TD still trains
    ONLY phi (sf_loss stop-grads b(s)=psi(s) in its target): psi makes the reward expressible,
    phi accumulates it. Division of labor, one shared psi network."""

    def _psi(self, goals):
        if self.goal_encoder is not None:
            goals = self.goal_encoder(goals)
        return self.psi(goals)   # LEARNED (vs GCFrozenBValue): grads arrive via sf_reward_regress

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


class GCIQLFlowAgent(GCIQLAgent):
    """GCIQL with FB's FlowBC actor. Inherits value_loss/critic_loss/update."""

    def _flow_rollout(self, obs, noise, params):
        """Euler-integrate the BC velocity field from noise -> action (flow_steps)."""
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
        zarg = (batch['value_goals'],) if self.config.get('fb_forward_z', False) else ()
        q1, q2 = self.network.select('target_critic')(
            batch['observations'], batch['value_goals'], batch['actions'], *zarg)
        q = jnp.minimum(q1, q2) if self.config['double_q'] else q1
        v = self.network.select('value')(batch['observations'], batch['value_goals'], params=grad_params)
        value_loss = self.expectile_loss(q - v, q - v, self.config['expectile']).mean()
        return value_loss, {'value_loss': value_loss, 'v_mean': v.mean(),
                            'v_max': v.max(), 'v_min': v.min()}

    def critic_loss(self, batch, grad_params):
        """IQL critic loss; thread z_goals when fb_forward_z (B1): the z-conditioned
        forward map takes an extra task arg, so Q(s,a,g)=<F(s,a,psi(g)),psi(g)>/sqrt(d)."""
        next_v = self.network.select('value')(batch['next_observations'], batch['value_goals'])
        q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v
        zarg = (batch['value_goals'],) if self.config.get('fb_forward_z', False) else ()
        q1, q2 = self.network.select('critic')(
            batch['observations'], batch['value_goals'], batch['actions'], *zarg, params=grad_params)
        critic_loss = ((q1 - q) ** 2 + (q2 - q) ** 2).mean()
        return critic_loss, {'critic_loss': critic_loss, 'q_mean': q.mean(),
                             'q_max': q.max(), 'q_min': q.min()}

    def actor_loss(self, batch, grad_params, rng=None):
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
        x_1 = actions
        x_0 = jax.random.normal(x0_rng, actions.shape)
        t = jax.random.uniform(t_rng, (actions.shape[0], 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0
        pred = self.network.select('actor_vf')(obs, x_t, t, params=grad_params)
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
        if self.config.get('actor_z_infer', False):
            # exp4: condition on the INFERRED task distribution z=E[r*psi] (sparse-reward
            # readout over the batch), not the exact goal embedding psi(g).
            psi_b = jax.lax.stop_gradient(self._actor_z(goals, grad_params))
            z_infer = self._infer_actor_z(psi_b, z_rng)
            actor_cond_input = z_infer
        elif self.config['actor_cond'] == 'z':
            actor_cond_input = jax.lax.stop_gradient(self._actor_z(goals, grad_params))
        else:
            actor_cond_input = goals
        actor_actions = self.network.select('actor')(obs, actor_cond_input, noise, params=grad_params)
        if z_infer is not None:
            # Q_z(s,a)=<phi(s,a), z>/sqrt(d): greedy readout on the inferred task vector.
            _, phi, _ = self.network.select('critic')(obs, goals, actor_actions, info=True)
            q = (phi * z_infer[None]).sum(-1) / jnp.sqrt(self.config['value_latent_dim'])
            q1, q2 = q[0], q[1]
        else:
            # fb_forward_z: q = F(s,a,z).B(g)/sqrt(d) with z=sg(psi(g)) -> the FB greedy readout.
            zarg = (goals,) if self.config.get('fb_forward_z', False) else ()
            q1, q2 = self.network.select('critic')(obs, goals, actor_actions, *zarg)
        q = jnp.minimum(q1, q2) if self.config['double_q'] else q1
        actor_loss = -q.mean()

        # 3) BC-distillation: one-shot actor -> frozen multi-step flow rollout (FQL
        # one-step distillation). Q-loss is normalized by |q| mean to be scale
        # invariant, exactly as in GCIQL's DDPG+BC actor loss.
        distill_error = 0.0
        if self.config['bc_coeff'] > 0:
            target = self._flow_rollout(obs, noise, jax.lax.stop_gradient(grad_params))
            distill_error = jnp.mean((actor_actions - target) ** 2)
            bc_loss = self.config['bc_coeff'] * distill_error
            actor_loss = actor_loss / jax.lax.stop_gradient(jnp.abs(q).mean() + 1e-6) + bc_loss
        actor_loss = actor_loss + bc_flow_loss

        return actor_loss, {
            'actor_loss': actor_loss,
            'bc_flow_loss': bc_flow_loss,
            'distill_loss': distill_error,
            'q_mean': q.mean(),
            'q_abs_mean': jnp.abs(q).mean(),
        }

    @staticmethod
    def _measure_terms(M, tM, gamma, scale_norm, eps=1e-6):
        """Reduce the online measure M and stop-gradded target tM into the FB (offdiag, diag)
        TD terms over the last two (BxB) axes (leading head axes preserved):
            diff = M - gamma*tM;  offdiag = 0.5*sum((diff*off)^2)/sum(off);  diag = -mean(diag(diff))
        With measure_scale_norm, divide by a stop-gradded measure scale so the gradient is
        DIMENSIONLESS -- the same trick the actor q_loss uses (q/sg[|q|]) -- which stops the
        unbounded <F,z> inflation from blowing up the shared critic (audit-20260706 §1):
            offdiag /= sg[mean(M^2)+eps];  diag /= sg[sqrt(mean(M^2))+eps].
        Ortho term is handled by the caller (already scale-bounded by psi)."""
        B = M.shape[-1]
        off = 1.0 - jnp.eye(B)
        diff = M - gamma * tM
        offdiag = 0.5 * jnp.sum((diff * off) ** 2, axis=(-2, -1)) / jnp.sum(off)
        diagt = -jnp.mean(jnp.diagonal(diff, axis1=-2, axis2=-1), axis=-1)
        if scale_norm:
            msq = jax.lax.stop_gradient(jnp.mean(M ** 2))
            offdiag = offdiag / (msq + eps)
            diagt = diagt / (jnp.sqrt(msq) + eps)
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
        if self.config.get('measure_legs', 'critic') == 'value':
            _, phi, psi = self.network.select('value')(obs, nobs, info=True, params=grad_params)
            if sg_phi: phi = jax.lax.stop_gradient(phi)
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
            zfrac = self.config.get('measure_z_random_frac', 0.0)
            zkw = {}
            z_mix = None
            if zarg and zfrac > 0:
                rng, zrng, prng = jax.random.split(rng, 3)
                _, _, psi_vg = self.network.select('critic')(vgoals, vgoals, acts, vgoals,
                                                             info=True, params=grad_params)
                z_goal = jax.lax.stop_gradient(psi_vg.mean(axis=0))       # head-mean [B,d]
                if self.config.get('project_z', False):
                    z_goal = _sphere(z_goal, d)
                z_rand = _sphere(jax.random.normal(zrng, (B, d)), d)
                pick = jax.random.bernoulli(prng, zfrac, (B, 1))
                z_mix = jnp.where(pick, z_rand, z_goal)
                zkw = {'z_vec': z_mix}
            _, phi, psi = self.network.select('critic')(obs, nobs, acts, *zarg, info=True,
                                                        params=grad_params, **zkw)
            if sg_phi: phi = jax.lax.stop_gradient(phi)
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
            offdiag, diagt = self._measure_terms(M, tM, gamma, scale_norm)                      # [2]
            Cov = jnp.einsum('hid,hjd->hij', psi, psi)                    # FB ortho on B:=psi(s') (grads on)
            o_off = 0.5 * jnp.sum((Cov * off[None]) ** 2, axis=(1, 2)) / jnp.sum(off)
            o_diag = -jnp.mean(jnp.diagonal(Cov, axis1=1, axis2=2), axis=1)
            orth = o_off + o_diag                                          # [2]
            lam = self.config.get('measure_ortho_coef', 0.0)
            loss = jnp.mean(offdiag + diagt + lam * orth)                  # per-head, then average
            offdiag, diagt, orth = jnp.mean(offdiag), jnp.mean(diagt), jnp.mean(orth)
        return loss, {'measure_loss': loss, 'measure_offdiag': offdiag, 'measure_diag': diagt,
                      'measure_orth': orth}

    def sf_loss(self, batch, grad_params, rng):
        """Successor-FEATURE vector TD on phi (SF-style, à la simple_successor_features), with
        an IDENTITY base b(s)=s. Trains ONLY phi (the goal/base leg psi=b is param-free):

            phi(s,a) <- b(s) + gamma * mask * phi_target(s',a'),   a' ~ pi(s',g)
            L_SF = mean_i || phi(s_i,a_i) - sg[ b(s_i) + gamma m_i phi_target(s'_i,a'_i) ] ||^2

        A PER-SAMPLE vector regression -- NOT the B x B successor-measure matrix. The readout
        Q(s,a,g)=<phi(s,a), psi(g)> drives the greedy flow actor. Base b(s): 'identity' -> b(s)=s
        (d==obs dim); 'frozen' -> b(s)=psi(s), the critic's frozen random features (d free)."""
        obs, nobs, acts = batch['observations'], batch['next_observations'], batch['actions']
        vgoals, mask = batch['value_goals'], batch['masks']
        gamma = self.config['discount']
        # fb_forward_z: task-condition phi([s,a,z=psi(g)]) -> thread z_goals=vgoals through the
        # phi, base-feature, and target-critic calls (same task g on both sides of the SF backup).
        zarg = (vgoals,) if self.config.get('fb_forward_z', False) else ()
        _, phi, _psi = self.network.select('critic')(obs, vgoals, acts, *zarg, info=True, params=grad_params)  # [2,B,d]
        if self.config.get('sf_base', 'identity') in ('frozen', 'learned'):
            # b(s) = psi(s): the base features of the CURRENT state (call psi via b_goals=obs).
            # Stop-gradded either way: the TD trains ONLY phi ('learned' psi trains solely via
            # the sf_reward_regress term below).
            _, _, b = self.network.select('critic')(obs, obs, acts, *zarg, info=True, params=grad_params)  # [2,B,d]
            b = jax.lax.stop_gradient(b)
        else:
            b = obs[None]                                             # identity base b(s)=s -> [1,B,d]
        if self.config['actor_kind'] == 'ddpgbc':
            # stock gaussian actor: a' = mode of pi(.|s',g) (or a sample if learned std).
            dist = self.network.select('actor')(nobs, vgoals)
            a_next = dist.mode() if self.config['const_std'] else dist.sample(seed=rng)
        else:
            noise = jax.random.normal(rng, acts.shape)
            a_next = self.network.select('actor')(nobs, vgoals, noise)          # a' ~ pi(s',g)
        a_next = jax.lax.stop_gradient(jnp.clip(a_next, -1, 1))
        _, tphi, _ = self.network.select('target_critic')(nobs, vgoals, a_next, *zarg, info=True)  # [2,B,d]
        target = jax.lax.stop_gradient(b + gamma * mask[None, :, None] * tphi)               # [2,B,d]
        td = jnp.mean((phi - target) ** 2)
        loss = td
        info = {'sf_loss': td, 'phi_absmax': jnp.max(jnp.abs(phi)), 'target_absmax': jnp.max(jnp.abs(target))}
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
        if self.config.get('sf_reward_regress', False):
            # E3 (corrected SF cell): ground the base in the ACTUAL reward -- bilinear reward
            # model r_hat(s',g) = <psi(s'), psi(g)>/sqrt(d) regressed on batch rewards (received
            # on ARRIVING at s', matching the SF telescope Q=sum gamma^t <b(s_t),w>). This is
            # psi's ONLY grad path (the TD target stop-grads b; the actor Q-term takes no critic
            # grads), and it is what makes the goal indicator linearly readable in the base.
            _, _, psi_ns = self.network.select('critic')(nobs, nobs, acts, *zarg, info=True, params=grad_params)  # psi(s')
            _, _, psi_g = self.network.select('critic')(obs, vgoals, acts, *zarg, info=True, params=grad_params)  # psi(g)
            r_hat = (psi_ns * psi_g).sum(axis=-1) / jnp.sqrt(psi_ns.shape[-1])   # [2,B]
            # sf_reward_shift: regress on r+shift. Raw OGBench rewards {-1 off-goal, 0 at-goal}
            # collapse the SYMMETRIC bilinear model to the psi=0 saddle (grad of a bilinear form
            # vanishes at psi=0, where the majority class is already fit); shift=1 gives {0, 1}
            # targets -- majority class 0 is saddle-consistent, goal pairs demand <psi,psi>=1
            # (a peaked kernel). A constant reward offset never changes the actor's argmax.
            r_tgt = batch['rewards'] + self.config.get('sf_reward_shift', 0.0)
            rew = jnp.mean((r_hat - r_tgt[None]) ** 2)
            loss = loss + self.config.get('sf_reward_coef', 1.0) * rew
            info['sf_reward_loss'] = rew
            info['sf_rhat_mean'] = jnp.mean(r_hat)
        return loss, info

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

        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        if self.config.get('sf_feature_td', False):    # SF-ONLY critic (no IQL): successor-feature vector TD on phi
            rng, sf_rng = jax.random.split(rng)
            sf_l, sf_info = self.sf_loss(batch, grad_params, sf_rng)
            for k, v in sf_info.items():
                info[f'sf/{k}'] = v
            return self.config.get('sf_coef', 1.0) * sf_l + actor_loss, info

        if self.config.get('measure_only', False):     # SM-ONLY critic (no IQL): FB objective
            rng, m_rng = jax.random.split(rng)
            m_loss, m_info = self.measure_loss(batch, grad_params, m_rng)
            for k, v in m_info.items():
                info[f'measure/{k}'] = v
            return self.config.get('measure_coef', 1.0) * m_loss + actor_loss, info

        value_loss, value_info = self.value_loss(batch, grad_params)
        for k, v in value_info.items():
            info[f'value/{k}'] = v
        critic_loss, critic_info = self.critic_loss(batch, grad_params)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v
        loss = value_loss + critic_loss + actor_loss
        if self.config.get('measure_coef', 0.0) > 0:   # graft FB measure onto GCIQL's rep
            rng, m_rng = jax.random.split(rng)
            m_loss, m_info = self.measure_loss(batch, grad_params, m_rng)
            loss = loss + self.config['measure_coef'] * m_loss
            for k, v in m_info.items():
                info[f'measure/{k}'] = v
        if self.config.get('iql_ortho_coef', 0.0) > 0:  # FB orthonormality ONLY (no measure)
            o_loss, o_info = self.ortho_loss(batch, grad_params)
            loss = loss + self.config['iql_ortho_coef'] * o_loss
            for k, v in o_info.items():
                info[f'ortho/{k}'] = v
        return loss, info

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
            # E3 guards: reward regression on a FIXED base is the audited confound (the linear
            # proxy r=<s,g> cannot express the goal indicator) -> refuse; a learned base without
            # the regression leaves psi with zero training signal -> refuse.
            if config.get('sf_reward_regress', False):
                assert config.get('sf_base', 'identity') == 'learned', \
                    "sf_reward_regress requires sf_base='learned' (fixed bases can't express the goal indicator)"
            if config.get('sf_base', 'identity') == 'learned':
                assert config.get('sf_reward_regress', False), \
                    "sf_base='learned' requires sf_reward_regress=True (else psi has no training signal)"
            config['actor_cond'] = 'goal'

        # Actor goal-input example. In z-conditioning (G4) the actor consumes a
        # d-dim task vector z=psi(goal) (B := value net's psi), not the raw goal
        # observation, so its goal-leg params must be sized to value_latent_dim.
        if config.get('actor_cond', 'goal') == 'z' or config.get('actor_z_infer', False):
            assert config.get('value_arch', 'mlp') == 'bilinear', \
                "actor_cond='z'/actor_z_infer requires value_arch='bilinear' (B := value net's psi)"
            ex_actor_goals = jnp.zeros((ex_observations.shape[0], config['value_latent_dim']))
        else:
            ex_actor_goals = ex_goals

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
        if config.get('value_arch', 'mlp') == 'bilinear':
            d = config['value_latent_dim']
            value_def = GCBilinearValue(
                hidden_dims=config['value_hidden_dims'], latent_dim=d,
                layer_norm=config['layer_norm'], ensemble=False,
                state_encoder=encoders.get('value'), goal_encoder=encoders.get('value'),
            )
            # fb_forward_z: task-condition the critic's forward map -> phi(s,a,z), the
            # FB-faithful F(s,a,z).B(s'). Only valid as a measure-only critic (its
            # extra z arg would break the IQL value/critic Bellman losses).
            critic_cls = GCBilinearValue
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
            critic_def = critic_cls(**crit_kwargs)
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
            value=(value_def, (ex_observations, ex_goals)),
            critic=(critic_def, critic_ex),
            target_critic=(copy.deepcopy(critic_def), critic_ex),
        )
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
            network_info['actor_vf'] = (actor_vf_def, (ex_observations, ex_actions, ex_t))
            network_info['actor'] = (actor_def, (ex_observations, ex_actor_goals, ex_noise))
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}
        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network_params
        params['modules_target_critic'] = params['modules_critic']

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = _gciql_get_config()
    with config.unlocked():
        config.agent_name = 'gciql_flow'
        # Value architecture: 'mlp' (canonical GCIQL) or 'bilinear' (inner-product, Direction A).
        config.value_arch = 'mlp'
        config.value_latent_dim = 512
        # Actor conditioning identity (G4 objective-isolation arm):
        # 'goal' = raw goal obs (stock); 'z' = z=stop_grad(value.psi(goal)),
        # FB-style single backward map. 'z' requires value_arch='bilinear'.
        config.actor_cond = 'goal'
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
        config.measure_z_random_frac = 0.0  # FB-fidelity: fraction of measure-loss task vectors z drawn as sqrt(d)-sphere gaussian directions (metamotivo sample_mixed_z) instead of psi(vgoals); needs fb_forward_z
        config.fb_forward_z = False  # task-condition the critic's forward map: phi(s,a,z), z=sg(psi(goal)) -> true F(s,a,z).B(s') (needs bilinear; runs under measure OR the IQL backup, B1)
        config.sf_feature_td = False  # SUCCESSOR-FEATURE vector TD on phi: phi(s,a)<-b(s)+gamma*phi(s',a'), Q=<phi,psi(g)> (train ONLY phi; needs bilinear)
        config.sf_coef = 1.0          # weight on the SF vector-TD loss (sf_feature_td)
        config.sf_base = 'identity'   # SF base feature b: 'identity' -> b(s)=s (d==obs dim); 'frozen' -> b(s)=psi(s) frozen random MLP (d free, e.g. 50); 'learned' -> psi trained by reward regression (needs sf_reward_regress)
        config.sf_reward_regress = False  # E3 corrected SF: learn the base by reward regression r_hat(s',g)=<psi(s'),psi(g)>/sqrt(d) on batch rewards (requires sf_base='learned')
        config.sf_reward_coef = 1.0   # weight on the reward-regression loss (sf_reward_regress)
        config.sf_reward_shift = 0.0  # regress on r+shift; use 1.0 with OGBench {-1,0} rewards to escape the psi=0 saddle of the symmetric bilinear reward model (argmax-invariant)
        config.sf_ortho_coef = 0.0    # aux orthonormality reg ||E[phi phi^T]-I|| on the SF vector phi (anti-collapse; sf_feature_td only)
        config.fwd_two_tower = False  # FB-style two-tower forward map F=trunk(MLP[s,a],MLP[s,z]) on the IQL bilinear critic (z=sg(psi(g)); architecture-only ablation, no measure loss)
        config.iql_ortho_coef = 0.0  # FB orthonormality ||E[psi psi^T]-I|| on the IQL bilinear critic's psi, with NO measure loss (isolates the ortho constraint on the ~90 IQL bilinear)
        config.project_z = False  # hard-project z and B=psi onto the sqrt(d) sphere (FB's ||B||=||z||=sqrt(d) Norm); applies to GCFBValue/GCTwoTowerCritic
        config.actor_z_from_critic = False  # source the flow actor's z from the CRITIC psi (=B, measure-trained), not the value net (audit #2 / D1); needs actor_cond='z'
        config.actor_z_infer = False  # exp4: train the flow actor on the INFERRED task dist z=E[r*psi] (sparse-reward readout over the batch), not exact psi(g); needs bilinear
    return config
