"""JAX/Flax CRL agent matching Unsupervised_Manipulation's SGCRL architecture.

Architecture mirrors PyTorchSGCRLAgent exactly:
  - Critic: dot-product bilinear (phi @ psi.T), CPC loss + logsumexp regularization
  - Actor:  MLP → tanh-normal → -diag(logits) objective with random-goal mixing
  - Separate Adam optimizers for actor (lr=6e-4) and critic (lr=3e-4)
"""

from __future__ import annotations

from typing import Any, Sequence

import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState


# ---------------------------------------------------------------------------
# Network modules
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)
        return x


class Actor(nn.Module):
    """Tanh-normal policy.  Input: [state ‖ goal].  Output: distribution over actions."""

    hidden_dims: Sequence[int]
    action_dim: int
    min_scale: float = 1e-6

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> distrax.Distribution:
        x = MLP(self.hidden_dims)(obs)
        loc = nn.Dense(self.action_dim)(x)
        loc = 10.0 * jnp.tanh(loc / 10.0)
        log_scale = nn.Dense(self.action_dim)(x)
        scale = nn.activation.softplus(log_scale) + self.min_scale
        return distrax.Transformed(
            distrax.Normal(loc, scale),
            distrax.Tanh(),
        )


class Critic(nn.Module):
    """Bilinear contrastive critic.

    phi = encoder(state ‖ action),  psi = encoder(goal)
    logits = phi @ psi.T   (B x B dot-product matrix)
    """

    hidden_dims: Sequence[int]
    repr_dim: int

    @nn.compact
    def __call__(
        self,
        state: jnp.ndarray,
        action: jnp.ndarray,
        goal: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        sa = jnp.concatenate([state, action], axis=-1)
        phi = nn.Dense(self.repr_dim)(MLP(self.hidden_dims)(sa))
        psi = nn.Dense(self.repr_dim)(MLP(self.hidden_dims)(goal))
        logits = phi @ psi.T
        return logits, phi, psi


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class CRLIWSAgent(flax.struct.PyTreeNode):
    """CRL agent with IWS-compatible update interface.

    Critic loss: CPC cross-entropy + logsumexp² regularization (matches UM).
    Actor  loss: −diag(logits) + entropy, with random-goal mixing.
    """

    rng: Any
    actor_state: Any
    critic_state: Any
    config: Any = flax.struct.field(pytree_node=False)

    # ------------------------------------------------------------------
    # Losses
    # ------------------------------------------------------------------

    def _critic_loss_fn(
        self,
        critic_params: Any,
        batch: dict[str, jnp.ndarray],
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        logits, _, _ = self.critic_state.apply_fn(
            {"params": critic_params},
            batch["states"],
            batch["actions"],
            batch["goals"],
        )
        batch_size = logits.shape[0]
        labels = jnp.arange(batch_size)
        xent = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
        logsumexp = jax.nn.logsumexp(logits, axis=1)
        loss = xent + self.config["logsumexp_coeff"] * jnp.mean(logsumexp ** 2)

        pos_mask = jnp.eye(batch_size, dtype=bool)
        info = {
            "critic_loss": loss,
            "logits_pos": jnp.sum(jnp.where(pos_mask, logits, 0.0)) / jnp.sum(pos_mask),
            "logits_neg": jnp.sum(jnp.where(~pos_mask, logits, 0.0)) / jnp.sum(~pos_mask),
            "categorical_accuracy": jnp.mean(
                (jnp.argmax(logits, axis=1) == labels).astype(jnp.float32)
            ),
            "binary_accuracy": jnp.mean(((logits > 0) == pos_mask).astype(jnp.float32)),
        }
        return loss, info

    def _actor_loss_fn(
        self,
        actor_params: Any,
        batch: dict[str, jnp.ndarray],
        rng: jax.Array,
        frozen_critic_params: Any,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        state, goal = batch["states"], batch["goals"]

        # random_goals=0.5: double batch — first half real goals, second half shuffled
        if self.config["random_goals"] == 0.5:
            state = jnp.concatenate([state, state], axis=0)
            goal = jnp.concatenate([goal, jnp.roll(goal, 1, axis=0)], axis=0)
        elif self.config["random_goals"] == 1.0:
            goal = jnp.roll(goal, 1, axis=0)

        obs = jnp.concatenate([state, goal], axis=-1)
        dist = self.actor_state.apply_fn({"params": actor_params}, obs)
        action = dist.sample(seed=rng)
        log_prob = dist.log_prob(action)

        scaled_action = self._scale_action(action)
        # Critic is frozen — stop gradients via frozen_critic_params
        logits, _, _ = self.critic_state.apply_fn(
            {"params": jax.lax.stop_gradient(frozen_critic_params)},
            state,
            scaled_action,
            goal,
        )

        loss = -jnp.diag(logits)
        entropy = -log_prob.mean()
        if self.config["use_action_entropy"]:
            loss = loss + self.config["entropy_coeff"] * log_prob
        mean_loss = loss.mean()
        return mean_loss, {
            "actor_loss": mean_loss,
            "policy_entropy": entropy,
        }

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    @jax.jit
    def update(
        self, batch: dict[str, jnp.ndarray]
    ) -> tuple["CRLIWSAgent", dict[str, float]]:
        rng, actor_rng = jax.random.split(self.rng)

        # Critic step
        (_, cinfo), cgrads = jax.value_and_grad(
            self._critic_loss_fn, argnums=0, has_aux=True
        )(self.critic_state.params, batch)
        new_critic_state = self.critic_state.apply_gradients(grads=cgrads)

        # Actor step (critic params frozen at pre-update values)
        (_, ainfo), agrads = jax.value_and_grad(
            self._actor_loss_fn, argnums=0, has_aux=True
        )(self.actor_state.params, batch, actor_rng, self.critic_state.params)
        new_actor_state = self.actor_state.apply_gradients(grads=agrads)

        info = {**{k: float(v) for k, v in cinfo.items()},
                **{k: float(v) for k, v in ainfo.items()}}
        return self.replace(
            rng=rng,
            actor_state=new_actor_state,
            critic_state=new_critic_state,
        ), info

    # ------------------------------------------------------------------
    # Action sampling
    # ------------------------------------------------------------------

    @jax.jit
    def sample_actions(
        self,
        states: jnp.ndarray,
        goals: jnp.ndarray,
        *,
        seed: jax.Array,
        temperature: float = 1.0,
    ) -> jnp.ndarray:
        obs = jnp.concatenate([states, goals], axis=-1)
        dist = self.actor_state.apply_fn({"params": self.actor_state.params}, obs)
        action = dist.sample(seed=seed)
        if temperature != 1.0:
            # approximate temperature by scaling the pre-tanh noise
            mean = dist.distribution.mode()
            noise = action - jnp.tanh(mean)
            action = jnp.tanh(mean + noise * temperature)
        return self._scale_action(action)

    def _scale_action(self, action: jnp.ndarray) -> jnp.ndarray:
        low = jnp.asarray(self.config["action_low"])
        high = jnp.asarray(self.config["action_high"])
        return low + 0.5 * (action + 1.0) * (high - low)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        *,
        seed: int,
        state_dim: int,
        goal_dim: int,
        action_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        hidden_dims: tuple[int, ...] = (256, 256),
        repr_dim: int = 64,
        actor_lr: float = 6e-4,
        critic_lr: float = 3e-4,
        entropy_coeff: float = 0.0,
        logsumexp_coeff: float = 0.1,
        random_goals: float = 0.5,
        use_action_entropy: bool = True,
    ) -> "CRLIWSAgent":
        rng = jax.random.PRNGKey(seed)
        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        actor_def = Actor(hidden_dims=hidden_dims, action_dim=action_dim)
        critic_def = Critic(hidden_dims=hidden_dims, repr_dim=repr_dim)

        ex_obs = jnp.zeros((1, state_dim + goal_dim))
        ex_state = jnp.zeros((1, state_dim))
        ex_action = jnp.zeros((1, action_dim))
        ex_goal = jnp.zeros((1, goal_dim))

        actor_params = actor_def.init(actor_rng, ex_obs)["params"]
        critic_params = critic_def.init(critic_rng, ex_state, ex_action, ex_goal)["params"]

        actor_state = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            tx=optax.adam(learning_rate=actor_lr),
        )
        critic_state = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=optax.adam(learning_rate=critic_lr),
        )

        config = {
            "state_dim": state_dim,
            "goal_dim": goal_dim,
            "action_dim": action_dim,
            "action_low": np.asarray(action_low, dtype=np.float32),
            "action_high": np.asarray(action_high, dtype=np.float32),
            "hidden_dims": tuple(hidden_dims),
            "repr_dim": repr_dim,
            "entropy_coeff": float(entropy_coeff),
            "logsumexp_coeff": float(logsumexp_coeff),
            "random_goals": float(random_goals),
            "use_action_entropy": bool(use_action_entropy),
        }

        return cls(rng=rng, actor_state=actor_state, critic_state=critic_state, config=config)
