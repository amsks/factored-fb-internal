"""Scalar-Gated In-Sample Forward-Backward (SG-ISFB) agent (JAX).

PAPER/fb-modified/isfb.tex. Imports IQL's in-sample improvement operator into the FB
successor measure, WITHOUT an out-of-distribution a+~pi bootstrap. On the shared
bilinear factorization (GCISFBValue: forward F, value-forward F_V, backward B=psi):

    M_ij = <F(s_i,a_i,z_i), B(s'_j)>/sqrt(d)     action successor measure
    N_ij = <F_V(s_i,z_i),   B(s'_j)>/sqrt(d)     value  successor measure (shares B)
    q_i  = <F(s_i,a_i,z_i), z_i>/sqrt(d)         scalar task value  Q_z(s,a)
    v_i  = <F_V(s_i,z_i),   z_i>/sqrt(d)         scalar task value  V_z(s)

  L_V = expectile_tau(sg q - v)                  the ONLY action-selection operator
  A_i = sg(q_i - v_i)                            scalar, task-aligned advantage
  w_i = tau-hard | AWR(exp(beta A))              improvement weights (per row)
  L_N = mean_i w_i * ||sg M_i - N_i||^2          weighted vector regression
        (coord variant: coordinatewise expectile of M -- the pre-refinement baseline)
  L_M = FB off-diag TD + diagonal, M bootstrapping through gamma*N^-(s'_i,.)  (no a+~pi)

The flow actor + z-conditioning are inherited from GCIQLFlowAgent unchanged; it goes
greedy on q=<F(s,a,z),z>. Note (isfb.tex F_V double-constraint): v_i=<F_V,z> ties the
scalar V head to the vector F_V, so L_V and L_N both shape F_V's z-column. Kept as
written; switch to a separate scalar V head if diagnostics show the two losses fight.
"""
import flax
import jax
import jax.numpy as jnp

from gciql_flow import GCIQLFlowAgent, GCISFBValue, _sphere, get_config as _gciql_flow_get_config

__all__ = ['ISFBAgent', 'GCISFBValue', 'get_config']


class ISFBAgent(GCIQLFlowAgent):
    """SG-ISFB: FB successor measure with IQL's in-sample improvement. Reuses
    GCIQLFlowAgent's create (via the isfb critic branch), flow actor, and z-cond."""

    def isfb_loss(self, batch, grad_params, rng):
        obs, nobs = batch['observations'], batch['next_observations']
        acts, vgoals = batch['actions'], batch['value_goals']
        d = self.config['value_latent_dim']; sqd = jnp.sqrt(d * 1.0)
        gamma = self.config['discount']; tau = self.config['expectile']
        B = obs.shape[0]; off = 1.0 - jnp.eye(B); offb = off[None]

        # Online forward quantities. z=sg(psi(vgoals)) is [B,d]; phi/fv/psi are [2,B,d].
        _, phi, psi, fv, z = self.network.select('critic')(
            obs, nobs, acts, vgoals, info=True, return_fv=True, params=grad_params)
        M = jnp.einsum('hid,hjd->hij', phi, psi) / sqd          # action measure  [2,B,B]
        N = jnp.einsum('hid,hjd->hij', fv, psi) / sqd           # value  measure  [2,B,B]
        q = jnp.einsum('hid,id->hi', phi, z) / sqd              # Q_z(s,a)        [2,B]
        v = jnp.einsum('hid,id->hi', fv, z) / sqd               # V_z(s)          [2,B]

        # (1) scalar gate: IQL expectile value loss -- the only action-selection operator.
        uV = jax.lax.stop_gradient(q) - v
        L_V = self.expectile_loss(uV, uV, tau).mean()
        A = jax.lax.stop_gradient(q - v).mean(axis=0)           # head-mean advantage [B]

        # (2) improve the vector measure N under the scalar gate.
        mode = self.config['measure_insample']
        if mode == 'coord':
            # coordinatewise expectile envelope of M (the pre-refinement IS-FB baseline).
            uN = jax.lax.stop_gradient(M) - N
            L_N = self.expectile_loss(uN, uN, tau).mean()
        else:
            if mode == 'scalar_gate_awr':
                w = jnp.clip(jnp.exp(self.config['awr_beta'] * A), 0.0, self.config['awr_wmax'])
            else:  # scalar_gate_hard
                w = tau * (A > 0) + (1.0 - tau) * (A <= 0)
            wbar = w / (w.mean() + 1e-6)                        # [B]
            L_N = (wbar[None, :, None] * (jax.lax.stop_gradient(M) - N) ** 2).mean()

        # (3) measure TD: M bootstraps through the target value measure at the next state.
        # Mask the bootstrap at goal-reached transitions (episodic termination, as in
        # GCIQL's r+gamma*mask*V'); masks[i] gates the transition (row i).
        masks = batch.get('masks', jnp.ones((B,), dtype=M.dtype))
        _, _, tpsi, tfv, _ = self.network.select('target_critic')(
            nobs, nobs, acts, vgoals, info=True, return_fv=True)
        Ntgt = jax.lax.stop_gradient(jnp.einsum('hid,hjd->hij', tfv, tpsi) / sqd)   # N^-  [2,B,B]
        Delta = M - gamma * masks[None, :, None] * Ntgt
        L_M_off = 0.5 * jnp.sum((Delta * offb) ** 2, axis=(1, 2)) / jnp.sum(off)
        L_M_diag = -jnp.mean(jnp.diagonal(Delta, axis1=1, axis2=2), axis=1)
        L_M = jnp.mean(L_M_off + L_M_diag)

        # (4) FB orthonormality on B=psi.
        Cov = jnp.einsum('hid,hjd->hij', psi, psi)
        o_off = 0.5 * jnp.sum((Cov * offb) ** 2, axis=(1, 2)) / jnp.sum(off)
        o_diag = -jnp.mean(jnp.diagonal(Cov, axis1=1, axis2=2), axis=1)
        L_ortho = jnp.mean(o_off + o_diag)

        loss = (self.config.get('isfb_alpha_v', 1.0) * L_V
                + self.config['isfb_alpha_n'] * L_N
                + self.config['isfb_alpha_m'] * L_M
                + self.config['measure_ortho_coef'] * L_ortho)
        return loss, {'L_V': L_V, 'L_N': L_N, 'L_M': L_M, 'L_ortho': L_ortho,
                      'q_mean': q.mean(), 'v_mean': v.mean(), 'adv_mean': A.mean(),
                      'adv_pos_frac': (A > 0).mean()}

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng
        rng, actor_rng, isfb_rng = jax.random.split(rng, 3)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v
        isfb, isfb_info = self.isfb_loss(batch, grad_params, isfb_rng)
        for k, v in isfb_info.items():
            info[f'isfb/{k}'] = v
        return isfb + actor_loss, info


def get_config():
    config = _gciql_flow_get_config()
    with config.unlocked():
        config.agent_name = 'isfb_flow'
        # SG-ISFB rides the shared-B bilinear factorization + z-conditioned flow actor.
        config.value_arch = 'bilinear'
        config.value_latent_dim = 50
        config.actor_cond = 'z'
        config.actor_z_from_critic = True     # actor z = sg(psi_critic(goal)) = B(g)
        config.fb_forward_z = True            # thread z_goals through the critic (create wiring)
        config.isfb = True                    # -> GCISFBValue critic (adds F_V leg)
        config.project_z = True               # FB sphere-Norm ||z||=||B||=sqrt(d)
        config.measure_ortho_coef = 1.0       # FB ortho at its harmless coef (NOT 1000)
        # improvement operator variant:
        #   'scalar_gate_hard' (default, isfb.tex) | 'scalar_gate_awr' | 'coord' (baseline)
        config.measure_insample = 'scalar_gate_hard'
        config.isfb_alpha_v = 1.0             # weight on the scalar-gate value loss L_V (set 0 for a clean coord baseline)
        config.isfb_alpha_n = 1.0             # weight on the value-measure regression L_N
        config.isfb_alpha_m = 1.0             # weight on the measure-TD L_M
        config.awr_beta = 3.0                 # AWR temperature (scalar_gate_awr)
        config.awr_wmax = 100.0               # AWR weight clip
    return config
