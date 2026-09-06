"""Factored agent: composes a Critic and an Actor over per-group TrainStates and drives
them with a generic staged update. Structure is OGBench-idiomatic (TrainState + nonpytree
static aux); FB math/RNG order are byte-identical to PSMFlows agents/fb.py."""
from typing import Any

import flax
import jax
import jax.numpy as jnp
import optax

from impls.utils.flax_utils import TrainState, soft_update, nonpytree_field
from impls.critics.base import check_combo
from impls.critics.fb import FBCritic
from impls.critics.gciql import GCIQLCritic
from impls.critics.gcivl import GCIVLCritic
from impls.critics.hiql import HIQLCritic
from impls.critics.iql_bilinear import IqlBilinearCritic
from impls.critics.crl import CRLCritic
from impls.critics.qrl import QRLCritic
from impls.critics.density_fb import DensityFBCritic
from impls.critics.psm import PSMCritic
from impls.critics.psm_density import PSMDensityCritic
from impls.critics.hilp import HILPCritic
from impls.critics.hilp_density import HILPDensityCritic
from impls.critics.rldp import RLDPCritic
from impls.critics.rldp_density import RLDPDensityCritic
from impls.actors.flowbc import FlowBCActor
from impls.actors.ddpgbc import DDPGBCActor
from impls.actors.ddpgbc_density import DDPGBCDensityActor
from impls.actors.ddpgbc_density_cpr import DDPGBCDensityCPRActor
from impls.actors.awr import AWRActor
from impls.actors.fql import FQLActor

CRITICS = {"fb": FBCritic, "gciql": GCIQLCritic, "gcivl": GCIVLCritic, "hiql": HIQLCritic, "iql_bilinear": IqlBilinearCritic, "crl": CRLCritic,
           "qrl": QRLCritic, "density_fb": DensityFBCritic, "psm": PSMCritic, "rldp": RLDPCritic,
           # Density-FB's categorical successor estimator under PSM's / RLDP's own
           # representation learners (impls/critics/{psm,rldp}_density.py).
           "psm_density": PSMDensityCritic, "rldp_density": RLDPDensityCritic,
           # ...and under HILP's, which qualifies for the same reason RLDP does: phi is
           # trained only by the Hilbert value loss, so swapping the successor estimator
           # changes exactly one thing (impls/critics/hilp_density.py).
           "hilp_density": HILPDensityCritic,
           # HILP (Park et al. 2024), transcribed from the official repo; the second
           # method of the ExoRL parity campaign. Both its stages are pinned by
           # tests/impls/test_hilp_seam.py.
           "hilp": HILPCritic}
# CONFIG-ONLY VARIANTS: a --critic name that selects a different yaml but the SAME critic
# class. `density_fb_pixels` is density_fb on image observations -- `DensityFBCritic.build`
# infers the pixel path from the example observation's rank, so nothing about the class
# changes; what changes is the config (encoder, frame stack, augmentation, and the privileged
# qpos measure channel), and it needs its own file because the composer merges
# domain-then-critic and a domain cannot override `measure_hit_spec`, which the pixel arm must.
#
# Kept OUT of CRITICS deliberately: that dict is enumerated by the seam's gates
# (tests/impls/test_boot_contract.py builds every critic x actor pair from it), and an alias
# would add a duplicate of an existing critic to every one of them without adding coverage.
CRITIC_ALIASES = {"density_fb_pixels": "density_fb",
                  # Exp B: density_fb with the spec-30 centered dual reward-sample index on
                  # (the yaml sets `measure_density_z: true`); same class, config-only.
                  "density_fb_z": "density_fb",
                  # Exp A: rldp_density with density_fb's task interface (the yaml sets
                  # `goal_kernel: true`); same RLDPDensityCritic class, config-only.
                  "rldp_density_gk": "rldp_density",
                  # 2026-09-06 goal-conditioned training variants (config-only aliases):
                  # density families via goal_kernel, vanilla via all-goal z sampling.
                  "rldp_density_gk_oc": "rldp_density",
                  "density_fb_z_state": "density_fb",
                  "hilp_density_gk": "hilp_density", "hilp_density_gk_oc": "hilp_density",
                  "hilp_gc": "hilp", "hilp_gc_oc": "hilp",
                  "rldp_gc": "rldp", "rldp_gc_oc": "rldp",
                  # Diag-screen repair arms: psm / psm_density at the torch sweep-winning
                  # cube config (batch 1024, discount 0.98, ortho_coef 1000); config-only.
                  "psm_repair": "psm",
                  "psm_density_repair": "psm_density",
                  # Object-coordinate backward-input arms: the yaml sets
                  # `backward_input: hit_spec`, so B (rldp family) / phi (psm family)
                  # consumes the eps-scaled hit-spec block slice of the observation
                  # instead of the full state (impls/critics/{rldp,psm}.py::build);
                  # same classes, config-only.
                  "rldp_oc": "rldp", "rldp_density_oc": "rldp_density",
                  "psm_oc": "psm", "psm_density_oc": "psm_density",
                  # ... and the HILP family, whose `_phi_in` choke point takes the same
                  # knob (impls/critics/hilp.py). These two yamls are also where the cube
                  # substrate lives for HILP -- official HILP values plus the td_jepa
                  # degeneracy fixes -- so they are not pure oc variants of critic/hilp.yaml
                  # the way rldp_oc is of critic/rldp.yaml.
                  "hilp_oc": "hilp", "hilp_density_oc": "hilp_density",
                  # `hilp_cube` is the matched state-input CONTROL for hilp_oc (same
                  # cube substrate, backward_input: state); `fb_oc` is plain FB with
                  # the same object-coordinate B input (impls/critics/fb.py::_b_in).
                  "hilp_cube": "hilp", "fb_oc": "fb",
                  # `hilp_tdjc` / `hilp_density_tdjc`: HILP and Density-HILP matched EXACTLY
                  # to td_jepa's OGBench (cube) configuration -- the left encoder, the
                  # generic ForwardArchi successor tower, batch 256, sf tau 0.005, phi lr
                  # 1e-4, copied (not re-initialised) target phis. Same classes as every
                  # other HILP arm; the whole delta is in the yamls.
                  "hilp_tdjc": "hilp", "hilp_density_tdjc": "hilp_density",
                  # `density_fb_state` is Density-FB with backward_input: state --
                  # the matched no-privileged-coordinates control the method never
                  # had (impls/critics/density_fb.py::_b_query). Config-only.
                  "density_fb_state": "density_fb",
                  # `hilp_density_cube` = Density-HILP at backward_input: state (the
                  # control for hilp_density_oc); `fb_ref` = plain FB at the reference
                  # stack's ortho_coef 1000. Both config-only.
                  "hilp_density_cube": "hilp_density", "fb_ref": "fb",
                  # `fb_tdj` = the minimal fb_tdj parity arm (docs/notes/2026-09-03-fb-vs-
                  # tdjepa.md #7): fb.yaml at ortho_coef 1000 (same value as fb_ref, kept as
                  # its own config so this arm's own runs/eval.csv/wandb group are not
                  # tangled with fb_ref's earlier audit-arm history). Config-only.
                  "fb_tdj": "fb",
                  # `rldp_portalign` = RLDP with the paper implementation's
                  # self-predictive window off-by-one reinstated (dataset flag
                  # rldp_sp_port_alignment). Config-only.
                  "rldp_portalign": "rldp",
                  # ExoRL official-parity arms (docs/specs/2026-08-28-exorl-official-
                  # parity-validation.md): the SAME RLDP / Density-RLDP classes with the
                  # six official knobs turned on in the yaml. Config-only, so the audited
                  # td_jepa arms (`rldp`, `rldp_density`) are untouched.
                  "rldp_official": "rldp",
                  # ExoRL ortho arms: identical to rldp_official except that the paper's
                  # cosine-similarity regularizer is ON. `a` keeps official's ortho_coef=1
                  # (= run 1, ratio 1.3e4); `b` uses 1e-4, where the ortho and SP gradients
                  # balance. Config-only; same RLDPCritic class.
                  "rldp_orthoa": "rldp", "rldp_orthob": "rldp",
                  "rldp_density_official": "rldp_density",
                  # td_jepa DMC replication arms. `rldp` mirrors td_jepa's OGBENCH
                  # launcher (512-wide, L_dim 50, batch 256, discount 0.99); their DMC
                  # launcher -- the one behind the ExoRL numbers we compare against -- is
                  # a different configuration, so `rldp_tdj` carries it. `hilp_tdj` is
                  # only a PARTIAL match (optimiser knobs yes, towers no); see the header
                  # of its yaml before quoting any comparison from it. Config-only.
                  "rldp_tdj": "rldp", "hilp_tdj": "hilp",
                  # The density ablation on that substrate: rldp_density_tdj.yaml is
                  # rldp_tdj.yaml plus the density knobs, so this pair differs in the
                  # successor estimator alone. Config-only; same RLDPDensityCritic.
                  "rldp_density_tdj": "rldp_density",
                  # The density ablation on the HILP substrate. hilp_density_tdj.yaml
                  # extends hilp_tdj.yaml, so the four td_jepa parity fixes are inherited
                  # rather than re-decided. Config-only; same HILPDensityCritic.
                  "hilp_density_tdj": "hilp_density"}

ACTORS = {"flowbc": FlowBCActor, "ddpgbc": DDPGBCActor, "awr": AWRActor, "fql": FQLActor,
          # bridge spec 22 (see impls/actors/ddpgbc_density.py): the ddpgbc extraction
          # wired to density_fb's value-goal bootstrap + centered q normalizer.
          "ddpgbc_density": DDPGBCDensityActor,
          # ddpgbc_density plus the hindsight-z-conditioned BC term (offline core of
          # FB-CPR's anchoring; impls/actors/ddpgbc_density_cpr.py). q_coeff=0 is the
          # pure hindsight-BC floor, the bcflow precedent.
          "ddpgbc_density_cpr": DDPGBCDensityCPRActor,
          # bcflow is FlowBCActor with the value term switched off (actor/bcflow.yaml sets
          # q_coeff: 0) — the pure-BC control, so it composes with exactly the same critics.
          "bcflow": FlowBCActor}

# per-group learning rates come from config; this maps group -> config key.
_LR_KEY = {"forward": "lr_f", "left_encoder": "lr_f", "backward": "lr_b",
           # rldp's self-predictive head: the reference appends _predictor to the
           # BACKWARD optimizer (RLDPAgent.setup_training), so it rides lr_b.
           "predictor": "lr_b",
           "value_forward": "lr_f",                       # isfb F_V rides the forward group
           "value": "lr", "critic": "lr", "center_b": "lr",  # gciql / centered_bilinear groups
           # density_fb's two towers. They ride the single `lr` unless the config carries
           # the FB-style split (`lr_f` / `lr_b`), which Meta Motivo's humanoid recipe does
           # -- lr_f 1e-4 against lr_b 1e-5. Resolved by _group_lr below, so a config
           # without those keys is bit-identical to before.
           "density_f": "lr", "density_b": "lr",
           "phi": "lr_phi", "sf_psi": "lr_sf", "psm_psi": "lr_sf",  # psm's basis + two psi towers
           # density_fb's index state under measure_density_z (default off, so these groups
           # only EXIST on that arm). Neither ever receives a gradient -- they are not in
           # any stage_groups and not in the actor's GROUPS -- so the optimizer built here
           # is never stepped; they live in `net` because that is the only container every
           # consumer (sample_cond, q_fn via `params`, infer_cond, map_eval_cond) can see,
           # and because `net` is what save_agent serializes.
           # Same story for the running observation statistics under `norm_obs`.
           "density_b_zenc": "lr", "density_zstats": "lr", "density_obsnorm": "lr",
           # hilp: groups are namespaced `hilp_*` ON PURPOSE. _LR_KEY is keyed by GROUP
           # NAME across every critic, so an unprefixed "sf_psi" here silently re-keyed
           # PSM's own sf_psi group from lr_sf to lr_f and broke all 8 PSM pairs --
           # caught by test_boot_contract's recorded baseline, which is exactly what it
           # is for. phi towers ride lr_b, the successor net lr_f; hilp_stats never steps.
           "hilp_phi1": "lr_b", "hilp_phi2": "lr_b", "hilp_sf": "lr_f", "hilp_stats": "lr_b",
           # td_jepa's left encoder sits in the SF optimizer's param list (sf/agent.py:91-92),
           # so it rides lr_sf -- our lr_f -- not the phi lr.
           "hilp_le": "lr_f",
           "goal_rep": "lr", "high_actor": "lr",                             # hiql's high-level policy, which
                                                           # the CRITIC owns (see critics/hiql.py)
           "actor": "lr_actor", "actor_vf": None,          # actor_vf uses actor.lr_actor_vf
           "actor_bc_flow": "lr_actor", "actor_onestep": "lr_actor"}  # fql groups


# density_fb's towers accept an optional FB-style per-tower split. Preferred key first,
# fallback second; absent preferred key => the fallback, i.e. every pre-existing run.
_LR_SPLIT = {"density_f": "lr_f", "density_b": "lr_b"}


def _group_opt(group, config):
    """(lr, weight_decay, use_adamw) for one optimizer group.

    Two references need per-GROUP optimizer settings that a single `optax.adam(lr)` cannot
    express, and in both cases the difference is on the representation towers specifically:

      * official RLDP builds its ENCODER optimizer as `AdamW(..., weight_decay=1e-4)`
        (hard-coded, baselines/rldp agent.py:66-71) while the forward and actor optimizers
        are plain Adam at `cfg.train.weight_decay` (0.0). Decoupled decay on exactly the
        towers whose weights we measured blowing up (||F|| 1486, M diag 576 vs an expected
        ~50) is not something to leave off.
      * official HILP gives phi its own learning rate, `lr_coef * lr` = 5e-4, against 1e-4
        for the successor net and actor (HILP sf.py:37,486-490).

    Config shape, both default-absent so every existing cell is bit-identical:
        group_weight_decay: {group: float}
        group_adamw:        [group, ...]
        group_lr:           {group: float}     # overrides the _LR_KEY lookup
    """
    lr = _group_lr(group, config)
    lr = float((config.get("group_lr") or {}).get(group, lr))
    wd = float((config.get("group_weight_decay") or {}).get(group, 0.0))
    adamw = group in set(config.get("group_adamw") or ())
    return lr, wd, adamw


def _group_tx(group, config):
    lr, wd, adamw = _group_opt(group, config)
    if adamw:
        # optax.adamw is torch's AdamW: decoupled decay, applied to the UPDATE.
        return optax.adamw(lr, weight_decay=wd)
    if wd:
        raise ValueError(
            f"group_weight_decay set for {group!r} but it is not in group_adamw. "
            f"optax.adam has no weight_decay argument, and folding decay into a plain "
            f"Adam is L2-into-the-gradient, which is NOT what torch AdamW does -- name "
            f"the group in group_adamw so the difference is explicit.")
    return optax.adam(lr)


def _group_lr(group, config):
    """The learning rate one optimizer group is built with."""
    if group == "actor_vf":                       # flowbc's value-flow head owns its own lr
        return float(config["actor"]["lr_actor_vf"])
    split = _LR_SPLIT.get(group)
    if split is not None and config.get(split) is not None:
        return float(config[split])
    return float(config[_LR_KEY[group]])


class Agent(flax.struct.PyTreeNode):
    rng: Any
    net: Any            # dict[group -> TrainState]   (online groups)
    target: Any         # dict[group -> params]        (target groups)
    cond_eval: Any      # eval-time conditioning (FB: {"z": ...})
    critic: Any = nonpytree_field()
    actor: Any = nonpytree_field()
    config: Any = nonpytree_field()

    # ---------------- core staged update (no drawing) ----------------
    def apply_update(self, batch, cond, boot):
        net = dict(self.net)
        target = dict(self.target)
        info = {}
        online = lambda: {g: net[g].params for g in net}

        # STAGE PARAMETER SOURCE. By default each stage re-reads `net`, so a later stage
        # sees the parameters an earlier one just stepped. Official HILP does NOT: its
        # `update_sf` computes the SF loss, the phi loss and target_phi all from the SAME
        # pre-step parameters and only then steps sf_opt and phi_opt (sf.py:610-666,
        # 766-770). With the default the SF target chases a phi that is always one Adam
        # step ahead. `stages_from_pre_step_params` reproduces the reference; it is
        # default-off so every multi-stage cell (psm, psm_density) stays bit-identical.
        pre_step = {g: net[g].params for g in net}
        from_pre = bool(self.config.get("stages_from_pre_step_params", False))
        pending = {}
        for stage in self.critic.STAGES:
            groups = self.critic.stage_groups(stage)
            base = pre_step if from_pre else None

            def stage_loss_fn(gp, _base=base):
                params = {**(_base if _base is not None else online()), **gp}
                return self.critic.stage_loss(stage, params, target, batch, cond, boot)

            (_, sinfo), grads = jax.value_and_grad(stage_loss_fn, has_aux=True)(
                {g: (pre_step[g] if from_pre else net[g].params) for g in groups})
            if from_pre:
                pending.update({g: grads[g] for g in groups})
            else:
                for g in groups:
                    net[g] = net[g].apply_gradients(grads=grads[g])
            info.update(sinfo)
        for g, gr in pending.items():          # all stages stepped together, after the fact
            net[g] = net[g].apply_gradients(grads=gr)

        # A critic may declare that this phase trains no actor at all. Only RLDP under
        # `rldp_two_phase` does: its official phase 1 runs the representation update alone
        # (baselines/rldp agent.py:174-197 calls update_mr and nothing else). The attribute
        # is read with a default of True, so every other critic is untouched and nothing
        # extra is traced for it. Skipping the block entirely -- rather than zeroing the
        # gradient -- is what keeps the actor's Adam moments frozen too.
        if getattr(self.critic, "TRAIN_ACTOR", True):
            def actor_loss_fn(gp):
                params = {**online(), **gp}
                return self.actor.loss(params, target, batch, self.critic, cond, boot)

            (_, ainfo), agrads = jax.value_and_grad(actor_loss_fn, has_aux=True)(
                {g: net[g].params for g in self.actor.GROUPS})
            for g in self.actor.GROUPS:
                net[g] = net[g].apply_gradients(grads=agrads[g])
            info.update(ainfo)

        for g, tau in self.critic.TARGETS.items():
            target[g] = soft_update(net[g].params, target[g], tau)

        # Optional critic hook for state that is neither a gradient target nor a polyak
        # target: running statistics and one-shot frozen draws a critic maintains itself
        # (density_fb's index moments mu_hat/C_hat, its slow-EMA index tower, and its frozen
        # eval pool). `self.net` is passed as the PRE-STEP online params because that is the
        # source discipline the bridge uses for all three (gciql_flow.py `update`, ~3849-3939
        # reads self.network.params, not the post-step new_network).
        #
        # Additive: a critic that does not define the method is untouched and nothing extra
        # is traced for it, so the 60-pair byte-identity gate is unaffected.
        if hasattr(self.critic, "update_stats"):
            net = self.critic.update_stats(net, self.net, batch)

        return self.replace(net=net, target=target), info

    def compute_static(self, batch, cond, boot):
        info = {}
        online = {g: self.net[g].params for g in self.net}
        for stage in self.critic.STAGES:
            _, sinfo = self.critic.stage_loss(stage, online, self.target, batch, cond, boot)
            info.update(sinfo)
        if getattr(self.critic, "TRAIN_ACTOR", True):
            _, ainfo = self.actor.loss(online, self.target, batch, self.critic, cond, boot)
            info.update(ainfo)
        return info

    # ---------------- live update (draws randomness) ----------------
    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)
        r_cond, r_boot = jax.random.split(rng)
        # A critic whose conditioning needs the TARGET towers declares it statically. Only
        # density_fb under measure_density_z does: its actor-q reward vector is read off the
        # target backward tower (bridge `_density_actor_q_z`), and q_fn -- which the actor
        # calls with online params only -- has no other way to reach it. A static attribute
        # rather than a try/except TypeError so no partially-traced call can be swallowed.
        if getattr(self.critic, "SAMPLE_COND_WANTS_TARGET", False):
            cond = self.critic.sample_cond(self.net, batch, r_cond, target=self.target)
        else:
            cond = self.critic.sample_cond(self.net, batch, r_cond)
        boot = self.actor.sample_bootstrap(self.net, batch, cond, r_boot)
        new_agent, info = self.apply_update(batch, cond, boot)
        return new_agent.replace(rng=new_rng), info

    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        # `goals` is used by goal-conditioned critics (gciql: cond["goal"]); FB ignores it
        # (it conditions on cond_eval["z"]). When goals is None, use the stored eval cond.
        seed = self.rng if seed is None else seed
        # `goals` is normally an array (the env's goal observation). It may also be a DICT of
        # conditioning CHANNELS, which is how a critic that needs more than one view of the
        # goal gets them: density_fb on pixels needs both the goal IMAGE (cond["goal"], what
        # the towers encode) and the privileged object-coordinate row (cond["goal_qpos"],
        # what the hit kernel slices), and there is no single array that is both.
        if goals is None:
            cond = self.cond_eval
        elif isinstance(goals, dict):
            cond = {**self.cond_eval, **goals}
        else:
            cond = {**self.cond_eval, "goal": goals}
        # Optional critic hook: some critics condition the actor on a FUNCTION of the goal
        # (iql_bilinear with actor_cond='z' feeds the actor psi(goal), width latent_dim),
        # which cannot be recovered from `cond` alone — it needs the critic's params.
        # Critics without the hook are unaffected.
        if hasattr(self.critic, "map_eval_cond"):
            # `observations` is passed for critics whose conditioning is joint in state and
            # goal (hiql: goal_rep([s; g]) via the high-level policy). Optional keyword, so
            # critics with the older two-argument hook (iql_bilinear) are untouched.
            try:
                cond = self.critic.map_eval_cond(self.net, cond, observations=observations)
            except TypeError:
                cond = self.critic.map_eval_cond(self.net, cond)
        return self.actor.sample_actions(self.net, observations, cond, seed, temperature)

    def infer_eval_cond(self, observations, rewards):
        return self.replace(cond_eval=self.critic.infer_cond(self.net, observations, rewards))

    # ---------------- factory ----------------
    @classmethod
    def create(cls, critic_name, actor_name, seed, ex_obs, ex_actions, config):
        config = {**config, "seed": seed}
        critic_cls = CRITICS[CRITIC_ALIASES.get(critic_name, critic_name)]
        actor_cls = ACTORS[actor_name]
        critic, cparams, targets = critic_cls.build(ex_obs, ex_actions, config)
        # Validate before building the actor: check_combo reads only class attributes, so
        # an invalid pair raises ComboError naming the real reason rather than dying on
        # whatever config key the actor's build() indexes first.
        check_combo(critic, actor_cls)
        config = {**config, "cvec_dim": int(critic.cvec_dim)}  # actors size conditioning on this
        # A critic whose actor sees something other than the raw observation declares its
        # width here (density_fb on pixels: the DrQ feature width, because the policy is
        # handed encoder features and the actor owns no encoder). Absent on every other
        # critic, and actors then size their observation leg from ex_obs exactly as before.
        actor_obs_dim = getattr(critic, "actor_obs_dim", None)
        if actor_obs_dim is not None:
            config = {**config, "actor_obs_dim": int(actor_obs_dim)}
        actor, aparams = actor_cls.build(ex_obs, ex_actions, config)

        params = {**cparams, **aparams}
        # Each group's nn.Module is an attribute named after the group on its owner.
        modules = {g: getattr(critic, g) for g in critic.GROUPS}
        modules.update({g: getattr(actor, g) for g in actor.GROUPS})
        net = {}
        for g, p in params.items():
            # actor_vf (flowbc) has its own lr; other groups map to a config lr key.
            net[g] = TrainState.create(modules[g], p, tx=_group_tx(g, config))

        cond_eval = critic.eval_cond0(ex_obs, config)
        return cls(rng=jax.random.PRNGKey(seed), net=net, target=targets,
                   cond_eval=cond_eval, critic=critic, actor=actor,
                   config=flax.core.FrozenDict(config))
