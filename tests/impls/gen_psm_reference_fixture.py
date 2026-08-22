"""Generator for tests/impls/fixtures/psm_reference.npz -- the PSM parity baseline.

THIS SCRIPT DOES NOT RUN IN THIS TREE'S VENV. It imports torch and the PyTorch PSM
reference (`agents/psm/`), which live in the repo's OTHER checkout under the torch
`.venv`; this tree is jax-only. Run it from there, pointing the output at this tree:

    PYTHONPATH=<torch checkout> <torch checkout>/.venv/bin/python \
        tests/impls/gen_psm_reference_fixture.py tests/impls/fixtures/psm_reference.npz

Everything is float64 on CPU. The two stochastic action sources are bypassed via the
`next_action=` injection hooks on PSMAgent._update_psm/_update_sf (the proto behaviour
sampler and the learned policy respectively), so every exported scalar is a deterministic
function of the exported weights + inputs -- nothing here depends on torch's RNG, which
JAX could not reproduce anyway. See tests/impls/test_psm_seam.py for how it is consumed.
"""
import sys
import numpy as np
import torch
import gymnasium as gym

torch.use_deterministic_algorithms(True)
DT = torch.float64

OBS_DIM, ACT_DIM, Z_DIM, MAX_LOG_SEED = 8, 5, 8, 6
PHI_HID, PSI_HID, NPAR, B = 16, 32, 2, 8
DISCOUNT = 0.98
PESS = 0.25            # NON-zero on purpose: exercises the uncertainty term
ACTOR_PESS = 0.5
ORTHO = 1.0

from agents.psm.agent import PSMAgent

a = PSMAgent(
    obs_space=gym.spaces.Box(-1, 1, (OBS_DIM,)), action_dim=ACT_DIM, batch_size=B,
    z_dim=Z_DIM, max_log_seed=MAX_LOG_SEED, num_parallel=NPAR, device="cpu",
    discount=DISCOUNT, lr_sf=1e-4, lr_phi=1e-4, lr_actor=1e-4, weight_decay=0.0,
    ortho_coef=ORTHO, pessimism_penalty=PESS, actor_pessimism_penalty=ACTOR_PESS,
    phi_cfg=dict(hidden_dim=PHI_HID, hidden_layers=2, norm=True, batch_norm=False),
    sf_cfg=dict(hidden_dim=PSI_HID, hidden_layers=1, embedding_layers=2),
    actor_cfg=dict(hidden_dim=PSI_HID, hidden_layers=1, embedding_layers=2),
)
a.model.double()
a.off_diag = (1 - torch.eye(B, B, dtype=DT))
a.off_diag_sum = a.off_diag.sum()

g = torch.Generator().manual_seed(0)
obs = torch.randn(B, OBS_DIM, generator=g, dtype=DT)
action = torch.rand(B, ACT_DIM, generator=g, dtype=DT) * 2 - 1
next_obs = torch.randn(B, OBS_DIM, generator=g, dtype=DT)
goal = next_obs.clone()                       # phi_input="s" => goal = next_obs
na_psm = torch.rand(B, ACT_DIM, generator=g, dtype=DT) * 2 - 1
na_sf = torch.rand(B, ACT_DIM, generator=g, dtype=DT) * 2 - 1
z_psm = torch.randint(0, 2, (B, MAX_LOG_SEED), generator=g).to(DT)
z_sf = torch.randn(B, Z_DIM, generator=g, dtype=DT)
z_sf = np.sqrt(Z_DIM) * torch.nn.functional.normalize(z_sf, dim=-1)   # project_z
terminated = torch.tensor([0., 0., 1., 0., 1., 0., 0., 0.], dtype=DT)
discount = (DISCOUNT * (1.0 - terminated)).reshape(-1, 1)

out = {}
# ---- weights (BEFORE any optimizer step) ----
for name, mod in (("phi", a.model.phi), ("sf_psi", a.model.sf_psi),
                  ("psm_psi", a.model.psm_psi), ("actor", a.model.actor)):
    for k, v in mod.state_dict().items():
        # .copy(): torch CPU state_dict tensors alias their storage and the
        # optimizer steps below mutate it IN PLACE -- without the copy every
        # exported weight silently becomes its post-step value.
        out[f"w__{name}.{k}"] = v.detach().cpu().numpy().copy()
# targets start as exact copies of the online nets, so the JAX side can reuse w__*.

# ---- raw network forwards (isolates net transcription from loss transcription) ----
with torch.no_grad():
    out["out__phi_goal"] = a.model.phi(goal).numpy()
    out["out__sf_psi"] = a.model.sf_psi(obs, z_sf, action).numpy()
    out["out__psm_psi"] = a.model.psm_psi(obs, z_psm, action).numpy()
    qs = (a.model.sf_psi(obs, z_sf, action) * z_sf).sum(-1)
    qm, qu = a.model.get_targets_uncertainty(qs)
    out["out__q"] = (qm - ACTOR_PESS * qu).numpy()
    out["out__qs"] = qs.numpy()
    # reward inference (infer_cond): z = reward^T phi(next_obs) / N, projected
    rew = torch.randn(B, 1, generator=g, dtype=DT)
    out["in__rewards"] = rew.numpy()
    # PSMModel.reward_inference body, minus its .float() casts (the model is float64
    # here) and minus the Identity normalizer/bw-encoder, which are no-ops for state.
    _phi_g = a.model.phi(next_obs)
    out["out__infer_z"] = a.model.project_z(
        torch.matmul(rew.T, _phi_g) / _phi_g.shape[0]).numpy()

# ---- losses (injected next_action; metrics are computed BEFORE the optim step) ----
m_psm = a._update_psm(obs, action, discount, next_obs, torch.arange(B), goal, z_psm,
                      next_action=na_psm)
# _update_psm stepped optim_phi/optim_psm_psi. The SF branch in the reference runs
# AFTER that step and reads the UPDATED phi -- exactly what the seam does (stage "sf"
# reads online params after stage "psm" applied its gradients). Export the post-step
# phi weights so the JAX side can reproduce the SF stage from the same state.
for k, v in a.model.phi.state_dict().items():
    out[f"w1__phi.{k}"] = v.detach().cpu().numpy().copy()
m_sf = a._update_sf(obs, action, discount, next_obs, goal, z_sf, next_action=na_sf)

for k, v in list(m_psm.items()) + list(m_sf.items()):
    out[f"loss__{k}"] = np.asarray(v.detach().cpu().numpy(), np.float64)

out.update({
    "in__observations": obs.numpy(), "in__actions": action.numpy(),
    "in__next_observations": next_obs.numpy(), "in__goal": goal.numpy(),
    "in__next_action_psm": na_psm.numpy(), "in__next_action_sf": na_sf.numpy(),
    "in__z_psm": z_psm.numpy(), "in__z_sf": z_sf.numpy(),
    "in__terminated": terminated.numpy(), "in__discount": discount.numpy(),
})
out.update({k: np.asarray(v) for k, v in dict(
    dim_obs=OBS_DIM, dim_act=ACT_DIM, dim_z=Z_DIM, dim_max_log_seed=MAX_LOG_SEED,
    dim_phi_hidden=PHI_HID, dim_psi_hidden=PSI_HID, dim_num_parallel=NPAR, dim_batch=B,
    cfg_discount=DISCOUNT, cfg_pessimism=PESS, cfg_actor_pessimism=ACTOR_PESS,
    cfg_ortho_coef=ORTHO,
).items()})

np.savez(sys.argv[1], **out)
print("wrote", sys.argv[1], len(out), "arrays")
for k in sorted(out):
    if k.startswith("loss__"):
        print(" ", k, float(out[k]))
