"""campaign/registry.py — the 19-dataset x 5-algorithm training matrix.

Datasets: the 18 collected Factor-FB tasks (8 Box2D + 5 Metaworld + 5 ManiSkill,
unified format) + OGBench cube-single (ogbench format). Algorithms: 2 native
(fb, fb_flowbc via train.py) + 3 JAX baselines (gciql, crl, crl_flowbc via
run_gciql / the OGBench pipeline). Training-only (no eval), 3 seeds, checkpoints.
"""

COLLECTED = "/raid/tongle/rl/fb-data/collected"     # unified per-episode datasets
OGBENCH = "/raid/tongle/rl/fb-data/datasets"        # ogbench-format cube-single
CAMPAIGN_OUT = "/raid/tongle/rl/fb-data/campaign"   # run outputs / checkpoints

SEEDS = [0, 1, 2]
NUM_TRAIN_STEPS = 500_000     # train-only budget (expert data); checkpoints every 100k
SAVE_EVERY = 100_000

# ---- DEV set: a small manip-focused subset for fast jax-unified iteration ----
# (goal: get FB or CRL working on manipulation). All have collected data already.
# Metaworld push/pick-place have no newt TD-MPC2 checkpoint -> swapped to bin-picking
# (pick) + button-press. Loco tasks (b2d-center-ctrl, goal-ctrl-maze) kept as sanity.
DEV_TASKS = [
    "cube-single",                                        # ogbench (play data; validation anchor 0.47)
    "mw-bin-picking", "mw-button-press",                 # metaworld
    "ms-pick-cube", "ms-push-cube",                      # maniskill
    "b2d-center-ctrl", "b2d-goal-ctrl-maze",             # box2d loco (sanity)
    "b2d-center-wall", "b2d-goal-ctrl-target", "b2d-center-ctrl-target",  # box2d manip
]

BOX2D = ["b2d-center-ctrl", "b2d-center-ctrl-target", "b2d-center-ctrl-target-hard",
         "b2d-center-wall", "b2d-center-wall-navigation", "b2d-goal-ctrl",
         "b2d-goal-ctrl-maze", "b2d-goal-ctrl-target"]
METAWORLD = ["mw-assembly", "mw-basketball", "mw-bin-picking", "mw-box-close", "mw-button-press"]
MANISKILL = ["ms-lift-peg", "ms-pick-cube", "ms-poke-cube", "ms-push-cube", "ms-stack-cube"]


def _ds(slug, family, domain, fmt, path):
    return dict(slug=slug, family=family, domain=domain, data_format=fmt, data_path=path)


DATASETS = (
    [_ds(t, "Box2D", f"Box2D/{t}", "unified", COLLECTED) for t in BOX2D]
    + [_ds(t, "Metaworld", f"Metaworld/{t}", "unified", COLLECTED) for t in METAWORLD]
    + [_ds(t, "ManiSkill", f"ManiSkill/{t}", "unified", COLLECTED) for t in MANISKILL]
    + [_ds("cube-single", "OGBench", "cube-single-play-v0", "ogbench", OGBENCH)]
)

# Native (PyTorch, train.py) agents. bc_coeff=0 for plain FB; 3.0 (manipulation) for FlowBC.
NATIVE_AGENTS = {
    "fb":        dict(agent="fb",        bc_coeff=0.0),
    "fb_flowbc": dict(agent="fb_flowbc", bc_coeff=3.0),
}

# JAX baselines (run via the OGBench pipeline / adapter). alpha per the cube recipe.
JAX_AGENTS = {
    "gciql":      dict(agent_file="agents/gciql.py",                 alpha=1.0),
    "crl":        dict(agent_file="agents/crl.py",                   alpha=3.0),
    "crl_flowbc": dict(agent_file="tools/wandb_mode_shim/crl_flowbc.py", bc_coeff=3.0),
}
