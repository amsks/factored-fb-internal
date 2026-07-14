"""Pinned constants for the value-function atlas. Every value here is TUNABLE;
change in one place. See spec section 10
(PAPER/fb-gciql-bridge/value-function-atlas-design.md)."""
SEED = 0
OBS_DIM = 28
ACTION_DIM = 5
CONTACT_IDX = 18                    # obs[18] = gripper contact channel
CUBE_SLICE = slice(14, 17)         # cube xyz in qpos

# action banks
N_MU = 256
N_PERTURB = 128
PERTURB_SIGMA = 0.1
N_GLOBAL = 128

# regularized whitening for support distance (Sigma_{mu,lambda})
COV_SHRINK_LAMBDA = 0.1
COV_EPS = 1e-4

# state bank
PHASES = ('P0', 'P1', 'P2', 'P3')
K_STATES_PER_PHASE = 32

# goal banks
N_EVAL_GOALS = 5
N_HINDSIGHT = 32
N_GRID = 64

# oracle (flow-proposed, return-selected MPPI)
MPPI_H = 40
MPPI_N_SEQ = 128
MPPI_TEMP = 1.0
MPPI_REPLAN = 1
MPPI_REPLICATES = 4

# analysis
EQUIV_MARGIN_DELTA = 0.05          # Spearman-rho equivalence interval
KNN_K = 8
TOPQ = 0.10
MIN_COMPONENT = 3
