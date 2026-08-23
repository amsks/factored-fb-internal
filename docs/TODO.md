# Pending work

State as of `820d5dd`. Written at the end of the branch's build-out, so the "why" behind
each item is still recoverable. Items are grouped by who can act on them, not by priority
within the codebase.

## Decisions that need a human

**GCIVL's AWR temperature.** Vendored `gcivl.py` uses `alpha=10.0`;
`impls/configs/actor/awr.yaml` applies `3.0` to every critic, and `load_config` merges the
ACTOR group last, so `critic/gcivl.yaml` cannot override it. Correct for hiql
(`low_alpha=3.0`), a >3x temperature error for gcivl. Either restructure config precedence
so a critic can pin an actor knob, or have the launcher emit
`--override actor.alpha=10.0` for gcivl cells. Reproducing vendored GCIVL needs one or the
other.

**`measure_hit_spec` outside cube.** `density_fb.build` defaults to cube-single's
`"19:22:0.4"`. `_check_rows` catches it on point_mass_maze (4 dims < 22, raises) but NOT on
dmc-walker (24 >= 22), where it silently slices velocity columns as "object coordinates" at
a cube-metre radius. `env/domains/dmc-walker.yaml` and `dmc-cheetah.yaml` leave the key
unset and say NOT VERIFIED. Cheetah's forward/backward tasks are velocity-specified, so the
right block may not be a position block at all -- this needs a per-domain decision, and a
guard that raises rather than slices whatever fits.

**`density_fb`'s B-centering term.** The docstring keeps it on the grounds that the
redundancy with reward-centering is an artifact of plug-in moments that "DISAPPEARS" under
an EMA. That reasoning is wrong: the redundancy follows from `sum_j(r_j - rbar) = 0` and
holds for ANY mu_hat, including the EMA'd one now shipped. So the term is mathematically
dead code. Either delete it or fix the claim -- but note the end-to-end invariance tests
stay green under either choice, so judge it by the formula-level tests.

## Nothing has trained

The cube anchor is the only trained result on this branch. Every other method has
loss-level parity and smoke tests and no learning curve.

- **T1** (288 jobs) -- gate cleared and the launcher works, but the MetaWorld/ManiSkill data
  is not on this machine. Needs `FB_DATA_ROOT` and `FB_HF_EXPORT` populated.
- **T2** (192 jobs) -- same blocker.
- **T3** (36 jobs) -- only `dmc-point-mass-maze` has data locally.
- **T4** (9 jobs) -- online is now verified runnable, so this one is unblocked.
- No success rate exists for psm, rldp, gcivl or hiql; no evidence the reward index
  reproduces the shim's .816.

## Environment

- **`nvidia-cudnn` is missing from `/dev/shm/.venv-jax`** -- `nvidia.cudnn` resolves to
  `None` and a bare `nn.Conv` fails with `CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH`, so the
  pixel path is CPU-only in that venv. The torch venv's GPU conv works fine. That venv is
  volatile (rebuilt into /dev/shm), which is probably how the wheel went missing.
- **`FB_MUJOCO_SITE_PACKAGES` unset** -- 8 errors in `tests/impls/test_exorl_env.py`, the
  only red in the suite. Needs a site-packages dir whose mujoco matches this dm_control.
- **ExoRL walker/cheetah buffers were never downloaded** -- their envs and all 12 reward
  tasks run against real mujoco, but the loader path was exercised against fixtures, so
  their observation/action layouts are INFERRED rather than checked.

## Known gaps in the code

- **`measure_density_z` + pixels raises.** The frozen eval pool holds object-coordinate
  columns the pixel B tower cannot consume, so the reward interface and the pixel path do
  not compose. Design change, not wiring.
- **`--infer_cond_episodes` is statistically weak on point_mass_maze.** Only ~0.08% of RND
  states fall inside `reach_top_left`'s full-credit radius, so a uniform M=256 draw contains
  a rewarded state ~18% of the time; the rest of the time the index is the neutral fallback
  and `infer_cond_noop_reason` does NOT warn, because it only catches exactly-zero. Zero-shot
  readouts here need many more samples or reward-aware sampling.
- **The byte-identity gate is threading-sensitive.** It reports ~194 moved arrays on
  UNMUTATED source under `OMP_NUM_THREADS=1` plus single-threaded XLA. It pins CPU-vs-GPU
  via `_require_cpu` and nothing pins intra-op parallelism. Documented in the README;
  a guard in the fixture would be better than a note.
- **Gate coverage is 40 of 60 pairs** -- the two actor variants `ddpgbc_density` and
  `bcflow` are never gated.

## Test quality

From the mutation audit (45 mutations, 14 survived, 10 killed by new tests):

- **Two tautological tests remain.** psm's proto-seed test recomputes `powers` in the test
  body and calls `proto_actions` directly instead of driving `sample_cond`; HGCDataset's
  overshoot test asserts only that the target is *some* dataset row, which is true of any
  index gather.
- **Target-network reads are systematically under-tested.** Collapsing every target net onto
  its online net leaves the psm, gcivl, hiql and rldp seam suites fully green. New tests
  cover density_fb and gcivl; the other two have none. Root cause is structural: `build()`
  initialises targets as exact copies, so any test that runs a single update cannot tell a
  target read from an online one.

## Housekeeping

- 13 agent worktrees are still registered under `.claude/worktrees/`.
- **`git stash` is shared across worktrees in this repo.** Two agents collided on it during
  the build-out -- one lost an edit, one popped another's work. Belongs in the brief for any
  future parallel work.

## Deliberately NOT to fix

- **RLDP's self-predictive off-by-one.** `agents/rldp/agent.py::update_fb` builds targets as
  `[B_t(next_obs)] + [B_t(future_obs[:, i])]` while actions advance, but `future_obs[:, 0]`
  IS `next_obs`, so from rollout step k>=2 the target lags one step behind its action.
  Transcribed verbatim into `impls/critics/rldp.py` so both implementations compute the same
  number. Correcting it means correcting both sides and expecting published RLDP numbers to
  move.
- **`qrl x awr` is BC-equivalent** (`adv = 0`): qrl's `q_fn` is `V(s,g)` ignoring actions, so
  the seam's default `q_fn - v_fn` advantage vanishes and AWR degenerates to behaviour
  cloning. Giving qrl an `adv_fn` would fix it and move already-published cells.
