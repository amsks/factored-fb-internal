# Seam Boot Contract (Workstream 0) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every critic in the seam composable with every actor family by giving `DDPGBCActor`, `AWRActor` and `FQLActor` the `next_action_vgoal` bootstrap key that only `FlowBCActor` emits today, and declaring which key each critic reads via `Critic.BOOT_COND` — without moving a single existing number.

**Architecture:** Purely additive. `boot` is drawn by `actor.sample_bootstrap`, but *which* bootstrap action a TD target is valid at is a **critic** property, and the answer genuinely differs per critic (`density_fb` needs the value goal; `td_fb` measures 0.920 at the actor goal vs 0.076 at the value goal). `FlowBCActor` already resolves this by emitting **both** keys, guarded on `"value_goal" in cond`, reusing its existing noise. The other three actors copy that pattern exactly. `BOOT_COND` is a *declaration* validated by `check_combo`; it never steers a draw, so no existing key, loss or info value can change.

**Tech Stack:** JAX 0.10.0 / Flax, `dataclasses.dataclass(frozen=True)` critics and actors, pytest.

## Global Constraints

- **Repo:** `~/git/Factored-FB-seam`, branch `feat/seam-boot-contract` (a git worktree of `~/git/Factored-FB`).
- **Interpreter:** `/dev/shm/.venv-jax/bin/python`. The repo's own `.venv` is torch-only and cannot run this tree.
- **Every command needs `PYTHONPATH=$PWD`.** Without it `import impls` resolves to `~/git/Factored-FB/third_party/ogbench/impls` and dies on `ModuleNotFoundError: No module named 'agents'`.
- **The bit-identity gate needs `JAX_PLATFORMS=cpu`.** The fixture is byte-identity and therefore platform-specific; `tests/impls/conftest.py` imports jax before any module-level `setdefault` can run. The gate fails loudly naming the backend if you forget.
- **Do NOT modify `impls/actors/ddpgbc_density.py` or `DDPGBCActor.loss`.** `DDPGBCDensityActor.loss` uses `q_scale = |qs|.mean()` off `q_fn`'s **second** return; stock `DDPGBCActor` uses `|q|.mean()` off the first. Those differ for every critic in the tree (`gciql` returns `(min(q1,q2), stack([q1,q2]))`, `sf_feature` returns `(min(q1,q2), v)`, `td_fb` returns `(q, (q - q.mean())[None])`). `ddpgbc_density` is the published `hard-ddpgbc 0.920/0.868/0.844` arm and stays.
- **Do NOT edit `tools/wandb_mode_shim/gciql_flow.py`** in the sibling repo. It is a read-only reference.
- **Commit messages carry no Claude co-author trailer.**
- Design of record: `docs/specs/2026-08-21-density-fb-agent-branch.md`.

**Run the full seam suite with:**
```bash
cd ~/git/Factored-FB-seam
PYTHONPATH=$PWD JAX_PLATFORMS=cpu /dev/shm/.venv-jax/bin/python -m pytest tests/impls -q
```

---

## File Structure

| file | responsibility | task |
|---|---|---|
| `tests/impls/gen_boot_contract_fixture.py` | builds every critic×actor pair at fixed seeds; writes the byte-identity baseline | 1 (done) |
| `tests/impls/fixtures/boot_contract_reference.npz` | the baseline: 49 pairs, 612 arrays | 1 (done) |
| `tests/impls/test_boot_contract.py` | the gate + the td_fb behavioural guard | 1 (done), extended in 2 |
| `impls/critics/base.py` | `Critic` protocol; `BOOT_COND` + `check_combo` validation | 2 |
| `impls/critics/density_fb.py` | declares `BOOT_COND = "value_goal"` | 2 |
| `impls/actors/ddpgbc.py` | emits `next_action_vgoal` | 3 |
| `impls/actors/awr.py` | emits `next_action_vgoal` | 4 |
| `impls/actors/fql.py` | emits `next_action_vgoal`, reusing `na_noise` | 5 |
| `tests/impls/test_density_fb_actors.py` | density_fb × all five actors build and update | 6 |

---

## Task 1: Bit-identity baseline — **ALREADY COMPLETE** (`519fa38`)

Landed before the plan was written, because the baseline must be captured from a tree with **no** workstream-0 edits and that window was now.

**Delivered:** `tests/impls/gen_boot_contract_fixture.py`, `tests/impls/fixtures/boot_contract_reference.npz` (49 pairs, 612 arrays), `tests/impls/test_boot_contract.py` (4 tests, all passing).

Three pairs are deliberately **absent** from the baseline because they raise today — `density_fb × {ddpgbc, awr, fql}` — which is exactly what Tasks 3–5 fix. They are covered by Task 6's tests, not by this gate.

To regenerate (only ever from a tree with no workstream-0 edits):
```bash
cd ~/git/Factored-FB-seam
JAX_PLATFORMS=cpu PYTHONPATH=$PWD /dev/shm/.venv-jax/bin/python tests/impls/gen_boot_contract_fixture.py
```

---

## Task 2: `BOOT_COND` declaration and its validation

**Files:**
- Modify: `impls/critics/base.py`
- Modify: `impls/critics/density_fb.py:139-142` (the class-attribute block)
- Test: `tests/impls/test_boot_contract.py`

**Interfaces:**
- Consumes: `ComboError`, `_name`, `check_combo` from `impls/critics/base.py`.
- Produces: `BOOT_KEY: dict[str, str]` and `boot_key(critic) -> str` in `impls/critics/base.py`. Tasks 3–5 do **not** import these — actors stay unaware of `BOOT_COND` by design. Task 6 uses `boot_key` in an assertion.

- [ ] **Step 1: Write the failing tests**

Append to `tests/impls/test_boot_contract.py`:

```python
def test_every_critic_declares_a_known_boot_cond():
    """BOOT_COND is a declaration of which bootstrap key a critic's TD target reads.
    Absent means the default, "actor_goal"."""
    from impls.agents import CRITICS
    from impls.critics.base import BOOT_KEY
    for name, critic in sorted(CRITICS.items()):
        cond = getattr(critic, "BOOT_COND", "actor_goal")
        assert cond in BOOT_KEY, f"{name} declares BOOT_COND={cond!r}"


def test_density_fb_is_the_only_value_goal_critic():
    """density_fb's forward tower is F(s,a,g), so its row is g-indexed and its target
    must bootstrap at that same g. Every other critic in the tree reads the actor-goal
    draw -- td_fb measurably so (0.920 vs 0.076)."""
    from impls.agents import CRITICS
    value_goal = {n for n, c in CRITICS.items()
                  if getattr(c, "BOOT_COND", "actor_goal") == "value_goal"}
    assert value_goal == {"density_fb"}


def test_boot_key_maps_the_declaration_to_the_dict_key():
    from impls.critics.base import boot_key
    from impls.critics.density_fb import DensityFBCritic
    from impls.critics.gciql import GCIQLCritic
    assert boot_key(DensityFBCritic) == "next_action_vgoal"
    assert boot_key(GCIQLCritic) == "next_action"


def test_check_combo_rejects_an_unknown_boot_cond():
    import pytest as _pytest
    from impls.critics.base import check_combo, ComboError
    from impls.actors.ddpgbc import DDPGBCActor

    class _Bogus:
        BOOT_COND = "somewhere_else"
        PROVIDES_V = True
        PROVIDES_COND = frozenset({"goal"})

    with _pytest.raises(ComboError, match="BOOT_COND"):
        check_combo(_Bogus, DDPGBCActor)


def test_check_combo_rejects_value_goal_the_critic_never_produces():
    """An actor only emits next_action_vgoal when "value_goal" in cond. A critic that
    declares BOOT_COND="value_goal" but whose sample_cond omits it would KeyError deep
    inside stage_loss; catch it at build time with a message that names the cause."""
    import pytest as _pytest
    from impls.critics.base import check_combo, ComboError
    from impls.actors.ddpgbc import DDPGBCActor

    class _Inconsistent:
        BOOT_COND = "value_goal"
        PROVIDES_V = True
        PROVIDES_COND = frozenset({"goal"})     # no "value_goal"

    with _pytest.raises(ComboError, match="value_goal"):
        check_combo(_Inconsistent, DDPGBCActor)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd ~/git/Factored-FB-seam
PYTHONPATH=$PWD JAX_PLATFORMS=cpu /dev/shm/.venv-jax/bin/python -m pytest \
  tests/impls/test_boot_contract.py -q -k "boot_cond or boot_key or value_goal_critic or check_combo"
```
Expected: FAIL — `ImportError: cannot import name 'BOOT_KEY' from 'impls.critics.base'`.

- [ ] **Step 3: Add `BOOT_COND` to the protocol and the validation**

In `impls/critics/base.py`, add to the `Critic` Protocol body, right after `PROVIDES_COND`:

```python
    BOOT_COND: str          # which cond the TD bootstrap action must be drawn at:
                            # "actor_goal" (default) -> boot["next_action"]
                            # "value_goal"           -> boot["next_action_vgoal"]
```

Then, after the `_name` helper, add:

```python
# A critic's TD target is only valid at one bootstrap action, and which one is a
# property of the critic, not of the actor drawing it. Actors emit BOTH keys (guarded
# on "value_goal" in cond); this maps a critic's declaration to the key it reads.
# Deliberately NOT consulted by any actor: keeping it out of the draw is what makes
# this change unable to move an existing number.
BOOT_KEY = {"actor_goal": "next_action", "value_goal": "next_action_vgoal"}


def boot_key(critic):
    """The boot dict key `critic`'s TD target reads. Default "actor_goal"."""
    cond = getattr(critic, "BOOT_COND", "actor_goal")
    if cond not in BOOT_KEY:
        raise ComboError(
            f"{_name(critic)}.BOOT_COND is {cond!r}; expected one of {sorted(BOOT_KEY)}"
        )
    return BOOT_KEY[cond]
```

And inside `check_combo`, before the existing `NEEDS_V` check:

```python
    boot_cond = getattr(critic, "BOOT_COND", "actor_goal")
    if boot_cond not in BOOT_KEY:
        raise ComboError(
            f"{_name(critic)}.BOOT_COND is {boot_cond!r}; expected one of {sorted(BOOT_KEY)}"
        )
    if boot_cond == "value_goal" and "value_goal" not in set(getattr(critic, "PROVIDES_COND", ())):
        raise ComboError(
            f"{_name(critic)}.BOOT_COND is 'value_goal' but its sample_cond does not "
            f"produce 'value_goal', so no actor will emit boot['next_action_vgoal'] "
            f"(every actor guards that draw on \"value_goal\" in cond)"
        )
```

- [ ] **Step 4: Declare it on `density_fb`**

In `impls/critics/density_fb.py`, in the class-attribute block alongside `GROUPS`/`STAGES`/`PROVIDES_COND` (around line 139-142), add:

```python
    # F(s, a, g): the row is g-indexed, so the TD target must bootstrap at that same g.
    # Reading the actor-goal action instead makes w_bar a goal-averaged (behaviour-like)
    # density -- measured density_ess_median ~310 against a healthy ~89.
    BOOT_COND = "value_goal"
```

- [ ] **Step 5: Run the new tests to verify they pass**

```bash
cd ~/git/Factored-FB-seam
PYTHONPATH=$PWD JAX_PLATFORMS=cpu /dev/shm/.venv-jax/bin/python -m pytest \
  tests/impls/test_boot_contract.py -q
```
Expected: PASS — 9 tests (the original 4 plus the 5 new ones).

- [ ] **Step 6: Commit**

```bash
cd ~/git/Factored-FB-seam
git add impls/critics/base.py impls/critics/density_fb.py tests/impls/test_boot_contract.py
git -c commit.gpgsign=false commit -m "feat(workstream 0): declare which bootstrap key each critic reads

Critic.BOOT_COND names the cond a critic's TD target is valid at -- 'actor_goal'
(default, boot['next_action']) or 'value_goal' (boot['next_action_vgoal']).
density_fb declares the latter: its forward tower is F(s,a,g), so its row is
g-indexed and bootstrapping at the actor goal makes w_bar a goal-averaged density.

It is a declaration, not control flow. No actor consults it; keeping it out of the
draw is what makes this change unable to move an existing number. check_combo now
rejects an unknown value, and rejects BOOT_COND='value_goal' on a critic whose
sample_cond omits 'value_goal' -- which would otherwise KeyError deep inside
stage_loss, since every actor guards that draw on \"value_goal\" in cond."
```

---

## Task 3: `DDPGBCActor` emits `next_action_vgoal`

**Files:**
- Modify: `impls/actors/ddpgbc.py:40-42` (`sample_bootstrap` only — **not** `loss`)
- Test: `tests/impls/test_density_fb_actors.py` (create)

**Interfaces:**
- Consumes: `_cvec(cond)` and `self._dist(params, obs, cvec, temperature=1.0)`, both already in `impls/actors/ddpgbc.py`.
- Produces: `boot["next_action_vgoal"]`, shape `(B, action_dim)`, clipped to `[-1, 1]`. Tasks 4 and 5 produce the same key from `awr` and `fql`.

- [ ] **Step 1: Write the failing test**

Create `tests/impls/test_density_fb_actors.py`:

```python
"""density_fb composes with every actor family (workstream 0, tasks 3-5).

Before this workstream only FlowBCActor emitted the value-goal bootstrap draw
density_fb requires, so density_fb x {ddpgbc, awr, fql} raised KeyError from
stage_loss. These are the tests for each actor gaining it.
"""
import numpy as np
import pytest

from tests.impls.gen_boot_contract_fixture import pair_statics


def _check(actor):
    boot, info = pair_statics("density_fb", actor)
    assert "next_action_vgoal" in boot, f"{actor} did not emit the value-goal draw"
    assert not np.allclose(boot["next_action"], boot["next_action_vgoal"]), (
        f"{actor}'s value-goal draw equals its actor-goal draw -- it was almost "
        f"certainly conditioned on the wrong goal"
    )
    assert np.isfinite(float(np.asarray(info["density_ce"]))), "density_ce is not finite"
    assert np.isfinite(float(np.asarray(info["actor_loss"]))), "actor_loss is not finite"


def test_density_fb_composes_with_ddpgbc():
    _check("ddpgbc")
```

- [ ] **Step 2: Run it to verify it fails**

```bash
cd ~/git/Factored-FB-seam
PYTHONPATH=$PWD JAX_PLATFORMS=cpu /dev/shm/.venv-jax/bin/python -m pytest \
  tests/impls/test_density_fb_actors.py::test_density_fb_composes_with_ddpgbc -q
```
Expected: FAIL — `KeyError: "density_fb needs boot['next_action_vgoal'] — the bootstrap action drawn at cond['value_goal'] …"`.

- [ ] **Step 3: Emit the key**

Replace `sample_bootstrap` in `impls/actors/ddpgbc.py` (currently lines 40-42) with:

```python
    def sample_bootstrap(self, net, batch, cond, rng):
        next_obs = batch["next_observations"]
        p = net["actor"].params
        boot = {"next_action": jnp.clip(self._dist(p, next_obs, _cvec(cond)).mode(), -1, 1)}
        if "value_goal" in cond:
            # a+ ~ pi(.|s', g_value) at frozen params, for critics whose TD target is only
            # valid at the VALUE goal (Critic.BOOT_COND == "value_goal"; density_fb's forward
            # tower is F(s,a,g), so its row is g-indexed). Emitted as a SEPARATE key -- every
            # critic reading `next_action` stays bit-identical. Mirrors
            # FlowBCActor.sample_bootstrap; this actor is deterministic (dist.mode()), so the
            # extra draw spends no randomness at all.
            boot["next_action_vgoal"] = jnp.clip(
                self._dist(p, next_obs, cond["value_goal"]).mode(), -1, 1)
        return boot
```

- [ ] **Step 4: Run the test and the bit-identity gate**

```bash
cd ~/git/Factored-FB-seam
PYTHONPATH=$PWD JAX_PLATFORMS=cpu /dev/shm/.venv-jax/bin/python -m pytest \
  tests/impls/test_density_fb_actors.py tests/impls/test_boot_contract.py -q
```
Expected: PASS. The gate proves no existing `× ddpgbc` cell moved.

- [ ] **Step 5: Commit**

```bash
cd ~/git/Factored-FB-seam
git add impls/actors/ddpgbc.py tests/impls/test_density_fb_actors.py
git -c commit.gpgsign=false commit -m "feat(workstream 0): DDPGBCActor emits the value-goal bootstrap draw

Mirrors FlowBCActor.sample_bootstrap: next_action stays the actor-goal draw, and
next_action_vgoal is added whenever the critic's cond carries 'value_goal'. This
actor is deterministic (dist.mode()), so the extra draw spends no randomness and
next_action is untouched -- the bit-identity gate confirms every existing x ddpgbc
cell is byte-identical.

Unblocks density_fb x ddpgbc, which raised KeyError from stage_loss before.
DDPGBCActor.loss is deliberately NOT touched: ddpgbc_density's centered |qs|.mean()
normalizer is a different actor loss, not a missing feature here."
```

---

## Task 4: `AWRActor` emits `next_action_vgoal`

**Files:**
- Modify: `impls/actors/awr.py:46-48` (`sample_bootstrap` only)
- Test: `tests/impls/test_density_fb_actors.py`

**Interfaces:**
- Consumes: `_cvec(cond)` and `self._dist(...)`, already in `impls/actors/awr.py`.
- Produces: `boot["next_action_vgoal"]`, same shape and clipping as Task 3.

- [ ] **Step 1: Write the failing test**

Append to `tests/impls/test_density_fb_actors.py`:

```python
def test_density_fb_composes_with_awr():
    _check("awr")
```

- [ ] **Step 2: Run it to verify it fails**

```bash
cd ~/git/Factored-FB-seam
PYTHONPATH=$PWD JAX_PLATFORMS=cpu /dev/shm/.venv-jax/bin/python -m pytest \
  tests/impls/test_density_fb_actors.py::test_density_fb_composes_with_awr -q
```
Expected: FAIL — the same `KeyError: "density_fb needs boot['next_action_vgoal'] …"`.

- [ ] **Step 3: Emit the key**

Replace `sample_bootstrap` in `impls/actors/awr.py` (currently lines 46-48) with:

```python
    def sample_bootstrap(self, net, batch, cond, rng):
        next_obs = batch["next_observations"]
        p = net["actor"].params
        boot = {"next_action": jnp.clip(self._dist(p, next_obs, _cvec(cond)).mode(), -1, 1)}
        if "value_goal" in cond:
            # See the block comment in impls/actors/ddpgbc.py::sample_bootstrap: a+ drawn at
            # the VALUE goal, for critics declaring BOOT_COND == "value_goal". Separate key,
            # deterministic draw, no randomness spent, next_action untouched.
            boot["next_action_vgoal"] = jnp.clip(
                self._dist(p, next_obs, cond["value_goal"]).mode(), -1, 1)
        return boot
```

- [ ] **Step 4: Run the tests and the gate**

```bash
cd ~/git/Factored-FB-seam
PYTHONPATH=$PWD JAX_PLATFORMS=cpu /dev/shm/.venv-jax/bin/python -m pytest \
  tests/impls/test_density_fb_actors.py tests/impls/test_boot_contract.py -q
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd ~/git/Factored-FB-seam
git add impls/actors/awr.py tests/impls/test_density_fb_actors.py
git -c commit.gpgsign=false commit -m "feat(workstream 0): AWRActor emits the value-goal bootstrap draw

Same additive change as DDPGBCActor: next_action unchanged, next_action_vgoal added
under the \"value_goal\" in cond guard. Deterministic draw, no randomness spent.
Unblocks density_fb x awr; the bit-identity gate confirms no existing x awr cell moved."
```

---

## Task 5: `FQLActor` emits `next_action_vgoal`

**Files:**
- Modify: `impls/actors/fql.py:58-65` (`sample_bootstrap` only)
- Test: `tests/impls/test_density_fb_actors.py`

**Interfaces:**
- Consumes: `_cvec(cond)` and `self._one(params, obs_c, noise)` (the one-step flow policy), already in `impls/actors/fql.py`.
- Produces: `boot["next_action_vgoal"]`, same shape and clipping as Tasks 3-4.

This is the only one of the three that spends randomness, so the noise-reuse discipline matters: the extra draw must reuse `na_noise` rather than splitting again, or `next_action` moves and the gate fails.

- [ ] **Step 1: Write the failing test**

Append to `tests/impls/test_density_fb_actors.py`:

```python
def test_density_fb_composes_with_fql():
    _check("fql")


def test_fql_reuses_its_bootstrap_noise():
    """fql is the only one of the three that draws randomness. If the value-goal draw
    split the key again instead of reusing na_noise, `next_action` -- and every existing
    x fql cell with it -- would move. The bit-identity gate catches that too; this names
    the cause directly."""
    import numpy as np
    from tests.impls.gen_boot_contract_fixture import FIXTURE, pair_statics
    baseline = dict(np.load(FIXTURE, allow_pickle=True))
    boot, _ = pair_statics("gciql", "fql")
    for key in ("next_action", "flow_x0", "flow_t", "actor_noise"):
        assert np.array_equal(boot[key], baseline[f"boot|gciql|fql|{key}"]), (
            f"fql's {key} moved -- the value-goal draw spent a split it should have reused"
        )
```

- [ ] **Step 2: Run them to verify they fail**

```bash
cd ~/git/Factored-FB-seam
PYTHONPATH=$PWD JAX_PLATFORMS=cpu /dev/shm/.venv-jax/bin/python -m pytest \
  tests/impls/test_density_fb_actors.py -q -k "fql"
```
Expected: `test_density_fb_composes_with_fql` FAILS with the `KeyError`; `test_fql_reuses_its_bootstrap_noise` PASSES (nothing has moved yet). Both must pass after Step 3.

- [ ] **Step 3: Emit the key, reusing the noise**

Replace `sample_bootstrap` in `impls/actors/fql.py` (currently lines 58-65) with:

```python
    def sample_bootstrap(self, net, batch, cond, rng):
        next_obs = batch["next_observations"]
        obs_c = jnp.concatenate([next_obs, _cvec(cond)], -1)
        B, A = batch["actions"].shape
        r_na, r_x0, r_t, r_n = jax.random.split(rng, 4)
        # hoisted out of the call below so the value-goal draw can REUSE it
        na_noise = jax.random.normal(r_na, (B, A))
        na = jnp.clip(self._one(net["actor_onestep"].params, obs_c, na_noise), -1, 1)
        boot = {"next_action": na, "flow_x0": jax.random.normal(r_x0, (B, A)),
                "flow_t": jax.random.uniform(r_t, (B, 1)), "actor_noise": jax.random.normal(r_n, (B, A))}
        if "value_goal" in cond:
            # See impls/actors/ddpgbc.py::sample_bootstrap. Unlike ddpgbc/awr this actor is
            # stochastic, so the draw REUSES na_noise -- one draw, no extra split, the same
            # four-way split as before. Splitting again would move `next_action` and every
            # existing x fql cell with it.
            obs_v = jnp.concatenate([next_obs, cond["value_goal"]], -1)
            boot["next_action_vgoal"] = jnp.clip(
                self._one(net["actor_onestep"].params, obs_v, na_noise), -1, 1)
        return boot
```

- [ ] **Step 4: Run the tests and the gate**

```bash
cd ~/git/Factored-FB-seam
PYTHONPATH=$PWD JAX_PLATFORMS=cpu /dev/shm/.venv-jax/bin/python -m pytest \
  tests/impls/test_density_fb_actors.py tests/impls/test_boot_contract.py -q
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd ~/git/Factored-FB-seam
git add impls/actors/fql.py tests/impls/test_density_fb_actors.py
git -c commit.gpgsign=false commit -m "feat(workstream 0): FQLActor emits the value-goal bootstrap draw

fql is the only one of the three that spends randomness, so the value-goal draw
REUSES na_noise rather than splitting again -- one draw, the same four-way split as
before, next_action untouched. A dedicated test compares fql's four random arrays
against the baseline fixture so a future refactor that adds a split fails with the
cause named rather than as an opaque gate failure.

Unblocks density_fb x fql. All 52 critic x actor-family pairs now compose."
```

---

## Task 6: Integration — density_fb across all five actors, and the full suite

**Files:**
- Test: `tests/impls/test_density_fb_actors.py`

**Interfaces:**
- Consumes: `boot_key` from `impls/critics/base.py` (Task 2); `pair_statics` from `tests/impls/gen_boot_contract_fixture.py` (Task 1).
- Produces: nothing consumed downstream. This is the closing gate for workstream 0.

- [ ] **Step 1: Write the failing test**

Append to `tests/impls/test_density_fb_actors.py`:

```python
@pytest.mark.parametrize("actor", ["flowbc", "ddpgbc", "awr", "fql", "ddpgbc_density"])
def test_density_fb_runs_a_real_update_with_every_actor(actor):
    """The point of workstream 0: density_fb had one usable actor and now has five.
    compute_static only evaluates losses -- this runs a real gradient step."""
    import jax.numpy as jnp
    from impls.agents import Agent
    from impls.utils.config_utils import load_config
    from tests.impls.gen_boot_contract_fixture import make_batch, B, OBS, ACT

    cfg = load_config("density_fb", actor, "cube_single")
    cfg["batch_size"] = B
    agent = Agent.create("density_fb", actor, 0, jnp.zeros((B, OBS)), jnp.zeros((B, ACT)), cfg)
    updated, info = agent.update(make_batch())
    assert np.isfinite(float(np.asarray(info["density_ce"])))
    assert np.isfinite(float(np.asarray(info["actor_loss"])))
    # the update actually moved the density towers
    before = agent.net["density_f"].params
    after = updated.net["density_f"].params
    leaves_changed = [not np.array_equal(np.asarray(a), np.asarray(b))
                      for a, b in zip(jax.tree_util.tree_leaves(before),
                                      jax.tree_util.tree_leaves(after))]
    assert any(leaves_changed), "density_f params did not move"


def test_density_fb_reads_the_key_its_declaration_names():
    """Ties Task 2's declaration to Tasks 3-5's emission: perturbing the key BOOT_COND
    names must change density_ce, and perturbing the other one must not."""
    import jax.numpy as jnp
    from impls.agents import Agent
    from impls.critics.base import boot_key
    from impls.critics.density_fb import DensityFBCritic
    from impls.utils.config_utils import load_config
    from tests.impls.gen_boot_contract_fixture import make_batch, B, OBS, ACT

    assert boot_key(DensityFBCritic) == "next_action_vgoal"

    cfg = load_config("density_fb", "ddpgbc", "cube_single")
    cfg["batch_size"] = B
    agent = Agent.create("density_fb", "ddpgbc", 0, jnp.zeros((B, OBS)), jnp.zeros((B, ACT)), cfg)
    batch = make_batch()
    cond = agent.critic.sample_cond(agent.net, batch, jax.random.PRNGKey(0))
    boot = agent.actor.sample_bootstrap(agent.net, batch, cond, jax.random.PRNGKey(1))

    def density_ce(b):
        return float(np.asarray(agent.compute_static(batch, cond, b)["density_ce"]))

    base = density_ce(boot)
    junk = jnp.zeros_like(boot["next_action"])
    assert density_ce({**boot, "next_action_vgoal": junk}) != base, \
        "density_fb must READ next_action_vgoal -- the key its BOOT_COND names"
    assert density_ce({**boot, "next_action": junk}) == base, \
        "density_fb must IGNORE next_action, the actor-goal draw it explicitly rejects"
```

Add `import jax` to the imports at the top of the file.

- [ ] **Step 2: Run to verify it fails**

```bash
cd ~/git/Factored-FB-seam
PYTHONPATH=$PWD JAX_PLATFORMS=cpu /dev/shm/.venv-jax/bin/python -m pytest \
  tests/impls/test_density_fb_actors.py -q -k "real_update or declaration_names"
```
Expected: FAIL if Tasks 2-5 are incomplete. If Tasks 2-5 are all done, these pass immediately — that is the correct outcome for an integration task, and it is the confirmation, not a defect.

- [ ] **Step 3: Fix anything the integration surfaces**

No implementation is planned here. If a parametrized case fails, the fault is in Tasks 3-5's emission or Task 2's declaration; fix it there and re-run, rather than special-casing it in this file.

- [ ] **Step 4: Run the full seam suite**

```bash
cd ~/git/Factored-FB-seam
PYTHONPATH=$PWD JAX_PLATFORMS=cpu /dev/shm/.venv-jax/bin/python -m pytest tests/impls -q
```
Expected: PASS, with no test count lower than before workstream 0 began. Pay particular attention to `tests/impls/test_density_fb.py`, whose five `ddpgbc_density` tests must be untouched.

- [ ] **Step 5: Commit**

```bash
cd ~/git/Factored-FB-seam
git add tests/impls/test_density_fb_actors.py
git -c commit.gpgsign=false commit -m "test(workstream 0): density_fb across all five actors, and the declaration<->emission tie

Runs a real gradient step for density_fb x {flowbc, ddpgbc, awr, fql,
ddpgbc_density} and asserts the density towers actually move -- compute_static only
evaluates losses, so on its own it would not have caught a no-op update.

The second test closes the loop between Task 2 and Tasks 3-5 behaviourally:
perturbing the key BOOT_COND names must change density_ce, and perturbing the
other one must not. Before workstream 0, density_fb had exactly one usable actor."
```

---

## Definition of done

- [ ] `tests/impls` passes in full with `JAX_PLATFORMS=cpu`.
- [ ] `tests/impls/test_boot_contract.py` is green: all 49 baseline pairs byte-identical.
- [ ] `density_fb` composes with `flowbc`, `ddpgbc`, `awr`, `fql`, `ddpgbc_density`.
- [ ] `impls/actors/ddpgbc_density.py` and `DDPGBCActor.loss` are unmodified (`git diff origin/feat/metaworld-runs -- impls/actors/ddpgbc_density.py` is empty).
- [ ] The cube anchor (`density_fb × flowbc × cube-single`, 500k, target ≈0.929) is launched. It gates the campaign, not this merge.

Once merged, workstreams 1-6 in `docs/specs/2026-08-21-density-fb-agent-branch.md` fan out in parallel worktrees.
