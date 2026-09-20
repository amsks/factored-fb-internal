"""Intervention probe, pure parts (no simulator, no checkpoints): the selection rules, the
per-rollout metrics and the per-phase aggregation."""
import numpy as np
import pandas as pd
import pytest

from scripts.probes.cube_bottleneck import intervention as IV


# ── selection rules ─────────────────────────────────────────────────────────────────────

def test_select_candidate_takes_the_argmax_and_keeps_the_actor_on_ties():
    assert IV.select_candidate([1.0, 5.0, 3.0], own_idx=0) == 1
    assert IV.select_candidate([5.0, 5.0, 3.0], own_idx=0) == 0      # tie at the max: keep own
    assert IV.select_candidate([0.0, 0.0, 0.0], own_idx=2) == 2      # all tie: keep own
    assert IV.select_candidate([5.0, 5.0, 3.0], own_idx=2) == 0      # own not tied: argmax


def _long(q, r, critic="fb_tdj", seed=0, bank="carrying", state=0):
    names = ["fb_policy", "density_policy", "secure", "release"][:len(q)]
    return pd.DataFrame([dict(critic=critic, seed=seed, bank=bank, state=state, task=1,
                              candidate=n, q_pred=qq, return_true=rr) for n, qq, rr in zip(names, q, r)])


def test_selection_table_picks_by_q_and_by_return_and_flags_agreement():
    long = pd.concat([
        _long(q=[1.0, 2.0, 9.0, 0.0], r=[3.0, 1.0, 0.0, 7.0], state=0),   # b -> secure, c -> release
        _long(q=[9.0, 2.0, 1.0, 0.0], r=[7.0, 1.0, 0.0, 3.0], state=1),   # both keep the actor
        _long(q=[1.0, 2.0, 3.0, 4.0], r=[0.0, 0.0, 0.0, 0.0], state=2),   # MC tie: c keeps the actor
    ], ignore_index=True)
    t = IV.selection_table(long).set_index("state")
    assert t.loc[0, "cand_actor"] == "fb_policy" and t.loc[0, "cand_critic"] == "secure" and t.loc[0, "cand_rollout"] == "release"
    assert t.loc[0, "mc_critic"] == 0.0 and t.loc[0, "mc_rollout"] == 7.0 and t.loc[0, "mc_actor"] == 3.0
    assert t.loc[0, "agree_critic_actor"] == 0 and t.loc[0, "agree_rollout_actor"] == 0 and t.loc[0, "agree_critic_rollout"] == 0
    assert t.loc[1, "agree_critic_actor"] == 1 and t.loc[1, "agree_rollout_actor"] == 1 and t.loc[1, "agree_critic_rollout"] == 1
    assert t.loc[2, "tie_mc"] == 1 and t.loc[2, "cand_rollout"] == "fb_policy" and t.loc[2, "cand_critic"] == "release"
    assert t.loc[0, "mc_best"] == 7.0 and t.loc[0, "mc_worst"] == 0.0


def test_selection_table_uses_the_critics_own_proposal():
    long = _long(q=[1.0, 2.0, 0.0, 0.0], r=[0.0, 0.0, 0.0, 0.0], critic="density_fb_z")
    t = IV.selection_table(long).iloc[0]
    assert t["cand_actor"] == "density_policy" and t["idx_actor"] == 1
    assert t["cand_critic"] == "density_policy"           # its own proposal is the argmax


# ── per-rollout metrics ─────────────────────────────────────────────────────────────────

def test_rollout_metrics_success_return_and_held():
    H, K = 5, 3
    hit = np.zeros((H, K), bool)
    hit[2:, 0] = True            # rollout 0: hits from step 2 to the end
    hit[1, 1] = True             # rollout 1: one hit at step 1, not at the end
    held = np.ones((H, K), bool)
    held[3, 1] = False           # rollout 1 drops the cube at step 3
    held[0, 2] = False           # rollout 2 loses it immediately
    m = IV.rollout_metrics(hit, held, gamma=0.5, held_steps=2)
    assert m["success_any"].tolist() == [1.0, 1.0, 0.0]
    assert m["success_final"].tolist() == [1.0, 0.0, 0.0]
    assert m["ret"][0] == pytest.approx(0.25 + 0.125 + 0.0625)
    assert m["ret"][1] == pytest.approx(0.5) and m["ret"][2] == 0.0
    assert m["held20"].tolist() == [1.0, 1.0, 0.0]        # first 2 steps
    assert m["held_all"].tolist() == [1.0, 0.0, 0.0]


# ── aggregation ─────────────────────────────────────────────────────────────────────────

def _rolls():
    rows = []
    # 2 states x 2 rollouts, one critic/bank; cond "actor": state means 0.5, 1.0; "rollout": 1.0, 1.0
    vals = {"actor": {0: [0.0, 1.0], 1: [1.0, 1.0]}, "rollout": {0: [1.0, 1.0], 1: [1.0, 1.0]},
            "critic": {0: [0.0, 0.0], 1: [1.0, 0.0]}}
    for cond, d in vals.items():
        for st, ks in d.items():
            for k, v in enumerate(ks):
                rows.append(dict(critic="fb_tdj", seed=0, bank="carrying", state=st, task=1, cond=cond,
                                 cand_idx=0, cand="fb_policy", k=k, K=2, H=4, gamma=0.99,
                                 success_any=v, success_final=v, ret=v * 2.0, held20=1.0, held_all=v))
    return pd.DataFrame(rows)


def _sel():
    return pd.DataFrame([
        dict(critic="fb_tdj", seed=0, bank="carrying", state=0, agree_critic_actor=1, agree_rollout_actor=0,
             agree_critic_rollout=0, tie_mc=0, mc_actor=1.0, mc_critic=1.0, mc_rollout=4.0),
        dict(critic="fb_tdj", seed=0, bank="carrying", state=1, agree_critic_actor=0, agree_rollout_actor=0,
             agree_critic_rollout=1, tie_mc=0, mc_actor=2.0, mc_critic=3.0, mc_rollout=3.0),
    ])


def test_aggregate_means_over_states_then_se_over_states_and_counts():
    t = IV.aggregate_rollouts(_rolls(), _sel()).set_index("cond")
    a = t.loc["actor"]
    assert a["n_states"] == 2 and a["n_rollouts"] == 4
    assert a["success_any_mean"] == pytest.approx(0.75)                        # (0.5 + 1.0) / 2
    assert a["success_any_se"] == pytest.approx(np.std([0.5, 1.0], ddof=1) / np.sqrt(2))
    assert a["ret_mean"] == pytest.approx(1.5) and a["held20_mean"] == 1.0
    r = t.loc["rollout"]
    assert r["success_any_mean"] == 1.0 and r["success_any_se"] == 0.0
    c = t.loc["critic"]
    assert c["success_any_mean"] == pytest.approx(0.25)
    # agreement and selection-time MC value of the chosen candidate
    assert a["frac_same_as_actor"] == 1.0 and a["mc_sel_mean"] == pytest.approx(1.5)
    assert c["frac_same_as_actor"] == pytest.approx(0.5) and c["mc_sel_mean"] == pytest.approx(2.0)
    assert r["frac_same_as_actor"] == 0.0 and r["mc_sel_mean"] == pytest.approx(3.5)
    assert r["frac_critic_eq_rollout"] == pytest.approx(0.5) and r["frac_tie_mc"] == 0.0
    assert list(t.index) == ["actor", "critic", "rollout"]


def test_aggregate_per_seed_keeps_seed_column_and_pools_across_seeds():
    r0 = _rolls(); r1 = _rolls(); r1["seed"] = 1
    r1.loc[r1["cond"] == "actor", ["success_any"]] = 0.0
    s0 = _sel(); s1 = _sel(); s1["seed"] = 1
    rolls, sel = pd.concat([r0, r1]), pd.concat([s0, s1])
    ts = IV.aggregate_rollouts(rolls, sel, by_seed=True)
    assert set(ts["seed"]) == {0, 1}
    a = ts[(ts["cond"] == "actor")].set_index("seed")["success_any_mean"]
    assert a.loc[0] == pytest.approx(0.75) and a.loc[1] == 0.0
    t = IV.aggregate_rollouts(rolls, sel).set_index("cond")
    assert t.loc["actor", "n_states"] == 4 and t.loc["actor", "success_any_mean"] == pytest.approx(0.375)


def test_paired_table_within_state_differences_and_changed_subset():
    rolls = _rolls()
    # make the critic condition pick a different candidate at state 1 only
    rolls.loc[(rolls["cond"] == "critic") & (rolls["state"] == 1), "cand_idx"] = 2
    rolls.loc[(rolls["cond"] == "rollout"), "cand_idx"] = 3
    t = IV.paired_table(rolls).set_index("pair")
    rc = t.loc["rollout-critic"]
    # state means: rollout 1.0, 1.0; critic 0.0, 0.5 -> diffs 1.0, 0.5
    assert rc["n"] == 2 and rc["n_changed"] == 2
    assert rc["success_any_diff"] == pytest.approx(0.75)
    assert rc["success_any_diff_se"] == pytest.approx(np.std([1.0, 0.5], ddof=1) / np.sqrt(2))
    ca = t.loc["critic-actor"]
    # actor 0.5, 1.0 ; critic 0.0, 0.5 -> diffs -0.5, -0.5; only state 1 changed candidate
    assert ca["success_any_diff"] == pytest.approx(-0.5) and ca["n_changed"] == 1
    assert ca["success_any_diff_changed"] == pytest.approx(-0.5)
    assert ca["ret_diff"] == pytest.approx(-1.0)
    ts = IV.paired_table(rolls, by_seed=True)
    assert "seed" in ts.columns and len(ts) == 3
