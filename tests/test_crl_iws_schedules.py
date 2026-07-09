"""Tests for aligned eval/checkpoint schedules."""

from agents.crl_iws.schedules import (
    advance_schedule_index,
    align_step,
    build_aligned_schedule,
    eval_milestones,
    legacy_nominal_to_aligned_map,
    next_pending_schedule_index,
    normalize_completed_eval_steps,
)


def test_align_step_matches_um_256_envs():
    assert align_step(500_000, 256) == 500_224
    assert align_step(1_000_000, 256) == 1_000_192
    assert align_step(1_500_000, 256) == 1_500_160


def test_eval_milestones_first_three():
    milestones = eval_milestones(total_env_steps=20_000_000, every_n_steps=500_000, align_to=256)
    assert milestones[:3] == [500_224, 1_000_192, 1_500_160]


def test_advance_schedule_index():
    schedule = [500_224, 1_000_192]
    assert advance_schedule_index(schedule, 0) == 0
    assert advance_schedule_index(schedule, 500_224) == 1
    assert advance_schedule_index(schedule, 1_000_192) == 2


def test_build_aligned_schedule_empty_when_disabled():
    assert build_aligned_schedule(1_000_000, 0, 256) == []


def test_legacy_nominal_to_aligned_map():
    mapping = legacy_nominal_to_aligned_map(
        total_env_steps=2_000_000,
        every_n_steps=500_000,
        align_to=256,
    )
    assert mapping[500_000] == 500_224
    assert mapping[1_000_000] == 1_000_192


def test_normalize_completed_eval_steps_maps_legacy():
    completed = normalize_completed_eval_steps(
        {500_000, 1_000_000},
        total_env_steps=20_000_000,
        every_n_steps=500_000,
        align_to=256,
    )
    assert completed == {500_224, 1_000_192}


def test_next_pending_schedule_index():
    schedule = [500_224, 1_000_192, 1_500_160]
    assert next_pending_schedule_index(schedule, set()) == 0
    assert next_pending_schedule_index(schedule, {500_224}) == 1
    assert next_pending_schedule_index(schedule, set(schedule)) == 3
