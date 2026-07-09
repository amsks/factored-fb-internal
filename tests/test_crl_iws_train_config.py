"""Tests for CRL/IWS training CLI defaults."""

from __future__ import annotations

import argparse
from types import SimpleNamespace

from scripts.data import run_crl_iws_train as train_mod


def _args(**overrides):
    defaults = train_mod.TrainConfig()
    base = {field: getattr(defaults, field) for field in defaults.__dataclass_fields__}
    base.update(overrides)
    return argparse.Namespace(**base)


def test_iws_uses_air_hockey_contact_defaults():
    args = _args(algorithm="iws", wandb_tags=[])
    train_mod._resolve_algorithm_config(args)
    assert args.use_iws is True
    assert args.context_repetition_factor == 8
    assert args.iws_full_episode_future is True
    assert args.contact_distance == 0.12
    assert args.contact_sigma == 30.0
    assert args.near_contact_distance == 0.24


def test_sgcrl_disables_iws_sampling():
    args = _args(algorithm="sgcrl", wandb_tags=[])
    train_mod._resolve_algorithm_config(args)
    assert args.use_iws is False
    assert args.context_repetition_factor == 1
    assert args.iws_full_episode_future is False


def test_checkpoint_dir_default_is_none():
    assert train_mod.TrainConfig().checkpoint_dir is None
