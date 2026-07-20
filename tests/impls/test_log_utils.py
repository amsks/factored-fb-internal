import os
from impls.utils.log_utils import get_exp_name, setup_wandb, CsvLogger


def test_exp_name_includes_pair_and_env():
    name = get_exp_name(3, critic="gciql", actor="fql", env_name="cube-single-play-v0")
    assert "gciql" in name and "fql" in name and "3" in name


def test_exp_name_backward_compatible():
    assert get_exp_name(3) == "seed_3"


def test_setup_wandb_disabled_returns_none():
    assert setup_wandb({"a": 1}, mode="disabled") is None


def test_csv_logger_roundtrip(tmp_path):
    p = tmp_path / "log.csv"
    lg = CsvLogger(str(p))
    lg.log({"loss": 1.0}, step=1)
    lg.log({"loss": 0.5}, step=2)
    lg.close()
    rows = p.read_text().strip().splitlines()
    assert rows[0] == "loss,step"
    assert rows[-1] == "0.5,2"
