import pytest

from impls.utils.log_utils import CsvLogger, get_exp_name, setup_wandb


def test_exp_name_includes_pair_and_env():
    name = get_exp_name(
        3, critic="density_fb", actor="ddpgbc", env_name="cube-single-play-v0"
    )
    assert "density_fb" in name and "ddpgbc" in name and "3" in name


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


def test_csv_logger_reopen_appends_instead_of_truncating(tmp_path):
    """Reopening a CsvLogger on an existing file appends rows under one header."""
    p = tmp_path / "log.csv"

    lg1 = CsvLogger(str(p))
    lg1.log({"loss": 1.0}, step=1)
    lg1.log({"loss": 0.9}, step=2)
    lg1.close()

    lg2 = CsvLogger(str(p))
    lg2.log({"loss": 0.5}, step=3)
    lg2.log({"loss": 0.4}, step=4)
    lg2.close()

    lines = p.read_text().strip().splitlines()
    assert lines.count("loss,step") == 1, "exactly one header line"
    assert lines[0] == "loss,step"
    body = lines[1:]
    assert body == ["1.0,1", "0.9,2", "0.5,3", "0.4,4"], (
        "pre-resume rows (steps 1-2) and post-resume rows (steps 3-4) must both survive"
    )


def test_csv_logger_warns_when_it_drops_a_column(tmp_path):
    """CsvLogger warns when it drops a column not in the frozen header."""
    p = tmp_path / "eval.csv"

    lg1 = CsvLogger(str(p))
    lg1.log({"success": 0.4}, step=100)
    lg1.close()

    lg2 = CsvLogger(str(p))
    with pytest.warns(UserWarning, match="inferred_cond"):
        lg2.log(
            {"success": 0.9, "inferred_cond": 1.0, "inferred_cond_norm": 3.7}, step=200
        )
    lg2.close()

    assert p.read_text().strip().splitlines()[0] == "success,step"


def test_csv_logger_does_not_warn_on_a_matching_row(tmp_path, recwarn):
    p = tmp_path / "train.csv"
    lg = CsvLogger(str(p))
    lg.log({"loss": 1.0}, step=1)
    lg.log({"loss": 0.9}, step=2)  # same keys -> nothing to drop
    lg.close()
    assert not [w for w in recwarn if "DROPPED" in str(w.message)]
