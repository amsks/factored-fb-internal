import os


def test_config_file_parses_and_selects_isfb():
    import yaml
    p = "configs/agent/isfb_flowbc.yaml"
    assert os.path.exists(p)
    with open(p) as f:
        cfg = yaml.safe_load(f)
    assert cfg["agent"] == "isfb_flowbc"
    assert cfg["measure_insample"] in ("coord", "scalar_gate_hard", "scalar_gate_awr")
    assert cfg["critic_mode"] == "measure"      # ISFB uses the measure-branch update() path
    assert float(cfg["ortho_coef"]) == 1.0
    assert float(cfg["train_goal_ratio"]) == 1.0  # Exp1 exact-goal


def test_train_has_isfb_branch():
    with open("train.py") as f:
        src = f.read()
    assert 'cfg.agent == "isfb_flowbc"' in src
    assert "ISFBAgent(" in src
