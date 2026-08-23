"""campaign/gen_impls.py — the campaign job generator.

The generator's whole job is to fail HERE rather than on a compute node forty minutes into
a 500k-step run, so most of this file is about the failure paths: unknown critic, unknown
actor, unknown domain, an unset ${oc.env:...} root, a flag impls/main.py does not define.

The rest pins spec section 7 (docs/specs/2026-08-21-density-fb-agent-branch.md): four
tables, 288 / 192 / 36 / 9 = 525 jobs (T3 lost td_fb with the critic cull).

    PYTHONPATH=$PWD JAX_PLATFORMS=cpu python -m pytest tests/impls/test_gen_impls.py -q
"""
import importlib
import os
import sys
from pathlib import Path

import pytest

from campaign import gen_impls as G

REPO = Path(__file__).resolve().parents[2]
# A domain whose yaml interpolates nothing, so job-building needs no data roots.
FREE_DOMAIN = "cube-single"
DATA_ROOTS = {"FB_DATA_ROOT": "/data/root", "FB_HF_EXPORT": "/data/hf"}


# ---------------------------------------------------------------------------------------
# spec section 7
# ---------------------------------------------------------------------------------------
def test_table_counts_match_spec_section_7():
    got = {name: len(t["critics"]) * len(t["actors"]) * len(t["domains"]) * len(t["seeds"])
           for name, t in G.TABLES.items()}
    assert got == {"T1": 288, "T2": 192, "T3": 36, "T4": 9}
    assert sum(got.values()) == 525
    # `expect` is what the summary line compares against; keep it honest.
    for name, t in G.TABLES.items():
        assert t["expect"] == got[name]


def test_t1_is_manipulation_only():
    """Section 7: env/domains also holds antmaze, pointmaze and five b2d-*, excluded from T1."""
    assert len(G.T1_DOMAINS) == 12
    assert len(set(G.T1_DOMAINS)) == 12
    excluded = {"antmaze", "pointmaze"} | {d for d in G.known_domains(REPO) if d.startswith("b2d-")}
    assert excluded, "sanity: env/domains should contain the excluded domains"
    assert not (set(G.T1_DOMAINS) & excluded)


def test_only_t4_is_online():
    assert [n for n, t in G.TABLES.items() if t["online"]] == ["T4"]


def test_table_summary_reports_every_table_and_the_total():
    text = G.table_summary()
    for name in G.TABLES:
        assert name in text
    assert "525 jobs" in text
    assert "MISMATCH" not in text


# ---------------------------------------------------------------------------------------
# registries: what a valid name is
# ---------------------------------------------------------------------------------------
def test_known_domains_is_the_union_env_wins():
    """env/config.py::_load_domain checks env/domains/ first, then impls/configs/domain/.
    The two sets differ, so validating against either alone would reject working names."""
    doms = G.known_domains(REPO)
    assert doms["cube-single"] == "env/domains"          # hyphenated, env-owned
    assert doms["cube_single"] == "impls/configs/domain"  # underscored, the impls fallback
    for name in G.T1_DOMAINS:
        assert name in doms, f"T1 domain {name} resolves nowhere"


def test_registry_names_all_validate():
    """Every registered critic and actor must pass validation on a data-root-free domain —
    otherwise the generator would reject a name impls.main accepts."""
    CRITICS, ACTORS, _, _ = G.registries()
    G.validate(sorted(CRITICS), sorted(ACTORS), [FREE_DOMAIN], fb_root=REPO)


def test_main_flags_scrapes_the_entrypoint():
    flags = G.main_flags(REPO)
    # the flags every job line carries
    for f in ("critic", "actor", "domain", "seed", "save_dir", "train_steps",
              "save_interval", "eval_interval", "eval_episodes", "agg_dir", "agg_group"):
        assert f in flags
    # flags added after the generator was last touched
    assert {"online", "num_video", "override", "infer_cond_episodes"} <= flags


def test_domain_env_requirements_finds_both_roots():
    """The 12 T1 domains need TWO different roots; hardcoding either lets half the table
    through with the other half unset."""
    need = G.domain_env_requirements(G.T1_DOMAINS, REPO)
    assert set(need) == {"FB_DATA_ROOT", "FB_HF_EXPORT"}
    assert "mw-pick-place" in need["FB_DATA_ROOT"]
    assert "mw-assembly" in need["FB_HF_EXPORT"]
    assert G.domain_env_requirements([FREE_DOMAIN, "scene", "cube-double"], REPO) == {}


# ---------------------------------------------------------------------------------------
# happy path: counts and well-formed command lines
# ---------------------------------------------------------------------------------------
def test_job_count_is_the_full_cross_product():
    critics, actors, domains, seeds = ["fb", "density_fb"], ["flowbc", "ddpgbc"], \
        [FREE_DOMAIN, "scene"], [0, 1, 2]
    lines, _ = G.build_jobs(critics, actors, domains, seeds, fb_root=REPO)
    assert len(lines) == 2 * 2 * 2 * 3
    assert len(set(lines)) == len(lines)


def test_seeds_are_innermost_so_a_group_is_contiguous():
    """campaign/wandb_agg_daemon.py fills an __agg run once all seeds report a step, which
    only happens early if the seeds of one group co-run."""
    lines, _ = G.build_jobs(["fb"], ["flowbc"], [FREE_DOMAIN, "scene"], [0, 1, 2], fb_root=REPO)
    seeds = [int(l.split("--seed ")[1].split()[0]) for l in lines]
    assert seeds == [0, 1, 2, 0, 1, 2]


def test_command_lines_are_well_formed():
    lines, agg_dir = G.build_jobs(["density_fb"], ["ddpgbc"], [FREE_DOMAIN], [0, 7],
                                  fb_root=REPO, campaign_out="/out", tag="tg")
    assert len(lines) == 2
    defined = G.main_flags(REPO)
    for line, seed in zip(lines, [0, 7]):
        assert " -m impls.main " in line
        assert "MUJOCO_GL=egl" in line and "XLA_PYTHON_CLIENT_PREALLOCATE=false" in line
        assert f"PYTHONPATH={REPO}" in line
        assert "--critic density_fb " in line
        assert "--actor ddpgbc " in line
        assert f"--domain {FREE_DOMAIN} " in line
        assert f"--seed {seed} " in line
        assert f"--save_dir /out/tg/density_fb_ddpgbc__{FREE_DOMAIN}__s{seed} " in line
        assert line.endswith(f"--agg_group density_fb_ddpgbc__{FREE_DOMAIN}")
        # Every flag the line emits must exist in impls/main.py, or absl aborts the job.
        emitted = {tok[2:] for tok in line.split() if tok.startswith("--")}
        assert emitted <= defined, f"impls.main defines no {sorted(emitted - defined)}"
    assert agg_dir == "/out/impls_eval/tg"


def test_optional_flags_are_off_by_default_and_emitted_when_asked():
    plain, _ = G.build_jobs(["fb"], ["flowbc"], [FREE_DOMAIN], [0], fb_root=REPO)
    assert "--online" not in plain[0]
    assert "--num_video" not in plain[0]
    assert "--override" not in plain[0]
    rich, _ = G.build_jobs(["fb"], ["flowbc"], [FREE_DOMAIN], [0], fb_root=REPO,
                           online=True, num_video=2, infer_cond_episodes=1024,
                           overrides=["batch_size=64", "lr=1e-4"])
    assert " --online " in rich[0]
    assert "--num_video 2" in rich[0]
    assert "--infer_cond_episodes 1024" in rich[0]
    assert "--override batch_size=64" in rich[0] and "--override lr=1e-4" in rich[0]


def test_required_env_roots_are_prefixed_onto_every_line():
    lines, _ = G.build_jobs(["fb"], ["flowbc"], ["mw-pick-place", "mw-assembly"], [0],
                            fb_root=REPO, env_vars=DATA_ROOTS)
    for line in lines:
        assert "FB_DATA_ROOT=/data/root" in line
        assert "FB_HF_EXPORT=/data/hf" in line
        assert line.index("FB_DATA_ROOT=") < line.index(" -m impls.main")


# ---------------------------------------------------------------------------------------
# --seeds
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("spec,expect", [
    ("3", [0, 1, 2]),          # bare integer = a COUNT
    ("1", [0]),
    ("0,1,2", [0, 1, 2]),      # comma = an explicit list
    ("7,9", [7, 9]),
    ("5,", [5]),               # one specific seed
    (" 0, 1 ,2 ", [0, 1, 2]),
])
def test_parse_seeds(spec, expect):
    assert G.parse_seeds(spec) == expect


@pytest.mark.parametrize("spec", ["", "  ", "abc", "0", "-2", "1,x", "2.5"])
def test_parse_seeds_rejects_junk(spec):
    with pytest.raises(G.GenError):
        G.parse_seeds(spec)


def test_seeds_flag_is_honoured_end_to_end(tmp_path, capsys):
    out = tmp_path / "jobs.txt"
    G.run(["--critics", "fb", "--actors", "flowbc", "--domains", FREE_DOMAIN,
           "--seeds", "4", "--out", str(out), "--campaign_out", str(tmp_path / "c")])
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 4
    assert [int(l.split("--seed ")[1].split()[0]) for l in lines] == [0, 1, 2, 3]

    G.run(["--critics", "fb", "--actors", "flowbc", "--domains", FREE_DOMAIN,
           "--seeds", "11,13", "--out", str(out), "--campaign_out", str(tmp_path / "c")])
    lines = out.read_text().strip().splitlines()
    assert [int(l.split("--seed ")[1].split()[0]) for l in lines] == [11, 13]


def test_seeds_overrides_a_table_preset(tmp_path):
    out = tmp_path / "jobs.txt"
    G.run(["--table", "T2", "--seeds", "1", "--out", str(out),
           "--campaign_out", str(tmp_path / "c"), "--data_root", "/d"])
    assert len(out.read_text().strip().splitlines()) == 8 * 4 * 2 * 1


# ---------------------------------------------------------------------------------------
# loud failure on unknown names
# ---------------------------------------------------------------------------------------
def test_unknown_critic_names_the_value_and_lists_the_valid_ones():
    CRITICS, _, _, _ = G.registries()
    with pytest.raises(G.GenError) as ei:
        G.validate(["biql"], ["flowbc"], [FREE_DOMAIN], fb_root=REPO)
    msg = str(ei.value)
    assert "biql" in msg
    assert f"({len(CRITICS)})" in msg
    for name in ("density_fb", "psm", "rldp"):
        assert name in msg


def test_unknown_actor_names_the_value_and_lists_the_valid_ones():
    _, ACTORS, _, _ = G.registries()
    with pytest.raises(G.GenError) as ei:
        G.validate(["fb"], ["ddpg"], [FREE_DOMAIN], fb_root=REPO)
    msg = str(ei.value)
    assert "ddpg" in msg
    for name in sorted(ACTORS):
        assert name in msg


def test_unknown_domain_lists_valid_ones_and_does_not_suggest_the_override_for_a_typo():
    with pytest.raises(G.GenError) as ei:
        G.validate(["fb"], ["flowbc"], ["cube_singel"], fb_root=REPO)
    msg = str(ei.value)
    assert "cube_singel" in msg
    assert "cube-single" in msg and "cube_single" in msg
    assert "will not rescue a typo" in msg


def test_pending_exorl_domains_explain_themselves():
    with pytest.raises(G.GenError) as ei:
        G.validate(["fb"], ["flowbc"], G.T3_DOMAINS, fb_root=REPO)
    msg = str(ei.value)
    assert "workstream 6" in msg
    assert "--allow_unknown_domains" in msg
    for d in G.T3_DOMAINS:
        assert d in msg


def test_allow_unknown_domains_emits_t3_and_t4(tmp_path, capsys):
    for table, n in (("T3", 36), ("T4", 9)):
        out = tmp_path / f"{table}.txt"
        G.run(["--table", table, "--allow_unknown_domains", "--out", str(out),
               "--campaign_out", str(tmp_path / "c")])
        lines = out.read_text().strip().splitlines()
        assert len(lines) == n
        assert "CANNOT run yet" in capsys.readouterr().err
    t4 = (tmp_path / "T4.txt").read_text().strip().splitlines()
    assert all(" --online " in l for l in t4)
    t3 = (tmp_path / "T3.txt").read_text().strip().splitlines()
    assert not any("--online" in l for l in t3)


def test_unset_interpolated_root_fails_before_launch(monkeypatch):
    for var in DATA_ROOTS:
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(G.GenError) as ei:
        G.build_jobs(["fb"], ["flowbc"], ["mw-pick-place", "mw-assembly"], [0],
                     fb_root=REPO, env_vars={})
    msg = str(ei.value)
    assert "FB_DATA_ROOT" in msg and "FB_HF_EXPORT" in msg
    assert "mw-pick-place" in msg and "mw-assembly" in msg
    assert "InterpolationResolutionError" in msg


def test_registered_critic_without_a_config_yaml_fails(monkeypatch, tmp_path):
    """A name in CRITICS with no impls/configs/critic/<name>.yaml would die in
    make_config(), not here — so the config file is checked too."""
    CRITICS, ACTORS, check_combo, ComboError = G.registries()
    monkeypatch.setattr(G, "registries",
                        lambda: ({**CRITICS, "ghost": CRITICS["fb"]}, ACTORS,
                                 check_combo, ComboError))
    with pytest.raises(G.GenError, match="no config at"):
        G.validate(["ghost"], ["flowbc"], [FREE_DOMAIN], fb_root=REPO)


def test_invalid_critic_x_actor_pair_is_rejected_at_generation_time(monkeypatch):
    """check_combo reads class attributes only, so the whole cross-product is validated
    without building a network. No pair in the tree is invalid today (102/102 compose), so
    the guard is exercised against a stub critic that declares a bootstrap it cannot serve."""
    CRITICS, ACTORS, check_combo, ComboError = G.registries()

    class BadCritic:                       # BOOT_COND='value_goal' without providing it
        BOOT_COND = "value_goal"
        PROVIDES_COND = frozenset({"z"})
        PROVIDES_V = False

    monkeypatch.setattr(G, "registries",
                        lambda: ({**CRITICS, "fb": BadCritic}, ACTORS, check_combo, ComboError))
    monkeypatch.setattr(Path, "exists", lambda self: True)
    with pytest.raises(G.GenError) as ei:
        G.validate(["fb"], ["flowbc"], [FREE_DOMAIN], fb_root=REPO)
    msg = str(ei.value)
    assert "fb x flowbc" in msg
    assert "check_combo" in msg and "ComboError" in msg


def test_empty_selection_is_rejected():
    with pytest.raises(G.GenError, match="no critics selected"):
        G.run(["--actors", "flowbc", "--domains", FREE_DOMAIN, "--dry_run"])
    with pytest.raises(G.GenError, match="no domains selected"):
        G.run(["--critics", "fb", "--actors", "flowbc", "--dry_run"])


def test_emitting_a_flag_impls_main_lacks_is_rejected(monkeypatch):
    monkeypatch.setattr(G, "main_flags", lambda fb_root=None: {"critic", "actor"})
    with pytest.raises(G.GenError) as ei:
        G.build_jobs(["fb"], ["flowbc"], [FREE_DOMAIN], [0], fb_root=REPO)
    assert "out of date with the entrypoint" in str(ei.value)


# ---------------------------------------------------------------------------------------
# paths: nothing from the old deployment box
# ---------------------------------------------------------------------------------------
def test_source_holds_no_hardcoded_deployment_path():
    src = Path(G.__file__).read_text()
    assert "/raid/" not in src


def test_no_raid_path_in_output_when_env_vars_are_unset(monkeypatch, tmp_path):
    """The generator used to bake in /raid/tongle/rl/... for FB, NEWT_PY and CAMPAIGN_OUT,
    none of which exist on this box. With every FB_* var unset the defaults must come from
    this checkout instead."""
    for var in ("FB_ROOT", "FB_PYTHON", "FB_CAMPAIGN_OUT", "FB_DATA_ROOT", "FB_HF_EXPORT"):
        monkeypatch.delenv(var, raising=False)
    mod = importlib.reload(G)
    try:
        assert mod.DEFAULT_FB_ROOT == str(REPO)
        assert mod.DEFAULT_PYTHON == sys.executable
        assert mod.DEFAULT_CAMPAIGN_OUT == str(REPO / "results" / "campaign")
        assert mod.DEFAULT_DATA_ROOT == ""
        for root in (mod.DEFAULT_FB_ROOT, mod.DEFAULT_PYTHON, mod.DEFAULT_CAMPAIGN_OUT):
            assert "/raid/" not in root

        out = tmp_path / "jobs.txt"
        mod.run(["--critics", "fb,density_fb", "--actors", "flowbc",
                 "--domains", f"{FREE_DOMAIN},scene", "--seeds", "3", "--out", str(out)])
        text = out.read_text()
        assert "/raid/" not in text
        assert "tongle" not in text
        assert len(text.strip().splitlines()) == 12
        # agg_dir defaulted under the checkout, not the old campaign root.
        assert str(REPO / "results" / "campaign") in text
    finally:
        importlib.reload(G)   # restore the module for anything importing it afterwards


def test_explicit_roots_beat_the_defaults():
    lines, agg = G.build_jobs(["fb"], ["flowbc"], [FREE_DOMAIN], [0], fb_root=REPO,
                              python="/opt/py", campaign_out="/camp", tag="t")
    assert "/opt/py -m impls.main" in lines[0]
    assert agg == "/camp/impls_eval/t"


def test_dry_run_writes_nothing(tmp_path, capsys):
    out = tmp_path / "jobs.txt"
    agg = tmp_path / "camp"
    G.run(["--critics", "fb", "--actors", "flowbc", "--domains", FREE_DOMAIN,
           "--out", str(out), "--campaign_out", str(agg), "--dry_run"])
    assert not out.exists()
    assert not agg.exists()
    assert "not writing" in capsys.readouterr().out


def test_env_flag_requires_var_equals_value(tmp_path):
    with pytest.raises(G.GenError, match="not VAR=VALUE"):
        G.run(["--critics", "fb", "--actors", "flowbc", "--domains", FREE_DOMAIN,
               "--env", "JUSTAVAR", "--out", str(tmp_path / "j.txt"), "--dry_run"])


def test_module_import_does_not_touch_the_filesystem_defaults():
    """The four table presets are pure data — a summary must not need a repo, a venv or a
    dataset root, so a human can sanity-check 525 before anything is installed."""
    assert isinstance(G.table_summary(), str)
