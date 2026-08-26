"""campaign/gen_impls.py — emit the job list that drives the FACTORED tree (impls.main,
critic x actor x domain x seed) for the spec-section-7 campaign.

    PYTHONPATH=$PWD /dev/shm/.venv-jax/bin/python campaign/gen_impls.py --table T1
    PYTHONPATH=$PWD /dev/shm/.venv-jax/bin/python campaign/gen_impls.py \
        --critics fb,density_fb --actors flowbc --domains cube-single --seeds 3

Each line is one (critic, actor, domain, seed). Eval/checkpoint cadence and the per-seed
eval jsonl match the previous campaigns, so campaign/wandb_agg_daemon.py aggregates the
seeds into {critic}_{actor}__{domain}__agg with no changes.

Everything is validated HERE, at generation time, against the real registries:

  * critic names        -> impls.agents.CRITICS        (+ impls/configs/critic/<name>.yaml)
  * actor names         -> impls.agents.ACTORS         (+ impls/configs/actor/<name>.yaml)
  * critic x actor      -> impls.critics.base.check_combo (class-attribute-only, so it runs
                           without building a single network)
  * domain names        -> env/domains/<name>.yaml, falling back to
                           impls/configs/domain/<name>.yaml -- which is exactly the lookup
                           env/config.py::_load_domain performs for `impls.main --domain`
  * impls.main flags    -> every flag emitted below is asserted to exist in impls/main.py

A bad name fails LOUDLY here, naming the bad value and listing the valid ones. The failure
mode this prevents is a job that dies forty minutes into a compute node.

PATHS. Nothing absolute is baked in. Each root reads an FB_* environment variable and falls
back to something derived from this checkout (the env/registry.py::COLLECTED convention):

    FB_ROOT         repo root                 <- this file's parent's parent
    FB_PYTHON       interpreter for the jobs  <- sys.executable
    FB_CAMPAIGN_OUT campaign output root      <- $FB_ROOT/results/campaign
    FB_DATA_ROOT    collected-dataset root    <- unset; REQUIRED by the mw-*/ms-* domains,
                                                whose yamls interpolate ${oc.env:FB_DATA_ROOT}
"""
import argparse
import os
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------------------
# Roots. Env var first (env/registry.py::COLLECTED convention), then a checkout-relative
# fallback. No absolute deployment path appears anywhere in this file.
# ---------------------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
DEFAULT_FB_ROOT = os.environ.get("FB_ROOT") or str(_HERE.parent)
DEFAULT_PYTHON = os.environ.get("FB_PYTHON") or sys.executable
DEFAULT_CAMPAIGN_OUT = (os.environ.get("FB_CAMPAIGN_OUT")
                        or str(Path(DEFAULT_FB_ROOT) / "results" / "campaign"))
DEFAULT_DATA_ROOT = os.environ.get("FB_DATA_ROOT") or ""

RUN_TAG = "impls_v1"
TRAIN_STEPS = 500_000
SAVE_INTERVAL = 100_000
EVAL_INTERVAL = 20_000
EVAL_EPISODES = 50
SEEDS = [0, 1, 2]

# ---------------------------------------------------------------------------------------
# The four tables of spec section 7 (docs/specs/2026-08-21-density-fb-agent-branch.md).
# Transcribed, not invented: the counts must come out 288 / 192 / 36 / 9 = 525 (T3 dropped td_fb with the critic cull).
# ---------------------------------------------------------------------------------------
# "8 of 8 baselines" from the progress log (spec section 12).
T1_METHODS = ["density_fb", "fb", "gciql", "iql_bilinear", "gcivl", "hiql", "psm", "rldp"]
# 12 manipulation domains. env/domains/ also holds antmaze, pointmaze and five b2d-*;
# section 7 excludes them explicitly -- T1 is manipulation-only.
T1_DOMAINS = ["cube-single", "cube-double", "scene",
              "mw-pick-place", "mw-push", "mw-assembly", "mw-basketball",
              "mw-bin-picking", "mw-box-close", "mw-button-press",
              "ms-pick-cube", "ms-push-cube"]
# "the four actor families" (spec section 4: bcflow and ddpgbc_density are variants).
T2_ACTORS = ["flowbc", "ddpgbc", "awr", "fql"]
T2_DOMAINS = ["cube-single", "mw-pick-place"]
# The five critics with a reward interface.
T3_METHODS = ["density_fb", "fb", "psm", "rldp"]   # the reward-interface family
# ExoRL / DMC on the RND exploration dataset. T4 uses the same three.
T3_DOMAINS = ["dmc-point-mass-maze", "dmc-walker", "dmc-cheetah"]

TABLES = {
    "T1": dict(critics=T1_METHODS, actors=["flowbc"], domains=T1_DOMAINS, seeds=SEEDS,
               online=False, expect=288,
               note="goal-conditioned manipulation"),
    "T2": dict(critics=T1_METHODS, actors=T2_ACTORS, domains=T2_DOMAINS, seeds=SEEDS,
               online=False, expect=192,
               note="actor study"),
    "T3": dict(critics=T3_METHODS, actors=["flowbc"], domains=T3_DOMAINS, seeds=SEEDS,
               online=False, expect=36,
               note="ExoRL, reward-specified"),
    "T4": dict(critics=["density_fb"], actors=["flowbc"], domains=T3_DOMAINS, seeds=SEEDS,
               online=True, expect=9,
               note="online Density-FB"),
}

# Domains a pending workstream will add but no yaml defines yet. Empty since workstream 6
# (the ExoRL env family) landed: env/domains/dmc-{point-mass-maze,walker,cheetah}.yaml all
# exist, so T3/T4 validate like any other table. Kept as the hook for the next such case --
# --allow_unknown_domains reads it to explain WHY a name is unknown.
PENDING_DOMAINS = {}


class GenError(ValueError):
    """A campaign spec that cannot be launched. Raised at generation time, on purpose."""


# ---------------------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------------------
def registries():
    """(CRITICS, ACTORS, check_combo, ComboError) from the live tree.

    Imported lazily so --help works under any interpreter; the import needs the jax venv
    (see the module docstring), and says so if it is missing."""
    try:
        from impls.agents import CRITICS, ACTORS
        from impls.critics.base import check_combo, ComboError
    except ImportError as e:                                  # noqa: BLE001 - re-raised loudly
        raise GenError(
            f"cannot import the impls registries ({type(e).__name__}: {e}). This generator "
            f"validates every name against them, so it needs the jax venv and PYTHONPATH:\n"
            f"    PYTHONPATH=<repo> /dev/shm/.venv-jax/bin/python campaign/gen_impls.py ..."
        ) from e
    return CRITICS, ACTORS, check_combo, ComboError


def known_domains(fb_root=None):
    """Domain names `impls.main --domain` accepts -> the directory each resolves from.

    env/config.py::_load_domain looks in env/domains/ FIRST and falls back to
    impls/configs/domain/, so the accepted set is the UNION and env/ wins ties. The two
    directories genuinely differ (env/domains has 18 hyphenated names; impls/configs/domain
    has only the underscored cube_single), so validating against either one alone would
    reject names that actually work."""
    root = Path(fb_root or DEFAULT_FB_ROOT)
    out = {}
    for p in sorted((root / "impls" / "configs" / "domain").glob("*.yaml")):
        out[p.stem] = "impls/configs/domain"
    for p in sorted((root / "env" / "domains").glob("*.yaml")):
        out[p.stem] = "env/domains"                            # env/ takes precedence
    return out


def main_flags(fb_root=None):
    """Flag names impls/main.py defines, scraped from its flags.DEFINE_* calls.

    Emitting a flag impls.main does not define is an absl UnrecognizedFlagError at job
    start; --online and friends were added after this generator was last touched, so the
    emitted command line is checked against the entrypoint rather than assumed."""
    root = Path(fb_root or DEFAULT_FB_ROOT)
    src = (root / "impls" / "main.py").read_text()
    return set(re.findall(r'flags\.DEFINE_\w+\(\s*["\'](\w+)["\']', src))


def domain_env_requirements(domains, fb_root=None):
    """{env var -> [domains that interpolate it]}, read out of the domain yamls.

    Not a fixed list: the 12 T1 domains alone need TWO different roots (mw-push/mw-pick-place
    and both ms-* use ${oc.env:FB_DATA_ROOT}; the other five mw-* use ${oc.env:FB_HF_EXPORT}),
    and hardcoding either would let half the table through with the other half unset.
    OmegaConf resolves these inside make_config(), so an unset var is an
    InterpolationResolutionError before a single gradient step -- once per job, in a log
    nobody reads until the queue has drained."""
    root = Path(fb_root or DEFAULT_FB_ROOT)
    need = {}
    for d in domains:
        p = root / "env" / "domains" / f"{d}.yaml"
        if not p.exists():
            continue
        for var in sorted(set(re.findall(r"\$\{oc\.env:(\w+)\}", p.read_text()))):
            need.setdefault(var, []).append(d)
    return need


# ---------------------------------------------------------------------------------------
# Parsing / validation
# ---------------------------------------------------------------------------------------
def parse_seeds(spec):
    """'0,1,2' -> [0,1,2] (explicit list);  '3' -> [0,1,2] (a COUNT).

    A bare integer is a count, which is the common case ("three seeds"). To name specific
    seeds always use a comma -- a one-element list is written '5,' and yields [5]."""
    spec = str(spec).strip()
    if not spec:
        raise GenError("--seeds is empty; pass a count (e.g. 3) or a list (e.g. 0,1,2)")
    if "," in spec:
        try:
            seeds = [int(s) for s in spec.split(",") if s.strip()]
        except ValueError:
            raise GenError(
                f"--seeds {spec!r} is not a comma-separated list of integers") from None
        if not seeds:
            raise GenError(f"--seeds {spec!r} names no seeds")
        return seeds
    try:
        n = int(spec)
    except ValueError:
        raise GenError(
            f"--seeds {spec!r} is neither an integer count nor a comma-separated list "
            f"(e.g. 3, or 0,1,2, or '5,' for the single seed 5)"
        ) from None
    if n < 1:
        raise GenError(f"--seeds {spec!r} is a count, so it must be >= 1")
    return list(range(n))


def _reject(kind, bad, valid, extra=""):
    show = ", ".join(sorted(valid))
    raise GenError(f"unknown {kind} {bad!r}. Valid {kind}s ({len(valid)}): {show}.{extra}")


def validate(critics, actors, domains, fb_root=None, allow_unknown_domains=False):
    """Fail loudly, here, on anything that would kill a job on a compute node.

    Checks, in order: critic names, actor names, their config yamls, every critic x actor
    pair (check_combo reads class attributes only, so nothing is built), and domain names."""
    CRITICS, ACTORS, check_combo, ComboError = registries()
    root = Path(fb_root or DEFAULT_FB_ROOT)

    for c in critics:
        if c not in CRITICS:
            _reject("critic", c, CRITICS)
        y = root / "impls" / "configs" / "critic" / f"{c}.yaml"
        if not y.exists():
            raise GenError(
                f"critic {c!r} is registered in impls.agents.CRITICS but has no config at "
                f"{y}; make_config() would raise before training starts"
            )
    for a in actors:
        if a not in ACTORS:
            _reject("actor", a, ACTORS)
        y = root / "impls" / "configs" / "actor" / f"{a}.yaml"
        if not y.exists():
            raise GenError(
                f"actor {a!r} is registered in impls.agents.ACTORS but has no config at "
                f"{y}; make_config() would raise before training starts"
            )

    bad_pairs = []
    for c in critics:
        for a in actors:
            try:
                check_combo(CRITICS[c], ACTORS[a])
            except ComboError as e:
                bad_pairs.append(f"  {c} x {a}: {e}")
    if bad_pairs:
        raise GenError(
            "these critic x actor pairs are structurally invalid (impls.critics.base."
            "check_combo) and would raise ComboError inside Agent.create:\n"
            + "\n".join(bad_pairs)
        )

    valid_domains = known_domains(root)
    unknown = [d for d in domains if d not in valid_domains]
    if unknown:
        pending = [d for d in unknown if d in PENDING_DOMAINS]
        why = ("\n" + "\n".join(f"  {d}: {PENDING_DOMAINS[d]}" for d in pending)) if pending else ""
        if not allow_unknown_domains:
            hint = ("\nPass --allow_unknown_domains to emit these jobs anyway (they will NOT "
                    "run until that workstream lands)." if pending else
                    "\n(--allow_unknown_domains exists for domains a pending workstream will "
                    "add; it will not rescue a typo.)")
            _reject("domain", unknown[0] if len(unknown) == 1 else unknown, valid_domains,
                    why + hint)
        print(f"[gen_impls] WARNING: --allow_unknown_domains emitted {len(unknown)} domain(s) "
              f"no yaml defines: {', '.join(unknown)}. These jobs CANNOT run yet.{why}",
              file=sys.stderr)
    return valid_domains


# ---------------------------------------------------------------------------------------
# Job lines
# ---------------------------------------------------------------------------------------
def build_jobs(critics, actors, domains, seeds, *, tag=RUN_TAG, fb_root=None, python=None,
               campaign_out=None, env_vars=None, train_steps=TRAIN_STEPS,
               save_interval=SAVE_INTERVAL, eval_interval=EVAL_INTERVAL,
               eval_episodes=EVAL_EPISODES, num_video=0, online=False, overrides=(),
               infer_cond_episodes=0):
    """-> (lines, agg_dir). Seed innermost so a group's seeds are contiguous and co-run,
    which is what lets the __agg run fill in early (campaign/wandb_agg_daemon.py)."""
    fb_root = str(fb_root or DEFAULT_FB_ROOT)
    python = str(python or DEFAULT_PYTHON)
    campaign_out = str(campaign_out or DEFAULT_CAMPAIGN_OUT)
    env_vars = dict(env_vars or {})
    agg_dir = f"{campaign_out}/impls_eval/{tag}"

    defined = main_flags(fb_root)
    emitted = ["critic", "actor", "domain", "seed", "save_dir", "train_steps",
               "save_interval", "eval_interval", "eval_episodes", "agg_dir", "agg_group"]
    if num_video:
        emitted.append("num_video")
    if online:
        emitted.append("online")
    if infer_cond_episodes:
        emitted.append("infer_cond_episodes")
    if overrides:
        emitted.append("override")
    missing = [f for f in emitted if f not in defined]
    if missing:
        raise GenError(
            f"impls/main.py defines no flag(s) {missing} — this generator is out of date "
            f"with the entrypoint and every job would die on absl's UnrecognizedFlagError. "
            f"Flags impls/main.py does define: {', '.join(sorted(defined))}"
        )

    # Resolve every ${oc.env:VAR} the selected domains interpolate: explicit env_vars first,
    # then this process's environment, else a loud failure naming the var AND the domains.
    required = domain_env_requirements(domains, fb_root)
    resolved, unresolved = {}, []
    for var, doms in sorted(required.items()):
        val = env_vars.get(var) or os.environ.get(var) or ""
        if val:
            resolved[var] = val
        else:
            unresolved.append(f"  {var}: needed by {', '.join(doms)}")
    if unresolved:
        raise GenError(
            "these domain yamls interpolate ${oc.env:...} variables that are unset, so "
            "every one of their jobs would die in make_config() with an "
            "InterpolationResolutionError before the first gradient step:\n"
            + "\n".join(unresolved)
            + "\nExport the variable(s) or pass --data_root (FB_DATA_ROOT) / --env VAR=VALUE."
        )
    # Any explicitly-passed var is emitted too, even if no selected domain reads it.
    resolved.update({k: v for k, v in env_vars.items() if v})

    extra = "".join(f" --override {o}" for o in overrides)
    lines = []
    for critic in critics:
        for actor in actors:
            for dom in domains:
                for s in seeds:
                    save_dir = f"{campaign_out}/{tag}/{critic}_{actor}__{dom}__s{s}"
                    envs = ["MUJOCO_GL=egl", "XLA_PYTHON_CLIENT_PREALLOCATE=false",
                            f"PYTHONPATH={fb_root}"]
                    envs += [f"{k}={v}" for k, v in sorted(resolved.items())]
                    lines.append(
                        f"{' '.join(envs)} {python} -m impls.main "
                        f"--critic {critic} --actor {actor} --domain {dom} --seed {s} "
                        f"--save_dir {save_dir} --train_steps {train_steps} "
                        f"--save_interval {save_interval} --eval_interval {eval_interval} "
                        f"--eval_episodes {eval_episodes} "
                        + (f"--num_video {num_video} " if num_video else "")
                        + ("--online " if online else "")
                        + (f"--infer_cond_episodes {infer_cond_episodes} "
                           if infer_cond_episodes else "")
                        + f"--agg_dir {agg_dir} "
                        f"--agg_group {critic}_{actor}__{dom}"
                        + extra
                    )
    return lines, agg_dir


def summarize(critics, actors, domains, seeds, n_jobs, label=""):
    head = f"[gen_impls] {label + ': ' if label else ''}"
    return (
        f"{head}{len(critics)} critic(s) x {len(actors)} actor(s) x {len(domains)} "
        f"domain(s) x {len(seeds)} seed(s) = {n_jobs} jobs\n"
        f"{head}  critics: {', '.join(critics)}\n"
        f"{head}  actors:  {', '.join(actors)}\n"
        f"{head}  domains: {', '.join(domains)}\n"
        f"{head}  seeds:   {', '.join(str(s) for s in seeds)}"
    )


def table_summary():
    """Counts per table for every table in spec section 7 — the human sanity check before
    ~534 runs go on a queue. Pure arithmetic: no registry import, no filesystem."""
    out, total = [], 0
    for name, t in TABLES.items():
        n = len(t["critics"]) * len(t["actors"]) * len(t["domains"]) * len(t["seeds"])
        total += n
        flag = "" if n == t["expect"] else f"  <-- MISMATCH, spec says {t['expect']}"
        out.append(f"  {name}  {t['note']:<52} {n:>4} jobs{flag}")
        out.append(f"        {len(t['critics'])} critics x {len(t['actors'])} actors x "
                   f"{len(t['domains'])} domains x {len(t['seeds'])} seeds"
                   + ("  [--online]" if t["online"] else ""))
    out.append(f"  {'TOTAL':<56} {total:>4} jobs")
    return "\n".join(out)


def build_parser():
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="tables (spec section 7):\n" + table_summary())
    ap.add_argument("--table", default=None, choices=sorted(TABLES) + ["all"],
                    help="preset from spec section 7. 'all' is a dry-run summary only. "
                         "--critics/--actors/--domains/--seeds override individual axes.")
    ap.add_argument("--critics", default=None,
                    help="comma list; validated against impls.agents.CRITICS")
    ap.add_argument("--actors", default=None,
                    help="comma list; validated against impls.agents.ACTORS")
    ap.add_argument("--domains", default=None,
                    help="comma list; validated against env/domains/ then impls/configs/domain/")
    ap.add_argument("--seeds", default=None,
                    help="count (3 -> 0,1,2) or explicit comma list (0,1,2). "
                         "For one specific seed write '5,'.")
    ap.add_argument("--out", default=None,
                    help="job file (default <fb_root>/campaign/jobs_impls.txt)")
    ap.add_argument("--tag", default=RUN_TAG)
    ap.add_argument("--fb_root", default=DEFAULT_FB_ROOT, help="$FB_ROOT, else this checkout")
    ap.add_argument("--python", default=DEFAULT_PYTHON, help="$FB_PYTHON, else sys.executable")
    ap.add_argument("--campaign_out", default=DEFAULT_CAMPAIGN_OUT,
                    help="$FB_CAMPAIGN_OUT, else <fb_root>/results/campaign")
    ap.add_argument("--data_root", default=DEFAULT_DATA_ROOT,
                    help="shorthand for --env FB_DATA_ROOT=...; needed by mw-push, "
                         "mw-pick-place and both ms-* domains")
    ap.add_argument("--env", action="append", default=[], metavar="VAR=VALUE",
                    help="extra env var prefixed onto every job line (repeatable). The "
                         "other five mw-* domains need FB_HF_EXPORT this way.")
    ap.add_argument("--train_steps", type=int, default=TRAIN_STEPS)
    ap.add_argument("--save_interval", type=int, default=SAVE_INTERVAL)
    ap.add_argument("--eval_interval", type=int, default=EVAL_INTERVAL)
    ap.add_argument("--eval_episodes", type=int, default=EVAL_EPISODES)
    ap.add_argument("--num_video", type=int, default=0)
    ap.add_argument("--infer_cond_episodes", type=int, default=0,
                    help="relabelled transitions used to infer eval conditioning (FB zero-shot)")
    ap.add_argument("--online", action="store_true", help="emit --online (T4)")
    ap.add_argument("--override", action="append", default=[],
                    help="passed through to impls.main --override (repeatable)")
    ap.add_argument("--allow_unknown_domains", action="store_true",
                    help="emit domains no yaml defines (T3/T4 until workstream 6 lands)")
    ap.add_argument("--dry_run", action="store_true", help="print the summary, write nothing")
    return ap


def run(argv=None):
    args = build_parser().parse_args(argv)

    if args.table == "all":
        print("[gen_impls] spec section 7 — counts per table (dry run, nothing written):")
        print(table_summary())
        return 0

    preset = TABLES.get(args.table, {})
    critics = ([c for c in args.critics.split(",") if c] if args.critics
               else list(preset.get("critics", [])))
    actors = ([a for a in args.actors.split(",") if a] if args.actors
              else list(preset.get("actors", [])))
    domains = ([d for d in args.domains.split(",") if d] if args.domains
               else list(preset.get("domains", [])))
    seeds = parse_seeds(args.seeds) if args.seeds else list(preset.get("seeds", SEEDS))
    online = args.online or bool(preset.get("online", False))

    for kind, vals in (("critic", critics), ("actor", actors), ("domain", domains)):
        if not vals:
            raise GenError(
                f"no {kind}s selected: pass --{kind}s or a --table preset "
                f"({', '.join(sorted(TABLES))})"
            )

    validate(critics, actors, domains, fb_root=args.fb_root,
             allow_unknown_domains=args.allow_unknown_domains)

    env_vars = {}
    if args.data_root:
        env_vars["FB_DATA_ROOT"] = args.data_root
    for item in args.env:
        if "=" not in item:
            raise GenError(f"--env {item!r} is not VAR=VALUE")
        k, v = item.split("=", 1)
        env_vars[k.strip()] = v

    lines, agg_dir = build_jobs(
        critics, actors, domains, seeds, tag=args.tag, fb_root=args.fb_root,
        python=args.python, campaign_out=args.campaign_out, env_vars=env_vars,
        train_steps=args.train_steps, save_interval=args.save_interval,
        eval_interval=args.eval_interval, eval_episodes=args.eval_episodes,
        num_video=args.num_video, online=online, overrides=args.override,
        infer_cond_episodes=args.infer_cond_episodes)

    label = args.table or "custom"
    print(summarize(critics, actors, domains, seeds, len(lines), label))
    expect = preset.get("expect")
    if expect is not None and not (args.critics or args.actors or args.domains or args.seeds):
        status = ("matches spec section 7" if len(lines) == expect
                  else f"MISMATCH — spec section 7 says {expect}")
        print(f"[gen_impls] {label}: {len(lines)} jobs, {status}")
    print(f"[gen_impls]   agg_dir: {agg_dir}")

    out = args.out or str(Path(args.fb_root) / "campaign" / "jobs_impls.txt")
    if args.dry_run:
        print(f"[gen_impls] --dry_run: not writing {out} (and not creating {agg_dir})")
        return 0
    Path(agg_dir).mkdir(parents=True, exist_ok=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(lines) + "\n")
    print(f"[gen_impls] wrote {len(lines)} jobs -> {out}")
    return 0


def main():
    try:
        sys.exit(run())
    except GenError as e:
        print(f"[gen_impls] ERROR: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
