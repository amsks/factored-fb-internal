#!/usr/bin/env python
"""Generate the spec-06 live experiment dashboard (HTML) from on-disk state.

Scrapes: GPU dispatch log + per-run eval.csv under ~/exp_spec06, the CPU-eval
JSONs/logs (item1 GPI, item3 rerank), and emits analysis/rescuing-fb/spec06_dashboard.html.
Cheap + deterministic so the monitoring loop can re-run it every refresh.

  /dev/shm/.venv-jax/bin/python  is NOT needed -- pure stdlib. Run with repo .venv or any python3.
"""
from __future__ import annotations
import csv, glob, html, json, re, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EXPROOT = Path("~/exp_spec06/factored-fb").expanduser()
LOGDIR = Path("/dev/shm/s06_logs")
DISPATCH = Path("/dev/shm/s06gpu_dispatch.log")
CPULOG = Path("/dev/shm/s06cpu.log")
ANALYSIS = REPO / "analysis" / "rescuing-fb"
OUT = ANALYSIS / "spec06_dashboard.html"

# group -> (spec item, kind, eval-metric meaning, pre-registered read)
GROUPS = {
    "RGFB-Cube-Single-State-d512-v0":         ("2", "method", "exact-z diag (zero-shot via eval_rgfb_zeroshot)",
                                               "zero-shot ≥0.88 explains the d-gap / ≤0.825 promote selection; anchors: G4 exact-z 0.921, v0 zs 0.825"),
    "RGFB-Cube-Single-State-d50-v1-taskprior": ("4", "method", "exact-z diag",
                                               "Dirac ≥0.80 non-reg (v0 0.825); two-goal mixture ≥0.60 / 0.30-0.60 / ≤0.30"),
    "GCIQL-State-Flow-Measure-FirstPassage-d50":  ("5", "mechanism", "greedy-on-M success",
                                               "≥0.30 = toxic axis / 0.10-0.30 partial / ≈0.10 axis-C exonerated (ceiling 0.08-0.10)"),
    "GCIQL-State-Flow-Measure-Expectile-FP-d50":  ("6", "mechanism", "greedy-on-M success",
                                               "≫0.10 = missing bootstrap / floor = axis-B exonerated"),
    "GCIQL-State-Flow-Measure-Expectile-d50":     ("6", "mechanism", "greedy-on-M success (unmasked variant)",
                                               "≫0.10 = missing bootstrap / floor = axis-B exonerated"),
    "GCIQL-State-Flow-Measure-HindsightTP-d50":   ("7", "mechanism", "greedy-on-M success",
                                               "≫0.10 validates the weighting half / null = weighting alone not the toxin"),
    "GCIQL-State-Flow-Measure-GCIQLCorner-d50":   ("★", "measure-win", "greedy-on-M success",
                                               "full-corner measure (fp+expectile+hindsight); at offdiag_weight→0, M=1+(1-γ)Q ≡ bilinear GC-IQL — the successor-measure-that-works arm"),
}
# extra exp roots to scan beyond spec06 (other-session + sota cells land here too)
EXTRA_ROOTS = [Path("~/exp_sota_impl/factored-fb").expanduser(),
               Path("~/exp_queue/factored-fb").expanduser()]

try:
    import sys as _sys
    _sys.path.insert(0, str(REPO / "scripts"))
    from results_map import MAP as _RESULTS_MAP
except Exception:
    _RESULTS_MAP = {}


def group_meta(group: str):
    """(spec-item, kind, eval-meaning, pre-registered read) for a group. Known groups use the
    curated read; unknown ones fall back to their results_map graft + a name-based kind."""
    if group in GROUPS:
        return GROUPS[group]
    low = group.lower()
    kind = ("method" if ("rgfb" in low or "latefusion" in low or "pmatched" in low)
            else "measure-win" if "corner" in low
            else "mechanism" if "measure" in low
            else "arch")
    graft = (_RESULTS_MAP.get(group) or {}).get("graft") or (_RESULTS_MAP.get(group) or {}).get("notes") or "—"
    meaning = "exact-z diag" if "rgfb" in low else "greedy-on-M / success"
    return ("·", kind, meaning, graft)


TOTAL_RUNS = 18


def last_eval(run_dir: Path):
    f = run_dir / "eval.csv"
    if not f.exists():
        return None
    try:
        rows = list(csv.DictReader(f.open()))
    except Exception:
        return None
    if not rows:
        return None
    r = rows[-1]
    ov = r.get("evaluation/overall_success")
    step = r.get("step")
    try:
        return {"overall": float(ov), "step": int(float(step)), "n_evals": len(rows),
                "per_task": [float(r.get(f"evaluation/task{i}_" + suf, "nan"))
                             for i, suf in zip(range(1, 6),
                             ["horizontal_success", "vertical1_success", "vertical2_success",
                              "diagonal1_success", "diagonal2_success"])]}
    except Exception:
        return None


def scrape_training():
    launched = 0
    if DISPATCH.exists():
        launched = DISPATCH.read_text().count("[launch]")
    crashed = []
    for lf in glob.glob(str(LOGDIR / "*.log")):
        try:
            t = Path(lf).read_text(errors="ignore")
        except Exception:
            continue
        if re.search(r"Traceback|Exception|AssertionError|CUDA_ERROR|RESOURCE_EXHAUSTED|out of memory", t):
            crashed.append(Path(lf).stem)
    # Auto-discover EVERY group dir across all exp roots (not just the curated ones), so the
    # dashboard reflects whatever is actually running (other-session corner/cellw cells too).
    groups = {}
    for root in [EXPROOT] + EXTRA_ROOTS:
        for gd in sorted(glob.glob(str(root / "*"))):
            gp = Path(gd)
            if not gp.is_dir():
                continue
            seeds = [{"run": Path(rd).name, "eval": last_eval(Path(rd))}
                     for rd in sorted(glob.glob(str(gp / "*"))) if Path(rd).is_dir()]
            if seeds:
                groups[gp.name] = seeds
    return launched, crashed, groups


def scrape_rerank():
    j = ANALYSIS / "rgfb_rerank_d50_v0.json"
    if j.exists():
        d = json.loads(j.read_text())
        return {"done": True, "agg": d.get("agg_mean_std", {}), "best": d.get("best_rerank_cell"),
                "best_succ": d.get("best_rerank_success"), "verdict": d.get("verdict_preregistered")}
    # partial: parse the log's per-task oneshot/flow lines -> mean over the latest complete pass
    if not CPULOG.exists():
        return {"done": False, "seen": 0}
    txt = CPULOG.read_text(errors="ignore")
    seeds = len(re.findall(r"\[rerank\] rgfb_v0_s\d+:", txt))
    fl = [float(x) for x in re.findall(r"flow_z_floor=([\d.]+)", txt)]
    of = [float(x) for x in re.findall(r"oneshot_z_floor=([\d.]+)", txt)]
    o64 = [float(x) for x in re.findall(r"oneshot_z_N64=([\d.]+)", txt)]
    fN = [float(x) for x in re.findall(r"flow_z_N32=([\d.]+)", txt)]
    return {"done": False, "seeds_started": seeds,
            "flow_floor": (sum(fl) / len(fl)) if fl else None,
            "flow_N32": (sum(fN) / len(fN)) if fN else None,
            "oneshot_floor": (sum(of) / len(of)) if of else None,
            "oneshot_N64": (sum(o64) / len(o64)) if o64 else None,
            "n_task_rows": len(fl)}


def scrape_gpi():
    out = {}
    for tag, pat in (("d50_v0_s0", "t6_gpi_*d50_v0_s0*.json"),
                     ("g4_d512", "t6_gpi_*g4_d512*.json"),
                     ("d50_v0_run", "t6_gpi_rgfb_v0_s0.json")):
        for p in glob.glob(str(ANALYSIS / pat)):
            try:
                d = json.loads(Path(p).read_text())
            except Exception:
                continue
            s = d.get("summary", {})
            gate = d.get("gate", {})
            out[tag] = {"gate": gate.get("mean_single_goal_gpi"),
                        "mixture": s.get("two_goal_mixture_gpi_success"),
                        "verdict": s.get("verdict_preregistered")}
    return out


def pill(text, cls):
    return f'<span class="pill {cls}">{html.escape(text)}</span>'


def succ_cls(v):
    if v is None:
        return "muted"
    if v >= 0.6:
        return "good"
    if v >= 0.3:
        return "warn"
    return "crit"


def fmt(v, d=3):
    return "—" if v is None else f"{v:.{d}f}"


RESULTS = REPO / "results"


def parse_results_csv(path: Path, has_graft: bool):
    rows = []
    if not path.exists():
        return rows
    for r in csv.DictReader(path.open()):
        try:
            mean = float(r["mean"]) if r.get("mean") else None
        except Exception:
            mean = None
        rows.append({"env": r.get("env", ""), "agent": r.get("agent", ""),
                     "actor": r.get("actor", ""), "variant": r.get("variant", ""),
                     "graft": r.get("graft", "") if has_graft else "",
                     "mean": mean, "ci95": r.get("ci95", ""), "n": r.get("n_seeds", ""),
                     "group": r.get("wandb_group", ""), "notes": r.get("notes", "")})
    return rows


def band(v):
    """Interpretation band on the cube scale (FB floor 0.42, GCIQL-bilinear ceiling ~0.90)."""
    if v is None:
        return "muted"
    if v >= 0.70:
        return "good"
    if v >= 0.40:
        return "warn"
    return "crit"


def render_results_table(rows, tid, with_graft):
    envs = sorted({r["env"] for r in rows if r["env"]})
    opts = "".join(f'<option value="{html.escape(e)}">{html.escape(e)}</option>' for e in envs)
    head = (['env', 'agent', 'actor', 'variant'] + (['graft'] if with_graft else [])
            + ['success@500k', 'ci95', 'seeds'])
    ths = "".join(
        f'<th data-sort="{"num" if h.startswith("success") else "str"}" data-col="{i}">{h} <span class="ar"></span></th>'
        for i, h in enumerate(head))
    trs = []
    for r in sorted(rows, key=lambda x: (x["mean"] is None, -(x["mean"] or 0))):
        m = r["mean"]
        cells = [r["env"], r["agent"], r["actor"], r["variant"]]
        if with_graft:
            cells.append(r["graft"] or r["notes"])
        cells_html = "".join(f'<td>{html.escape(str(c))}</td>' for c in cells)
        mcell = (f'<td class="num {band(m)}" data-v="{m if m is not None else -1}"><b>{m:.3f}</b></td>'
                 if m is not None else '<td class="num muted" data-v="-1">—</td>')
        trs.append(f'<tr data-env="{html.escape(r["env"])}" data-text="{html.escape((" ".join(str(c) for c in cells)).lower())}">'
                   f'{cells_html}{mcell}<td>{html.escape(r["ci95"])}</td><td>{html.escape(r["n"])}</td></tr>')
    return (f'<div class="controls">'
            f'<input class="filter" data-target="{tid}" placeholder="filter (agent, graft, variant…)">'
            f'<select class="envfilter" data-target="{tid}"><option value="">all envs</option>{opts}</select>'
            f'<span class="count" id="{tid}-count"></span></div>'
            f'<div class="scroll"><table class="rtable" id="{tid}"><thead><tr>{ths}</tr></thead>'
            f'<tbody>{"".join(trs)}</tbody></table></div>')


RTABLE_JS = """
<script>
(function(){
  function apply(tid){
    var t=document.getElementById(tid); if(!t) return;
    var f=(document.querySelector('.filter[data-target="'+tid+'"]')||{}).value||'';
    var e=(document.querySelector('.envfilter[data-target="'+tid+'"]')||{}).value||'';
    f=f.toLowerCase(); var n=0, rows=t.tBodies[0].rows;
    for(var i=0;i<rows.length;i++){var r=rows[i];
      var ok=(!f||r.getAttribute('data-text').indexOf(f)>-1)&&(!e||r.getAttribute('data-env')===e);
      r.style.display=ok?'':'none'; if(ok)n++;}
    var c=document.getElementById(tid+'-count'); if(c)c.textContent=n+' rows';
  }
  document.querySelectorAll('.filter,.envfilter').forEach(function(el){
    el.addEventListener('input',function(){apply(el.getAttribute('data-target'));});
    el.addEventListener('change',function(){apply(el.getAttribute('data-target'));});
  });
  document.querySelectorAll('table.rtable thead th').forEach(function(th){
    th.style.cursor='pointer';
    th.addEventListener('click',function(){
      var t=th.closest('table'), tb=t.tBodies[0], idx=th.cellIndex,
          num=th.getAttribute('data-sort')==='num',
          asc=th.getAttribute('data-asc')==='1';
      th.setAttribute('data-asc',asc?'0':'1');
      var rs=Array.prototype.slice.call(tb.rows);
      rs.sort(function(a,b){
        var x,y;
        if(num){x=parseFloat(a.cells[idx].getAttribute('data-v'));y=parseFloat(b.cells[idx].getAttribute('data-v'));}
        else{x=a.cells[idx].textContent.toLowerCase();y=b.cells[idx].textContent.toLowerCase();}
        return (x<y?-1:x>y?1:0)*(asc?1:-1);
      });
      rs.forEach(function(r){tb.appendChild(r);});
    });
  });
  document.querySelectorAll('table.rtable').forEach(function(t){apply(t.id);});
})();
</script>
"""


def build():
    launched, crashed, groups = scrape_training()
    rerank = scrape_rerank()
    gpi = scrape_gpi()
    now = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    bench = parse_results_csv(RESULTS / "summary_agents.csv", False)
    bridge = parse_results_csv(RESULTS / "bridge_grafts.csv", True)
    bench_html = render_results_table(bench, "benchTable", False)
    bridge_html = render_results_table(bridge, "bridgeTable", True)
    n_bench, n_bridge = len(bench), len(bridge)
    best_by_env = {}
    for r in bench:
        if r["mean"] is not None and r["mean"] > best_by_env.get(r["env"], (-1, ""))[0]:
            best_by_env[r["env"]] = (r["mean"], f'{r["agent"]}·{r["actor"]}·{r["variant"]}')
    anchors = " · ".join(f'<b>{e}</b> best {v:.2f} <span class="muted">({who})</span>'
                         for e, (v, who) in sorted(best_by_env.items()))

    # KPIs (auto-discovered groups; method vs measure-win/mechanism)
    n_seeds_total = sum(len(s) for s in groups.values())
    n_groups = len(groups)
    live = 0; best_method = None; mech_ceiling = None
    for g, seeds in groups.items():
        kind = group_meta(g)[1]
        for s in seeds:
            ev = s["eval"]
            if ev is None:
                continue
            live += 1
            if kind == "method":
                best_method = ev["overall"] if best_method is None else max(best_method, ev["overall"])
            else:
                mech_ceiling = ev["overall"] if mech_ceiling is None else max(mech_ceiling, ev["overall"])

    kind_order = {"method": 0, "measure-win": 1, "mechanism": 2, "arch": 3}
    rows = []
    for g in sorted(groups, key=lambda x: (kind_order.get(group_meta(x)[1], 9), x)):
        item, kind, meaning, read = group_meta(g)
        seeds = groups.get(g, [])
        cells = []
        for s in seeds:
            ev = s["eval"]
            if ev is None:
                cells.append('<td class="muted">init</td>')
            else:
                cells.append(f'<td class="{succ_cls(ev["overall"])}">'
                             f'<b>{ev["overall"]:.3f}</b><span class="sub">@{ev["step"]//1000}k</span></td>')
        while len(cells) < 3:
            cells.append('<td class="muted">—</td>')
        rows.append(
            f'<tr><td class="grp"><span class="tag {kind}">{html.escape(item)} · {kind}</span>'
            f'<div class="gname">{html.escape(g)}</div>'
            f'<div class="meaning">{html.escape(meaning)}</div></td>'
            + "".join(cells) +
            f'<td class="read">{html.escape(read)}</td></tr>')
    train_table = "\n".join(rows)

    # rerank card
    if rerank.get("done"):
        rr = (f'<div class="big {succ_cls(rerank.get("best_succ"))}">{fmt(rerank.get("best_succ"))}</div>'
              f'<div class="cap">best rerank cell: {html.escape(str(rerank.get("best")))}</div>'
              f'<div>{pill(str(rerank.get("verdict")), "crit" if "BELOW" in str(rerank.get("verdict")) else "good")}</div>')
    else:
        rr = (f'<div class="cap">running · {rerank.get("seeds_started",0)}/3 seeds, {rerank.get("n_task_rows",0)} task-rows</div>'
              f'<table class="mini"><tr><th></th><th>floor</th><th>reranked</th></tr>'
              f'<tr><td>BC flow (flow_z)</td><td>{fmt(rerank.get("flow_floor"),2)}</td><td class="crit">{fmt(rerank.get("flow_N32"),2)} <span class="sub">N32</span></td></tr>'
              f'<tr><td>one-shot actor</td><td>{fmt(rerank.get("oneshot_floor"),2)}</td><td class="warn">{fmt(rerank.get("oneshot_N64"),2)} <span class="sub">N64</span></td></tr></table>'
              f'<div class="note">emerging read: selection HURTS — BC flow →0, one-shot declines with N (off-support peak)</div>')

    # gpi card
    gpi_rows = []
    for tag, d in gpi.items():
        gpi_rows.append(f'<tr><td>{html.escape(tag)}</td>'
                        f'<td class="{succ_cls(d.get("gate"))}">{fmt(d.get("gate"))}</td>'
                        f'<td class="{succ_cls(d.get("mixture"))}">{fmt(d.get("mixture"))}</td>'
                        f'<td>{html.escape(str(d.get("verdict") or "—"))}</td></tr>')
    gpi_body = ("".join(gpi_rows) if gpi_rows else
                '<tr><td colspan="4" class="muted">full-stage GPI running (gate smoke: 0.933 PASS)</td></tr>')

    crash_pill = pill("0 crashed", "good") if not crashed else pill(f"{len(crashed)} CRASHED: " + ", ".join(crashed), "crit")

    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Spec-06 experiments — live</title>
<style>
:root {{
  --paper:#f6f8fa; --card:#ffffff; --ink:#0f1720; --muted:#5b6b7a; --line:#dce3ea;
  --accent:#0ea5b7; --good:#1f9d63; --warn:#b8801a; --crit:#cf3b52; --shadow:0 1px 2px rgba(15,23,32,.06);
}}
@media (prefers-color-scheme: dark) {{
  :root {{ --paper:#0c1116; --card:#141b22; --ink:#e8eef3; --muted:#8296a6; --line:#243039;
    --accent:#2bc2d4; --good:#3fce8a; --warn:#e0a83a; --crit:#f4667c; --shadow:0 1px 2px rgba(0,0,0,.3); }}
}}
:root[data-theme="dark"] {{ --paper:#0c1116; --card:#141b22; --ink:#e8eef3; --muted:#8296a6; --line:#243039;
  --accent:#2bc2d4; --good:#3fce8a; --warn:#e0a83a; --crit:#f4667c; --shadow:0 1px 2px rgba(0,0,0,.3); }}
:root[data-theme="light"] {{ --paper:#f6f8fa; --card:#ffffff; --ink:#0f1720; --muted:#5b6b7a; --line:#dce3ea;
  --accent:#0ea5b7; --good:#1f9d63; --warn:#b8801a; --crit:#cf3b52; --shadow:0 1px 2px rgba(15,23,32,.06); }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--paper); color:var(--ink);
  font:15px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; }}
.wrap {{ max-width:1100px; margin:0 auto; padding:28px 20px 60px; }}
h1 {{ font-size:1.5rem; margin:0 0 2px; letter-spacing:-.01em; text-wrap:balance; }}
.subhead {{ color:var(--muted); font-size:.9rem; margin-bottom:20px; }}
.subhead b {{ color:var(--ink); }}
.tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:12px; margin-bottom:24px; }}
.tile {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:14px 16px; box-shadow:var(--shadow); }}
.tile .lab {{ font-size:.72rem; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); }}
.tile .val {{ font-size:1.7rem; font-weight:650; font-variant-numeric:tabular-nums; margin-top:2px; }}
.tile .val .sub {{ font-size:.8rem; color:var(--muted); font-weight:400; }}
section {{ background:var(--card); border:1px solid var(--line); border-radius:14px; padding:18px 18px 6px;
  margin-bottom:20px; box-shadow:var(--shadow); }}
section > h2 {{ font-size:.95rem; margin:0 0 12px; letter-spacing:-.005em; display:flex; align-items:center; gap:10px; }}
.scroll {{ overflow-x:auto; }}
table {{ width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; }}
th, td {{ text-align:left; padding:9px 10px; border-bottom:1px solid var(--line); vertical-align:top; font-size:.87rem; }}
th {{ font-size:.7rem; text-transform:uppercase; letter-spacing:.05em; color:var(--muted); font-weight:600; }}
td b {{ font-size:1rem; }}
td .sub {{ color:var(--muted); font-size:.75rem; margin-left:4px; }}
.grp {{ min-width:230px; }}
.gname {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.78rem; color:var(--ink); margin-top:3px; }}
.meaning {{ color:var(--muted); font-size:.74rem; margin-top:2px; }}
.read {{ color:var(--muted); font-size:.76rem; max-width:280px; }}
.good {{ color:var(--good); }} .warn {{ color:var(--warn); }} .crit {{ color:var(--crit); }} .muted {{ color:var(--muted); }}
.tag {{ font-size:.66rem; text-transform:uppercase; letter-spacing:.05em; padding:2px 7px; border-radius:20px; border:1px solid var(--line); color:var(--muted); }}
.tag.method {{ color:var(--accent); border-color:color-mix(in srgb, var(--accent) 40%, var(--line)); }}
.tag.measure-win {{ color:var(--good); border-color:color-mix(in srgb, var(--good) 45%, var(--line)); font-weight:600; }}
.pill {{ display:inline-block; font-size:.72rem; font-weight:600; padding:3px 9px; border-radius:20px; }}
.pill.good {{ background:color-mix(in srgb,var(--good) 16%,transparent); color:var(--good); }}
.pill.warn {{ background:color-mix(in srgb,var(--warn) 18%,transparent); color:var(--warn); }}
.pill.crit {{ background:color-mix(in srgb,var(--crit) 16%,transparent); color:var(--crit); }}
.cards {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; }}
@media (max-width:720px) {{ .cards {{ grid-template-columns:1fr; }} }}
.big {{ font-size:2rem; font-weight:700; font-variant-numeric:tabular-nums; }}
.cap {{ color:var(--muted); font-size:.8rem; margin:2px 0 8px; }}
.note {{ font-size:.78rem; color:var(--ink); background:color-mix(in srgb,var(--accent) 8%,transparent);
  border-left:3px solid var(--accent); padding:7px 10px; border-radius:0 6px 6px 0; margin-top:10px; }}
table.mini td, table.mini th {{ padding:5px 8px; font-size:.8rem; }}
.foot {{ color:var(--muted); font-size:.8rem; }}
ul.log {{ margin:6px 0 12px; padding-left:18px; }} ul.log li {{ margin:3px 0; font-size:.82rem; }}
.controls {{ display:flex; gap:10px; align-items:center; margin:2px 0 12px; flex-wrap:wrap; }}
.controls input, .controls select {{ font:inherit; font-size:.85rem; padding:6px 10px; border:1px solid var(--line);
  border-radius:8px; background:var(--paper); color:var(--ink); }}
.controls input {{ min-width:230px; flex:1; }}
.controls .count {{ color:var(--muted); font-size:.78rem; }}
table.rtable th {{ white-space:nowrap; user-select:none; }}
table.rtable th .ar {{ font-size:.6rem; color:var(--muted); }}
table.rtable td {{ font-size:.83rem; }}
table.rtable td.num {{ font-weight:650; text-align:right; }}
table.rtable tbody tr:hover {{ background:color-mix(in srgb,var(--accent) 6%,transparent); }}
.anchbar {{ font-size:.8rem; color:var(--muted); background:var(--card); border:1px solid var(--line);
  border-radius:10px; padding:10px 14px; margin-bottom:16px; }}
.legend {{ display:inline-flex; gap:14px; font-size:.75rem; color:var(--muted); margin-left:auto; }}
.legend b {{ padding:1px 7px; border-radius:20px; font-weight:600; }}
.legend .good {{ background:color-mix(in srgb,var(--good) 16%,transparent); color:var(--good); }}
.legend .warn {{ background:color-mix(in srgb,var(--warn) 18%,transparent); color:var(--warn); }}
.legend .crit {{ background:color-mix(in srgb,var(--crit) 16%,transparent); color:var(--crit); }}
</style></head><body><div class="wrap">
<h1>Factored-FB — project results &amp; live status</h1>
<div class="subhead">FB↔GCIQL bridge · rescuing-fb · ogbench-sota · atlas — all results + what's running now · last refresh <b>{now}</b> · auto-updated by the monitoring loop</div>
<div class="anchbar">Reading the numbers (cube scale): <b>FB-flow floor ≈ 0.42</b> · <b>GCIQL-bilinear ceiling ≈ 0.90</b> · the whole bridge/rescuing-fb program asks what closes that gap.
<span class="legend"><b class="crit">&lt;0.40 weak</b><b class="warn">0.40–0.70 mid</b><b class="good">≥0.70 strong</b></span></div>

<div class="tiles">
  <div class="tile"><div class="lab">runs on disk</div><div class="val">{n_seeds_total}<span class="sub">across {n_groups} groups</span></div></div>
  <div class="tile"><div class="lab">runs reporting evals</div><div class="val">{live}</div></div>
  <div class="tile"><div class="lab">health</div><div class="val" style="font-size:1rem;padding-top:8px">{crash_pill}</div></div>
  <div class="tile"><div class="lab">best method exact-z</div><div class="val {succ_cls(best_method)}">{fmt(best_method,3)}</div></div>
  <div class="tile"><div class="lab">mechanism-cell ceiling</div><div class="val {succ_cls(mech_ceiling)}">{fmt(mech_ceiling,3)}</div></div>
</div>

<section>
  <h2>Training campaigns <span class="foot">· per-seed latest eval (success @ step)</span></h2>
  <div class="scroll"><table>
    <tr><th>campaign</th><th>seed 0</th><th>seed 1</th><th>seed 2</th><th>pre-registered read</th></tr>
    {train_table}
  </table></div>
</section>

<div class="cards">
  <section>
    <h2>Item 3 · selection / rerank <span class="tag method">method</span></h2>
    {rr}
  </section>
  <section>
    <h2>Item 1 · GPI over goal candidates <span class="tag method">method</span></h2>
    <div class="scroll"><table class="mini">
      <tr><th>checkpoint</th><th>gate</th><th>2-goal mix</th><th>verdict</th></tr>
      {gpi_body}
    </table></div>
    <div class="note">gate ≥0.85 = GPI path sound; two-goal mixture ≥0.60 = GPI fixes actor coverage</div>
  </section>
</div>

<section>
  <h2>Not yet launched / gated / fixed</h2>
  <ul class="log">
    <li><b>v1d50</b> (item 4) — FAILED to launch (config passed <code>task_prior_probs</code> as a list, which ml_collections flag-override rejects). <b>Config fixed</b> (drop the override; default is 0.6/0.2/0.2) → relaunchable via <code>ARMS="v1d50"</code>.</li>
    <li><b>v1d512</b> (item 4 @ d512) — gated on item 2 (d512-v0) reaching ≥0.88; held.</li>
    <li><b>ogbench-sota cells</b> (latefusion / pmatched / phi_norm_cap / bounded / scene-split) — coded + tested, <code>launch_sota_impl.sh</code> DRY_RUN; bounded needs psi_bound_scale calibration, scene-split needs value_split_dims.</li>
    <li><b>item 8</b> readout head — jax RG-FB arm runnable now; torch plain-FB arm needs a plain-FB checkpoint.</li>
    <li>Method-chain stop rule: zero-shot ≥0.90 via any of items 2/3/4 ⇒ freeze &amp; write.</li>
  </ul>
</section>

<section>
  <h2>Canonical benchmark <span class="foot">· {n_bench} agent×env×actor cells @500k (summary_agents.csv)</span></h2>
  <div class="foot" style="margin-bottom:10px">{anchors}</div>
  {bench_html}
</section>

<section>
  <h2>Bridge / rescuing-fb program <span class="foot">· {n_bridge} grafts &amp; ablations (bridge_grafts.csv) — sort by success, filter by env, search the graft</span></h2>
  {bridge_html}
</section>

<div class="foot">Sources: <code>~/exp_spec06/factored-fb/&lt;group&gt;/*/eval.csv</code> (RG-FB overall = exact-z diagnostic; zero-shot is post-hoc via <code>eval_rgfb_zeroshot.py</code>) · <code>/dev/shm/s06cpu.log</code> · <code>analysis/rescuing-fb/*.json</code>. Regenerated by <code>scripts/rescuing_fb/spec06_dashboard.py</code>.</div>
</div>{RTABLE_JS}</body></html>"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(doc)
    print(f"[dashboard] {now}: launched={launched} reporting={live} crashed={len(crashed)} "
          f"best_method={fmt(best_method)} mech_ceiling={fmt(mech_ceiling)} -> {OUT}")


if __name__ == "__main__":
    build()
