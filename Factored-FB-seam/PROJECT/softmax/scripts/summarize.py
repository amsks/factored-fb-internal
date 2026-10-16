"""Summarise occupancy_rank.py json results into a markdown table.
Usage: python scripts/summarize.py results/ogbench/*.json [--md out.md]"""
import json, sys, glob, argparse
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("files", nargs="+"); ap.add_argument("--md"); ap.add_argument("--eps", type=float, default=0.1)
a = ap.parse_args()
rows = []
for f in sorted(sum([glob.glob(x) for x in a.files], [])):
    j = json.load(open(f)); s = j["stats"]; ranks = j["ranks"]
    def eps_rank(model):
        for r in ranks:
            if j[model][str(r)]["tv"] <= a.eps: return r
        return f">{ranks[-1]}"
    def at(model, r, m="tv"): return j[model][str(r)][m] if str(r) in j[model] else float("nan")
    rows.append(dict(tag=j["tag"], K=s["K"], conc=s["concentration"], ess=s["ess_median"], essK=s["ess_over_K"],
                     maxR=s["maxR_median"], p1self=s["P1_self"], HP1=s["H_P1_mean"],
                     lin16=at("linear",16), log16=at("log",16), lin64=at("linear",64), log64=at("log",64),
                     lin_top16=at("linear",16,"top1"), log_top16=at("log",16,"top1"),
                     lin_g16=at("linear",16,"goal_value_relerr"), log_g16=at("log",16,"goal_value_relerr"),
                     lin_eps=eps_rank("linear"), log_eps=eps_rank("log"),
                     sv_decay=np.array(j["singular_values"])[:64].sum()/np.array(j["singular_values"]).sum()))
hdr = ["tag","K","conc","ess_med","ess/K","maxR_med","P1_self","H_P1","TV lin@16","TV log@16","TV lin@64","TV log@64","top1 lin@16","top1 log@16","goalV err lin@16","goalV err log@16",f"rank lin TV<={a.eps}",f"rank log TV<={a.eps}"]
keys = ["tag","K","conc","ess","essK","maxR","p1self","HP1","lin16","log16","lin64","log64","lin_top16","log_top16","lin_g16","log_g16","lin_eps","log_eps"]
lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
for r in rows:
    lines.append("| " + " | ".join(f"{r[k]:.3g}" if isinstance(r[k], float) else str(r[k]) for k in keys) + " |")
out = "\n".join(lines); print(out)
if a.md: open(a.md, "w").write(out + "\n")
