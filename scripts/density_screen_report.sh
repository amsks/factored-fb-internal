#!/bin/bash
# Read the density dev screens against the collapse spec's graduation gate:
#   500k final >= 0.55  AND  no >0.25 drop from that arm's own peak.
#
#   bash scripts/density_screen_report.sh [run-root ...]
#
# Reads the job .o logs, not eval.csv: eval.csv can carry null bytes and duplicate-pass rows
# (both hit this campaign already). Highest job id per job name wins.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
"${VENV:-$SCRATCH/.venv-jax-fb}/bin/python" - "$@" <<'PY'
import ast, glob, re, sys, collections

pat = sys.argv[1:] or ["logs/DP-*.o*"]
best = {}
for p in pat:
    for f in glob.glob(p):
        m = re.match(r".*/(.+)\.o(\d+)$", f)
        if not m:
            continue
        name, jid = m.group(1), int(m.group(2))
        if name not in best or jid > best[name][0]:
            best[name] = (jid, f)

if not best:
    sys.exit(f"no job logs matched {pat}")

print(f"{'arm':<28}{'job':>9}{'evals':>7}{'peak':>9}{'@step':>9}{'final':>9}{'drop':>8}  gate")
print("-" * 92)
for name in sorted(best):
    jid, f = best[name]
    ev = {}
    for line in open(f, errors="ignore"):
        if line.startswith("eval "):
            parts = line.split(None, 4)
            try:
                ev[int(parts[1])] = float(ast.literal_eval(parts[4].strip())["success"])
            except Exception:
                pass
    ev.pop(1, None)                      # the step-1 eval is the untrained agent
    if not ev:
        print(f"{name:<28}{jid:>9}{0:>7}   (no eval lines yet)")
        continue
    final_step = max(ev)
    final = ev[final_step]
    peak_step = max(ev, key=ev.get)
    peak = ev[peak_step]
    drop = peak - final
    ok = final >= 0.55 and drop <= 0.25
    verdict = "GRADUATES" if ok else ("fail: final<0.55" if final < 0.55 else "fail: decays")
    if final < 0.55 and drop > 0.25:
        verdict = "fail: low+decays"
    print(f"{name:<28}{jid:>9}{len(ev):>7}{peak:>9.3f}{peak_step//1000:>8}k{final:>9.3f}"
          f"{drop:>8.2f}  {verdict}")
print("\ngate: 500k final >= 0.55 AND peak-final <= 0.25 (collapse spec §4).")
print("If NO arm clears 0.5, the structural-failure clause applies: report the instability,")
print("not the ceiling, as the deficit — with the peak disclosed in the table caption (R4).")
PY
