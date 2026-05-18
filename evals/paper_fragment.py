"""evals/paper_fragment.py — render an \\input-able LaTeX section from
the phase-probe + representation-profile aggregates. No LaTeX is
compiled here; this only generates text + copies figures.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Sequence

_ESCAPE = {"&": r"\&", "%": r"\%", "#": r"\#", "_": r"\_",
           "$": r"\$", "{": r"\{", "}": r"\}"}

_DEFAULT_FIGS = ("value_vs_dist.png", "sparsity_bars.png",
                 "b_resolution_bars.png", "coverage_curves.png")
_PHASE_FIGS = ("funnel.png", "fail_phase_composition.png")


def latex_escape(s: str) -> str:
    return "".join(_ESCAPE.get(ch, ch) for ch in str(s))


def df_to_booktabs(df, caption: str, label: str) -> str:
    cols = list(df.columns)
    head = " & ".join(latex_escape(c) for c in cols) + r" \\"
    body = []
    for _, row in df.iterrows():
        cells = []
        for v in row:
            cells.append(f"{v:.3g}" if isinstance(v, float)
                         else latex_escape(v))
        body.append(" & ".join(cells) + r" \\")
    return "\n".join([
        r"\begin{table}[t]", r"\centering",
        r"\begin{tabular}{" + "l" * len(cols) + "}",
        r"\toprule", head, r"\midrule", *body, r"\bottomrule",
        r"\end{tabular}",
        r"\caption{" + caption + "}", r"\label{" + label + "}",
        r"\end{table}", ""])


def _fig(figdir_rel: str, name: str, present: bool, caption: str) -> str:
    if not present:
        return f"% [figure missing] {name}\n"
    return "\n".join([
        r"\begin{figure}[t]", r"\centering",
        r"\includegraphics[width=0.7\linewidth]{"
        + f"{figdir_rel}/{name}" + "}",
        r"\caption{" + caption + "}",
        r"\end{figure}", ""])


def build_fragment(phase_root, repr_root, out_dir, subdir: str,
                   figures: Sequence[str] = _DEFAULT_FIGS) -> Path:
    phase_root, repr_root = Path(phase_root), Path(repr_root)
    base = Path(out_dir) / subdir
    figdir = base / "figures"
    tabdir = base / "tables"
    for d in (figdir, tabdir):
        d.mkdir(parents=True, exist_ok=True)

    for missing in (phase_root, repr_root):
        if not missing.exists():
            raise FileNotFoundError(
                f"Aggregate input not found: {missing}. Produce it with "
                f"the *_aggregate scripts before building the fragment.")

    # copy figures (phase + repr); track which exist
    present = {}
    for src_root, names in ((phase_root, _PHASE_FIGS),
                            (repr_root, figures)):
        for n in names:
            src = src_root / n
            if src.exists():
                shutil.copyfile(src, figdir / n)
                present[n] = True
            else:
                present[n] = False

    # tables from repr aggregate parquet
    table_inputs = []
    try:
        import pandas as pd
        for pq in sorted(repr_root.glob("T*.parquet")):
            df = pd.read_parquet(pq).reset_index()
            tname = pq.stem
            (tabdir / f"{tname}.tex").write_text(
                df_to_booktabs(df, caption=tname.replace("_", " "),
                               label=f"tab:{tname}"))
            table_inputs.append(f"tables/{tname}.tex")
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] table generation skipped: {e}")

    story = ""
    sp = repr_root / "story.md"
    if sp.exists():
        story = latex_escape(sp.read_text())

    figs_phase = "".join(
        _fig("figures", n, present.get(n, False),
             "Phase-probe: " + n.replace("_", " ").replace(".png", ""))
        for n in _PHASE_FIGS)
    figs_repr = "".join(
        _fig("figures", n, present.get(n, False),
             n.replace("_", " ").replace(".png", ""))
        for n in figures)
    tbls = "".join(r"\input{" + t + "}" + "\n" for t in table_inputs)

    tex = f"""% Auto-generated; regenerate via scripts/build_paper_fragment.py
\\section{{Where and why FB fails on cube manipulation}}
\\label{{sec:fb-cube-failure}}

\\subsection{{Experiment design}}
We decompose each cube episode into reach $\\to$ grasp/lift $\\to$
transport, and probe the policy from three start states: S0 (baseline),
S1 (skip-reach), S2 (pre-grasped, physically clamped, no weld). The
phase-probe over seeds s3--s8, s10 localises failure.
{figs_phase}

\\subsection{{Methods}}
Four representation probes on the FB checkpoints: (i) value landscape
$V(s)=Q_z(s,\\pi(s))$ vs cube--goal distance; (ii) z-decoding and
relabel sparsity; (iii) ridge $R^2$ of $B(s)\\!\\to\\!d$ and
placed-vs-near separability; (iv) off-support nearest-neighbour
distance. Probes are aggregated (mean$\\pm$std) over the seven seeds.

\\subsection{{Results}}
{figs_repr}{tbls}

\\subsection{{Discussion}}
{story}
"""
    (base / "fb_cube_failure.tex").write_text(tex)
    (base / "standalone.tex").write_text(
        "\\documentclass{article}\n"
        "\\usepackage{graphicx}\\usepackage{booktabs}\n"
        "\\begin{document}\n"
        "\\input{fb_cube_failure.tex}\n"
        "\\end{document}\n")
    return base
