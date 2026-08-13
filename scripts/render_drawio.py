#!/usr/bin/env python
"""Render a draw.io (mxGraph) diagram to PNG with matplotlib.

Covers the subset of mxGraph styling used by the paper's method schematic:
rounded/plain rects, ellipses, text cells, and straight or elbowed edges with
block arrowheads. Rich labels are handled as runs of (text, colour, size) so a
single box can mix colours, and known label strings are mapped to mathtext so
that bars, hats and Greek render properly.

    python scripts/render_drawio.py <in.drawio> <out.png> [--dpi 400]
"""
from __future__ import annotations

import argparse
import html
import re
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse, Polygon
from matplotlib.lines import Line2D

# Labels that read better as mathtext than as raw unicode (combining macrons and
# unicode subscripts do not shape correctly in the default font).
MATHTEXT = {
    "F(s,a,z)": r"$F(s,a,z)$",
    "ψ(g)": r"$\psi(g)$",
    "Q": r"$Q$",
    "M": r"$M$",
    "y = h + γ (1 − h) Q̄": r"$y = h + \gamma\,(1-h)\,\bar{Q}$",
    "y = h + γ M̄": r"$y = h + \gamma\,\bar{M}$",
    "h ∈ {0,1}": r"$h \in \{0,1\}$",
    "N̂": r"$\hat{N}$",
    "zᵣ = E[ r · ψ / N̂ ]": r"$z_r = \mathrm{E}[\,r\cdot\psi\,/\,\hat{N}\,]$",
    "act on zᵣ": r"act on $z_r$",
    "⟨·,·⟩": r"$\langle\cdot,\cdot\rangle$",
}

PT = 1.0 / 1.34  # mxGraph px -> matplotlib pt, so drawn text matches the editor


def parse_style(s: str) -> dict:
    out = {}
    for part in (s or "").split(";"):
        if not part:
            continue
        k, _, v = part.partition("=")
        out[k.strip()] = v.strip()
    return out


def runs_from_label(value: str, base_size: float, base_color: str) -> list[dict]:
    """Split an mxGraph HTML label into styled runs, one per line."""
    if not value:
        return []
    v = value.replace("&nbsp;", " ")
    v = re.sub(r"<br\s*/?>", "\n", v, flags=re.I)

    lines, size, color, bold = [[]], base_size, base_color, False
    stack = []
    for tok in re.split(r"(<[^>]+>)", v):
        if not tok:
            continue
        if tok.startswith("<"):
            if tok.startswith("</"):
                if stack:
                    size, color, bold = stack.pop()
            else:
                stack.append((size, color, bold))
                m = re.search(r"font-size:\s*(\d+(?:\.\d+)?)", tok)
                if m:
                    size = float(m.group(1))
                m = re.search(r'color="(#[0-9A-Fa-f]{6})"', tok)
                if m:
                    color = m.group(1)
                if re.match(r"<b[\s>]", tok) or tok.startswith("<b>"):
                    bold = True
            continue
        text = html.unescape(tok)
        for i, chunk in enumerate(text.split("\n")):
            if i:
                lines.append([])
            if chunk:
                lines[-1].append(
                    {"text": chunk, "size": size, "color": color, "bold": bold}
                )
    return [ln for ln in lines if ln]


def draw_runs(ax, fig, lines, box, align, spacing_left, valign="middle"):
    """Place styled runs inside box=(x, y, w, h) in mxGraph coordinates."""
    x, y, w, h = box
    renderer = fig.canvas.get_renderer()
    n = len(lines)
    for li, runs in enumerate(lines):
        merged, out = "".join(r["text"] for r in runs).strip(), []
        # A whole line that matches a known label renders as one mathtext run.
        if merged in MATHTEXT:
            runs = [
                {
                    "text": MATHTEXT[merged],
                    "size": max(r["size"] for r in runs),
                    "color": runs[0]["color"],
                    "bold": any(r["bold"] for r in runs),
                }
            ]
        else:
            for r in runs:
                t = r["text"]
                r = dict(r)
                r["text"] = MATHTEXT.get(t.strip(), t)
                runs.append(r) if False else None
                out.append(r)
            runs = out

        line_h = max(r["size"] for r in runs) * 1.45
        cy = y + h / 2 + (li - (n - 1) / 2) * line_h

        artists, widths = [], []
        for r in runs:
            t = ax.text(
                0,
                0,
                r["text"],
                color=r["color"],
                fontsize=r["size"] * PT,
                fontweight="bold" if r["bold"] else "normal",
                ha="left",
                va="center",
                zorder=6,
            )
            bb = t.get_window_extent(renderer=renderer)
            widths.append(bb.width / fig.dpi * 72 / PT * (1 / 72) * fig.dpi / fig.dpi)
            # convert display px -> data units
            inv = ax.transData.inverted()
            p0 = inv.transform((0, 0))
            p1 = inv.transform((bb.width, 0))
            widths[-1] = abs(p1[0] - p0[0])
            artists.append(t)

        total = sum(widths)
        if align == "left":
            cx = x + spacing_left
        elif align == "right":
            cx = x + w - spacing_left - total
        else:
            cx = x + w / 2 - total / 2
        for t, wd in zip(artists, widths):
            t.set_position((cx, cy))
            cx += wd


def anchor(cell, fx, fy):
    x, y, w, h = cell["geom"]
    return (x + fx * w, y + fy * h)


def edge_endpoint(cell, style, prefix, fallback):
    if cell is None:
        return fallback
    if f"{prefix}X" in style:
        return anchor(cell, float(style[f"{prefix}X"]), float(style[f"{prefix}Y"]))
    return None  # resolved later against the other end


def nearest_side(cell, toward):
    x, y, w, h = cell["geom"]
    cx, cy = x + w / 2, y + h / 2
    dx, dy = toward[0] - cx, toward[1] - cy
    if abs(dx) * h > abs(dy) * w:
        return (x + w if dx > 0 else x, cy)
    return (cx, y + h if dy > 0 else y)


def render(src: str, out: str, dpi: int, pad: float = 14.0):
    root = ET.parse(src).getroot()
    model = root if root.tag == "mxGraphModel" else root.find(".//mxGraphModel")
    cells = model.find("root")

    vertices, order = {}, []
    for c in cells.findall("mxCell"):
        g = c.find("mxGeometry")
        style = parse_style(c.get("style"))
        item = {
            "id": c.get("id"),
            "style": style,
            "value": c.get("value") or "",
            "edge": c.get("edge") == "1",
            "geom": None,
            "cell": c,
        }
        if g is not None and g.get("width"):
            item["geom"] = (
                float(g.get("x", 0)),
                float(g.get("y", 0)),
                float(g.get("width")),
                float(g.get("height")),
            )
            vertices[item["id"]] = item
        order.append(item)

    xs = [v["geom"][0] for v in vertices.values()] + [
        v["geom"][0] + v["geom"][2] for v in vertices.values()
    ]
    ys = [v["geom"][1] for v in vertices.values()] + [
        v["geom"][1] + v["geom"][3] for v in vertices.values()
    ]
    x0, x1, y0, y1 = min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad
    W, H = x1 - x0, y1 - y0

    fig = plt.figure(figsize=(W / 100, H / 100), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)  # mxGraph y grows downwards
    ax.axis("off")
    fig.canvas.draw()

    for item in order:
        st, val = item["style"], item["value"]
        if item["edge"]:
            src_c = vertices.get(item["cell"].get("source"))
            tgt_c = vertices.get(item["cell"].get("target"))
            g = item["cell"].find("mxGeometry")
            pts = []
            arr = g.find("Array") if g is not None else None
            if arr is not None:
                pts = [
                    (float(p.get("x")), float(p.get("y")))
                    for p in arr.findall("mxPoint")
                ]
            sp = tp = None
            for p in g.findall("mxPoint") if g is not None else []:
                if p.get("as") == "sourcePoint":
                    sp = (float(p.get("x")), float(p.get("y")))
                elif p.get("as") == "targetPoint":
                    tp = (float(p.get("x")), float(p.get("y")))

            # A connected end always wins over a stale sourcePoint/targetPoint,
            # which draw.io leaves behind after a reconnect.
            start = edge_endpoint(src_c, st, "exit", sp) or (None if src_c else sp)
            end = edge_endpoint(tgt_c, st, "entry", tp) or (None if tgt_c else tp)
            if start is None and src_c:
                start = nearest_side(src_c, pts[0] if pts else (end or (0, 0)))
            if end is None and tgt_c:
                end = nearest_side(tgt_c, pts[-1] if pts else start)
            if start is None or end is None:
                continue

            path = [start] + pts + [end]
            # de-duplicate consecutive identical points
            clean = [path[0]]
            for p in path[1:]:
                if abs(p[0] - clean[-1][0]) > 1e-6 or abs(p[1] - clean[-1][1]) > 1e-6:
                    clean.append(p)
            if len(clean) < 2:
                continue

            color = st.get("strokeColor", "#77848F")
            lw = float(st.get("strokeWidth", 1)) * 0.9
            dash = (0, (4, 2.5)) if st.get("dashed") == "1" else "solid"
            ax.add_line(
                Line2D(
                    [p[0] for p in clean],
                    [p[1] for p in clean],
                    color=color,
                    linewidth=lw,
                    linestyle=dash,
                    solid_capstyle="butt",
                    zorder=4,
                )
            )
            if st.get("endArrow", "block") not in ("none", "0"):
                (ax1, ay1), (ax2, ay2) = clean[-2], clean[-1]
                dx, dy = ax2 - ax1, ay2 - ay1
                L = (dx * dx + dy * dy) ** 0.5 or 1.0
                ux, uy = dx / L, dy / L
                s = 7.0
                ax.add_patch(
                    Polygon(
                        [
                            (ax2, ay2),
                            (ax2 - s * ux - s * 0.42 * uy, ay2 - s * uy + s * 0.42 * ux),
                            (ax2 - s * ux + s * 0.42 * uy, ay2 - s * uy - s * 0.42 * ux),
                        ],
                        closed=True,
                        facecolor=color,
                        edgecolor="none",
                        zorder=5,
                    )
                )
            continue

        if item["geom"] is None:
            continue
        x, y, w, h = item["geom"]
        fill = st.get("fillColor", "none")
        stroke = st.get("strokeColor", "none")
        is_text = st.get("text") is not None or "text" in (item["style"] or {})
        shape_none = fill in ("none", None) and stroke in ("none", None)

        if not shape_none and not (fill == "none" and stroke == "none"):
            fc = "none" if fill in ("none", None) else fill
            ec = "none" if stroke in ("none", None) else stroke
            lw = float(st.get("strokeWidth", 1)) * 0.9
            ls = (0, (4, 3)) if st.get("dashed") == "1" else "solid"
            if st.get("ellipse") is not None or "ellipse" in st:
                ax.add_patch(
                    Ellipse(
                        (x + w / 2, y + h / 2),
                        w,
                        h,
                        facecolor=fc,
                        edgecolor=ec,
                        linewidth=lw,
                        zorder=3,
                    )
                )
            else:
                r = 3.5 if st.get("rounded") == "1" else 0.0
                ax.add_patch(
                    FancyBboxPatch(
                        (x + r, y + r),
                        max(w - 2 * r, 0.1),
                        max(h - 2 * r, 0.1),
                        boxstyle=f"round,pad={r},rounding_size={r}"
                        if r
                        else "square,pad=0",
                        facecolor=fc,
                        edgecolor=ec,
                        linewidth=lw,
                        linestyle=ls,
                        mutation_aspect=1,
                        zorder=3,
                    )
                )

        lines = runs_from_label(
            val, float(st.get("fontSize", 12)), st.get("fontColor", "#17212C")
        )
        if lines:
            draw_runs(
                ax,
                fig,
                lines,
                (x, y, w, h),
                st.get("align", "center"),
                float(st.get("spacingLeft", 0)),
            )

    fig.savefig(out, dpi=dpi, transparent=False, facecolor="white")
    print(f"wrote {out}  ({W:.0f}x{H:.0f} units @ {dpi} dpi)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--dpi", type=int, default=400)
    a = ap.parse_args()
    render(a.src, a.out, a.dpi)
