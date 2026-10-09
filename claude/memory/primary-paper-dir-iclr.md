---
name: primary-paper-dir-iclr
description: As of 2026-09-02 the PRIMARY paper directory is PAPER/ICLR/ — everything related to Density-FB lives there; supersedes PAPER/Multistep-FB/ and PAPER/rlbrew/
metadata: 
  node_type: memory
  type: project
  originSessionId: fa4a4fd2-4abd-4ec7-ae98-b6d15b962f10
  modified: 2026-09-03T02:15:30.982Z
---

User instruction, 2026-09-02: **`PAPER/ICLR/` is the primary directory, and it holds
everything related to Density-FB.** New tables, results and writeups for the
Density-FB program go there.

Supersedes [[primary-paper-dir-multistep-fb]] (which named `PAPER/Multistep-FB/`
on 2026-08-06). `CLAUDE.md` still names `PAPER/rlbrew/`; that line has now been
stale twice — don't trust it on this point, and don't edit CLAUDE.md unasked.

**MOVED 2026-09-03 (user): absolute path is now
`/home/adityamohan/git/Factored-FB-seam/PAPER/ICLR/`** — the SEAM worktree,
because the latest results live there. Seeded by full copy from the main
worktree's `Factored-FB/PAPER/ICLR/` (which the cube session had been
writing — ⚠ if that session keeps updating `table_privcoord.tex` in the
main worktree, re-sync before quoting; see the dir's README.md for the
write-side rules of `table_main_1m.tex` / `table_benchmarks_autogen.tex`).
`PAPER/` is gitignored in BOTH worktrees, so nothing here is under version
control (the tracked `decoding-bilinear/` files are a pre-ignore anomaly).

Layout:
- `main.tex` — ICLR-2027 style (`iclr2027_conference`), `\input{math_commands.tex}`,
  per-author comment macros `\aditya{}` / `\tongle{}` gated by `\ifcomments`.
  Defines `\DensityFB`. `content/experiments.tex` and `content/related-work.tex`
  are currently commented OUT of main.tex.
- `content/` — abstract, introduction, prelims, method, experiments, related-work,
  and `content/appendix/{theory,implementation}.tex`.
- `table.tex` — STANDALONE (`\documentclass[10pt]{article}`, its own `\begin{document}`),
  not `\input` by main.tex. The headline table: rows = datasets, columns =
  BC / FB / GCIQL / Density-FB.
- `table_privcoord.tex` — STANDALONE, added 2026-09-02: the privileged-object-coordinate
  matrix (8 methods x {state, object coords} x {cube-single, cube-double}).
  See [[density-fb-privileged-coords-20260902]].
- `figures/`, `bib/`, `notes/`.

Build with tectonic ([[latex-engine-tectonic]]); each standalone table compiles on
its own (`tectonic table_privcoord.tex`).
