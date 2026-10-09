# PAPER/ICLR — the ICLR paper home (moved here 2026-09-03, user decision)

Canonical home for all ICLR-relevant content, IN THIS SEAM WORKTREE because
the latest results live here. Seeded 2026-09-03 from the main worktree's
`Factored-FB/PAPER/ICLR` (which the cube session had been using — if that
session keeps writing there, re-sync `table_privcoord.tex` before quoting).

Tables:
- `table_privcoord.tex` — privileged-coordinates table (cube campaign;
  09-03 copy).
- `table_main_1m.tex` — 1M-only tab:main (BC/GCIQL/biGCIQL/Density-FB,
  95% CI). WRITE SIDE = `../decoding-bilinear/table.tex`; re-copy after
  regeneration, never edit here.
- `table_benchmarks_autogen.tex` — ManiSkill/Metaworld table; write side =
  `../../results/table.tex` via `scripts/make_results_table.py`.
- `table.tex` — 08-20 table carried over from the main worktree.

Build: tectonic (no system texlive).
