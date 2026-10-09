---
name: primary-paper-dir-multistep-fb
description: As of 2026-08-06 the PRIMARY paper directory is PAPER/Multistep-FB/ (ICLR-2027 style) — NOT PAPER/rlbrew/ as CLAUDE.md still says; it holds main.tex (multi-step measure TD) + delta-measure.tex (spec 17 variants A/B)
metadata: 
  node_type: memory
  type: project
  originSessionId: 576b53a0-d404-4a89-9380-6fece089b102
  modified: 2026-08-06T11:04:42.892Z
---

User instruction, 2026-08-06: **`PAPER/Multistep-FB/` is the primary paper
directory going forward.** New theory/results notes for the successor-measure
program go there, not into `PAPER/rescuing-fb/` (which stays as the ledger +
specs archive) and not `PAPER/rlbrew/` (what `CLAUDE.md` still names as the
current draft — that line is stale; CLAUDE.md is gitignored and user-managed,
so don't edit it unasked, but don't trust it on this point either).

Directory conventions (match these when adding a document):
- `\documentclass{article}` + `\usepackage{iclr2027_conference,times}` +
  `\input{math_commands.tex}` + `\iclrfinalcopy`; bibliography is
  `\bibliography{references}` / `\bibliographystyle{iclr2027_conference}`.
- `references.bib` was EMPTY until 2026-08-06 (main.tex cites nothing); I
  seeded it with 7 entries copied from `PAPER/rescuing-fb/references.bib`
  (park2024ogbench, dayan1993improving, blier2021learning, touati2021learning,
  touati2023does, kostrikov2021offline, park2025flow). Add there, don't fork.
- Build with `tectonic <file>.tex` from inside the directory (no system
  texlive — see [[latex-engine-tectonic]]).

Contents:
- `specs/` — the spec-17 family LIVES HERE now (moved 2026-08-06):
  `17-delta-measure.md`, `17a-delta-measure-impl.md`,
  `17b-delta-partition-impl.md`. Commit `fb4e614` repointed every provenance
  comment in configs/scripts/tests to `PAPER/Multistep-FB/specs/17...`.
  Specs 15 and 16 are still under `PAPER/rescuing-fb/specs/` (not moved).
- `main.tex` — multi-step successor-measure TD (spec 16): k-step Bellman
  operator, off-policy bias gated by the dwell-invariance lemma, horizon choice.
- `delta-measure.tex` — spec 17 variants: A (\textsc{Delta}, temporal
  differencing, identity J=(1-γ)M−h(x0) + the zero-flux relocation proposition)
  and B (\textsc{Part}, fixed-partition simplex measure), plus the first
  cube-single wave. Ported here 2026-08-06 from `PAPER/rescuing-fb/`; the
  rescuing-fb copy was DELETED so there is one source of truth.

The two notes are deliberately framed as rival explanations of the same
one-step collapse (starved bootstrap vs level-dominated regression), designed
to be run head-to-head in identical slots.

Related: [[spec17-delta-measure-20260806]], [[spec16-multistep-measure-20260806]],
[[latex-engine-tectonic]].
