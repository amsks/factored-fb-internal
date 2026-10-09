---
name: latex-engine-tectonic
description: "How PAPER/*.tex PDFs are built on this box — tectonic, not a system texlive"
metadata: 
  node_type: memory
  type: reference
  originSessionId: 00be5f9a-855e-4726-bc21-154ca4e4bf29
---

PAPER PDFs are compiled with **tectonic** (v0.16.9), NOT a system TeX install —
there is no `pdflatex`/`latexmk`/texlive on this box. tectonic downloads `.sty`
files and fonts on demand (needs network), caching to `~/.cache/tectonic`.

Binary installed at **`~/.local/bin/tectonic`** (copied from `/tmp/tectonic`).
Build a paper with: `cd PAPER/<name> && ~/.local/bin/tectonic <name>.tex`
(single command does the multi-pass; writes `<name>.pdf`).

If the binary is ever gone, copies have historically sat at `/tmp/tectonic` and
in old session scratchpads. `pdftoppm`/poppler is also absent, so PDFs can't be
rasterized to images on this box for visual inspection.
