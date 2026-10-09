---
name: tdjepa-published-ogbench-table-20260907
description: "TD-JEPA paper (arXiv 2510.00739) Table 1, PROPRIO (state) block, zero-shot success % (mean±se): cube-single HILP 74.2±3.5 / FB 49.6±3.8 / RLDP 19.8±2.4 / TD-JEPA 34.2 / BYOL-γ 79.4; cube-double HILP 20.0±2.7 / FB 2.6 / RLDP 3.8 / TD-JEPA 3.6; scene HILP 43.8 / FB 12.8 / RLDP 11.6 / TD-JEPA 38.4; antmaze-mn HILP 83.6 / FB 73.0 / RLDP 74.6. Appendix depth table: HILP_shallow/deep cube-single 84.0/88.0, cube-double 27.1/29.2. DMC proprio: FB walker 811.5, HILP 796.4, RLDP 723.9, TD-JEPA 785.2. Source: arxiv HTML table parsed 2026-09-07 (scratchpad tdjepa.html)."
metadata:
  type: reference
---

Full proprio row set (cube-single / cube-double / scene / puzzle):
Laplacian 15.11/2.00/7.80/2.80 · ICVF* 20.40/5.00/45.40/16.60 · HILP 74.20/20.00/43.80/2.80 ·
FB 49.60/2.60/12.80/4.80 · RLDP 19.80/3.80/11.60/14.60 · BYOL* 22.00/4.40/15.40/13.20 ·
BYOL-γ* 79.40/2.60/14.40/18.80 · TD-JEPA 34.20/3.60/38.44/15.60.
Antmaze proprio (mn/ln/ms/ls/me): HILP 83.6/52.6/50.6/12.2/2.0; FB 73.0/36.8/70.4/49.8/51.6;
RLDP 74.6/36.4/58.4/19.6/4.8; TD-JEPA 70.4/57.2/61.6/40.6/20.2.
Protocol (from the repo): 1M steps, reward-inference z from 10k relabelled samples, best cell
of a small HP sweep over 10 seeds (launch_*_ogbench.py). Our matched arms: HILP tdjc cube-single
.83 (≥ published .742), cube-double .08 (≪ .200); FB tdj cube-single .38 (< .496); RLDP
cube-single .23 @500k (≈ .198). Related: [[hilp-tdjc-cube-not-inert-20260905]].
