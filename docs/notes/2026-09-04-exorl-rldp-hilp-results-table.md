# RLDP / HILP results — one table (assembled 2026-09-04)

Cube numbers: `PAPER/ICLR/table_privcoord.tex` (already a table). ExoRL numbers were
scattered across `exp/paper_eval*.jsonl`, `docs/notes/2026-08-31-*`, and memory; this
note aggregates them (mean ± sd over seeds of the per-task mean return). Source file
named per block so a number can be traced.

## ExoRL walker, td_jepa protocol (500k, 10 inference-k samples, 10 episodes, 3 seeds)
Source: `exp/paper_eval_tdj_final.jsonl`; td_jepa reference from their own runs at 500k
(`docs/notes/2026-08-31-tdjepa-replication-screen.md`).

| arm | stand | walk | run | spin | mean(4) | vs td_jepa |
|---|---|---|---|---|---|---|
| td_jepa RLDP (theirs) | 763 ± 52 | 509 ± 88 | 226 ± 45 | 917 ± 56 | **604** | — |
| ours `rldp_tdj` | 795 ± 39 | 595 ± 108 | 293 ± 61 | 613 ± 294 | **574** | 0.95× |
| ours `rldp_density_tdj` | 444 ± 117 | 357 ± 62 | 116 ± 36 | 532 ± 278 | **362** | 0.60× |
| td_jepa HILP (theirs) | 864 ± 65 | 873 ± 52 | 301 ± 47 | 981 ± 2 | **755** | — |
| ours `hilp_tdj` | 879 ± 33 | 884 ± 16 | 338 ± 14 | 972 ± 15 | **768** | 1.02× |
| ours `hilp_density_tdj` ⚠ | 892 ± 2 | 728 ± 10 | 317 ± 55 | 982 ± 3 | **730** | 0.97× |

⚠ `hilp_density_tdj` on disk trained at discount 0.96 and a 1M budget (config-delivery bug,
`docs/notes/2026-09-03-delta-audit.md` F2); the 500k checkpoint is scored here but the pair
is not clean until relaunched. Both `rldp_tdj` arms ran actor lr 3e-4 instead of 1e-4 (F1);
internally matched, but the replication is 3× hot on the actor. `exp/paper_eval_tdj.jsonl`
holds an earlier, discarded `hilp_tdj` (591.6) — do not quote.

## ExoRL, official-RLDP protocol (5M steps, 4 seeds, 50 episodes walker / 10 cheetah)
Source: `exp/paper_eval_r2_walker.jsonl`, `exp/paper_eval_ortho.jsonl`, `exp/paper_eval_r2.jsonl`.

| arm | walker stand | walk | run | flip | mean(4) | vs RLDP Table 1 |
|---|---|---|---|---|---|---|
| RLDP Table 1 (published) | 878 ± 45 | 791 ± 68 | 325 ± 55 | 492 ± 37 | **621** | — |
| `rldp_official` run 2 (ortho inert, as shipped) | 365 ± 133 | 107 ± 76 | 67 ± 28 | 125 ± 78 | **166** | 0.27× |
| `rldp_orthoa` (ortho live, coef 1.0) | 828 ± 92 | 490 ± 176 | 225 ± 37 | 347 ± 96 | **472** | 0.76× |
| `rldp_orthob` (ortho live, coef 1e-4) | 148 ± 18 | 41 ± 29 | 25 ± 3 | 28 ± 4 | **60** | 0.10× |

Cheetah, `rldp_official` run 2: run 14, run_backward 3, walk 65, walk_backward 18
(published 236 / 322 / 895 / 985) — not reproduced; `orthoa` never run on cheetah/quadruped.

## Not run
- HILP official-protocol arms (Wave B, `exp/exorl-hilp/` is empty).
- Density-RLDP on the official substrate (Wave C, gated on A).
- Density-FB on ExoRL: walker replicates Touati (756 ± 46 vs 760 ± 19, compendium
  2026-08-25); cheetah does not (112 vs 247).

## Reading
- Same-protocol replication is fine for both methods (`rldp_tdj` 0.95×, `hilp_tdj` 1.02×
  of td_jepa's own 500k numbers). The official-RLDP shortfall is the dead regularizer,
  partly recovered by running it live (0.27× → 0.76×).
- The density retrofit does not help on locomotion and hurts RLDP (574 → 362). Consistent
  with the perspective note: nothing dwells on walker, so normalizing the target has
  nothing to fix, and a frozen representation plus an uncentered index is what the
  retrofit adds.
