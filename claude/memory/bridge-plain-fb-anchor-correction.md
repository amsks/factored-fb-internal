---
name: bridge-plain-fb-anchor-correction
description: "The fb-gciql-bridge doc's plain-FB anchor was 43 but is actually 48 @1M (wandb-confirmed); 43 is stale. Affects the 43->93 narrative everywhere."
metadata: 
  node_type: memory
  type: project
  originSessionId: 29d24434-2cf7-47b2-89cc-f7b4407c6f44
---

**The `PAPER/fb-gciql-bridge` plain-FB anchor was WRONG.** The doc reported plain FB
@1M = **43±6** and built the whole narrative on "43→93". On 2026-06-29 a clean
re-extraction (to build the 3-seed/10-seed split) could not reproduce 43 from ANY source:
- local `fb_plain_ext_20260611` @1M = **48.4**, @1.5M = 48.0 (so NOT post-peak decay)
- original `fb_state_plain` runs' own @1M eval = 52.0
- **wandb `amsks/factored-fb` group `fb_state_plain`** (authoritative, the source the doc
  cites): history @ `_step=1000000` = **48.4** (per-seed exactly matches fb_plain_ext:
  .48/.50/.46/.48/.36/.56/.54/.40/.56/.50), summary(=1.5M) = 48.0.

Every OTHER bridge row reproduced the doc's 10-seed number exactly (GCIQL d50 92.6, d128
93.9, d5 14.3, tau05 88.9, tau095 90.6, singleq 90.2, ddpgbc 56.6, d512 94.5, G4 93.4,
CRL-flow 25.8, CRL-gauss 16.8, FB zHER 32.0, FB-IQL q-loss 38.8, FB-IQL critic 23.6) —
plain FB was the lone outlier. ⇒ **43 was stale; correct anchor = 48±5 (n=10), 3-seed
{0,1,2} = 48.0±5.0.** Corrected to 48 throughout the tex (the "43→93" jump becomes "48→93").

**Extraction pipeline (validated):** `scratchpad/extract_seeds.py` reads GCIQL
`eval.csv` (`evaluation/overall_success` at step 1e6) + FB stdout
(`/dev/shm/bridge1m_logs/fb_state_*__s*__{resume,fresh}.log` `success/overall=`; plain FB
`fb_plain_ext_*/sN.log` `eval/reward/eval/success=` at `[step  1000000]`). Seeds 0,1,2 =
preliminary (dates: GCIQL originals Jun 9-17, fresh seeds 3-9 Jun 22-23). Stock GCIQL cube
(MLP+Gaussian, 67.4) has NO local per-seed data — 12-seed reuse from elsewhere; pull from
wandb if its 3-seed subset is needed. Relates: [[handoff-iwr-program]].
