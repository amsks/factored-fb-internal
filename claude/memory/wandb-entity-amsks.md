---
name: wandb-entity-amsks
description: "All wandb runs in this repo MUST log to entity amsks, never the al-laq-fb team"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 52c7de9a-dc0d-44e3-9c0a-6711de243819
---

Every wandb run in Factored-FB must go to entity **`amsks`** (the user's personal
workspace), NEVER **`al-laq-fb`** (the team). The user is logged in as `amsks (al-laq-fb)`,
so `wandb_entity: null` / unset makes wandb default to the **team al-laq-fb** — runs then
don't appear in the amsks workspace where everything else lives.

**Why:** the user has repeatedly required amsks; al-laq-fb runs are effectively "lost"
and a recurring frustration.

**How to apply:** before launching ANY run that logs to wandb, confirm the entity
resolves to amsks. Levers:
- PyTorch FB (`train.py`): `wandb_entity: amsks` in `configs/train.yaml`; launchers
  pass `wandb_entity=amsks` (e.g. `run_fb_reweight.sh` WANDB_ENTITY=amsks). Fixed
  2026-05-25 (was `null` -> al-laq-fb).
- JAX/OGBench CRL: the shim (`run_gciql.build_child_env`) sets GCIQL_WANDB_ENTITY=amsks
  (project factored-fb) — already correct.
After launch, grep the run log for `wandb.ai/amsks/` to verify. See [[fb-coverage-balanced]].
