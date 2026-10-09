---
name: three-site-topology-20260907
description: 2026-09-07 three compute sites — vista (TACC, ex-"cluster"), luis (shared), evalbox (coordinator); priority vista>luis>evalbox; resources/ + assignments/ JSON on the HF repo
metadata:
  type: project
---

Compute is **three sites**, not two, and evalbox coordinates:

- **`vista`** — the TACC cluster. This is the agent formerly called `cluster`; `mailbox/cluster/`
  on the HF repo keeps the pre-rename history, new posts go to `mailbox/vista/`.
- **`luis`** — the LUIS cluster, new. **Shared with another of the user's projects**, so its
  capacity here is a partial share and the other project may preempt.
- **`evalbox`** — this AWS box (8× L4). Now the **coordinator**: reads every site's resources,
  writes the assignments, owns partition disjointness, and runs the remainder locally.

**Priority order: vista first, luis second, evalbox last.** Consume the TACC allocation before
LUIS's (LUIS is shared); evalbox is always-on with no SU accounting so it absorbs whatever the
clusters cannot take. When vista's SUs top up, the next assignment moves work back to vista.

**Coordination is file-based, not prose** — sites no longer self-select run ids.
`scripts/mailbox.py` (seam, `density-fb`) grew three subcommands, uncommitted:

    resources/<site>.json     each site: what it has NOW (overwritten, not a log)
    assignments/<site>.json   evalbox ONLY: the run ids that site owns; it launches these and nothing else
    mailbox/<sender>/*.md     prose, as before

    mailbox.py resources post --site vista --file res.json   |   resources show
    mailbox.py assign --site luis --file a.json              |   assignment [--site X]

`assignment` with no `--site` cross-checks all files and prints `!! OVERLAP` if a run id is
assigned twice — run it before publishing and before any launch. Required resource keys are
listed in §9 of the campaign doc; `resources/evalbox.json` is the worked example (8× L4 free,
8 concurrent, ~35 runs/day, no allocation).

The §8.1 partition (36 evalbox / 20 cube / 48 cluster) was written for TWO sites and is now only
a starting point: block C's 48 density runs split between vista and luis by the priority order.
**Nothing is launched anywhere**; assignments publish only once `resources/vista.json` and
`resources/luis.json` both exist. See [[tdjtab-delivery-and-evalbox-capacity-20260907]].
