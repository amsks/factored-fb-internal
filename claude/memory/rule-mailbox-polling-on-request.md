---
name: rule-mailbox-polling-on-request
description: "User rule (2026-09-07): do NOT poll the cluster mailbox on a schedule. Check mailbox/status (scripts/mailbox.py on HF repo amsks/factored-fb-checkpoints) only when the user asks, or when they say a deployment/run campaign is in progress and to 'restart the message-check routine' — then recreate the 20-min CronCreate poll (13,33,53 * * * *)."
metadata:
  type: feedback
---

**Why:** the recurring 20-minute poll produced a stream of "nothing new" turns the user did not
want. **How to apply:** no CronCreate mailbox poll by default. On request, or on "we are in the
middle of deployment/runs, restart the message check", schedule it again with the same prompt as
before (read --sender cluster --last 3; status; summarize new items; answer spec-covered questions
on mailbox/evalbox). Related: [[partb-campaign-handoff-20260906]].
