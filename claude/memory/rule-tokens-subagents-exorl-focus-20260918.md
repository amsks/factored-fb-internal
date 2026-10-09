---
name: rule-tokens-subagents-exorl-focus-20260918
description: USER RULE 2026-09-18 (emphatic): preserve main-context tokens, delegate to subagents as much as possible; after the Luis mailbox posts, spend tokens ONLY on the ExoRL job = better Density-FB results at matched hyperparameters with no bugs in the method implementation, so the paper table can be completed
metadata:
  type: feedback
---

User instruction, 2026-09-18, all caps: "PRESERVE YOUR TOKENS AND USE SUBAGENTS AS MUCH AS POSSIBLE. AFTER THE
MESSAGE IS POSTED, ONLY USE TOKENS FOR THE EXORL JOB. OUR GOAL IS TO GET BETTER RESULTS WITH MATCHED HPS AND
WITHOUT ANY BUGS IN OUR METHOD IMPL, SO THAT I CAN COMPLETE THE TABLE IN THE PAPER."

**Why:** the main session context is the scarce resource; the user left the session unattended.
**How to apply:** fan every analysis/implementation/scoring task out to subagents with precise briefs; keep the
main thread to relaying results and decisions. Once the rebalance + baseline-seed handoffs are posted to the HF
mailbox for luis, the only work is ExoRL: matched-HP Density-FB runs (see [[exorl-cheetah-maze-disc-20260917]]),
implementation audits for bugs, scoring, and updating tab:manipulation's ExoRL row.
