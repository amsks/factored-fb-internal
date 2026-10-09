---
name: rule-no-pushforward-formalization-20260831
description: "STANDING RULE (2026-08-31, user, emphatic): NEVER use the pushforward formalization (encode_#(P), ν_C = pushforward measures, etc.) — not in analyses, not in visualizations, not in the paper, not in docs. Absolute ban."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 74028a31-0c8e-407a-a040-e7761054a936
  modified: 2026-08-31T22:52:46.168Z
---

The user's words: "NEW RULE — we will NEVER use the pushforward
formalization. This is absolutely bullshit. We will never visualize it,
never mention it in the paper."

**Why:** Measure-theoretic dress-up ("the pushforward ν_C = encode_#(P_C)")
of what is plainly "the distribution of inferred z's for a class of reward
functions" is exactly the invented-jargon failure mode the user has banned
before ([[simple-means-plain-rl-language]]). The formalization adds no
content over the empirical statement and would leak into paper prose.

**How to apply:** Describe such analyses in plain RL words only: "sample
rewards from class C, infer z for each, compare where those z's live
against the z's the actor trained on." Ban the words "pushforward",
"ν_C", "encode_#" from every doc, figure caption, spec, and prompt to
subagents. If a subagent produces them, strip before anything ships. The
underlying empirical analysis (z-distribution vs training prior) is NOT
banned — the framing is; confirm scope with the user if that ever seems
in doubt.
