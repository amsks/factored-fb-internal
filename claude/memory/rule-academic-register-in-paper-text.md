---
name: rule-academic-register-in-paper-text
description: "USER RULE (2026-09-12, emphatic): paper prose must be written in a plain academic register — no metaphors or colloquial images (\"budget\", \"cheap to get wrong\", \"asks the critic\", \"lands where the reward is\"), no anthropomorphizing the model; and when the user gives a paragraph structure, RETAIN it and edit within it (each change followed by why it addresses the failure), do not restructure."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: be24e170-3bbf-4591-adc3-bb535ce89254
  modified: 2026-09-12T15:59:37.763Z
---

**Rule.** In any text destined for the paper: standard academic phrasing only. Banned: "budget", "cheap", "asks", "wants", "lands", "share out", and similar figurative or conversational wording. Say "training signal", "weight", "objective", "predicted distribution". When the user supplies a paragraph and asks for an update, keep their sentence order and structure; insert the explanatory clause after each change; do not rewrite from scratch.

**Why:** the user rejected two rewrites on 2026-09-12: one for restructuring the paragraph, one for "budget"-style wording, and had already asked for academic writing earlier.

**How to apply:** draft, then scan for figurative nouns/verbs and replace with the technical term. Related: [[rule-no-seed-count-caveats]], [[simple-means-plain-rl-language]].
