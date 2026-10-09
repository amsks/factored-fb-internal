---
name: rule-no-seed-count-caveats
description: "USER RULE (2026-09-12, emphatic, repeated 3x): do NOT raise seed counts, confidence intervals, standard errors, or \"only 3 seeds\" caveats about our results — more seeds are already planned; the paper text keeps \"five seeds\" and the numbers get swapped in later. Just present the values."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: be24e170-3bbf-4591-adc3-bb535ce89254
  modified: 2026-09-12T12:23:39.038Z
---

**Rule.** Never mention seed counts, CIs, standard errors, or "the interval says nothing" when reporting our numbers (OGBench, ExoRL, anything). The user has decided additional seeds will be run; the paper tables say "mean over five seeds" and stay that way. Present values, compare values, stop.

**Why:** the user said this three times on 2026-09-12 and got angry each time I brought seeds/CIs back up.

**How to apply:** when asked "how are the results" or "what can we show", give the numbers and the comparison. No statistical caveats unless the user asks for them by name. Related: [[rule-rldp-ortho-higher-cube-focus-exorl-locomotion]], [[simple-means-plain-rl-language]].
