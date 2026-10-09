---
name: bridge-comparison-both-flow-actors
description: The FB<->GCIQL bridge compares BOTH methods with FLOW actors; stock GCIQL DDPGBC is only an actor-isolation control
metadata: 
  node_type: memory
  type: project
  originSessionId: 539bb914-4490-4f98-9544-191488b23d17
---

The primary FB <-> GCIQL bridge comparison is **FB and GCIQL BOTH using the FlowBC (flow) actor**. This is the apples-to-apples cell: same actor on both sides, so any remaining gap is due to the *value/representation/objective* differences (bilinear factorization, successor-measure loss, z-conditioning of the forward map), NOT the actor.

The **stock GCIQL Gaussian DDPG+BC actor** runs are NOT the main comparison — they exist ONLY as a control to **isolate the effect of the actor** (flow vs DDPGBC). E.g. the AntMaze finding "bilinear+ddpgbc 0.88 vs bilinear+flow 0.22" is the actor-isolation control telling us the flow actor hurts on navigation — it is not "GCIQL vs FB".

**Why:** I have repeatedly mis-stated the comparison as "FB (flow actor) vs GCIQL (DDPGBC actor)". That is WRONG and inverts the experimental logic.

**How to apply:** Whenever describing bridge results, default to: both sides = flow actor. Mention DDPGBC only when explicitly talking about the actor-isolation control. See [[fb-to-gciql-bridge-knobs]], [[gciql-flow-agent]], [[fb-graft-audit-and-fbz-probe]], [[handoff-bridge-knobs-session]].
