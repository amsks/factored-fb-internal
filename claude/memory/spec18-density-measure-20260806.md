---
name: spec18-density-measure-20260806
description: "2026-08-06 spec18 IMPLEMENTED+COMMITTED (74e861d) and cube-single wave COLLECTED: measure_density (continuous normalized successor density, two-tower F·B/τ over batch-resampled queries) BEATS 17b partition — density .896/.928/.964 vs part .884, and region-family zero-shot .26-.28 (hard .47) vs part's .04-.18; G1+G2 pass, G3 fails at P=4096 but passes at P>=16384; K-sweep NOT flat (K64 .556)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 576b53a0-d404-4a89-9380-6fece089b102
  modified: 2026-08-06T16:46:34.166Z
---

Bridge track. Spec `PAPER/Multistep-FB/specs/18-density-measure.md`; write-up sibling
of [[spec17-delta-measure-20260806]]. Code `74e861d`, 90 tests green.

**What it is.** 17b's fixed 256 prototypes exist only because `delta_head` was an
MLP with K output logits — they are NOT on the list of things 17b's win is
attributed to. Spec 18 removes the fixed support: `MLP([s,a,g])→K logits` becomes
a two-tower score `f(s,a,g,u)=F(s,a,g)ᵀB(u)/τ` over queries **resampled from ρ_x
every update** (round 1 = the batch's own next-state object coords), softmax-
normalized. CE target keeps 17b's exact structure `sg[(1−γ)k + γw̄⁺]`, with `k`
literally `_delta_phi` at `protos := queries`. Query count becomes a Monte-Carlo
sample size, not a spatial resolution. No .npy, so 17b's checkpoint↔file coupling
disappears structurally.

**Results (cube-single, 500k, 3 seeds density + 1 hard + K-sweep):**
- goal success: density **.896 / .928 / .964**; hard (k=I) .876; partK64 .556,
  K512 .872, K1024 .868; (K256 = .884 from spec 17). Anchor fp-noortho = .967.
- zero-shot 25-ep: density dirac .67/.75/.89, dense .64/.67/.86, **region .26-.28**;
  hard dirac .86 dense .86 **region .47**; partition region only .04-.18.
  ⇒ spec §7's directional read CONFIRMED — removing the fixed support pays off
  exactly on extended-region rewards, where a finite prototype set must
  interpolate. On point-goal families density ≈ partition, not better.
- **K-sweep is NOT flat**: K64 .556 ≪ K256 .884 ≈ K512 .872 ≈ K1024 .868 ⇒ the
  prototype count IS load-bearing below ~256 and saturates above. The answer to
  "isn't the discretization arbitrary?" is "arbitrary above a threshold".

**Gates.** G1 PASS (target ≈ part's .884). G2 PASS (ess_median 87-91 density,
5.14 hard, gate ≥4; k_self .057-.059 = the healthy baseline, NOT coverage
failure). G3 **fails at the script default P=4096** (max per-task drift .120)
but **passes at P≥16384** (drift .000, 16384→32768) ⇒ the readout converges;
4096 is under-sampled; move the eval default, don't touch the method. G4
(Spearman) STILL NOT RUN for either part K256 or density.

**⚠ Method lessons that cost me real time here:**
- **Never compare mid-run evals.** At 450k the hard ablation showed .984 and I
  reported it as best ("smoothing is not load-bearing"); at 500k it is .876.
  Adjacent evals swing .05-.12 on 250 episodes. Only multi-seed, final-step, or
  last-k-averaged numbers are quotable. density > partition survives (3 seeds,
  both K≥256); hard > density does NOT (1 seed).
- The zero-shot eval is **fully deterministic** (bit-identical repeats) — that is
  what let me attribute the P=4096 drift 100% to the readout with zero rollout
  contamination. Use a repeat-run as the noise floor before blaming a gate.
- Pool CONTENT matters slightly at fixed size (relabel-size 50000 vs 60000 moved
  t1 1.00→.96) — pin the pool seed AND relabel-size when quoting.
- `eval_density_zeroshot.py` **OOM-kills at P=65536** on a 728G box (the SECONDARY
  ridge readout scales badly). Practical ceiling ~32768.
- Pre-launch review caught: `density_ess` logged a MEAN while gate G2 pre-registers
  a MEDIAN (a one-head/bimodal collapse would have read healthy) — same class as
  spec 17's msq-computed-and-discarded. Instrument the gate stat, not a proxy.

**Infra:** `d17_cs_part_s1` died TWICE with `CUDA_ERROR_STREAM_CAPTURE_INVALIDATED`
from a `device_get` at logging time (70k on GPU4, 230k on GPU0 ⇒ not a bad card).
Fix baked into both launchers: `XLA_FLAGS=--xla_gpu_enable_command_buffer=`.
Also: never write a pgrep-based liveness monitor whose pattern appears in its own
command line — mine self-matched and hid the first crash for ~6.5h.

**Still owed:** G4 Spearman (both methods); a BC-only control (so far NO cube-single
number here is attributed to the measure rather than the bc_coeff=3.0 cloning term);
my own review pass on `eval_density_zeroshot.py` + `launch_spec18_cs.sh`; results_map
entries + CSV refresh for the `d18_*` groups; cube-double/scene (gated on G1, now passed).

Related: [[spec17-delta-measure-20260806]], [[spec16-multistep-measure-20260806]],
[[primary-paper-dir-multistep-fb]], [[results-csv-pipeline]].
