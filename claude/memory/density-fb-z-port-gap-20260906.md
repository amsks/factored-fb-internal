---
name: density-fb-z-port-gap-20260906
description: "2026-09-06 ROOT CAUSE of the z-variant collapse: the 0.82 cube-single z number came from the BRIDGE stack (~/exp-spec/exp_spec30_p1/.../ra2_cs_centdual, gciql_flow.py) which trained with measure_density_z_actor_reward='kernel' (actor q-term integrates the encoded goal's exact kernel reward; random-sphere-z rows masked out) and measure_density_hard=False (soft target). The seam port density_fb_z uses the LATENT actor reward B̃ᵀz/√d for all rows ('kernel' cell deliberately not ported, density_fb.py:940) and hard=True. Seam z arm: 0.43 own-route @500k (peak .51), 0.22-0.28 @1M (3 seeds, cluster Tier 1), unstable curve; bridge: monotone .60→.82. Everything else (encoder knobs, batch, lr, arch, goal sampling) matches; only residual: actor lr 1e-4 (seam) vs 3e-4 (bridge agent.lr)."
metadata:
  type: project
---

Evidence: `exp/cube-single-play-v0_density_fb_z_flowbc_seed_0/eval.csv` (.18/.26/.34/.51/.32/.43
over 250k-500k); Tier-1 cluster runs `exp/partb-tier1/*density_fb_z*` eval.csv at 1M .27/.22/.28;
bridge `flags.json` + `eval.csv` (.60@50k → .82@500k); bridge code
`tools/wandb_mode_shim/gciql_flow.py:1509-1527, 1750-1770` (kernel reward + q_row_mask=z_hindsight).

**Consequences.** (1) The paper's method text (r_z = B̄ᵀz/√d latent reward) describes the variant
that does NOT reproduce 0.82; the working variant trains the actor on the encoded goal's kernel
reward. Must be stated. (2) The td_jepa relabel-route 0.28 was measured on a checkpoint already
at 0.43 — the encoder/pool question is SECONDARY to the port gap. (3) Fix path: port
`measure_density_z_actor_reward='kernel'` + random-row q mask into seam density_fb.py, then a
2x2 {actor_reward latent/kernel} x {hard/soft} at 500k, 1 seed each, on the local box; gate =
seam z arm reaches the bridge's ~.8 own-route. Then re-run the relabel route on it.
Related: [[partb-campaign-handoff-20260906]], [[spec30-wave1-finals-20260819]], [[dfb-port-comparison-20260819]].

**2026-09-06 FIX IMPLEMENTED + SCREEN LAUNCHED (commit da74bfd, density-fb).** New knob
`measure_density_z_actor_reward` in density_fb.py: 'latent' (old: fast target tower,
uncentered) | 'latent_centered' (slow index tower B_xi minus mu_hat -- the encoder's own
basis). density_fb_z.yaml now defaults to latent_centered; code default stays 'latent' so the
cluster Tier-1 runs' recorded configs mean the old path. Tests: 5 new, all green (the one red
test in that file, update_stats noop for hilp, is the other track's WIP). Screen: tmux `zfix`,
GPUs 0-4, 500k, cube_single: lc_hard s0/1/2, lc_soft s0, l_soft s0 -> exp/zfix/<arm>/...;
control = existing latent+hard s0 (.43). Gate: own-route eval ≥ ~.8 at 500k.

**Soft arms KILLED by user (focus on hard first).** Only lc_hard s0/1/2 running (GPUs 0-2). To run soft later: `bash scripts/launch_zfix.sh` with ARMS trimmed to lc_soft/l_soft, or the equivalent two impls.main lines with measure_density_hard=false and --save_dir exp/zfix/<arm>.

**FIX CONFIRMED (mid-run, 2026-09-06, 200k-250k of 500k).** lc_hard (latent_centered + hard),
cube-single, own-route eval: s0 .57/.71/.85/.74, s1 .73/.69/.70/.80, s2 .76/.76/.76/.76 at
50k/100k/150k/200k; td_jepa relabel route: s0 .76/.76/.86/.82, s1 .68/.76/.80/.94, s2
.88/.78/.84/.78. Control (latent+hard s0) at the same steps: .69/.38/.20/.27. ⇒ the actor's
reward basis WAS the whole gap; under the td_jepa route the fixed z variant scores ≥ its own
route (~.8-.9), i.e. the "pool-composition sensitivity" was the broken checkpoint, not the
encoder. Final 500k numbers pending; quote those, not these.

**State+index implemented** (commit after da74bfd): pool stores full obs rows under backward_input=state, kernel slices at use, infer_cond/map_eval_cond/update_stats route B's input by mode; guard removed; tests green; CPU smoke on real data OK. Launched lc_hard_state s0/1/2 (density_fb_z + backward_input=state, hard, latent_centered) on GPUs 3-5, exp/zfix/lc_hard_state/.

**FINAL 500k (2026-09-07), cube-single, 3 seeds:** lc_hard (obj coords) own .74±.05 / td_jepa .76±.03; lc_hard_state (full state) own .78±.09 / td_jepa .92±.07. Cluster tier-1 at 1M (Vista): z obj own .60/.79/.86, td_jepa .68/.92/.86; z state own .80/.90/.83, td_jepa .92/.92/.92; goal obj .91; goal state .90-.92. cube-double: z .2-.4 vs goal-obj .69-.77; goal-state collapses .11-.15.
