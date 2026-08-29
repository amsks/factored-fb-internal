"""results_map.py — the curated wandb-group -> (csv, env, agent, actor, variant) mapping
behind the two living results CSVs (scripts/compile_results_csvs.py).

CSV1 `summary_agents`  : canonical agents x envs x actors (FB, GCIQL, GCIVL, HIQL + the
                         other baseline agents), uniform 500k protocol.
CSV2 `bridge_grafts`   : every graft/ablation in the FB<->GCIQL bridge program (and FB/GCIQL
                         self-ablations), same axes + graft-descriptor columns.
EXCLUDED               : smokes, HP sweeps superseded by a tuned row, probe stubs — each with
                         a reason so the backward audit can account for 100% of groups.

Actor vocabulary (one label per extraction mechanism):
  ddpgbc   : stock GCIQL gaussian DDPG+BC
  fql_flow : FQL-style flow actor w/ Q guidance ("Flow Q-Learning"; jax *-Flow-* cells)
  flowbc   : torch FB-family FlowBC actor (BC flow + Q-guided z; fb/psm/rldp/crl-torch)
  awr      : native AWR extraction (GCIVL/HIQL native; HIQL = hierarchical AWR)
  awr_flow : IQL AWR extraction on a goal-conditioned flow-BC policy (awr_flow actor_kind)
  planner  : TD-MPC2 MPPI planner
Env vocabulary: cube | scene | antmaze | cube-pixel.
Update discipline: EVERY new wandb group must land in exactly one bucket; UNMAPPED groups
make compile_results_csvs.py fail loudly. Add an entry here per new campaign.
"""

# ---- explicit per-group entries -------------------------------------------------------
# fields: csv ('summary'|'bridge'|'exclude'), env, agent, actor, variant (short label),
#         graft (bridge only: direction+what), notes. Group name is the dict key.

E = 'exclude'

MAP = {
    # ======================= CSV1: canonical agents =======================
    # --- FB (torch; metric success/overall) ---
    'FB-State-Flow':        dict(csv='summary', env='cube', agent='FB', actor='flowbc', variant='plain'),
    'FB-State-Flow-L256':   dict(csv='summary', env='cube', agent='FB', actor='flowbc', variant='L256',
                                 notes='misnamed-group gotcha resolved: this IS L256; FB-State-Flow is plain'),
    'fb_state_plain':       dict(csv='summary', env='cube', agent='FB', actor='flowbc', variant='plain-1.5M',
                                 notes='same cell as FB-State-Flow, later 1.5M campaign; @500k point extracted'),
    'fb_state_ddpgbc':      dict(csv='summary', env='cube', agent='FB', actor='ddpgbc', variant='plain'),
    'fb_antmaze_medium_500k': dict(csv='summary', env='antmaze', agent='FB', actor='flowbc', variant='plain'),
    'fb_scene_500k':        dict(csv='summary', env='scene', agent='FB', actor='flowbc', variant='plain'),
    # --- 2026-08-05: the "---" cells of the ICLR draft's tab:main (launch_missing_baselines.sh).
    # "Tuned FB"  = domain-default ortho_coef 1.0, matching FB-State-Flow / fb_scene_500k.
    # "FB (ref.)" = ortho_coef 1000, matching exp_spec10/torchfb/fb_stock_s0 (cube 0.532).
    'mb_tunedfb_cdbl_s42': dict(csv='summary', env='cube-double', agent='FB', actor='flowbc', variant='plain'),
    'mb_tunedfb_cdbl_s43': dict(csv='summary', env='cube-double', agent='FB', actor='flowbc', variant='plain'),
    'mb_fbref_cdbl_s42':   dict(csv='summary', env='cube-double', agent='FB', actor='flowbc', variant='ortho1000-ref'),
    'mb_fbref_cdbl_s43':   dict(csv='summary', env='cube-double', agent='FB', actor='flowbc', variant='ortho1000-ref'),
    'mb_fbref_scn_s42':    dict(csv='summary', env='scene', agent='FB', actor='flowbc', variant='ortho1000-ref'),
    'mb_fbref_scn_s43':    dict(csv='summary', env='scene', agent='FB', actor='flowbc', variant='ortho1000-ref'),
    # tab:main plain-FB DDPG+BC cells (filled 2026-08-13/14). Confirmed via .hydra/config.yaml:
    # agent: fb (Gaussian+BC actor, NOT fb_flowbc) -- exp_fbddpg/fbd_{cd,sc}_s{0,1,2}.
    'fbd_fbd_cd': dict(csv='summary', env='cube-double', agent='FB', actor='ddpgbc', variant='plain',
                        notes='tab:main plain-FB DDPG+BC, cube-double (.013 +- .029)'),
    'fbd_fbd_sc': dict(csv='summary', env='scene', agent='FB', actor='ddpgbc', variant='plain',
                        notes='tab:main plain-FB DDPG+BC, scene (.047 +- .057)'),
    # "FB loss on regions" on the multi-object envs: stock FB's loss at MATCHED
    # conjunctive support (hitdirac hit_spec, added 2026-08-05). cube-single = 0.284.
    'fr_fbregion_cdbl_s42': dict(csv='summary', env='cube-double', agent='FB', actor='flowbc', variant='fbloss-on-regions'),
    'fr_fbregion_cdbl_s43': dict(csv='summary', env='cube-double', agent='FB', actor='flowbc', variant='fbloss-on-regions'),
    'fr_fbregion_scn_s42':  dict(csv='summary', env='scene', agent='FB', actor='flowbc', variant='fbloss-on-regions'),
    'fr_fbregion_scn_s43':  dict(csv='summary', env='scene', agent='FB', actor='flowbc', variant='fbloss-on-regions'),
    # 2026-08-05 method-reproducibility waves (CLAIMS-AUDIT.md). nf_* = region measure
    # + FBAgent native TD3+BC actor (no flow; actor-class control). tz_* = region measure
    # + flow actor + train_goal_mode=traj (grounded actor z's; the jax<->torch bridge).
    'nf_nf_cs_s42':   dict(csv='summary', env='cube', agent='RegionFB', actor='td3bc', variant='p1-noflow'),
    'nf_nf_cs_s43':   dict(csv='summary', env='cube', agent='RegionFB', actor='td3bc', variant='p1-noflow'),
    'nf_nf_cdbl_s42': dict(csv='summary', env='cube-double', agent='RegionFB', actor='td3bc', variant='p1-noflow'),
    'nf_nf_cdbl_s43': dict(csv='summary', env='cube-double', agent='RegionFB', actor='td3bc', variant='p1-noflow'),
    'nf_nf_scn_s42':  dict(csv='summary', env='scene', agent='RegionFB', actor='td3bc', variant='p1-noflow'),
    'nf_nf_scn_s43':  dict(csv='summary', env='scene', agent='RegionFB', actor='td3bc', variant='p1-noflow'),
    'tz_tz_cs_s42':   dict(csv='summary', env='cube', agent='RegionFB', actor='flowbc', variant='p1-trajz'),
    'tz_tz_cs_s43':   dict(csv='summary', env='cube', agent='RegionFB', actor='flowbc', variant='p1-trajz'),
    'tz_tz_cdbl_s42': dict(csv='summary', env='cube-double', agent='RegionFB', actor='flowbc', variant='p1-trajz'),
    'tz_tz_cdbl_s43': dict(csv='summary', env='cube-double', agent='RegionFB', actor='flowbc', variant='p1-trajz'),
    'tz_tz_scn_s42':  dict(csv='summary', env='scene', agent='RegionFB', actor='flowbc', variant='p1-trajz'),
    'tz_tz_scn_s43':  dict(csv='summary', env='scene', agent='RegionFB', actor='flowbc', variant='p1-trajz'),
    'FB-Pixel-DrQ-Flow':    dict(csv='summary', env='cube-pixel', agent='FB', actor='flowbc', variant='drq'),
    # --- GCIQL (jax; metric evaluation/overall_success) ---
    'GCIQL-State-DDPGBC':          dict(csv='summary', env='cube', agent='GCIQL', actor='ddpgbc', variant='mlp'),
    'GCIQL-State-Flow':            dict(csv='summary', env='cube', agent='GCIQL', actor='fql_flow', variant='mlp'),
    'GCIQL-State-Flow-Bilinear':   dict(csv='summary', env='cube', agent='GCIQL', actor='fql_flow', variant='bilinear-d512'),
    'GCIQL-State-Flow-Bilinear-d128': dict(csv='summary', env='cube', agent='GCIQL', actor='fql_flow', variant='bilinear-d128'),
    'GCIQL-State-Flow-Bilinear-d5':   dict(csv='summary', env='cube', agent='GCIQL', actor='fql_flow', variant='bilinear-d5'),
    'GCIQL-State-Flow-Bilinear-d28':  dict(csv='summary', env='cube', agent='GCIQL', actor='fql_flow', variant='bilinear-d28',
                                          notes='rank control for the identity-psi cell (d==cube obs dim)'),
    'GCIQL-State-Flow-Bilinear-d50':  dict(csv='summary', env='cube', agent='GCIQL', actor='fql_flow', variant='bilinear-d50'),
    'GCIQL-State-Flow-Bilinear-d50-ddpgbc': dict(csv='summary', env='cube', agent='GCIQL', actor='ddpgbc', variant='bilinear-d50'),
    'GCIQL-State-AWRFlow-Bilinear-d50': dict(csv='summary', env='cube', agent='GCIQL', actor='awr_flow', variant='bilinear-d50',
                                             notes='parallel SOTA session cell; may still be running'),
    'GCIQL-AntMaze-Medium-Stock-DDPGBC':   dict(csv='summary', env='antmaze', agent='GCIQL', actor='ddpgbc', variant='mlp'),
    'GCIQL-AntMaze-Medium-Bilinear-d50-DDPGBC': dict(csv='summary', env='antmaze', agent='GCIQL', actor='ddpgbc', variant='bilinear-d50',
                                                     notes='0.837 is the anchor (SOTA config comment 83.7); earlier 0.88 citations were imprecise'),
    'GCIQL-AntMaze-Medium-Flow-MLP':       dict(csv='summary', env='antmaze', agent='GCIQL', actor='fql_flow', variant='mlp'),
    'GCIQL-AntMaze-Medium-Flow-Bilinear-d50': dict(csv='summary', env='antmaze', agent='GCIQL', actor='fql_flow', variant='bilinear-d50'),
    'GCIQL-AntMaze-Medium-Flow-MLP-bc03':  dict(csv='summary', env='antmaze', agent='GCIQL', actor='fql_flow', variant='mlp-bc03',
                                                notes='td_jepa-correct bc_coeff=0.3 rerun (parallel session); may be running'),
    'GCIQL-AntMaze-Medium-Flow-Bilinear-d50-bc03': dict(csv='summary', env='antmaze', agent='GCIQL', actor='fql_flow', variant='bilinear-d50-bc03',
                                                        notes='may be running'),
    'GCIQL-AntMaze-Medium-AWRFlow-Bilinear-d50': dict(csv='summary', env='antmaze', agent='GCIQL', actor='awr_flow', variant='bilinear-d50',
                                                      notes='may be running'),
    'GCIQL-Scene-Stock-DDPGBC':     dict(csv='summary', env='scene', agent='GCIQL', actor='ddpgbc', variant='mlp'),
    'GCIQL-Scene-Bilinear-d50-DDPGBC': dict(csv='summary', env='scene', agent='GCIQL', actor='ddpgbc', variant='bilinear-d50'),
    'GCIQL-Scene-Flow-Bilinear-d50':   dict(csv='summary', env='scene', agent='GCIQL', actor='fql_flow', variant='bilinear-d50'),
    'GCIQL-Pixel-DrQ-DDPGBC':   dict(csv='summary', env='cube-pixel', agent='GCIQL', actor='ddpgbc', variant='drq'),
    'GCIQL-Pixel-DrQ-Flow':     dict(csv='summary', env='cube-pixel', agent='GCIQL', actor='fql_flow', variant='drq'),
    'GCIQL-Pixel-IMPALA-DDPGBC': dict(csv='summary', env='cube-pixel', agent='GCIQL', actor='ddpgbc', variant='impala'),
    # --- GCIVL ---
    'GCIVL-State-Stock':          dict(csv='summary', env='cube', agent='GCIVL', actor='awr', variant='mlp'),
    'GCIVL-State-Bilinear-d50':   dict(csv='summary', env='cube', agent='GCIVL', actor='awr', variant='bilinear-d50'),
    'GCIVL-State-Flow-Stock':     dict(csv='summary', env='cube', agent='GCIVL', actor='fql_flow', variant='mlp'),
    'GCIVL-State-Flow-Bilinear-d50': dict(csv='summary', env='cube', agent='GCIVL', actor='fql_flow', variant='bilinear-d50',
                                          notes='BIMODAL group (audit-20260707): all-seeds mean ~0.475, healthy-batch reading 0.792 — filter by per_seed'),
    'GCIVL-AntMaze-Medium-Flow-Bilinear-d50': dict(csv='summary', env='antmaze', agent='GCIVL', actor='fql_flow', variant='bilinear-d50',
                                                   notes='may be running'),
    'GCIVL-Pixel-DrQ-AWR':    dict(csv='summary', env='cube-pixel', agent='GCIVL', actor='awr', variant='drq'),
    'GCIVL-Pixel-IMPALA-AWR': dict(csv='summary', env='cube-pixel', agent='GCIVL', actor='awr', variant='impala'),
    # --- HIQL (native = hierarchical AWR). sg10 = OGBench-correct subgoal_steps; non-sg10 cube cells ran sg25 (mis-tuned) ---
    'HIQL-State-Stock':                dict(csv='summary', env='cube', agent='HIQL', actor='awr', variant='mlp-sg25-mistuned'),
    'HIQL-State-Bilinear-d50':         dict(csv='summary', env='cube', agent='HIQL', actor='awr', variant='bilinear-d50-sg25-mistuned'),
    'HIQL-State-Native-MLP-sg10':      dict(csv='summary', env='cube', agent='HIQL', actor='awr', variant='mlp-sg10'),
    'HIQL-State-Native-Bilinear-d50-sg10': dict(csv='summary', env='cube', agent='HIQL', actor='awr', variant='bilinear-d50-sg10'),
    'HIQL-State-Native-Bilinear-d50-schaul-sg10': dict(csv='summary', env='cube', agent='HIQL', actor='awr', variant='bilinear-schaul-sg10'),
    'HIQL-State-Native-Bilinear-d50-rep50-sg10':  dict(csv='summary', env='cube', agent='HIQL', actor='awr', variant='bilinear-rep50-sg10'),
    'HIQL-State-Native-HighBil-d50-sg10': dict(csv='summary', env='cube', agent='HIQL', actor='awr', variant='high-bilinear-sg10'),
    'HIQL-State-Native-BothBil-d50-sg10': dict(csv='summary', env='cube', agent='HIQL', actor='awr', variant='both-bilinear-sg10'),
    'HIQL-State-Flow-both-MLP-sg10':      dict(csv='summary', env='cube', agent='HIQL', actor='fql_flow', variant='mlp-sg10'),
    'HIQL-State-Flow-both-Bilinear-d50-sg10': dict(csv='summary', env='cube', agent='HIQL', actor='fql_flow', variant='bilinear-d50-sg10'),
    'HIQL-State-Flow-both-HighBil-d50-sg10':  dict(csv='summary', env='cube', agent='HIQL', actor='fql_flow', variant='high-bilinear-sg10'),
    'HIQL-State-Flow-both-BothBil-d50-sg10':  dict(csv='summary', env='cube', agent='HIQL', actor='fql_flow', variant='both-bilinear-sg10'),
    'HIQL-State-Flow-both-Bilinear-d50': dict(csv='summary', env='cube', agent='HIQL', actor='fql_flow', variant='bilinear-d50-sg25-mistuned'),
    'HIQL-State-Flow-high-Bilinear-d50': dict(csv='summary', env='cube', agent='HIQL', actor='fql_flow', variant='bilinear-high-only-sg25-mistuned'),
    'HIQL-State-Flow-low-Bilinear-d50':  dict(csv='summary', env='cube', agent='HIQL', actor='fql_flow', variant='bilinear-low-only-sg25-mistuned'),
    'HIQL-AntMaze-Medium-MLP':          dict(csv='summary', env='antmaze', agent='HIQL', actor='awr', variant='mlp'),
    'HIQL-AntMaze-Medium-Bilinear-d50': dict(csv='summary', env='antmaze', agent='HIQL', actor='awr', variant='bilinear-d50'),
    'HIQL-AntMaze-Medium-Flow-both-HighBil-d50': dict(csv='summary', env='antmaze', agent='HIQL', actor='fql_flow', variant='high-bilinear',
                                                      notes='may be running'),
    # --- other baseline agents (failure-analysis project; filter by agent) ---
    'CRL-State-Flow':    dict(csv='summary', env='cube', agent='CRL', actor='flowbc', variant='plain'),
    'crl_state_ddpgbc':  dict(csv='summary', env='cube', agent='CRL', actor='ddpgbc', variant='plain'),
    # 2026-08-05: CRL on the multi-object envs, same recipe as CRL-State-Flow (cube 0.311),
    # only env_name and the 500k budget differ. Fills two "---" cells of tab:main.
    'factored-fb-crl-flowbc-cube-double': dict(csv='summary', env='cube-double', agent='CRL', actor='flowbc', variant='plain'),
    'factored-fb-crl-flowbc-scene':       dict(csv='summary', env='scene', agent='CRL', actor='flowbc', variant='plain'),
    'PSM-State-Flow':    dict(csv='summary', env='cube', agent='PSM', actor='flowbc', variant='plain'),
    'psm_z50_ortho1000': dict(csv='summary', env='cube', agent='PSM', actor='flowbc', variant='z50-ortho1000-tuned'),
    'psm_state_tuned__ortho_coef1__lr_phi1e-5':    dict(csv='summary', env='cube', agent='PSM', actor='flowbc', variant='tuned-ortho1'),
    'psm_state_orthohi__ortho_coef1000__lr_phi1e-5': dict(csv='summary', env='cube', agent='PSM', actor='flowbc', variant='tuned-ortho1000'),
    'psm_state_orthohi__ortho_coef100__lr_phi1e-5':  dict(csv='summary', env='cube', agent='PSM', actor='flowbc', variant='tuned-ortho100'),
    'PSM-Pixel-DrQ-Flow': dict(csv='summary', env='cube-pixel', agent='PSM', actor='flowbc', variant='drq'),
    'RLDP-State-Flow':    dict(csv='summary', env='cube', agent='RLDP', actor='flowbc', variant='plain'),
    'RLDP-State-Flow-Tuned-Ortho0.001-LrB1e-5': dict(csv='summary', env='cube', agent='RLDP', actor='flowbc', variant='tuned'),
    'RLDP-Pixel-DrQ-Flow': dict(csv='summary', env='cube-pixel', agent='RLDP', actor='flowbc', variant='drq'),
    'TDMPC2-State-Planner': dict(csv='summary', env='cube', agent='TDMPC2', actor='planner', variant='plain'),
    'OneStepFB-State-Flow': dict(csv='summary', env='cube', agent='OneStepFB', actor='flowbc', variant='plain'),

    # ======================= CSV2: bridge grafts =======================
    # --- FB-side morphs (torch): import GCIQL pieces into FB ---
    'fb_state_iql':          dict(csv='bridge', env='cube', agent='FB', actor='flowbc', variant='fb-iql-v0', graft='FB<-IQL objective (early)'),
    'fb_state_iqlcritic':    dict(csv='bridge', env='cube', agent='FB', actor='flowbc', variant='fb-iql-critic', graft='FB<-IQL critic (old incidentals)'),
    'fb_state_iqlcritic_fix': dict(csv='bridge', env='cube', agent='FB', actor='flowbc', variant='fb-iql-critic-fix', graft='FB<-IQL critic (lr1e-4, ortho1000)'),
    'fb_bridge_r0_base':     dict(csv='bridge', env='cube', agent='FB', actor='flowbc', variant='r0-base', graft='FB<-IQL + goal-match + GCIQL mix'),
    'fb_bridge_r1_decond':   dict(csv='bridge', env='cube', agent='FB', actor='flowbc', variant='r1-decond', graft='r0 + de-condition F->phi'),
    'fb_bridge_r2_dnorm':    dict(csv='bridge', env='cube', agent='FB', actor='flowbc', variant='r2-dnorm', graft='r0 + sqrt(d) norm'),
    'fb_bridge_r3_allon':    dict(csv='bridge', env='cube', agent='FB', actor='flowbc', variant='r3-allon', graft='r0 + decond + dnorm'),
    'fb_bridge_r3_bilinearv': dict(csv='bridge', env='cube', agent='FB', actor='flowbc', variant='r3-bilinearv', graft='r3 + bilinear-V head (refuted gap-closer)'),
    'fb_bridge_a0_cubexyz':  dict(csv='bridge', env='cube', agent='FB', actor='flowbc', variant='a0-cubexyz', graft='privileged cube-xyz reward'),
    'fb_bridge_idB_dnorm':   dict(csv='bridge', env='cube', agent='FB', actor='flowbc', variant='idB-dnorm', graft='identity-B (z=g) + dnorm'),
    'fb_bridge_idB_decond_dnorm': dict(csv='bridge', env='cube', agent='FB', actor='flowbc', variant='idB-decond-dnorm', graft='identity-B + decond + dnorm (collapse)'),
    'fb_cube_zg_iql':        dict(csv='bridge', env='cube', agent='FB', actor='flowbc', variant='zg-iql', graft='z=g identity-FB + IQL'),
    'fb_cube_zg_her':        dict(csv='bridge', env='cube', agent='FB', actor='flowbc', variant='zg-her', graft='z=g identity-FB + HER'),
    'fb_cube_iql_learnedB_r1': dict(csv='bridge', env='cube', agent='FB', actor='flowbc', variant='learnedB-iql', graft='learned-B + IQL (phase2 control)'),
    'fb_cube_her_learnedB_r1': dict(csv='bridge', env='cube', agent='FB', actor='flowbc', variant='learnedB-her', graft='learned-B + HER'),
    'fb_cube_her_idnorm_r1':   dict(csv='bridge', env='cube', agent='FB', actor='flowbc', variant='idnorm-her', graft='identity-norm + HER'),
    'fb_cube_decouple':      dict(csv='bridge', env='cube', agent='FB', actor='flowbc', variant='decouple', graft='decouple actor-z'),
    'fb_cube_decouple_iql':  dict(csv='bridge', env='cube', agent='FB', actor='flowbc', variant='decouple-iql', graft='decouple actor-z + IQL'),
    'fb_cube_decouple_zher_traj': dict(csv='bridge', env='cube', agent='FB', actor='flowbc', variant='decouple-zher', graft='decouple actor-z + zHER-traj'),
    'fb_iql_ontraj_flow':    dict(csv='bridge', env='cube', agent='FB', actor='flowbc', variant='iql-ontraj', graft='FB-IQL on-trajectory goals'),
    'fb_iql_ontraj_ddpgbc':  dict(csv='bridge', env='cube', agent='FB', actor='ddpgbc', variant='iql-ontraj', graft='FB-IQL on-trajectory goals, ddpgbc actor'),
    'fb_state_iwr':          dict(csv='bridge', env='cube', agent='FB', actor='flowbc', variant='iwr', graft='IWR reweight'),
    'fb_state_iwr_wz':       dict(csv='bridge', env='cube', agent='FB', actor='flowbc', variant='iwr-wz', graft='IWR reweight w/ z'),
    'fb_state_zher':         dict(csv='bridge', env='cube', agent='FB', actor='flowbc', variant='zher', graft='z-HER relabel'),
    'fb_z128_ortho1000':     dict(csv='bridge', env='cube', agent='FB', actor='flowbc', variant='z128-ortho1000', graft='FB capacity/ortho variant'),
    'FB-State-Flow-GoalCond': dict(csv='bridge', env='cube', agent='FB', actor='flowbc', variant='goal-cond-v1', graft='explicit goal obs into F + actor'),
    'FB-State-Flow-USFA':     dict(csv='bridge', env='cube', agent='FB', actor='flowbc', variant='fixed-b-cubexyz-v2', graft='privileged USFA B=cube xyz'),
    'FB-State-Flow-EncMatch': dict(csv='bridge', env='cube', agent='FB', actor='flowbc', variant='encmatch', graft='critic-encoder match (KNOWN design confound)'),
    'covbal_a0.5':            dict(csv='bridge', env='cube', agent='FB', actor='flowbc', variant='covbal', graft='coverage-balanced rho reweight'),
    'covbal_prio_wz_a0.5':    dict(csv='bridge', env='cube', agent='FB', actor='flowbc', variant='covbal-prio-wz', graft='coverage-balanced prioritized w/ z'),
    'cube-single-video-replay-ortho1000-lrb1e-4': dict(csv='bridge', env='cube', agent='FB', actor='flowbc', variant='video-replay', graft='FB data ablation (video replay)'),
    'cube-single-onestep-ortho1000-lrb1e-4': dict(csv='bridge', env='cube', agent='OneStepFB', actor='flowbc', variant='ortho1000', graft='one-step FB variant'),
    # T7/T8 (2026-07-20): the last measure-side arms (matched ortho1000/lr_b1e-4, 1M)
    'fb_state_t7_min':   dict(csv='bridge', env='cube', agent='FB', actor='flowbc', variant='t7-pess0.5-min', graft='T7 conservatism: pessimism 0.5 == exact min-aggregation'),
    'fb_state_t7_extra': dict(csv='bridge', env='cube', agent='FB', actor='flowbc', variant='t7-pess1.0', graft='T7 conservatism: pessimism 1.0, beyond-min (VC-FB-style)'),
    'fb_state_t8_zcond': dict(csv='bridge', env='cube', agent='OneStepFB', actor='flowbc', variant='t8-zcond', graft='T8 faithful one-step FB: SARSA bootstrap, F kept z-conditioned'),
    # FB incidental-HP ladder (kept: they anchor the knob-ladder story)
    'cube-single-ortho100-lrb1e-4':  dict(csv='bridge', env='cube', agent='FB', actor='flowbc', variant='hp-ortho100-lrb1e-4', graft='FB incidental-HP ladder'),
    'cube-single-ortho100-lrb1e-5':  dict(csv='bridge', env='cube', agent='FB', actor='flowbc', variant='hp-ortho100-lrb1e-5', graft='FB incidental-HP ladder'),
    'cube-single-ortho1000-lrb1e-4': dict(csv='bridge', env='cube', agent='FB', actor='flowbc', variant='hp-ortho1000-lrb1e-4', graft='FB incidental-HP ladder'),
    'cube-single-ortho1000-lrb1e-5': dict(csv='bridge', env='cube', agent='FB', actor='flowbc', variant='hp-ortho1000-lrb1e-5', graft='FB incidental-HP ladder'),
    'antmaze-medium-ortho100-lrb1e-4':  dict(csv='bridge', env='antmaze', agent='FB', actor='flowbc', variant='hp-ortho100-lrb1e-4', graft='FB incidental-HP ladder'),
    'antmaze-medium-ortho100-lrb1e-5':  dict(csv='bridge', env='antmaze', agent='FB', actor='flowbc', variant='hp-ortho100-lrb1e-5', graft='FB incidental-HP ladder'),
    'antmaze-medium-ortho1000-lrb1e-4': dict(csv='bridge', env='antmaze', agent='FB', actor='flowbc', variant='hp-ortho1000-lrb1e-4', graft='FB incidental-HP ladder'),
    'antmaze-medium-ortho1000-lrb1e-5': dict(csv='bridge', env='antmaze', agent='FB', actor='flowbc', variant='hp-ortho1000-lrb1e-5', graft='FB incidental-HP ladder'),
    # --- GCIQL-side morphs (jax): impose FB pieces on GCIQL ---
    'gciql_g4_zcond': dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='g4-zcond', graft='actor_cond=z from value psi (G4)'),
    'GCIQL-State-Flow-Bilinear-d50-TwoTower':      dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='two-tower', graft='FB two-tower forward map, IQL objective'),
    'GCIQL-State-Flow-Bilinear-d50-B-TwoTowerProjZ': dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='B-twotower-projz', graft='two-tower + sphere-Norm'),
    'GCIQL-State-Flow-Bilinear-d50-B-TwoTowerOrtho1': dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='B-twotower-ortho1', graft='two-tower + ortho1.0'),
    'GCIQL-State-Flow-Bilinear-d50-B1-FBzIQL':     dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='B1-fbz-iql', graft='fb_forward_z under IQL (USFA-IQL core)'),
    'GCIQL-State-Flow-Bilinear-d50-B4-FBstruct':   dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='B4-full-struct', graft='two-tower + z-fwd + sphere + ortho1, IQL'),
    'GCIQL-Bil-MeasureOnly':        dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='measure-only', graft='FB successor-measure objective only'),
    'GCIQL-Bil-MeasureOnly-O1000':  dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='measure-only-ortho1000', graft='measure only + ortho1000'),
    'GCIQL-Bil-FBz-O1000':          dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='C1-fbz-ortho1000', graft='measure + fb_forward_z (C1)'),
    'GCIQL-State-Flow-Bilinear-d50-C-FBz-O1':       dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='C-fbz-ortho1', graft='measure + fbz + ortho1.0'),
    'GCIQL-State-Flow-Bilinear-d50-C-FBzProjZ-O1000': dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='C-fbz-projz-ortho1000', graft='measure + fbz + sphere + ortho1000'),
    'GCIQL-State-Flow-Bilinear-d50-C2-MeasureTwoTower': dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='C2-measure-twotower', graft='measure + two-tower'),
    'GCIQL-State-Flow-Bilinear-d50-C4-FBrebuild':  dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='C4-fb-rebuild', graft='measure + fbz + sphere + ortho1 (full C stack)'),
    'GCIQL-State-Flow-Bilinear-d50-C5-Zmix':       dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='C5a-zmix', graft='C4 + FB mixed z-sampling (random sqrt(d) directions)'),
    'GCIQL-State-Flow-Bilinear-d50-C5-Zmix-ActorZ': dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='C5b-zmix-actorz', graft='C5a + actor_cond=z + z-consistent bootstrap'),
    'GCIQL-State-Flow-Bilinear-d50-D1-ActorZCritic': dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='D1-actor-z-critic', graft='actor z from critic psi under measure'),
    'GCIQL-State-Flow-Bilinear-d50-Exp4-ActorZInfer': dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='exp4-actor-z-infer', graft='inferred-z actor (ARTIFACT: z=99% noise, see audit)'),
    'GCIQL-Bilinear-Measure-mc0.3':  dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='iql+measure-mc0.3-RAW', graft='IQL+measure interpolation, NO scale ctrl (18/18 crashed)'),
    'GCIQL-Bilinear-Measure-mc1.0':  dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='iql+measure-mc1.0-RAW', graft='crashed (scale blowup)'),
    'GCIQL-Bilinear-Measure-mc10.0': dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='iql+measure-mc10-RAW', graft='crashed (scale blowup)'),
    'GCIQL-Bil-MeasureSA-mc0.3':  dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='measureSA-mc0.3', graft='(s,a)-measure legs, crashed <=15k'),
    'GCIQL-Bil-MeasureSA-mc1.0':  dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='measureSA-mc1.0', graft='crashed'),
    'GCIQL-Bil-MeasureSA-mc10.0': dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='measureSA-mc10', graft='crashed'),
    'GCIQL-State-Flow-Bilinear-d50-MeasureMix-l001': dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='E2-mix-l0.01', graft='IQL + scale-normed measure, shared legs'),
    'GCIQL-State-Flow-Bilinear-d50-MeasureMix-l01':  dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='E2-mix-l0.1', graft='shared legs'),
    'GCIQL-State-Flow-Bilinear-d50-MeasureMix-l1':   dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='E2-mix-l1', graft='shared legs'),
    'GCIQL-State-Flow-Bilinear-d50-MeasureMix-l10':  dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='E2-mix-l10', graft='shared legs (absorption limit test)'),
    'GCIQL-State-Flow-Bilinear-d50-MeasureMix-l01-sgphi': dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='E2-mix-l0.1-sgphi', graft='stopgrad-phi (measure->psi only)'),
    'GCIQL-State-Flow-Bilinear-d50-MeasureMix-l1-sgphi':  dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='E2-mix-l1-sgphi', graft='stopgrad-phi'),
    'GCIQL-State-Flow-Bilinear-d50-MeasureMix-l3-sgphi':  dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='E2-mix-l3-sgphi', graft='stopgrad-phi dose'),
    'GCIQL-State-Flow-Bilinear-d50-MeasureMix-l10-sgphi': dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='E2-mix-l10-sgphi', graft='stopgrad-phi dose (1 q-band violation)'),
    'GCIQL-State-Flow-Bilinear-d50-MeasureMix-l30-sgphi': dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='E2-mix-l30-sgphi', graft='stopgrad-phi dose (3/3 q-band violations, UNINTERPRETABLE)'),
    'GCIQL-AntMaze-Medium-Bilinear-d50-MeasureOnly-DDPGBC': dict(csv='bridge', env='antmaze', agent='GCIQL', actor='ddpgbc', variant='measure-only', graft='measure objective only, domain-flip control'),
    'GCIQL-State-Flow-Bilinear-d50-Ortho1000': dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='iql-ortho1000', graft='FB ortho coef on IQL critic (kills it)'),
    'GCIQL-State-Flow-Bilinear-SG-d50': dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='bilinear-sg-d50',
                                             graft='psi(s,g) not psi(g): same inner-product head, no low-rank GOAL bottleneck',
                                             notes='2/3 seeds DIVERGE late (value_loss 0.15->1-5, q_mean crosses positive); '
                                                   'the one stable seed lands at MLP level. Mean is divergence-driven, not a policy quality.'),
    'GCIQL-State-Flow-Bilinear-d50-singleq': dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='single-q', graft='GCIQL self-ablation'),
    'GCIQL-State-Flow-Bilinear-d50-tau05':   dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='tau0.5', graft='GCIQL self-ablation'),
    'GCIQL-State-Flow-Bilinear-d50-tau095':  dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='tau0.95', graft='GCIQL self-ablation'),
    'GCIQL-State-Flow-IdentityPsi-Measure':     dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='identity-psi-measure', graft='REVERTED mis-build (identity_psi); crashed/partial'),
    'GCIQL-State-Flow-IdentityPsi-Measure-FBz': dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='identity-psi-measure-fbz', graft='REVERTED mis-build; crashed/partial'),
    # --- E3: corrected SF, reward IN the loop via a learned base (sf_base=learned) ---
    # The two pre-fix runs regressed the reward on psi(s') while GCDataset ties reward AND mask
    # to the current index (fixed 2026-07-09, fb83d68). Both are dead numbers; the -Fix / -Cond
    # groups are the citable cells. Their 100k-400k curves are still emitted by the compiler.
    'GCIQL-State-Flow-SF-FeatureTD-RewReg':       dict(csv=E, reason='E3 v1: psi=0 saddle ({-1,0} rewards x symmetric bilinear model) AND the psi(s") reward bug; killed @260k'),
    'GCIQL-State-Flow-SF-FeatureTD-RewReg-Shift': dict(csv=E, reason='E3 v2: escaped the saddle but still carried the psi(s") reward bug; killed @330k. Superseded by -Fix'),
    'GCIQL-State-Flow-SF-FeatureTD-RewReg-Shift-Fix':  dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='sf-td-rewreg-shift', graft='E3: SF vector TD + reward-regressed learned base (corrected r_hat(s,g))'),
    'GCIQL-State-Flow-SF-FeatureTD-RewReg-Shift-Cond': dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='sf-td-rewreg-shift-cond', graft='+ task-conditioned phi (audit pt3: goal-dependent mask)'),

    # --- 1f: annealed SM finetuning from the converged bilinear-GCIQL anchor (0.904). lambda(t)
    # ramps 0 -> lam_max over 300k then holds; dense evals make each run a success-vs-lambda dose
    # curve. Separates fixed-point incompatibility from an optimization-path artifact.
    'GCIQL-State-Flow-Anneal-Zero':   dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='anneal-zero', graft='CONTROL: finetune the 0.904 anchor 400k more steps with NO measure loss (validates restore; isolates extra IQL steps)'),
    'GCIQL-State-Flow-Anneal-SgPhi-IQLfade': dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='anneal-sgphi-iqlfade', graft='warm-start + crossfade IQL OUT as lambda ramps to 10 (endpoint = measure-only): does the good basin survive without L_IQL?'),
    'GCIQL-State-Flow-Anneal-SgPhi':  dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='anneal-sgphi', graft='warm-start + lambda 0->10 ramp on the stopgrad-phi measure (crosses the cold cliff at lambda=3)'),
    'GCIQL-State-Flow-Anneal-Shared': dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='anneal-shared', graft='warm-start + lambda 0->100 ramp on the shared-legs measure (cold cliff is above 10, never located)'),

    # --- 1e: task-weighted successor moments (research note). N1 swaps the measure loss's B x B
    # test states from rho (next_observations) to nu (value_goals); N2 additionally collapses the
    # test set to rank 1. Ortho and Dirac stay on psi(next_observations) in both.
    'GCIQL-State-Flow-Bilinear-d50-C-TaskNorm': dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='C-tasknorm', graft='FB measure, residual tested in the TASK norm (test states ~ value_goals) instead of the replay norm'),
    'GCIQL-State-Flow-Bilinear-d50-C-TaskProj': dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='C-taskproj', graft='+ rank-1 test set: project each residual onto its own z_i=psi(g_i); fixed point = scalar TD with a kernel reward'),

    # --- 1d: GC-SimpleSF (Chua et al.). Rung 2 isolates the architecture against bilinear GCIQL
    # (0.904); rung 3 grounds psi in the reward and isolates the grounding against rung 2.
    'GCIQL-State-Flow-SimpleSF-Arch':     dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='simplesf-arch', graft='SimpleSF trunk split (b=h/||h||, Psi=head(h)), psi learned end-to-end; scalar IQL TD'),
    'GCIQL-State-Flow-SimpleSF-Grounded': dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='simplesf-grounded', graft="Chua et al.'s cell: + psi trained ONLY by reward regression <sg[b],psi(g)>"),
    'GCIQL-State-Flow-SimpleSF-Anneal':   dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='simplesf-anneal', graft='SimpleSF curriculum: cross-fade true-reward TD OUT while reward-regression grounding fades IN (rung 2 geometry -> rung 3 endpoint); tests optimization-path vs representability'),

    # --- 1a: joint-psi SF, FB ortho + sphere (spec 2026-07-10). Cell A is the estimator-axis
    # control: identical to GCIQL-State-Flow-Bilinear-d50-C-FBzProjZ-O1000 (0.231) except the
    # BxB measure loss is replaced by the per-sample vector TD.
    'GCIQL-State-Flow-SF-JointPsi-Ortho1k':        dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='sf-joint-ortho1k', graft="SF vector TD trains psi too; FB ortho1000 + sphere; b=psi(s'), no mask (matched to C-fbz-projz-ortho1000 except the estimator)"),
    'GCIQL-State-Flow-SF-JointPsi-Ortho1k-Absorb': dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='sf-joint-ortho1k-absorb', graft='+ absorbing goal mask: Q=gamma^T (first-hitting) instead of a visitation density'),

    # --- 1b: readout-vs-TD crux -- SCALAR IQL TD with the SF cells' crippled goal leg ---
    'GCIQL-State-Flow-Bilinear-d50-FrozenPsi':  dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='frozen-psi-d50', graft='scalar IQL TD + frozen random psi (isolates the vector TD from the frozen readout)'),
    'GCIQL-State-Flow-Bilinear-IdentityPsi':    dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='identity-psi-d28', graft='scalar IQL TD + psi(g)=g (read vs bilinear-d28, not d50)'),

    # --- SF-feature-TD cells (SEMANTICALLY CONFOUNDED per audit-20260706: no reward in loop) ---
    'GCIQL-State-Flow-SF-FeatureTD-IdentityB':   dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='sf-td-identity', graft='SF vector TD, identity base (CONFOUNDED, see audit)'),
    'GCIQL-State-Flow-SF-FeatureTD-Frozen50':    dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='sf-td-frozen50', graft='SF vector TD, frozen base (CONFOUNDED)'),
    'GCIQL-State-Flow-SF-FeatureTD-IdentityCond': dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='sf-td-identity-cond', graft='+ task-conditioned phi (CONFOUNDED)'),
    'GCIQL-State-Flow-SF-FeatureTD-FrozenCond':   dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='sf-td-frozen-cond', graft='+ task-conditioned phi (CONFOUNDED)'),
    'GCIQL-State-Flow-SF-FeatureTD-Identity-Ortho1k':     dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='sf-td-identity-ortho1k', graft='+ phi ortho (kills critic)'),
    'GCIQL-State-Flow-SF-FeatureTD-Frozen-Ortho1k':       dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='sf-td-frozen-ortho1k', graft='+ phi ortho'),
    'GCIQL-State-Flow-SF-FeatureTD-IdentityCond-Ortho1k': dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='sf-td-identity-cond-ortho1k', graft='+ phi ortho'),
    'GCIQL-State-Flow-SF-FeatureTD-FrozenCond-Ortho1k':   dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='sf-td-frozen-cond-ortho1k', graft='+ phi ortho'),
    'GCIQL-State-Flow-SF-FeatureTD-IdentityB-DDPGBC':    dict(csv='bridge', env='cube', agent='GCIQL', actor='ddpgbc', variant='sf-td-identity', graft='no-flow-actor control (CONFOUNDED)'),
    'GCIQL-State-Flow-SF-FeatureTD-Frozen50-DDPGBC':     dict(csv='bridge', env='cube', agent='GCIQL', actor='ddpgbc', variant='sf-td-frozen50', graft='no-flow-actor control'),
    'GCIQL-State-Flow-SF-FeatureTD-IdentityCond-DDPGBC': dict(csv='bridge', env='cube', agent='GCIQL', actor='ddpgbc', variant='sf-td-identity-cond', graft='no-flow-actor control'),
    'GCIQL-State-Flow-SF-FeatureTD-FrozenCond-DDPGBC':   dict(csv='bridge', env='cube', agent='GCIQL', actor='ddpgbc', variant='sf-td-frozen-cond', graft='no-flow-actor control'),
    # --- ISFB (SG-ISFB, FB-modified) ---
    'ISFB-State-Coord-d50':        dict(csv='bridge', env='cube', agent='ISFB', actor='fql_flow', variant='coord', graft='SG-ISFB jax coord readout'),
    'ISFB-State-ScalarGateAWR-d50': dict(csv='bridge', env='cube', agent='ISFB', actor='fql_flow', variant='scalargate-awr', graft='SG-ISFB jax'),
    'ISFB-State-ScalarGateHard-d50': dict(csv='bridge', env='cube', agent='ISFB', actor='fql_flow', variant='scalargate-hard', graft='SG-ISFB jax'),
    'ISFB-Torch-Exp1-fixed':       dict(csv='bridge', env='cube', agent='ISFB', actor='flowbc', variant='torch-exp1', graft='faithful SG-ISFB torch'),

    # ======================= EXCLUDED (reason required) =======================
    'fb_antmaze_medium_smoke': dict(csv=E, reason='50k smoke'),
    'fbd_fbd_am': dict(csv=E, reason='failed at launch (missing dataset path, per .hydra/config.yaml agent=fb_flowbc, '
                                      'exp_fbddpg/fbd_am_s{0,1,2}): 0 checkpoints, no training history -- not a completed run'),
    'factored-fb-gciql':       dict(csv=E, reason='step-1 stubs (2 runs)'),
    'NONE':                    dict(csv=E, reason='ungrouped runs (early FB-pixel gotcha per wandb-group-naming memory) — audit lists them'),
    'ISFB-Torch-Exp1':         dict(csv=E, reason='superseded by ISFB-Torch-Exp1-fixed (crashed ~340k)'),
    'GCIQL-State-Flow-SF-FeatureTD-Identity-Ortho10k':     dict(csv=E, reason='1 metricless stub'),
    'GCIQL-State-Flow-SF-FeatureTD-IdentityCond-Ortho10k': dict(csv=E, reason='2 stubs, no completed seed'),
    'psm_isolation_300k':      dict(csv=E, reason='300k isolation probe, off-protocol'),
    # PSM/RLDP HP sweeps: superseded by the tuned rows in CSV1
    'PSM-State-Flow-FBmatch-LrCheck-Ortho1-LrPhi1e-4': dict(csv=E, reason='HP sweep (tuned row kept)'),
    'PSM-State-Flow-FBmatch-LrCheck-Ortho1-LrPhi1e-5': dict(csv=E, reason='HP sweep'),
    'PSM-State-Flow-FBmatch-Tau0.01-LrPhi1e-4': dict(csv=E, reason='HP sweep'),
    'PSM-State-Flow-FBmatch-Tau0.01-LrPhi1e-5': dict(csv=E, reason='HP sweep'),
    'PSM-State-Flow-Pess0-LrPhi1e-5':           dict(csv=E, reason='HP sweep (readout ablation)'),
    'PSM-State-Flow-Sweep-Ortho0.1-LrPhi1e-4':  dict(csv=E, reason='HP sweep'),
    'PSM-State-Flow-Sweep-Ortho0.1-LrPhi1e-5':  dict(csv=E, reason='HP sweep'),
    'PSM-State-Flow-Sweep-Ortho1-LrPhi1e-4':    dict(csv=E, reason='HP sweep'),
    'PSM-State-Flow-Sweep-Ortho1-LrPhi1e-5':    dict(csv=E, reason='HP sweep'),
    'PSM-State-Flow-Sweep-Ortho10-LrPhi1e-4':   dict(csv=E, reason='HP sweep'),
    'PSM-State-Flow-Sweep-Ortho10-LrPhi1e-5':   dict(csv=E, reason='HP sweep'),
    'RLDP-State-Flow-FBmatch-Ortho1000-LrB1e-4': dict(csv=E, reason='HP sweep'),
    'RLDP-State-Flow-Sweep-Ortho0.001-LrB1e-4':  dict(csv=E, reason='HP sweep'),
    'RLDP-State-Flow-Sweep-Ortho0.001-LrB1e-5':  dict(csv=E, reason='HP sweep'),
    'RLDP-State-Flow-Sweep-Ortho0.01-LrB1e-4':   dict(csv=E, reason='HP sweep'),
    'RLDP-State-Flow-Sweep-Ortho0.01-LrB1e-5':   dict(csv=E, reason='HP sweep'),
    'RLDP-State-Flow-Sweep-Ortho0.1-LrB1e-4':    dict(csv=E, reason='HP sweep'),
    'RLDP-State-Flow-Sweep-Ortho0.1-LrB1e-5':    dict(csv=E, reason='HP sweep'),
    # pixel infra groups (project-level default groups, mixed content; audited separately)
    'factored-fb-crl-flowbc':          dict(csv=E, reason='project-default group, contents duplicated in named groups'),
    'factored-fb-crl-flowbc-pixel-drq': dict(csv=E, reason='project-default group'),
    'factored-fb-gciql-pixel':         dict(csv=E, reason='project-default group'),
    'factored-fb-gciql-pixel-drq':     dict(csv=E, reason='project-default group'),
    'factored-fb-gcivl-pixel':         dict(csv=E, reason='project-default group'),
    'factored-fb-gcivl-pixel-drq':     dict(csv=E, reason='project-default group'),
    '_redirect_validate':              dict(csv=E, reason='infra stub'),

    # ---- rescuing-fb program (2026-07-16..20): RG-FB + T10 centered measure ----
    'RGFB-Cube-Single-State-d50-v0':         dict(csv='bridge', env='cube', agent='RG-FB', actor='fql_flow', variant='rgfb-v0',
                                                  graft='RG-FB v0: FB interface (z-actor + moment inference), pure scalar IQL grounding; zero-shot 0.825'),
    'RGFB-Cube-Single-State-d50-vfnone-ctl': dict(csv='bridge', env='cube', agent='RG-FB', actor='fql_flow', variant='rgfb-vfnone-ctl',
                                                  graft='RG-FB v0 isolation control: actor_vf_cond=none'),
    'GCIQL-State-Flow-CenteredMeasure-Aux-d50':           dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='t10a-centered-aux',
                                                  graft='T10-a: IQL + centered-measure advantage aux (0.823, costs ~8pts vs anchor)'),
    'GCIQL-State-Flow-CenteredMeasure-Only-Anchored-d50': dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='t10b2-centered-only-anchored',
                                                  graft='T10-b2: measure-only, centered + FB scale anchors + SARSA base leg (dead band 0.06-0.10)'),
    'GCIQL-State-Flow-CenteredMeasure-Only-d50': dict(csv=E, reason='T10-b v1: scale-runaway (q->1e13), killed @160k; superseded by Anchored arm'),
    # ---- spec 06 phenomenon implementations (2026-07-20): pre-registered, not yet run ----
    # Direction I (method chain). Eval-only scripts (GPI item1, rerank item3, readout item8)
    # produce analysis JSONs, not wandb training groups, so they are not mapped here.
    'RGFB-Cube-Single-State-d512-v0':          dict(csv='bridge', env='cube', agent='RG-FB', actor='fql_flow', variant='rgfb-v0-d512',
                                                  graft='spec06 item2: RG-FB v0 at latent_dim=512 (isolates the d-gap vs d50 0.825)'),
    'RGFB-Cube-Single-State-d50-v1-taskprior':  dict(csv='bridge', env='cube', agent='RG-FB', actor='fql_flow', variant='rgfb-v1-taskprior-d50',
                                                  graft='spec06 item4: RG-FB v1 task-prior (Dirac+mixture+random-linear scalar grounding at the sampled z)'),
    'RGFB-Cube-Single-State-d512-v1-taskprior': dict(csv='bridge', env='cube', agent='RG-FB', actor='fql_flow', variant='rgfb-v1-taskprior-d512',
                                                  graft='spec06 item4: RG-FB v1 task-prior at d512 (if item2 won)'),
    # Direction II (mechanism cells) -- single-axis moves on the plain measure-only cell.
    'GCIQL-State-Flow-Measure-CellW0-NoOrtho-d50': dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='measure-cellw-w0-noortho',
                                                  graft='CELL W w=0, ortho=0 (scale-fix rerun): the TRUE GC-IQL-equivalence run after the ortho-scale confound'),
    'GCIQL-State-Flow-Measure-FirstPassage-NoOrtho-d50': dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='measure-first-passage-noortho',
                                                  graft='first-passage axis-C cell, ortho=0 (was 0.225 with a ~100x oversized ortho term)'),
    'GCIQL-State-Flow-Measure-CellW0-d50':         dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='measure-cellw-w0',
                                                  graft='CELL W w=0: first-passage SM, hindsight-paired diagonal only == bilinear GC-IQL under M=1+(1-g)Q (theorem-backed win)'),
    'GCIQL-State-Flow-Measure-CellW03-d50':        dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='measure-cellw-w03',
                                                  graft='CELL W w=0.3: + off-diagonal matrix supervision at a GC-IQL-like task mix (the measure-native win candidate)'),
    'GCIQL-State-Flow-Measure-GCIQLCorner-d50':    dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='measure-gciql-corner',
                                                  graft='FULL-CORNER measure: first-passage + expectile backup + hindsight test points (all 3 axes) -- the headline rescuing-fb arm'),
    'GCIQL-State-Flow-Measure-FirstPassage-d50':   dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='measure-first-passage',
                                                  graft='spec06 item5: first-passage (absorbing) successor-measure residual, cube-xyz hit'),
    # spec08 E7 -- the two-phase curriculum on a SECOND/THIRD env (measure_hit_spec knob).
    # The *-1Cube / *-CubeOnly cells are SPARSITY CONTROLS: same objective, cube-single-sized
    # region density (18.6 / 61.9 off-diag positives per row at B=1024) against the faithful
    # conjunctions (0.51 / 3.07). They deconfound "reach-first fails here" from "the reward is
    # 27x sparser here"; their regions ignore factors the native eval scores, so read their
    # TRAINING traces, not their overall success.
    'GCIQL-CubeDouble-Flow-Bilinear-d50':          dict(csv='summary', env='cube-double', agent='GCIQL', actor='fql_flow', variant='bilinear-d50'),
    'GCIQL-CubeDouble-Flow-Measure-FirstPassage-NoOrtho-d50': dict(csv='bridge', env='cube-double', agent='GCIQL', actor='fql_flow', variant='measure-first-passage-noortho',
                                                  graft='spec08 E7 phase 1 on cube-double; goal region = BOTH cubes within 4cm (env-faithful conjunction)'),
    'GCIQL-CubeDouble-Flow-Measure-FP-NoOrtho-1Cube-d50': dict(csv='bridge', env='cube-double', agent='GCIQL', actor='fql_flow', variant='measure-fp-noortho-1cube',
                                                  graft='spec08 E7 sparsity control: cube-double phase 1 with the region on cube 0 only'),
    'GCIQL-CubeDouble-Flow-Measure-BallVisit-FrozenPsi-d50': dict(csv='bridge', env='cube-double', agent='GCIQL', actor='fql_flow', variant='measure-ballvisit-frozenpsi',
                                                  graft='spec08 E7 phase 2 on cube-double: non-absorbing visitation TD on the frozen phase-1 psi'),
    'GCIQL-Scene-Flow-Measure-FirstPassage-NoOrtho-d50': dict(csv='bridge', env='scene', agent='GCIQL', actor='fql_flow', variant='measure-first-passage-noortho',
                                                  graft='spec08 E7 phase 1 on scene; goal region = cube+2 buttons+drawer+window (env-faithful 5-way conjunction)'),
    'GCIQL-Scene-Flow-Measure-FP-NoOrtho-CubeOnly-d50': dict(csv='bridge', env='scene', agent='GCIQL', actor='fql_flow', variant='measure-fp-noortho-cubeonly',
                                                  graft='spec08 E7 sparsity control: scene phase 1 with the region on the cube ball only'),
    'GCIQL-Scene-Flow-Measure-BallVisit-FrozenPsi-d50': dict(csv='bridge', env='scene', agent='GCIQL', actor='fql_flow', variant='measure-ballvisit-frozenpsi',
                                                  graft='spec08 E7 phase 2 on scene: non-absorbing visitation TD on the frozen phase-1 psi'),

    # ledger §9.16 p2fix arms -- phase-2 regression mechanism fixes on cube-double + scene
    # (launch_p2fix_arms.sh, 2026-07-30). Mechanism: visitation Q collapses to the hindsight
    # dwell constant when off-diag grounding is sparse; three single-knob arms against it.
    'GCIQL-CubeDouble-Flow-Measure-BallVisit-FrzActor-d50': dict(csv='bridge', env='cube-double', agent='GCIQL', actor='fql_flow', variant='measure-ballvisit-frzactor',
                                                  graft='§9.16 arm 1: phase 2 with the actor optimizer masked (E2 recipe) — regression erased by construction'),
    'GCIQL-Scene-Flow-Measure-BallVisit-FrzActor-d50': dict(csv='bridge', env='scene', agent='GCIQL', actor='fql_flow', variant='measure-ballvisit-frzactor',
                                                  graft='§9.16 arm 1: phase 2 with the actor optimizer masked (E2 recipe) — regression erased by construction'),
    'GCIQL-CubeDouble-Flow-Measure-BallVisit-EpsDM-d50': dict(csv='bridge', env='cube-double', agent='GCIQL', actor='fql_flow', variant='measure-ballvisit-epsdm',
                                                  graft='§9.16 arm 2: phase-2 training region density-matched to cube-single (eps 0.96 -> 1.21% grounding); eval unchanged'),
    'GCIQL-Scene-Flow-Measure-BallVisit-EpsDM-d50': dict(csv='bridge', env='scene', agent='GCIQL', actor='fql_flow', variant='measure-ballvisit-epsdm',
                                                  graft='§9.16 arm 2: phase-2 training region density-matched to cube-single (eps x1.9 -> 1.29% grounding); eval unchanged'),
    'GCIQL-CubeDouble-Flow-Measure-BallVisit-Hindsight-d50': dict(csv='bridge', env='cube-double', agent='GCIQL', actor='fql_flow', variant='measure-ballvisit-hindsight',
                                                  graft='§9.16 arm 3: phase 2 with hindsight test-point columns (sparsity-independent grounding)'),
    'GCIQL-Scene-Flow-Measure-BallVisit-Hindsight-d50': dict(csv='bridge', env='scene', agent='GCIQL', actor='fql_flow', variant='measure-ballvisit-hindsight',
                                                  graft='§9.16 arm 3: phase 2 with hindsight test-point columns (sparsity-independent grounding)'),

    # spec08 E8 -- the two-phase curriculum on the FULL OGBench state manipulation suite
    # (spec PAPER/rescuing-fb/specs/09-e8-full-manipulation-suite.md). 3 seeds, 500k, d=50
    # everywhere; the ONLY per-env variable is measure_hit_spec, each region mirroring that
    # env's own success criterion (verified by scripts/rescuing_fb/verify_hit_specs.py).
    # Every env also gets a matched GC-IQL bilinear-d50 anchor: the bar is env-specific
    # (cube-single 0.90 vs scene 0.292), and a shared floor across cell+anchor is what
    # separates "hard env" from "the measure objective fails here".
    # Region density at B=1024 (off-diagonal positives/row) is noted per env -- the bottom
    # envs are ~1500x sparser than cube-single, so a floor must be read against the anchor.
    'GCIQL-CubeTriple-Flow-Bilinear-d50':          dict(csv='summary', env='cube-triple', agent='GCIQL', actor='fql_flow', variant='bilinear-d50'),
    'GCIQL-CubeTriple-Flow-Measure-FirstPassage-NoOrtho-d50': dict(csv='bridge', env='cube-triple', agent='GCIQL', actor='fql_flow', variant='measure-first-passage-noortho',
                                                  graft='spec08 E8 phase 1 on cube-triple; region = all 3 cubes within 4cm (density 0.03/row)'),
    'GCIQL-CubeTriple-Flow-Measure-BallVisit-FrozenPsi-d50': dict(csv='bridge', env='cube-triple', agent='GCIQL', actor='fql_flow', variant='measure-ballvisit-frozenpsi',
                                                  graft='spec08 E8 phase 2 on cube-triple: non-absorbing visitation TD on the frozen phase-1 psi'),

    'GCIQL-CubeQuad-Flow-Bilinear-d50':            dict(csv='summary', env='cube-quadruple', agent='GCIQL', actor='fql_flow', variant='bilinear-d50'),
    'GCIQL-CubeQuad-Flow-Measure-FirstPassage-NoOrtho-d50': dict(csv='bridge', env='cube-quadruple', agent='GCIQL', actor='fql_flow', variant='measure-first-passage-noortho',
                                                  graft='spec08 E8 phase 1 on cube-quadruple; region = all 4 cubes within 4cm (density 0.01/row)'),
    'GCIQL-CubeQuad-Flow-Measure-BallVisit-FrozenPsi-d50': dict(csv='bridge', env='cube-quadruple', agent='GCIQL', actor='fql_flow', variant='measure-ballvisit-frozenpsi',
                                                  graft='spec08 E8 phase 2 on cube-quadruple: non-absorbing visitation TD on the frozen phase-1 psi'),

    'GCIQL-Puzzle3x3-Flow-Bilinear-d50':           dict(csv='summary', env='puzzle-3x3', agent='GCIQL', actor='fql_flow', variant='bilinear-d50'),
    'GCIQL-Puzzle3x3-Flow-Measure-FirstPassage-NoOrtho-d50': dict(csv='bridge', env='puzzle-3x3', agent='GCIQL', actor='fql_flow', variant='measure-first-passage-noortho',
                                                  graft='spec08 E8 phase 1 on puzzle-3x3; region = all 9 button one-hots match (density 2.04/row)'),
    'GCIQL-Puzzle3x3-Flow-Measure-BallVisit-FrozenPsi-d50': dict(csv='bridge', env='puzzle-3x3', agent='GCIQL', actor='fql_flow', variant='measure-ballvisit-frozenpsi',
                                                  graft='spec08 E8 phase 2 on puzzle-3x3: non-absorbing visitation TD on the frozen phase-1 psi'),

    'GCIQL-Puzzle4x4-Flow-Bilinear-d50':           dict(csv='summary', env='puzzle-4x4', agent='GCIQL', actor='fql_flow', variant='bilinear-d50'),
    'GCIQL-Puzzle4x4-Flow-Measure-FirstPassage-NoOrtho-d50': dict(csv='bridge', env='puzzle-4x4', agent='GCIQL', actor='fql_flow', variant='measure-first-passage-noortho',
                                                  graft='spec08 E8 phase 1 on puzzle-4x4; region = all 16 button one-hots match (density 0.06/row)'),
    'GCIQL-Puzzle4x4-Flow-Measure-BallVisit-FrozenPsi-d50': dict(csv='bridge', env='puzzle-4x4', agent='GCIQL', actor='fql_flow', variant='measure-ballvisit-frozenpsi',
                                                  graft='spec08 E8 phase 2 on puzzle-4x4: non-absorbing visitation TD on the frozen phase-1 psi'),

    'GCIQL-Puzzle4x5-Flow-Bilinear-d50':           dict(csv='summary', env='puzzle-4x5', agent='GCIQL', actor='fql_flow', variant='bilinear-d50'),
    'GCIQL-Puzzle4x5-Flow-Measure-FirstPassage-NoOrtho-d50': dict(csv='bridge', env='puzzle-4x5', agent='GCIQL', actor='fql_flow', variant='measure-first-passage-noortho',
                                                  graft='spec08 E8 phase 1 on puzzle-4x5; region = all 20 button one-hots match (density 0.01/row)'),
    'GCIQL-Puzzle4x5-Flow-Measure-BallVisit-FrozenPsi-d50': dict(csv='bridge', env='puzzle-4x5', agent='GCIQL', actor='fql_flow', variant='measure-ballvisit-frozenpsi',
                                                  graft='spec08 E8 phase 2 on puzzle-4x5: non-absorbing visitation TD on the frozen phase-1 psi'),

    'GCIQL-Puzzle4x6-Flow-Bilinear-d50':           dict(csv='summary', env='puzzle-4x6', agent='GCIQL', actor='fql_flow', variant='bilinear-d50'),
    'GCIQL-Puzzle4x6-Flow-Measure-FirstPassage-NoOrtho-d50': dict(csv='bridge', env='puzzle-4x6', agent='GCIQL', actor='fql_flow', variant='measure-first-passage-noortho',
                                                  graft='spec08 E8 phase 1 on puzzle-4x6; region = all 24 button one-hots match (density 0.01/row)'),
    'GCIQL-Puzzle4x6-Flow-Measure-BallVisit-FrozenPsi-d50': dict(csv='bridge', env='puzzle-4x6', agent='GCIQL', actor='fql_flow', variant='measure-ballvisit-frozenpsi',
                                                  graft='spec08 E8 phase 2 on puzzle-4x6: non-absorbing visitation TD on the frozen phase-1 psi'),

    # cube-single phase 2 (the note's 0.952/0.928); seed 2 added by E8 wave B
    'GCIQL-State-Flow-Measure-BallVisit-FrozenPsi-d50': dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='measure-ballvisit-frozenpsi',
                                                  graft='phase 2 on cube-single: non-absorbing visitation TD on the frozen phase-1 psi (the note headline)'),

    'GCIQL-State-Flow-Measure-Expectile-FP-d50':   dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='measure-expectile-fp',
                                                  graft='spec06 item6: expectile-measure (state-only V_M + in-sample backup) composed with first-passage'),
    'GCIQL-State-Flow-Measure-Expectile-d50':      dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='measure-expectile-unmasked',
                                                  graft='spec06 item6: expectile-measure, unmasked variant (FB sphere+ortho anchors)'),
    'GCIQL-State-Flow-Measure-HindsightTP-d50':    dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='measure-hindsight-testpoints',
                                                  graft='spec06 item7: hindsight-weighted (value_goals) measure test points; no rho importance correction'),
    # ---- ogbench-sota implementations (2026-07-20, PAPER/ogbench-sota/implementations.md) ----
    # Why the bilinear form works (Layer 0/1/3 mechanism cells). Probe scripts (t5c srank item3,
    # t3 peak-geometry item4a, t9 advantage-rank item6) produce analysis JSONs, not groups.
    'GCIQL-State-Flow-LateFusion-d50':   dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='latefusion-d50',
                                             graft='sota item1: MLP readout on concat[phi;psi] (Layer-1 control); rerank@32 is the read'),
    'GCIQL-State-Flow-MLP-PMatched':     dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='mlp-pmatched-x1.44',
                                             graft='sota item2: param-matched MLP (widths x1.44 ~2.06x, closes the param confound)'),
    'GCIQL-State-DDPGBC-Bilinear-d50-PhiNormCap': dict(csv='bridge', env='cube', agent='GCIQL', actor='ddpgbc', variant='bilinear-d50-phinormcap',
                                             graft='sota item4b: ||phi|| capped to p99 (residual-relocation surgery on the off-support peak)'),
    'GCIQL-State-Flow-Bilinear-d50-PhiNormCap':   dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='bilinear-d50-phinormcap',
                                             graft='sota item4b: phi_norm_cap flow control (the cap must not cost the ordering)'),
    'GCIQL-State-Flow-BilinearSGBounded-d50':     dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='bilinear-sg-bounded-d50',
                                             graft='sota item5: bounded psi(s,g) tanh output (BVN regime-inversion test)'),
    'GCIQL-Scene-Flow-BilinearSplit':    dict(csv='bridge', env='scene', agent='GCIQL', actor='fql_flow', variant='bilinear-split',
                                             graft='sota item7: factored Q=sum_k phi_k.psi_k per scene factor (compositional span test)'),
    'GCIQL-Scene-DDPGBC-BilinearSplit':  dict(csv='bridge', env='scene', agent='GCIQL', actor='ddpgbc', variant='bilinear-split',
                                             graft='sota item7: split-head, ddpgbc actor'),
    # ---- successor-measure offdiag_weight sweep (2026-07-21): the middle of the curve
    # between w=0 (M=1+(1-g)Q, degenerate bilinear Q) and w=1 (full measure, floor 0.10) ----
    'GCIQL-State-Flow-Measure-CellW010-d50': dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='measure-offdiag-0.1',
                                             graft='offdiag_weight sweep w=0.1: how much genuine occupancy structure holds control'),
    'GCIQL-State-Flow-Measure-CellW025-d50': dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='measure-offdiag-0.25',
                                             graft='offdiag_weight sweep w=0.25'),
    'GCIQL-State-Flow-Measure-CellW030-d50': dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='measure-offdiag-0.3',
                                             graft='offdiag_weight sweep w=0.3'),
    'GCIQL-State-Flow-Measure-CellW050-d50': dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='measure-offdiag-0.5',
                                             graft='offdiag_weight sweep w=0.5'),
    # ---- warm-start experiment (2026-07-21): does the Q basin survive the measure objective ----
    'GCIQL-State-Flow-Measure-WarmFirstPassage-d50': dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='warm-firstpassage',
                                             graft='warm-start first-passage measure from bilinear-d50 (~0.90): does warm lift the 0.23 cold pulse?'),
    'GCIQL-State-Flow-Measure-WarmCellW0-d50': dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='warm-offdiag-0.0',
                                             graft='warm-start M=1+(1-g)Q (offdiag 0) from bilinear-d50: does the Q basin survive the measure objective?'),
    'GCIQL-State-Flow-Measure-WarmCellW025-d50': dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='warm-offdiag-0.25',
                                             graft='warm-start offdiag 0.25 from bilinear-d50 (warm sweep point)'),
    'GCIQL-State-Flow-BilinearMLPV-d50': dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='bilinear-mlpv',
                                             graft='ogbench-sota ablation: bilinear Q + MLP V -- is Vs bilinearity load-bearing?'),
    'GCIQL-State-Flow-Occupancy-d50': dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='occupancy-iql',
                                             graft='Occupancy-IQL (occupancy.tex): shared-B + CRL occupancy reward on GCIQLs IQL operator -- does occupancy grounding WORK when it rides on policy improvement?'),
    # ---- spec 07 (2026-07-22): capability on top of the FP-NoOrtho win ----
    'GCIQL-State-Flow-Measure-FP-NoOrtho-TPActor-d50': dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='fp-noortho-tpactor',
                                             graft='spec07 item1: actor z-prior (Dirac/mix/random 0.6/0.2/0.2) + forced z-actor on the FP-NoOrtho critic; Dirac band FAILED (0.29-0.56 vs anchor 0.967, never reaches it)'),
    'GCIQL-State-Flow-Measure-FP-NoOrtho-TPActorCriticMix-d50': dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='fp-noortho-tpactor-criticmix',
                                             graft='spec07 item4 linearity-conflict probe: +scalar FP TD at z_mix; tracked the ANCHOR 0.84-0.96 through 250k then collapsed to 0.33 (conflict confirmed late; early phase = off-Dirac grounding helps the z-readout)'),

    # ---- spec 17 (2026-08-06): delta / delta-partition region successor measures on cube-single,
    # d17_cs_* run names, 3 seeds, 500k, PAPER/Multistep-FB/specs/17{,a,b}-*.md. Bookkeeping item
    # flagged owed by spec 18 §7 -- added here, not launched by this session.
    'GCIQL-CubeSingle-Flow-Measure-Delta-d50': dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='measure-delta',
                                             graft="spec17a (variant A): signed event target hit(x')-hit(x) on the region successor measure "
                                                   "(temporally-centered, non-absorbing; deletes the dwell-plateau level channel from supervision)"),
    'GCIQL-CubeSingle-Flow-Measure-DeltaPartition-d50': dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='measure-delta-partition',
                                             graft='spec17b (variant B): one-stage K=256 simplex successor measure over fixed hit-spec prototypes '
                                                   '(delta_head CE-TD to sg[(1-gamma)phi(x)+gamma*p_bar]; actor q = p.phi(goal), frozen head)',
                                             notes='d17_cs_part_s1 crashed twice with a CUDA fault (dirs preserved as '
                                                   'd17_cs_part_s1.crashed_70k_cuda_1043 / .crashed_230k_cuda_1138) and is re-running from scratch '
                                                   'as of 2026-08-06; those two dead dirs are NOT completed runs -- see compile_audit.txt partial handling'),

    # ---- spec 18 (2026-08-06): continuous normalized successor density + the 17b K-sweep
    # control, cube-single, d18_cs_* run names, /home/adityamohan/exp_spec18/factored-fb/.
    # PAPER/Multistep-FB/specs/18-density-measure.md sec.7. P0.3 bookkeeping item (spec 19
    # sec.1) -- added here, not launched by this session.

    # ---- spec 19 (2026-08-06/07): ORS program -- occupancy reward shaping from the
    # BEHAVIOR-mode density head (no goal conditioning, SARSA bootstrap) feeding a
    # g-conditioned GCIQL extractor. d19_* run names, /home/adityamohan/exp_spec19/factored-fb/.
    # PAPER/Multistep-FB/specs/19{,a,b}-ors-*.md. Confirmed via flags.json (agent_name=
    # gciql_flow, actor_kind=flow -> actor='fql_flow' throughout).
    'GCIQL-CubeSingle-Flow-BCOnly-d50': dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='bc-only',
        graft='spec19 P0 control: BC-only floor (actor_bc_only=True, q-term zeroed), cube-single'),
    'GCIQL-CubeDouble-Flow-BCOnly-d50': dict(csv='bridge', env='cube-double', agent='GCIQL', actor='fql_flow', variant='bc-only',
        graft='spec19 P0 control: BC-only floor, cube-double'),
    'GCIQL-Scene-Flow-BCOnly-d50': dict(csv='bridge', env='scene', agent='GCIQL', actor='fql_flow', variant='bc-only',
        graft='spec19 P0 control: BC-only floor, scene'),
    'GCIQL-CubeSingle-Flow-Measure-DensityBehavior-d50': dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='measure-density-behavior',
        graft='spec19a: behavior-mode density head (measure_density_behavior=True, no goal conditioning, SARSA bootstrap) -- '
              'trains the reward model consumed by the ORS extractor, cube-single',
        notes='single seed (reward-model training run, not a policy-eval cell)'),
    'GCIQL-CubeDouble-Flow-Measure-DensityBehavior-d50': dict(csv='bridge', env='cube-double', agent='GCIQL', actor='fql_flow', variant='measure-density-behavior',
        graft='spec19a: behavior-mode density head, cube-double', notes='single seed'),
    'GCIQL-CubeSingle-Flow-ORS-d50': dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='ors-scale4.0',
        graft='spec19b: ORS extractor -- GCIQL trained on the occupancy-reward-shaping readout from the frozen behavior '
              'density head (ors_reward_scale=4.0, ors_pool_size=4096), cube-single'),
    'GCIQL-CubeDouble-Flow-ORS-d50': dict(csv='bridge', env='cube-double', agent='GCIQL', actor='fql_flow', variant='ors-scale4.0',
        graft='spec19b: ORS extractor, cube-double (ors_reward_scale=4.0)'),
    'GCIQL-CubeSingle-Flow-ORS-scale1.0-d50': dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='ors-scale1.0',
        graft='spec19b ORS reward-scale ablation: ors_reward_scale=1.0 (vs the main cell 4.0)', notes='single seed'),
    'GCIQL-CubeSingle-Flow-ORS-scale1.333333-d50': dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='ors-scale1.333',
        graft='spec19b ORS reward-scale ablation: ors_reward_scale=1.333333', notes='single seed'),

    # --- 2026-08-11/13: density cross-env, actor ablation, hard variant, z-index ---
    'GCIQL-CubeDouble-Flow-Measure-Density-d50': dict(csv='bridge', env='cube-double', agent='GCIQL', actor='fql_flow', variant='measure-density',
        graft='spec18 density, cube-double (the discriminating env)'),
    'GCIQL-Scene-Flow-Measure-Density-d50': dict(csv='bridge', env='scene', agent='GCIQL', actor='fql_flow', variant='measure-density',
        graft='spec18 density, scene'),
    'GCIQL-CubeDouble-Flow-Measure-DensityHard-d50': dict(csv='bridge', env='cube-double', agent='GCIQL', actor='fql_flow', variant='measure-density-hard',
        graft='spec21 verdict: hard immediate target, cube-double',
        exclude_runs=('hx_cd_hard_s0_re',),
        exclude_runs_reason='restore-continuation of s0 (restore_epoch=100k, ran to 600k) — not an independent budget-matched seed'),
    'GCIQL-Scene-Flow-Measure-DensityHard-d50': dict(csv='bridge', env='scene', agent='GCIQL', actor='fql_flow', variant='measure-density-hard',
        graft='spec21 verdict: hard immediate target, scene'),
    'GCIQL-CubeDouble-Flow-Measure-Density-DDPGBC-d50': dict(csv='bridge', env='cube-double', agent='GCIQL', actor='ddpgbc', variant='measure-density',
        graft='density soft + stock DDPG+BC actor (actor-independence)'),
    'GCIQL-Scene-Flow-Measure-Density-DDPGBC-d50': dict(csv='bridge', env='scene', agent='GCIQL', actor='ddpgbc', variant='measure-density',
        graft='density soft + stock DDPG+BC actor'),
    'GCIQL-CubeDouble-Flow-Measure-DensityHard-DDPGBC-d50': dict(csv='bridge', env='cube-double', agent='GCIQL', actor='ddpgbc', variant='measure-density-hard',
        graft='density hard + stock DDPG+BC actor'),
    'GCIQL-Scene-Flow-Measure-DensityHard-DDPGBC-d50': dict(csv='bridge', env='scene', agent='GCIQL', actor='ddpgbc', variant='measure-density-hard',
        graft='density hard + stock DDPG+BC actor'),
    'GCIQL-CubeSingle-Flow-Measure-DensityHard-DDPGBC-d50': dict(csv='bridge', env='cube', agent='GCIQL', actor='ddpgbc', variant='measure-density-hard',
        graft='density hard + stock DDPG+BC actor, cube-single'),
    'GCIQL-CubeSingle-DDPGBC-Measure-Density-d50': dict(csv='bridge', env='cube', agent='GCIQL', actor='ddpgbc', variant='measure-density',
        graft='spec22 mirror: density soft + DDPG+BC on cube-single'),
    'GCIQL-CubeSingle-AWRFlow-Measure-Density-d50': dict(csv='bridge', env='cube', agent='GCIQL', actor='awr_flow', variant='measure-density',
        graft='spec22 screen: density soft + AWR-flow, cube-single'),
    'GCIQL-AntMaze-Medium-Flow-Measure-Density-d50': dict(csv='bridge', env='antmaze', agent='GCIQL', actor='fql_flow', variant='measure-density',
        graft='spec22: density on locomotion (no dwell asymmetry)'),
    'GCIQL-AntMaze-Medium-DDPGBC-Measure-Density-d50': dict(csv='bridge', env='antmaze', agent='GCIQL', actor='ddpgbc', variant='measure-density',
        graft='spec22: density + DDPG+BC on locomotion'),
    'GCIQL-AntMaze-Medium-AWRFlow-Measure-Density-d50': dict(csv='bridge', env='antmaze', agent='GCIQL', actor='awr_flow', variant='measure-density',
        graft='spec22 screen: density + AWR-flow on locomotion'),
    'GCIQL-AntMaze-Medium-Flow-Measure-DensityHard-d50': dict(csv='bridge', env='antmaze', agent='GCIQL', actor='fql_flow', variant='measure-density-hard',
        graft='density hard on locomotion'),
    'GCIQL-AntMaze-Medium-Flow-Measure-DensityHard-DDPGBC-d50': dict(csv='bridge', env='antmaze', agent='GCIQL', actor='ddpgbc', variant='measure-density-hard',
        graft='density hard + DDPG+BC on locomotion'),
    'GCIQL-PointMassMaze-RND-Flow-Measure-Density-d50': dict(csv='bridge', env='point-mass-maze', agent='GCIQL', actor='fql_flow', variant='measure-density',
        graft='spec23 P0: density on ExoRL point-mass-maze (RND buffer)'),
    'GCIQL-PointMassMaze-RND-Flow-Bilinear-d50': dict(csv='bridge', env='point-mass-maze', agent='GCIQL', actor='fql_flow', variant='plain',
        graft='spec23 P0 control: stock GCIQL-flow'),
    'GCIQL-PointMassMaze-RND-Flow-BCOnly-d50': dict(csv='bridge', env='point-mass-maze', agent='GCIQL', actor='fql_flow', variant='bc-only',
        graft='spec23 P0 control: BC-only floor (actor q-term zeroed)'),
    'GCIQL-PointMassMaze-RND-Flow-Measure-DensityZ-d50': dict(csv='bridge', env='point-mass-maze', agent='GCIQL', actor='fql_flow', variant='measure-density-z',
        graft='spec24: z-indexed density (FB latent index instead of goals)'),
    'GCIQL-CubeDouble-Flow-Measure-DensityZ-d50': dict(csv='bridge', env='cube-double', agent='GCIQL', actor='fql_flow', variant='measure-density-z',
        graft='spec24: z-indexed density, cube-double'),

    # ---- spec 25 stage 1 (2026-08-12/13): z-indexed density mechanism cells on top of the
    # z24 revived-index-sensitivity checkpoint. z25_cs_* run names,
    # /home/adityamohan/exp_spec25_stage1/factored-fb/. Knobs isolated via flags.json diff
    # (measure_density_z_tilt / _rdecode_coef / _enc_tau / _bc_hindsight); C3Actor and Full
    # are the checkpoints Q2 (spec27) targets for the E2 reward-readout eval.
    'GCIQL-CubeSingle-Flow-Measure-DensityZ-C1Tilt-d50': dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='z25-c1-tilt',
        graft='spec25 stage1 C1: z-indexed density + measure_density_z_tilt=1.0 only (isolates the tilt term)'),
    'GCIQL-CubeSingle-Flow-Measure-DensityZ-C2Dec-d50': dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='z25-c2-decode',
        graft='spec25 stage1 C2: z-indexed density + measure_density_z_rdecode_coef=1.0 only (reward-decode auxiliary)'),
    'GCIQL-CubeSingle-Flow-Measure-DensityZ-C3Actor-d50': dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='z25-c3-actor',
        graft='spec25 stage1 C3: z-indexed density + measure_density_z_enc_tau=5e-4 + _bc_hindsight=True (actor-extraction '
              'arm -- the checkpoint spec27 Q2 targets for the E2a/E2b reward-readout eval)'),
    'GCIQL-CubeSingle-Flow-Measure-DensityZ-Full-d50': dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='z25-full',
        graft='spec25 stage1 Full: tilt + decode + enc_tau + bc_hindsight combined (C1+C2+C3 stacked)'),

    # ---- spec 29 R-A (2026-08-14): the UNCENTERED dual index. s29_cs_ra_* run names,
    # /home/adityamohan/exp_spec29_ra/factored-fb/. The .812 cell is the KernelRew one --
    # and spec 30 sec.4.8 established that its native goal eval went through the ANALYTIC
    # POINT PROMPT, not the reward-sample encoder, so it validates the dual GOAL code only.
    'GCIQL-CubeSingle-Flow-Measure-DensityZ-RA-Dual-KernelRew-d50': dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='ra-dual-kernel',
        graft='spec29 R-A/kernel: uncentered dual index zhat=(Sigma_hat+lam I)^-1 m_hat + external kernel readout '
              '(cube-single .812; eval used the point prompt -- see spec 30 sec.4.8)'),
    'GCIQL-CubeSingle-Flow-Measure-DensityZ-RA-Dual-LatentRew-d50': dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='ra-dual-latent',
        graft='spec29 R-A/latent: same uncentered dual index + the closed-form B~^T zhat readout'),

    # ---- spec 30 P1 / spec 30a Item 1 (2026-08-16): the CENTERED index screen. ra2_cs_*
    # run names, /home/adityamohan/exp_spec30_p1/factored-fb/. Spec 30 re-measured the
    # geometry and found spec 28's "rank-1 collapse" was largely a softmax-gauge MEAN
    # offset (||mu_B|| 33-103x the centered signal; centered covariance eff rank 6-21/50);
    # centering alone lifts point-index reconstruction AUC .43-.86 -> .99+, and the
    # centered encoder is exactly invariant to r -> r+beta where the uncentered one rotates
    # to near-orthogonal. The two arms UNBUNDLE centering from dualization (spec 30 P2.3).
    # ⚠ CentDual's native goal eval goes through the REWARD-SAMPLE API with an M=256 kernel
    # pool (the point shortcut is a separate diagnostic column only) -- unlike spec 29's
    # .812, so these numbers are not interchangeable with it.
    'GCIQL-CubeSingle-Flow-Measure-DensityZ-RA2-CentPoint-d50': dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='ra2-centered-point',
        graft='spec30 P1 arm A: analytic CENTERED POINT prompt z=sqrt(d)*norm(B(phi(g))-mu_hat) + external kernel '
              'readout -- does mean subtraction alone repair goal prompting?'),
    'GCIQL-CubeSingle-Flow-Measure-DensityZ-RA2-CentDual-d50': dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='ra2-centered-dual',
        graft='spec30 P1 arm B: centered REWARD-SAMPLE index zhat=sqrt(d)*norm((C_hat+lam I)^-1 m_c), M>1, eval '
              'through the M=256 reward-sample pool + external kernel readout -- the general reward interface'),

    # ---- spec 30a Items 2 & 3 (2026-08-17): the two ISOLATION BASELINES, cube-double
    # 500k s0. Run names tdi_cd_exact_s0 / lsif_cd_phi_s0,
    # /home/adityamohan/exp_spec30_i23/factored-fb/. Both exist to remove an alternative
    # explanation for Density-FB's numbers, so BOTH are controls, not candidate methods:
    #   * TD InfoNCE (the ESTIMATOR control) -- spec 30 found the HARD density loss reduces
    #     to its categorical Bellman update under matched conventions, while our flagship
    #     cells are SOFT. The paper-exact arm answers "how does the published method do
    #     here?"; the matched-extractor arm (their critic, our kernel readout + flow actor)
    #     separates estimator from extraction and is the PROMOTION wave, not wave 1.
    #     ⚠ the exact arm runs THEIR conventions throughout (l2-normalized towers, marginal
    #     negative pool from the dataset future-state marginal, goal-contrastive actor with
    #     an MSE goal-BC regularizer at lambda=0.5, geometric goal relabeling), so its
    #     actor/extraction is NOT comparable to the fql_flow density rows.
    #   * LSIF (the PRIVILEGED-COORDINATES control) -- FB's own unnormalized objective on
    #     Density-FB's phi coordinates, matched in everything but LSIF-vs-CE and
    #     unnormalized-vs-normalized. Expected well below Density-FB; if it closes the gap,
    #     the normalization story weakens (spec 30's decision tree).
    # Anchors on this env: density soft .736 | density hard .680 | plain FB .00 | BC .019.
    'GCIQL-CubeDouble-TDInfoNCE-Exact-d50': dict(csv='bridge', env='cube-double', agent='TD-InfoNCE', actor='ddpgbc', variant='paper-exact',
        graft='spec30a Item 2: PAPER-EXACT TD InfoNCE (arXiv:2310.20141) Eq. 10 critic -- immediate term labels the '
              'dynamics next state, bootstrap term a marginal draw soft-labelled by the stop-grad target softmax, ONE '
              'shared negative pool drawn iid from the dataset future-state marginal, l2-normalized towers (implicit '
              'temperature) -- plus their Eq. 11 / App. D.3 actor (goal-contrastive CE at the commanded goal + MSE '
              'goal-BC at lambda=0.5, geometric future relabeling), on Density-FB phi coordinates',
        notes='single seed; estimator control, NOT a variant of ours -- a faithful arm scoring well is a real finding'),
    'GCIQL-CubeDouble-TDInfoNCE-Matched-d50': dict(csv='bridge', env='cube-double', agent='TD-InfoNCE', actor='fql_flow', variant='matched-extractor',
        graft='spec30 P2.1 second arm: their Eq. 10 critic with OUR extraction (external kernel readout over the '
              'l2-normalized classifier + flow actor + actor_q_center) -- separates estimator from extraction',
        notes='PROMOTION WAVE, not launched in wave 1'),
    'GCIQL-CubeDouble-Flow-LSIF-Phi-d50': dict(csv='bridge', env='cube-double', agent='GCIQL', actor='fql_flow', variant='lsif-outcome-space',
        graft='spec30a Item 3: goal-conditioned outcome-space LSIF m(s,a,g,x)=F(s,a,g)^T B(phi(x)) fit by FB\'s own '
              'least-squares importance objective (no softmax) and extracted by external kernel reward integration '
              'q=E_x[m*kg] -- matched to spec18 density in everything except LSIF-vs-CE and unnormalized-vs-normalized',
        notes='single seed; privileged-coordinates confound control'),

    'GCIQL-CubeSingle-Flow-Measure-Density-d50': dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='measure-density',
                                             graft='spec18 (PRIMARY): continuous normalized successor density, two-tower score '
                                                   "f(s,a,g,u)=F(s,a,g).B(u)/tau over queries resampled from rho_x every step "
                                                   '(replaces 17b\'s fixed K-prototype simplex with a Monte-Carlo query pool)'),
    'GCIQL-CubeSingle-Flow-Measure-DensityHard-d50': dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='measure-density-hard',
                                             graft='spec18 ablation: measure_density_hard=True, immediate term = one-hot at the row\'s '
                                                   'own successor (k=I) instead of the eps-kernel softmax -- attributes smoothing vs simplex-TD'),
    'GCIQL-CubeSingle-Flow-Measure-DeltaPartition-K64-d50': dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='measure-delta-partition-k64',
                                             graft='spec18 sec.7 K-sweep control: 17b partition head at K=64 prototypes -- '
                                                   '"prototypes-are-not-load-bearing" check alongside K256/K512/K1024'),
    'GCIQL-CubeSingle-Flow-Measure-DeltaPartition-K512-d50': dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='measure-delta-partition-k512',
                                             graft='spec18 sec.7 K-sweep control: 17b partition head at K=512 prototypes'),
    'GCIQL-CubeSingle-Flow-Measure-DeltaPartition-K1024-d50': dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='measure-delta-partition-k1024',
                                             graft='spec18 sec.7 K-sweep control: 17b partition head at K=1024 prototypes'),
}
