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

    # --- 1d: GC-SimpleSF (Chua et al.). Rung 2 isolates the architecture against bilinear GCIQL
    # (0.904); rung 3 grounds psi in the reward and isolates the grounding against rung 2.
    'GCIQL-State-Flow-SimpleSF-Arch':     dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='simplesf-arch', graft='SimpleSF trunk split (b=h/||h||, Psi=head(h)), psi learned end-to-end; scalar IQL TD'),
    'GCIQL-State-Flow-SimpleSF-Grounded': dict(csv='bridge', env='cube', agent='GCIQL', actor='fql_flow', variant='simplesf-grounded', graft="Chua et al.'s cell: + psi trained ONLY by reward regression <sg[b],psi(g)>"),

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
}
