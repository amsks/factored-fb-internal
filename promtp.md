 I want to start sbatch jobs on metaword Metaworld picks (answered above, short version): mw-bin-picking as the hard task — the exact structural analog of cube-single (grasp-transport, interaction variable, object-pos hit coordinate), avoiding assembly's insertion-precision confound — and mw-button-press as the easy one, deliberately chosen because it has no grasp-transport, making it the antmaze-style domain-flip control. Pre-registerable prediction for the expert-data caveat: occupancy grounding should be less toxic on unimodal expert trajectories, so if it still floors on bin-picking, the mixing-mass mechanism needs sharpening. Run the triplet (bilinear GCIQL / FP-NoOrtho / occupancy control) × 3 seeds × ~300k steps per task — fits your half-day budget.


  Agents: FB + FlowBC ad DDPGBC, Bilinear GCIQL + FlowBC and DDPGBC, normal GCIQL wiht FLowBC and DDPGBC

  the dataset is  Published dataset in https://huggingface.co/datasets/TommyS
    hen/Factor-FB-Offline,
    tdmpc2 checkpoints of box2d:
    https://huggingface.co/TommyShen/Factor-FB-Offline-tdmpc2