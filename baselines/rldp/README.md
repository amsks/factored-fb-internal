# RLDP code [Submitted to ICLR 2025]

This repository provides a PyTorch implementation for RLDP. 


## Dependencies 
```
"safetensors>=0.4.5",
"torch>=2.3",
"tyro>=0.9.0",
"wandb>=0.19.0"
"tensordict>=0.6.0"
```


## Download ExoRL datasets

[ExoRL](https://github.com/denisyarats/exorl) has been widely used to train offline algorithms. We provide the code for training RLDP on standard domains such as `walker`, `cheetah`, `quadruped` and `pointmass`. We use the standard tasks in `dm_control`, but you can easily update the script to run the full set of tasks defined in `ExoRL`.

## Train RLDP representations + zero shot RL policy

```
export PYTHONPATH=$PYTHONPATH:.
```

```
python examples/rldp_train_dmc.py --domain_name walker --dataset_root <exorl_dataset_path> 
```


## Train RLDP representations + zero-shot RL policies


# Citation
This code is primarily built on the following repository which is licensed the CC BY-NC 4.0 license:

```
@article{tirinzoni2024metamotivo,
  title={Zero-shot Whole-Body Humanoid Control via Behavioral Foundation Models},
  author={Tirinzoni, Andrea and Touati, Ahmed and Farebrother, Jesse and Guzek, Mateusz and Kanervisto, Anssi and Xu, Yingchen and Lazaric, Alessandro and Pirotta, Matteo},
}
```

