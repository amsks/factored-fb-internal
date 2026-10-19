# Density Matters: Normalized Successor Measures for Zero-Shot Control

## Overview

Density-FB is a forward-backward critic for offline RL whose successor measure is a
distribution over the batch's next states, trained by cross-entropy to a TD target.
A task is given either as a goal state or as reward samples, from which a latent task
vector z is inferred. The repository also contains our FB, HILP and RLDP baselines.

## Installation

The code is based on JAX and tested with Python 3.11.

```bash
uv venv --python 3.11 .venv && uv pip install -e '.[exorl,figures,dev]'
```

OGBench datasets are downloaded on first use.

## Usage

The main implementation of Density-FB is in [impls/critics/density_fb.py](impls/critics/density_fb.py),
and the baselines (FB, HILP and RLDP) are in the same directory. Every run pairs a critic
with an actor (`flowbc` or `ddpgbc`) and a domain:

```bash
# Density-FB on OGBench cube-single, task given as a goal state
python -m impls.main --critic density_fb --actor flowbc --domain cube_single --override 'goal_spec="19:22:0.4"'
# Density-FB with the task inferred from reward samples
python -m impls.main --critic density_fb_z --actor flowbc --domain cube_single --override 'goal_spec="19:22:0.4"' --override eval_relabel_size=10000
# Baselines
python -m impls.main --critic fb --actor flowbc --domain cube_double
python -m impls.main --critic hilp --actor flowbc --domain cube_double
python -m impls.main --critic rldp --actor flowbc --domain cube_double
```

Configs live in `impls/configs/` and `env/domains/`; any value can be changed with
`--override key=value`.

## Acknowledgments

This codebase is built on [OGBench](https://github.com/seohongpark/ogbench)'s reference
implementations. The FB, HILP and RLDP baselines follow the
[TD-JEPA](https://github.com/facebookresearch/td_jepa) implementations and settings.

## Citation

```bibtex
@misc{mohan2026density,
  title={Density Matters: Normalized Successor Measures for Zero-Shot Control},
  author={Mohan, Aditya and Shen, Tongle and Feng, Fan and Agarwal, Siddhant and Chuck, Caleb and Eysenbach, Benjamin and Zhang, Amy},
  year={2026}
}
```

## License

BSD 3-Clause, see [LICENSE](LICENSE).
