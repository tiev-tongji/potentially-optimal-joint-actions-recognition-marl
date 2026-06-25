# Potentially Optimal Joint Actions Recognition for Cooperative Multi-Agent Reinforcement Learning

This repository contains the code release for the paper "Potentially Optimal Joint Actions Recognition for Cooperative Multi-Agent Reinforcement Learning".

The implementation is based on PyMARL2 and includes the POWQMIX algorithm, baseline configurations, and environments used in the experiments.

## Repository Layout

- `src/`: training code, agents, learners, controllers, environment wrappers, and experiment configs.
- `src/config/algs/`: algorithm configuration files.
- `src/config/envs/`: environment configuration files.
- `install_dependencies.sh`: Python and PyTorch dependency installer.
- `install_sc2.sh`: StarCraft II 4.10 and SMAC map installer.

## Environment Setup

The experiments were run with Python 3.8.18.

```bash
conda create -n powqmix2025 python=3.8 -y
conda activate powqmix2025
bash ./install_dependencies.sh
```

For SMAC and SMACv2 experiments, install StarCraft II and the bundled maps:

```bash
bash ./install_sc2.sh
```

For the highway intersection environment, also install:

```bash
pip install gym==0.26
pip install ray
```

## Running Experiments

Matrix game:

```bash
python3 src/main.py --config=powqmix --env-config=one_step_matrix_game with epsilon_start=1 epsilon_finish=1 no_weighted_steps=0
```

Difficulty-enhanced predator-prey:

```bash
python3 src/main.py --config=powqmix --env-config=stag_hunt with env_args.map_name=stag_hunt env_args.miscapture_punishment=-3 epsilon_anneal_time=500000 t_max=5050000 weighted_qmix_threshold=1
```

SMAC:

```bash
python3 src/main.py --config=powqmix --env-config=sc2 with env_args.map_name='3s_vs_5z' epsilon_anneal_time=500000 t_max=10050000
```

SMACv2:

```bash
python3 src/main.py --config=powqmix --env-config=sc2_v2_zerg with epsilon_anneal_time=500000 t_max=10050000
```

Intersection:

```bash
python3 src/main.py --config=powqmix --env-config=intersec with epsilon_start=0.1 epsilon_finish=0.1 t_max=5050000 weighted_qmix_threshold=0.1
```

## Computing Infrastructure

The paper experiments used the following setup:

- CPU: AMD Ryzen 9 7950X 16-Core Processor
- GPU: NVIDIA GeForce RTX 3090 24GB
- Memory: 512GB
- Storage: 1TB
- Operating system: Ubuntu 20.04.6 LTS
- Python: 3.8.18

Library versions are specified in `install_dependencies.sh` and `install_sc2.sh`.

## Notes

- StarCraft II version: 4.10.
- SMAC difficulty: 7.
- Experiment configs are resolved from `src/config/algs` and `src/config/envs`.
- Training outputs are ignored by Git via `.gitignore`.

## Acknowledgement

This codebase builds on PyMARL2, the open-source implementation for "Rethinking the Implementation Tricks and Monotonicity Constraint in Cooperative Multi-Agent Reinforcement Learning".

```bibtex
@article{hu2021rethinking,
  title={Rethinking the Implementation Tricks and Monotonicity Constraint in Cooperative Multi-Agent Reinforcement Learning},
  author={Jian Hu and Siyang Jiang and Seth Austin Harding and Haibin Wu and Shih-wei Liao},
  year={2021},
  eprint={2102.03479},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```
