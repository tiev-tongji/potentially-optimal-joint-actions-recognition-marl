#!/bin/bash
# Install PyTorch and Python Packages

# conda create -n powqmix2025 python=3.8 -y
# conda activate powqmix2025

echo 'POWQMIX: Install PyTorch and Python dependencies...'

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch   -y
pip install protobuf==3.19.5 sacred==0.7.5 numpy==1.23.1 scipy gym==0.10.8 matplotlib seaborn \
    pyyaml==5.3.1 pygame pytest probscale imageio snakeviz tensorboard-logger pymongo

# pip install git+https://github.com/oxwhirl/smac.git
# Do not need install SMAC anymore. We have integrated SMAC-V1 and SMAC-V2 in pymarl3/envs.

pip install "pysc2>=3.0.0"
pip install "s2clientprotocol>=4.10.1.75800.0"
pip install "absl-py>=0.1.0"
pip install "protobuf<3.21"