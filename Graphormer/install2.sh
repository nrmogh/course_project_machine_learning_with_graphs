#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


# Replace the first section of install.sh with:
pip install torch==2.2.0+cu121 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install lmdb
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
pip install torch-geometric
pip install tensorboardX==2.4.1
pip install ogb==1.3.2
pip install rdkit
pip install dgl -f https://data.dgl.ai/wheels/cu121/repo.html

# Then continue with fairseq installation
cd fairseq
pip install -e . 
python setup.py build_ext --inplace
