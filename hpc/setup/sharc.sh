#!/bin/bash

module load apps/python/conda
module load dev/gcc/10.1

conda create --name darwin -y
conda install -n darwin pip wheel -y
source activate darwin

pip install -r requirements.txt
