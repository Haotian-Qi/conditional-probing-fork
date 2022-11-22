#!/bin/bash

module load Anaconda3/2019.07
module load GCC/10.2.0

conda create --name darwin -y
conda install -n darwin pip wheel -y
source activate darwin

pip install -r requirements.txt
