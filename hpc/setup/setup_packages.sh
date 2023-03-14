#!/bin/bash

source "$(dirname $0)/setup_env.sh"

conda create --name darwin -y
conda install -n darwin pip wheel -y
source activate darwin

pip install -r requirements.txt
