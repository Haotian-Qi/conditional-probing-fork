#!/bin/bash

module load Anaconda3/5.3.0
module load GCC/10.2.0

conda create --name darwin -y
conda install -n darwin pip -y
source active darwin

pip install -r requirements.txt
"$(dirname $0)/copy_data.sh"
