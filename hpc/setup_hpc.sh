#!/bin/bash

module load apps/python/conda
module load dev/gcc/10.1

conda create --name python3 -y
source active python3

pip install -r requirements.txt
"$(dirname $0)/copy_data.sh"
