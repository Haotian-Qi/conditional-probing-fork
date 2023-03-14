#!/bin/bash

. "$(dirname $0)/activate_modules.sh"

conda create --name darwin -y
conda install -n darwin pip wheel -y
. "$(dirname $0)/activate_env.sh"

pip install -r requirements.txt
