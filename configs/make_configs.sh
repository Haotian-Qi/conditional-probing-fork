#!/bin/bash

TASKS=("dep_rel" "named_entities" "ptb_pos" "sst2" "upos" "word_sense")

# Print help and exit
help() {
cat <<EOF
Usage: $0 model_string layers

Creates probing configs for the given model string for the tasks dep_rel, 
named_entities, ptb_pos, sst2, upos and word_sense.

Creates for layer{0..n} and layers{1..n}-0, where n is the number of layers
in the model.

You must have the bert768 configs in the configs/ directory for each task.

Parameters:

model_string    the HuggingFace model string to use
layers          the number of layers in the model

Optional flags:

-h, --help      show this help message
EOF

exit 0
}

output_config () {
    new_filename=$(echo $(basename $bert768_config_path) | sed "s/bert768/${model_name}/" -)
    cat $bert768_config_path | sed "s,google/bert_uncased_L-12_H-768_A-12,$model_string,g" - > "${model_task_dir}/${new_filename}"
}

# Print help if no arguments or flags passed
if [[ -z "$*" ]]; then
    help
fi

# Collect parameters
declare -a params=()

# Iterate over command line arguments
for arg; do
    case $arg in
    -h|--help)
        help
        ;;
    -*)
        >&2 echo "error: unknown flag $arg"
        exit 1
        ;;
    *)
        params+=($arg)
        ;;
    esac
done

# Check number of arguments
if [[ ${#params[@]} -ne 2 ]]; then
    >&2 echo "error: too many arguments, expected 2"
fi

model_string="${params[0]}"
layers="${params[1]}"

# Check number of layers
if [[ $layers -gt 12 ]]; then
    >&2 echo "error: Number of layers ($layers) is greater than number of layers in bert768 (12)"
fi

model_name=$(echo "${model_string}" | cut -d '/' -f2)

for task in "${TASKS[@]}"; do
    # Set up paths
    task_dir="configs/${task}"
    model_task_dir="${task_dir}/${model_name}"
    bert768_task_dir="${task_dir}/bert768"
    mkdir -p "${model_task_dir}"

    # Copy configs

    # Baseline
    for layer in $(seq 0 $layers); do
        bert768_config_path=$bert768_task_dir/*layer$layer.yaml
        output_config
    done

    # Conditional
    for layer in $(seq 1 $layers); do
        bert768_config_path=$bert768_task_dir/*layer$layer-0.yaml
        output_config
    done
done
