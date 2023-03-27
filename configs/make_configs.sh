#!/bin/bash

TASKS=("dep_rel" "named_entities" "ptb_pos" "sst2" "upos" "word_sense")

# Print help and exit
help() {
cat <<EOF
Usage: $0 model_string layers [directory_name]

Creates probing configs for the given model string for the tasks dep_rel, 
named_entities, ptb_pos, sst2, upos and word_sense.

Creates configs for layer{0..n} and layers{1..n}-0, where n is the number
of layers in the model.

You must have the bert768 configs in the configs/ directory for each task.

Parameters:

model_string    the HuggingFace model string to use
layers          the number of layers in the model
[label_name]    (optional) the name to use instead of the model name

Optional flags:

-h, --help      show this help message
EOF

exit 0
}

output_baseline_config() {
    template_filename="${task}-template-layer0.yaml"
    template_file_path="${template_dir}/${template_filename}"
    new_config_filename="${task}-${label_name}-layer${index}.yaml"
    new_config_file_path="${model_task_dir}/${new_config_filename}"

    cat $template_file_path \
    | sed "s|<<tokenizer_model_string>>|${tokenizer_model_string}|g" \
    | sed "s|<<model_string>>|${model_string}|g" - \
    | sed "s|<<index>>|${layer}|g" \
    > "${new_config_file_path}"
}

output_conditional_config() {
    template_filename="${task}-template-layer0-0.yaml"
    template_file_path="${template_dir}/${template_filename}"
    new_config_filename="${task}-${label_name}-layer${index_1}-${index_2}.yaml"
    new_config_file_path="${model_task_dir}/${new_config_filename}"

    cat $template_file_path \
    | sed "s|<<tokenizer_model_string>>|${tokenizer_model_string}|g" \
    | sed "s|<<model_string>>|${model_string}|g" - \
    | sed "s|<<index_1>>|${index_1}|g" \
    | sed "s|<<index_2>>|${index_2}|g" \
    > "${new_config_file_path}"
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

label_name=$(echo "${model_string}" | cut -d '/' -f2)

for task in "${TASKS[@]}"; do
    # Set up paths
    task_dir="configs/${task}"
    template_dir="${task_dir}/template"
    model_task_dir="${task_dir}/${label_name}"
    tokenizer_model_string="google/bert_uncased_L-12_H-768_A-12"
    mkdir -p "${model_task_dir}"

    ## Copy configs
    # Baseline
    for layer in $(seq 0 $layers); do
        index=$layer
        output_baseline_config
    done

    # Conditional against embedding layer
    for layer in $(seq 1 $layers); do
        index_1=$layer
        index_2=0
        output_conditional_config
    done

    # Adjacent conditional
    for layer in $(seq 0 $(($layers - 1))); do
        index_1=$layer
        index_2=$(($layer + 1))
        output_conditional_config
    done
done
