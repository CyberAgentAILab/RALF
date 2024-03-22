source scripts/bin/setup.sh

# User provided args
JOB_DIR="${1:-""}"
DEBUG=${DEBUG:-"False"}

echo -e "\033[31mStart inference_single_data\033[0m"
echo -e "Inference_single_data: \033[0;32m${JOB_DIR}\033[0m with debug=\033[0;32m${DEBUG}\033[0m"
echo "DATASET: ${DATASET}"

OPTIONAL_ARGS=${@:2}
echo "OPTIONAL_ARGS: ${OPTIONAL_ARGS}"

poetry run python3 -m image2layout.train.inference_single_data \
    job_dir=$JOB_DIR \
    result_dir=$JOB_DIR \
    dataset_path="${DATA_ROOT}/${DATASET}" \
    +sampling="top_k" \
    debug=$DEBUG \
    hydra/hydra_logging=none \
    hydra/job_logging=none \
    ${@:2}