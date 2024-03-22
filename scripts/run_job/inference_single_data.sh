source scripts/bin/setup.sh

DEBUG="False"

JOB_DIR="${2:-"./cache/training_logs/ralf_uncond_cgl"}"
DATASET="${3:-"cgl"}"

SAMPLE_ID="${4:-""}"

if [ -z "$SAMPLE_ID" ]; then
    SAMPLE_ID="None"
fi

# User provided args
GPU_ID="${1:-"0"}"
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Print args
printf "\n=== Show user-defined variables===\n"
TEMPLATE="%s=\033[0;32m%s\033[0m\n"
varnames=("GPU_ID" "JOB_DIR" "DATASET")
for varname in "${varnames[@]}"; do
    declare value="${!varname}"
    printf "$TEMPLATE" "$varname" "$value"
done
printf "===========\n\n"

_ARGS="cond_type=uncond test_split=with_no_annotation num_seeds=1 sample_id=${SAMPLE_ID}"

source scripts/bin/inference_single_data.sh $JOB_DIR $_ARGS
