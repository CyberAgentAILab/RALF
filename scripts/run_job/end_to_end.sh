source scripts/bin/setup.sh

# User provided args
GPU_ID="${1:-"0"}"
export CUDA_VISIBLE_DEVICES=$GPU_ID
EXPERIMENT="${2:-"cglgan"}"
DATASET="${3:-"pku10"}"
TASK_NAME="${4:-"001"}"

# Load pre-defined config file
CONFIG_PATH="configs/${EXPERIMENT}_${DATASET}/${TASK_NAME}.sh"
if [ -f "${CONFIG_PATH}" ]; then
    echo "Load CONFIG_PATH: ${CONFIG_PATH}"
    source "${CONFIG_PATH}"
else
    echo "Read no additional config file."
    exit 1
fi

# Print args
printf "\n=== Show user-defined variables===\n"
TEMPLATE="%s=\033[0;32m%s\033[0m\n"
varnames=("GPU_ID" "EXPERIMENT" "DATASET" "TASK_NAME" "ADDITIONAL_ARGS" "DEBUG")
for varname in "${varnames[@]}"; do
    declare value="${!varname}"
    printf "$TEMPLATE" "$varname" "$value"
done
printf "===========\n\n"

# Jobs
### Training
source scripts/bin/train.sh $DATASET $EXPERIMENT
# ### Inference to generate layouts for an evaluation
# source scripts/bin/inference.sh $JOB_DIR "cond_type=${TASK_NAME}"
# ### Evaluation
# source scripts/bin/eval.sh $JOB_DIR
# ### Export a score to LaTeX
# poetry run python -m image2layout.train.helpers.export_score_to_tex --root $JOB_DIR