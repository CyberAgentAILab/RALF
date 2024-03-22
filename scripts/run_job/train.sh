source scripts/bin/setup.sh

# User provided args
GPU_ID="${1:-"0"}"
export CUDA_VISIBLE_DEVICES=$GPU_ID
EXPERIMENT="${2:-"cglgan"}"
DATASET="${3:-"pku10"}"
EXP_ID="${4:-"001"}"

# Load pre-defined config file
CONFIG_PATH="configs/${EXPERIMENT}_${DATASET}/${EXP_ID}.sh"
echo "CONFIG_PATH: ${CONFIG_PATH}"
if [ -f "${CONFIG_PATH}" ]; then
    echo "Load ${CONFIG_PATH}"
    source "${CONFIG_PATH}"
else
    echo "Read no additional config file."
    exit;
fi

# Print args
printf "\n=== Show user-defined variables===\n"
TEMPLATE="%s=\033[0;32m%s\033[0m\n"
varnames=("GPU_ID" "EXPERIMENT" "DATASET" "EXP_ID" "ADDITIONAL_ARGS" "DEBUG")
for varname in "${varnames[@]}"; do
    declare value="${!varname}"
    printf "$TEMPLATE" "$varname" "$value"
done
printf "===========\n\n"

# Jobs
source scripts/bin/train.sh $DATASET $EXPERIMENT
