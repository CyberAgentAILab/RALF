source scripts/bin/setup.sh

DEBUG="False"

GPU_ID=$1
JOB_DIR=$2
COND_TYPE=$3
DATASET=$4

# User provided args
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


TEST_SPLIT="test"


_ARGS="cond_type=${COND_TYPE} test_split=${TEST_SPLIT}"

if [[ $COND_TYPE == "refinement" ]]; then
    _ARGS="${_ARGS} sampling=deterministic"
fi

source scripts/bin/inference.sh $JOB_DIR $_ARGS

EXTRA_ARGS="--run-on-local"
source scripts/bin/eval.sh $JOB_DIR $EXTRA_ARGS

poetry run python -m image2layout.train.helpers.export_score_to_tex --root $JOB_DIR