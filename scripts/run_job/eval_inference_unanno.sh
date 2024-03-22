source scripts/bin/setup.sh

DEBUG=False

GPU_ID=$1
JOB_DIR=$2
COND_TYPE=$3
DATASET=$4

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

_cond_type="uncond"
test_split="with_no_annotation"
_ARGS="cond_type=${_cond_type} test_split=${test_split} no_anno_dataset_name=${DATASET}"

source scripts/bin/inference_unanno.sh $JOB_DIR $_ARGS
source scripts/bin/eval_unanno.sh $JOB_DIR    

poetry run python -m image2layout.train.helpers.export_score_to_tex_unanno --root $JOB_DIR

