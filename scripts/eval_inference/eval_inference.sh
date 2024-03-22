GPU_ID=$1
JOB_DIR=$2
COND_TYPE=$3
DATASET=$4

bash scripts/run_job/eval_inference.sh $GPU_ID $JOB_DIR $COND_TYPE $DATASET
# for example,
# bash scripts/run_job/eval_inference.sh 0 "cache/training_logs/autoreg_uncond_pku10" uncond pku10