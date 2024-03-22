GPU_ID=$1
JOB_DIR=$2
COND_TYPE=$3
DATASET=$4

bash scripts/run_job/eval_inference_unanno.sh $GPU_ID $JOB_DIR $COND_TYPE $DATASET