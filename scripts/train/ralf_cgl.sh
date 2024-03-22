GPU_ID=$1
TASK=$2
bash scripts/run_job/end_to_end.sh $GPU_ID ralf cgl $TASK
