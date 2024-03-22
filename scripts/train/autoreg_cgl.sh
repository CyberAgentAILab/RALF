GPU_ID=$1
TASK=$2
bash scripts/run_job/end_to_end.sh $GPU_ID autoreg cgl $TASK
# bash scripts/run_job/end_to_end.sh 2 dsgan cgl uncond