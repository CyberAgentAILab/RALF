GPU_ID=$1

# Autoreg -- CGL
DATASET="cgl"
JOB_DIR="./cache/training_logs/autoreg_uncond_cgl"
COND_TYPE="uncond"
bash scripts/run_job/eval_inference_unanno.sh $GPU_ID $JOB_DIR $COND_TYPE $DATASET


# # Autoreg -- PKU
# DATASET="pku10"
# JOB_DIR="./cache/training_logs/autoreg_uncond_pku10"
# COND_TYPE="uncond"
# bash scripts/run_job/eval_inference_unanno.sh $GPU_ID $JOB_DIR $COND_TYPE $DATASET


# RALF -- CGL
DATASET="cgl"
JOB_DIR="./cache/training_logs/ralf_uncond_cgl"
COND_TYPE="uncond"
bash scripts/run_job/eval_inference_unanno.sh $GPU_ID $JOB_DIR $COND_TYPE $DATASET


# # RALF -- PKU
# DATASET="pku10"
# JOB_DIR="./cache/training_logs/ralf_uncond_pku10"
# COND_TYPE="uncond"
# bash scripts/run_job/eval_inference_unanno.sh $GPU_ID $JOB_DIR $COND_TYPE $DATASET
