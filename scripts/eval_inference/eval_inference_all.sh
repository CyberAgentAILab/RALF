GPU_ID=$1

# Autoreg -- CGL
DATASET="cgl"
JOB_DIR="./cache/training_logs/autoreg_uncond_cgl"
COND_TYPE="uncond"
bash scripts/run_job/eval_inference.sh $GPU_ID $JOB_DIR $COND_TYPE $DATASET


JOB_DIR="./cache/training_logs/autoreg_c_cgl"
COND_TYPE="c"
bash scripts/run_job/eval_inference.sh $GPU_ID $JOB_DIR $COND_TYPE $DATASET


JOB_DIR="./cache/training_logs/autoreg_cwh_cgl"
COND_TYPE="cwh"
bash scripts/run_job/eval_inference.sh $GPU_ID $JOB_DIR $COND_TYPE $DATASET



JOB_DIR="./cache/training_logs/autoreg_partial_cgl"
COND_TYPE="partial"
bash scripts/run_job/eval_inference.sh $GPU_ID $JOB_DIR $COND_TYPE $DATASET


JOB_DIR="./cache/training_logs/autoreg_refinement_cgl"
COND_TYPE="refinement"
bash scripts/run_job/eval_inference.sh $GPU_ID $JOB_DIR $COND_TYPE $DATASET


JOB_DIR="./cache/training_logs/autoreg_relation_cgl"
COND_TYPE="relation"
bash scripts/run_job/eval_inference.sh $GPU_ID $JOB_DIR $COND_TYPE $DATASET



# Autoreg -- PKU
DATASET="pku10"
JOB_DIR="./cache/training_logs/autoreg_uncond_pku10"
COND_TYPE="uncond"
bash scripts/run_job/eval_inference.sh $GPU_ID $JOB_DIR $COND_TYPE $DATASET


JOB_DIR="./cache/training_logs/autoreg_c_pku10"
COND_TYPE="c"
bash scripts/run_job/eval_inference.sh $GPU_ID $JOB_DIR $COND_TYPE $DATASET


JOB_DIR="./cache/training_logs/autoreg_cwh_pku10"
COND_TYPE="cwh"
bash scripts/run_job/eval_inference.sh $GPU_ID $JOB_DIR $COND_TYPE $DATASET


JOB_DIR="./cache/training_logs/autoreg_partial_pku10"
COND_TYPE="partial"
bash scripts/run_job/eval_inference.sh $GPU_ID $JOB_DIR $COND_TYPE $DATASET


JOB_DIR="./cache/training_logs/autoreg_refinement_pku10"
COND_TYPE="refinement"
bash scripts/run_job/eval_inference.sh $GPU_ID $JOB_DIR $COND_TYPE $DATASET


JOB_DIR="./cache/training_logs/autoreg_relation_pku10"
COND_TYPE="relation"
bash scripts/run_job/eval_inference.sh $GPU_ID $JOB_DIR $COND_TYPE $DATASET



# RALF -- CGL
DATASET="cgl"
JOB_DIR="./cache/training_logs/ralf_uncond_cgl"
COND_TYPE="uncond"
bash scripts/run_job/eval_inference.sh $GPU_ID $JOB_DIR $COND_TYPE $DATASET

JOB_DIR="./cache/training_logs/ralf_c_cgl"
COND_TYPE="c"
bash scripts/run_job/eval_inference.sh $GPU_ID $JOB_DIR $COND_TYPE $DATASET

JOB_DIR="./cache/training_logs/ralf_cwh_cgl"
COND_TYPE="cwh"
bash scripts/run_job/eval_inference.sh $GPU_ID $JOB_DIR $COND_TYPE $DATASET

JOB_DIR="./cache/training_logs/ralf_partial_cgl"
COND_TYPE="partial"
bash scripts/run_job/eval_inference.sh $GPU_ID $JOB_DIR $COND_TYPE $DATASET

JOB_DIR="./cache/training_logs/ralf_refinement_cgl"
COND_TYPE="refinement"
bash scripts/run_job/eval_inference.sh $GPU_ID $JOB_DIR $COND_TYPE $DATASET

JOB_DIR="./cache/training_logs/ralf_relation_cgl"
COND_TYPE="relation"
bash scripts/run_job/eval_inference.sh $GPU_ID $JOB_DIR $COND_TYPE $DATASET




# # RALF -- PKU
DATASET="pku10"
# JOB_DIR="./cache/training_logs/ralf_uncond_pku10"
# COND_TYPE="uncond"
# bash scripts/run_job/eval_inference.sh $GPU_ID $JOB_DIR $COND_TYPE $DATASET

# JOB_DIR="./cache/training_logs/ralf_c_pku10"
# COND_TYPE="c"
# bash scripts/run_job/eval_inference.sh $GPU_ID $JOB_DIR $COND_TYPE $DATASET

# JOB_DIR="./cache/training_logs/ralf_cwh_pku10"
# COND_TYPE="cwh"
# bash scripts/run_job/eval_inference.sh $GPU_ID $JOB_DIR $COND_TYPE $DATASET

# JOB_DIR="./cache/training_logs/ralf_partial_pku10"
# COND_TYPE="partial"
# bash scripts/run_job/eval_inference.sh $GPU_ID $JOB_DIR $COND_TYPE $DATASET

# JOB_DIR="./cache/training_logs/ralf_refinement_pku10"
# COND_TYPE="refinement"
# bash scripts/run_job/eval_inference.sh $GPU_ID $JOB_DIR $COND_TYPE $DATASET

JOB_DIR="./cache/training_logs/ralf_relation_pku10"
COND_TYPE="relation"
bash scripts/run_job/eval_inference.sh $GPU_ID $JOB_DIR $COND_TYPE $DATASET
