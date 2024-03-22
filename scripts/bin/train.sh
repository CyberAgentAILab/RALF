#!/usr/bin/env bash

echo -e "\033[31mStart training\033[0m"

# load environment variables
source scripts/bin/setup.sh
echo $*

# some constants
NOW=$(date "+%Y%m%d%H%M%S")
DATASETS=("pku10" "cgl")

# optional variables
DEBUG=${DEBUG:-"False"}  # comsume less data and resume in 1 epoch
SEEDS=${SEEDS:-1}  # "0,1,2" if you want to try seed 0-2

EXP_ID=${EXP_ID:-""}

DATASET=$1
EXPERIMENT=$2
if [ "${DATASET}" = "" ] || [ "${EXPERIMENT}" = "" ]; then
    echo "Please launch with {DATASET} {EXPERIMENT} {OTHER_ARGS}"
    exit;
fi
if [[ ! $(echo ${DATASETS[@]} | fgrep -w $DATASET) ]]; then
    echo "DATASET: ${DATASET} is not implemented"
    exit;
fi
if [ "${3}" != "" ]; then
    ADDITIONAL_ARGS="${ADDITIONAL_ARGS} ${@:3}"
fi

JOB_NAME="${DATASET}_${EXPERIMENT}_${EXP_ID}"
JOB_NAME=$(echo $JOB_NAME | tr "_" "-")
DATA_DIR="${DATA_ROOT}/${DATASET}"

echo "Debug is ${DEBUG}"
if [[ $DEBUG == "True" ]]; then
    JOB_DIR="tmp/jobs_${EXPERIMENT}_debug/${DATASET}/${EXP_ID}/${EXPERIMENT}_${NOW}"
else
    JOB_DIR="tmp/jobs_${EXPERIMENT}/${DATASET}/${EXP_ID}/${EXPERIMENT}_${NOW}"
fi


SHARED_DEFAULT_ARGS="+experiment=${EXPERIMENT} job_dir=${JOB_DIR} dataset=${DATASET} dataset.data_dir=${DATA_DIR} seed=${SEEDS}"
ARGS="${SHARED_DEFAULT_ARGS} ${ADDITIONAL_ARGS} debug=${DEBUG}"
echo "ARGS=$ARGS"

PYTHON_MODULE="image2layout.train.train"

export MASTER_PORT=$RANDOM

OMP_NUM_THREADS=${OMP_NUM_THREADS} poetry run python -m ${PYTHON_MODULE} ${ARGS}
