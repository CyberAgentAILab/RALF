source scripts/bin/setup.sh

# User provided args
JOB_DIR="${1:-""}"

echo -e "\033[31mStart evaluation\033[0m"
DATASET=${DATASET:-"pku10"}
DEBUG=${DEBUG:-"False"}


# Result directory
echo -e "Evaluation: \033[0;32m${JOB_DIR}\033[0m with debug=\033[0;32m${DEBUG}\033[0m"

SAVE_SCORE_DIR="${PWD}/tmp/scores/${DATASET}"

EXTRA_ARGS=${EXTRA_ARGS:-""}
if [[ $DEBUG == True ]]; then
    EXTRA_ARGS+=" --debug"
fi

for dir in "$JOB_DIR"/no_anno_data_*"$DATASET"*; do

    echo "dir: ${dir}"
    if [ -d "$dir" ]; then
        INPUT_DIR=$dir

        varnames=("SAVE_SCORE_DIR" "EXTRA_ARGS" "INPUT_DIR" "DATASET")
        for varname in "${varnames[@]}"; do
            declare value="${!varname}"
            printf "$TEMPLATE" "$varname" "$value"
        done

        poetry run python eval_unanno.py \
            --input-dir "${INPUT_DIR}" \
            --save-score-dir "${SAVE_SCORE_DIR}" \
            --dataset-path "${DATA_ROOT}/${DATASET}" \
            $EXTRA_ARGS
    fi
done
