source scripts/bin/setup.sh

# User provided args
JOB_DIR="${1:-""}"

echo -e "\033[31mStart evaluation\033[0m"
DATASET=${DATASET:-"pku10"}
DEBUG=${DEBUG:-"False"}


# Result directory
echo -e "Evaluation: \033[0;32m${JOB_DIR}\033[0m with debug=\033[0;32m${DEBUG}\033[0m"

SAVE_SCORE_DIR="${PWD}/tmp/scores/${DATASET}"
FID_WEIGHT_DIR="${PWD}/cache/PRECOMPUTED_WEIGHT_DIR/fidnet/${DATASET}"

EXTRA_ARGS=${EXTRA_ARGS:-""}
if [[ $DEBUG == True ]]; then
    EXTRA_ARGS+=" --debug"
fi

for dir in "$JOB_DIR"/generated_samples_*; do

    echo "dir: ${dir}"
    if [ -d "$dir" ]; then
        INPUT_DIR=$dir

        varnames=("SAVE_SCORE_DIR" "FID_WEIGHT_DIR" "EXTRA_ARGS" "INPUT_DIR")
        for varname in "${varnames[@]}"; do
            declare value="${!varname}"
            printf "$TEMPLATE" "$varname" "$value"
        done

        poetry run python eval.py \
            --input-dir "${INPUT_DIR}" \
            --fid-weight-dir "${FID_WEIGHT_DIR}" \
            --save-score-dir "${SAVE_SCORE_DIR}" \
            --dataset-path "${DATA_ROOT}/${DATASET}" \
            $EXTRA_ARGS
    fi
done
