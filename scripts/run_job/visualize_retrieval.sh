GPU_ID="${1:-"0"}"
export CUDA_VISIBLE_DEVICES=$GPU_ID

OMP_NUM_THREADS=2 poetry run python3 -m  image2layout.train.visualize_retrieval