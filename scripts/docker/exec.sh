docker run --gpus all \
    --rm -it --name img2layout_$(($RANDOM % 1000 + 1000)) \
    --shm-size=300g \
    --memory=300g \
    -e HOSTNAME=$HOSTNAME \
    -v $PWD:/src \
    img2layout bash