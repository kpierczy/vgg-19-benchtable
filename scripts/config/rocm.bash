# ====================================================================================================================================
# @file       rocm.bash
# @author     Krzysztof Pierczyk (krzysztof.pierczyk@gmail.com)
# @maintainer Krzysztof Pierczyk (krzysztof.pierczyk@gmail.com)
# @date       Wednesday, 11th November 2020 4:09:10 pm
# @modified   Sunday, 26th June 2022 4:24:52 pm
# @project    vgg-19-testbench
# @brief      Builds and runs the docker image used to work with AMD-GPU-base systems.
#
# @copyright Krzysztof Pierczyk © 2022
# ====================================================================================================================================

# Build the docker image with the preconfigured environment
if [[ "$(sudo docker images $DOCK_IMG:$DOCK_IMG_TAG | wc -l)" == "1" ]]; then
    printf "\nLOG: Building a docker image for ROCm environment.\n"
    builder="sudo docker build                                      \
        -f $PROJECT_HOME/scripts/docker/rocm.Dockerfile             \
        -t $DOCK_IMG:$DOCK_IMG_TAG                                  \
        --build-arg PROJECT_HOME=${PROJECT_HOME}                    \
        --build-arg TF_VERSION=${TF_VERSION}                        \
        --build-arg DATASET=${DATASET}                              \
        --build-arg KAGGLE_CONFIG_DIR={$PROJECT_HOME/config/kaggle} \
        --build-arg DOCK_IMG_TAG=${DOCK_IMG_TAG}"

    if ! $builder .; then
        printf "\nERR: Building a docker image failes.\n"
        return
    fi
fi

# Run the container
printf "\nLOG: Running virtual environment for ROCm tensorflow \n\n"
sudo docker run                            \
    -it                                    \
    --rm                                   \
    --name $DOCK_IMG                       \
    --network=host                         \
    --device=/dev/kfd                      \
    --device=/dev/dri                      \
    --ipc=host                             \
    --shm-size 16G                         \
    --group-add video                      \
    --cap-add=SYS_PTRACE                   \
    --security-opt seccomp=unconfined      \
    -v $HOME/dockerx:/dockerx              \
    -v $PROJECT_HOME:$PROJECT_HOME         \
    $DOCK_IMG:$DOCK_IMG_TAG
