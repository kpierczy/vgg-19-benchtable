# ====================================================================================================================================
# @file       aliases.bash
# @author     Krzysztof Pierczyk (krzysztof.pierczyk@gmail.com)
# @maintainer Krzysztof Pierczyk (krzysztof.pierczyk@gmail.com)
# @date       Wednesday, 11th November 2020 4:09:10 pm
# @modified   Sunday, 26th June 2022 4:25:51 pm
# @project    vgg-19-testbench
# @brief      Sourcing this script will provide the user's terminal with a set of handy commands used widely
#             during working on the project. The main reason to use these commands is to reduce number of
#             click's performed by the user when interacting with AMD-GPU based system :*
# 
# @note Modify the file to fit your needs!
# @copyright Krzysztof Pierczyk © 2022
# ====================================================================================================================================

# ======================================================== Docker utilities ======================================================== #

if [[ $TF_VERSION == "amd" ]]; then
if which docker > /dev/null; then

    # Show running containers
    alias dps='sudo docker ps'

    # Stop and remove all containers
    alias drm='                                                  \
        if [[ $(sudo docker ps -a -q) != "" ]] > /dev/null; then \
            sudo docker stop $(sudo docker ps -a -q) &&          \
            sudo docker rm $(sudo docker ps -a -q);              \
        fi'

    # Stop and remove all containers. Prune intermediate images.
    alias prune='                                                \
        if [[ $(sudo docker ps -a -q) != "" ]] > /dev/null; then \
            sudo docker stop $(sudo docker ps -a -q) &&          \
            sudo docker rm $(sudo docker ps -a -q);              \
        fi && sudo docker image prune'

    # Show docker images
    alias dimg='sudo docker images'

    # Remove a docker image
    alias dimgrm='sudo docker rmi'

    # Executes an additional bash in the running environment
    if [[ $DOCK_IMG != "" ]]; then
        alias dexec="sudo docker exec -it $(dps | awk -v i=$DOCK_IMG '/i/ {print $1}') bash"
    else
        alias dexec="sudo docker exec -it $(dps | awk '/snr-rocm/ {print $1}') bash"
    fi
fi
fi

# ====================================================== Neural nets workflow ====================================================== #

# Clear all models logs, history files and test evaluations from the given model's directory
nncl() {
    if [[ $2 == "" ]]; then
        sudo rm -rf models/$1/*
    else
        sudo rm -rf models/$1/$2/logs
        sudo rm -rf models/$1/$2/history
        sudo rm -rf models/$1/$2/test
        sudo rm -rf models/$1/$2/weights
    fi
}

# Opens tensorboard with data of the given model's directory
tboard() {
    SPEC="$2/training:models/$1/$2/logs,$2/test:models/$1/$2/test"
    for run in "$@"; do
        if [[ $run == "$0" || $run == "$1" || $run == "$2" ]]; then
            continue
        fi
        SPEC="${SPEC},$run/training:models/$1/$run/logs,$run/test:models/$1/$run/test"
    done
    tensorboard --logdir_spec $SPEC
}

# ================================================================================================================================== #
