# ====================================================================================================================================
# @file       source_me.bash
# @author     Krzysztof Pierczyk (krzysztof.pierczyk@gmail.com)
# @maintainer Krzysztof Pierczyk (krzysztof.pierczyk@gmail.com)
# @date       Wednesday, 11th November 2020 4:09:10 pm
# @modified   Sunday, 26th June 2022 4:54:26 pm
# @project    vgg-19-testbench
# @brief      Script gathers all functionalities of impl/* scripts. Configuration variables, defined under this header, 
#             should be set appropriately before running. After running the script the whole environment for working with
#             the project will be prepared. Lacking tools and python packages will be installed and the environment variables
#             will be set. Console used to source this script will be ready to use project's python scripts.
#                 
#             In cases when a CPU/Nvidia GPU is used, the console sourcing the script will be switched to the python virtual
#             environment. Thanks to it, the user space will be not cluttered with additional python packages required by the
#             project.
#                 
#             In case of AMD GPU, the script will run a docker container with a fully-isolated environment used to play
#             with project's code. The virtual machine will have the project's folder mounted and the user will be 
#             automatically switched to it so that they can instantly run the training. 
#    
# @note Read 'config/kaggle/README.md' before sourcing this file
#
# @note You should 'source' this script instead of executing. For future compatibility source it from the project's home directory.
#
# @warning Sourcing this script will install lacking tools with sudo, without asking user at the every  installation. Before sourcing
#    the script, read content of the scripts/impl/*.bash files or source this script in an isolated environment.
#
# @note The script assumes that `python3` will be used to run project's code and all python packages are#     installed using 
#   `python3 -m pip install`
#
# @note In case of AMD workflow, the virtual machine is run only when script is sourced with the 'run' argument.
#             
# @copyright Krzysztof Pierczyk © 2022
# ====================================================================================================================================

# Project's home directory
export PROJECT_HOME="$(dirname "$(readlink -f "$BASH_SOURCE")")"

# Type of the tensorflow installation ['c': CPU, 'nvd': NVIDIA GPU, 'amd': AMD GPU]
export TF_VERSION=amd

# Dataset to be downloaded from Kaggle (in form <owner>/<dataset-name>)
export DATASET=moltean/fruits

# Name of the docker image (if AMD GPU used)
export DOCK_IMG='snr-rocm'
export DOCK_IMG_TAG='rocm3.10-tf2.4-rc3-dev'

# ============================================================== Setup ============================================================= #

# Initialize submodules
git submodule update --init --recursive

# Handy aliases
source $PROJECT_HOME/scripts/aliases.bash

# ========================================================== Dependencies ========================================================== #

# Download data set
$PROJECT_HOME/scripts/install/data.bash

# AMD GPU environment
if [[ $TF_VERSION == "amd" ]]; then

    # Install docker
    $PROJECT_HOME/scripts/install/docker_install.bash

    # Build the image and run the container
    if [[ "$1" == "run" ]]; then
        source $PROJECT_HOME/scripts/config/rocm.bash
    fi

# CPU / Nvidia GPU environment
elif [[ $TF_VERSION == "c" || $TF_VERSION == "nvd" ]]; then

    # Install dependancies
    source $PROJECT_HOME/scripts/install/dependencies.bash

    # Setup the environment
    source $PROJECT_HOME/scripts/config/venv.bash    

else
    echo "ERR: Wrong tensorflow version ('$TF_VERSION=c' unknown)"
fi

# ================================================================================================================================== #
