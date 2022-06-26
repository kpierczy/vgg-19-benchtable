# ====================================================================================================================================
# @file       docker.bash
# @author     Krzysztof Pierczyk (krzysztof.pierczyk@gmail.com)
# @maintainer Krzysztof Pierczyk (krzysztof.pierczyk@gmail.com)
# @date       Wednesday, 11th November 2020 4:09:10 pm
# @modified   Sunday, 26th June 2022 4:22:54 pm
# @project    vgg-19-testbench
# @brief      Downloads and installs lates stable docker version. If the docker command is found, the script
#             will return instantly.
# 
# @note The script installs docker for amd64 architercture. To change it, modify the docker repository added
#     in the 3rd step.
# @copyright Krzysztof Pierczyk © 2022
# ====================================================================================================================================

# Install docker
if ! which docker > /dev/null; then

    echo -e "\nLOG: Installing docker\n"

    # 1. Install dependancies and download docker's key
    sudo apt-get update
    sudo apt-get install -y  \
            apt-transport-https \
            ca-certificates     \
            curl                \
            gnupg-agent         \
            software-properties-common

    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

    # 2. Check docker key's fingerprint
    if ! sudo apt-key fingerprint 0EBFCD88; then
        echo -e "\nERR: Wrong docker's key.\n"
        return 1
    fi
    
    # 3. Set the 'stable' repository for the docker
    sudo add-apt-repository -y \
        "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
        $(lsb_release -cs) \
        stable"

    # 4. Install the docker
    sudo apt-get update -y
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io

fi