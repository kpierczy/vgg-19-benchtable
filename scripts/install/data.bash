# ====================================================================================================================================
# @file       data.bash
# @author     Krzysztof Pierczyk (krzysztof.pierczyk@gmail.com)
# @maintainer Krzysztof Pierczyk (krzysztof.pierczyk@gmail.com)
# @date       Wednesday, 11th November 2020 4:09:10 pm
# @modified   Sunday, 26th June 2022 4:18:37 pm
# @project    vgg-19-testbench
# @brief      Downloads a 'fruits-360' dataset from the kaggle.com. The dataset will be downloaded and extracted
#             to the data/ folder.
# 
# @note This script used Kaggle Python API. If it is not present in the system, the script will install it automatically.
# @note Read 'config/kaggle/README.md' before running this file
#
# @copyright Krzysztof Pierczyk © 2022
# ====================================================================================================================================

# Check if python 3 is available 
if ! which python3 > /dev/null; then
    echo -e "\nERR: No 'python3' available. Cannot run kaggle API. Exiting...\n"
    return 1
fi

# ======================================================= Dependencies check ======================================================= #

# Install Kaggle API
if ! which kaggle > /dev/null; then
    echo -e "\nLOG: kaggle API will be installed\n"
    python3 -m pip install --user kaggle
fi

# ======================================================== Dataset download ======================================================== #

# Download dataset
if [[ ! -d $PROJECT_HOME/data/fruits-360 && ! -f $PROJECT_HOME/data/fruits.zip ]]; then
    echo -e "\nLOG: Downloading $DATASET dataset\n"
    kaggle datasets download -p $PROJECT_HOME/data $DATASET
fi

# Unzip downloaded data
if [[ ! -d $PROJECT_HOME/data/fruits-360 ]]; then
    
    # Check if unzip is available
    if ! which unzip > /dev/null; then
        echo -e "\nERR: 'unzip' tool unavailable. Exiting...\n"
        exit 1
    fi

    echo -e "\nLOG: Unzipping dataset\n"
    unzip $PROJECT_HOME/data/fruits.zip -d $PROJECT_HOME/data > /dev/null
fi

# Delete zip file
if [[ -f $PROJECT_HOME/data/fruits.zip ]]; then
    echo -e "\nLOG: Removing data/fruits.zip\n"
    rm $PROJECT_HOME/data/fruits.zip
fi

# ================================================================================================================================== #
