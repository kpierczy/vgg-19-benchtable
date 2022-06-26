# ====================================================================================================================================
# @file       rocm.bash
# @author     Krzysztof Pierczyk (krzysztof.pierczyk@gmail.com)
# @maintainer Krzysztof Pierczyk (krzysztof.pierczyk@gmail.com)
# @date       Wednesday, 11th November 2020 4:09:10 pm
# @modified   Sunday, 26th June 2022 4:22:12 pm
# @project    vgg-19-testbench
# @brief      Sets virtual environment for work with Keras (desired to source if CPU or Nvidia GPU tensorflow version used).
#
# @note You should 'source' this script instead of executing.
# @copyright Krzysztof Pierczyk © 2022
# ====================================================================================================================================

# Create venv folder, if needed 
if [[ ! -d $PROJECT_HOME/config/venv ]]; then
    python3 -m virtualenv $PROJECT_HOME/config/env/venv
fi

# Source virtual environment
source $PROJECT_HOME/config/env/venv/bin/activate

# Update pip
python3 -m pip install --upgrade pip

# Install required packages in the virtual environment
echo -e "\nLOG: Installing required packages in the virtual environment\n"
python3 -m pip install -r $PROJECT_HOME/config/env/requirements.py

# Install required packages
if [[ $TF_VERSION == "c" ]]; then
    python3 -m pip install -r $PROJECT_HOME/config/env/requirements_cpu.py
elif [[ $TF_VERSION == "nvd" ]]; then
    python3 -m pip install -r $PROJECT_HOME/config/env/requirements_nvd.py
fi
