# ====================================================================================================================================
# @file       dirs.py
# @author     Krzysztof Pierczyk (krzysztof.pierczyk@gmail.com)
# @maintainer Krzysztof Pierczyk (krzysztof.pierczyk@gmail.com)
# @date       Wednesday, 6th January 2021 2:17:57 am
# @modified   Sunday, 26th June 2022 4:31:15 pm
# @project    vgg-19-testbench
# @brief      Configuration file containing input and output directories for the training process. Paths are given
#             relative to the PROJECT_HOME environment variable.
# 
# @copyright Krzysztof Pierczyk © 2022
# ====================================================================================================================================

dirs = {

    # Path to the Training data directory
    'training' : 'data/fruits-360/Training',

    # Path to the Validation/Test data directory
    'validation' : 'data/fruits-360/Test',

    # Path to the directory that output files (model's weights, logs, etc.) will be saved to (created as needed)
    'output' : 'models/simplified/run_1',
}