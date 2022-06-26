# ====================================================================================================================================
# @file       devices.py
# @author     Krzysztof Pierczyk (krzysztof.pierczyk@gmail.com)
# @maintainer Krzysztof Pierczyk (krzysztof.pierczyk@gmail.com)
# @date       Wednesday, 6th January 2021 2:17:57 am
# @modified   Sunday, 26th June 2022 4:29:08 pm
# @project    vgg-19-testbench
# @brief      Configuration of physical devices
# 
# @copyright Krzysztof Pierczyk © 2022
# ====================================================================================================================================

gpu_params = {

    # Limit of the GPU's memory usage [MB]
    "memory_cap_mb" : 3072,

    # Verbosity mode of TF data placement in context of devices
    "tf_device_verbosity" : False,
}