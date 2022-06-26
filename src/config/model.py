# ====================================================================================================================================
# @file       model.py
# @author     Krzysztof Pierczyk (krzysztof.pierczyk@gmail.com)
# @maintainer Krzysztof Pierczyk (krzysztof.pierczyk@gmail.com)
# @date       Wednesday, 6th January 2021 2:17:57 am
# @modified   Sunday, 26th June 2022 4:31:20 pm
# @project    vgg-19-testbench
# @brief      Configuration of the model's parameters
# 
# @copyright Krzysztof Pierczyk © 2022
# ====================================================================================================================================

model_params = {
    
    # Path to the initial model's weights (relative to PROJECT_HOME environment variable)
    'base_model' : None,

    # TF Kernel and bias intiializers' identifier
    'initializer' : {
        'kernel' : 'glorot_normal',
        'bias' : 'glorot_normal'
    },
    
    # Number of _last_ original VGG layers that should be removed
    'vgg_to_remove' : 5,

    # Number of _last_ original VGG _conv_ layers to be retrained [None to train all layers]
    'vgg_conv_to_train' : None,
}