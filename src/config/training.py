# ====================================================================================================================================
# @file       training.py
# @author     Krzysztof Pierczyk (krzysztof.pierczyk@gmail.com)
# @maintainer Krzysztof Pierczyk (krzysztof.pierczyk@gmail.com)
# @date       Wednesday, 6th January 2021 2:17:57 am
# @modified   Sunday, 26th June 2022 4:31:12 pm
# @project    vgg-19-testbench
# @brief      Configuration file containing settings of the training's parameters
# 
# @copyright Krzysztof Pierczyk © 2022
# ====================================================================================================================================

training_params = {
    
    # Batch size
    'batch_size' : 64,

    # Optimization algorithm settings
    'optimization' : {

        # TF Optimizer's identifier
        'optimizer' : 'adam',

        # TF Loss function's identifier
        'loss' : 'categorical_crossentropy',

        # Learning rate settings (@see tf.keras.callbacks.ReduceLROnPlateau)
        'learning_rate' : {

            # Reduction's indicator 
            'indicator' : 'val_loss',

            # Initial value
            'init': 1e-4,

            # Minimal value
            'min' : 1e-7,

            # Minimal indicator change to be noticed
            'min_delta' : 5e-2,

            # Reduction factor
            'reduce_factor': 2e-1,

            # Patience (in epochs)
            'patience' : 4,

            # Cooldown
            'cooldown' : 0,

            # Changes' verbosity
            'verbosity' : 1
        }
    },

    # Training's length
    'epochs' : 40,

    # Index of the initial epoch (handy for training's continuation)
    'initial_epoch' : 0,

    # Number of batches per epoch (None if the whole dataset should be proceeded)
    'steps_per_epoch' : None,

    # Training workers
    'workers' : 4,

    # Training's verbosity
    'verbosity' : 1,
}