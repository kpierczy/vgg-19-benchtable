# ====================================================================================================================================
# @file       pipe.py
# @author     Krzysztof Pierczyk (krzysztof.pierczyk@gmail.com)
# @maintainer Krzysztof Pierczyk (krzysztof.pierczyk@gmail.com)
# @date       Wednesday, 6th January 2021 2:17:57 am
# @modified   Sunday, 26th June 2022 4:31:24 pm
# @project    vgg-19-testbench
# @brief      Configuration file containing settings of the input data pipeline
# 
# @copyright Krzysztof Pierczyk © 2022
# ====================================================================================================================================

pipeline_params = {

    # Ratio of the validation:training datasets' sizes (like valid_test_split[0]:valid_test_split[1])
    'valid_test_split' : [1, 1],

    # Size of the shuffle buffer for training pipeline (1 for not shuffling)
    'shuffle_buffer_size' : 5000,

    # Size of the prefetch buffers (None for autotune)
    'prefetch_buffer_size' : None,

    # Number of parallel threads used to load data (None for autotune)
    'parallel_calls' : None,

    # Pipeline's augmentation (operations in order)
    'augmentation' : {
        
        # ------------------------------------------------------------------------------------------
        # Note
        # ----
        # All ranges are expressed in form [min, max). Range expressed as [x,x] pair will result
        # in deterministic generation of x value instead of uniform random generation.
        # ------------------------------------------------------------------------------------------

        # Data augmentation is skipped if False
        'on' : True,

        # Random brightness variations' range (in [0,255]) [float] (@see tf.image.adjust_brightness)
        'brightness_range' : [0, 0],

        # Random contrast variations' range [float] (@see tf.image.adjust_contrast)
        'contrast_range' : [0, 0],

        # Random vertical flips
        'vertical_flip' : True,

        # Random horizontal flips
        'horizontal_flip' : True,

        # Random zoom range (in [0, 1)) (@see tf.image.crop_and_resize)
        'zoom_range' : [0, 0],

        # Random rotations' range (in degrees) [int] (@see tfa.image.rotate)
        'rotation_range' : [-20, 20],

        # Random width-wise shifs range (in pixels) [int] (@see tfa.image.translate)
        'width_shift_range' : [0, 0],

        # Random height-wise shifs range (in pixels) [int] (@see tfa.image.translate)
        'height_shift_range' : [0, 0],

        # Random x-wise shear variations' range [float] (@see tfa.image.shear_x)
        'shear_x_range' : [0, 0],

        # Random y-wise shear variations' range [float] (@see tfa.image.shear_y)
        'shear_y_range' : [0, 0],

        # Value of the filling pixels for translation, rotation & shear operations [int]
        'fill' : 255
    }

}