# ====================================================================================================================================
# @file       main.py
# @author     Krzysztof Pierczyk (krzysztof.pierczyk@gmail.com)
# @maintainer Krzysztof Pierczyk (krzysztof.pierczyk@gmail.com)
# @date       Wednesday, 6th January 2021 2:17:57 am
# @modified   Sunday, 26th June 2022 4:34:16 pm
# @project    vgg-19-testbench
# @details
#       
#     Script implements an configfile-driven envirenment used to fit a VGG19 model. Before using the
#     script, the 'source_me.bash' script should be sourced to prepare appropriate environment variables.
#     Alternatively, required 'PROJECT_HOME' variable can be set manually to the absolute path of the
#     project's home folder.
#
#     The script was prepared so that learning workflow could be run without training code's modifications.
#     All parameters configuring the learning process can be adjusted from config/*.py config files.
#     Only modifications of the model's structure require direct interference into the script's code.
#
#     As the learning rate scheduler the tf.keras.callbacks.ReduceLROnPlateau object is used. It's parameters
#     can be tuned from the training.py file.
# 
# @requirements All required python packages was listed in config/env/requirements*.py files
# @copyright Krzysztof Pierczyk © 2022
# ====================================================================================================================================

# ============================================================= Imports ============================================================ #

# Tensorflow
import tensorflow as tf
# Keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Model
# Configuration files
from config.dirs     import dirs
from config.devices  import gpu_params
from config.logs     import logging_params
from config.model    import model_params
from config.pipe     import pipeline_params
from config.training import training_params
# training API
from api.NeuralTrainer import NeuralTrainer
# Images manipulation
from PIL import Image
# Utilities
from glob import glob
import os

# ==================================================== Environment configuration =================================================== #

# Get project's path
PROJECT_HOME = os.environ.get('PROJECT_HOME')

# Resolve all relative paths to absolute paths
dirs["training"] = os.path.join(PROJECT_HOME, dirs["training"])
dirs["validation"] = os.path.join(PROJECT_HOME, dirs["validation"])
dirs["output"] = os.path.join(PROJECT_HOME, dirs["output"])

# Get number of classes
num_classes = len(glob(os.path.join(PROJECT_HOME, os.path.join(dirs['training'], '*'))))

# Load an example image from training dataset to establish input_shape
input_shape = \
    list( Image.open(glob(os.path.join(PROJECT_HOME, os.path.join(dirs['training'], '*/*.jp*g')))[0]).size ) + [3]

# Limit GPU's memory usage
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [
            tf.config.experimental.VirtualDeviceConfiguration(
                memory_limit=gpu_params['memory_cap_mb']
            )
        ])
    except RuntimeError as e:
        print(e)

# Verbose data placement info
tf.debugging.set_log_device_placement(gpu_params['tf_device_verbosity'])


# =========================================================== Build model ========================================================== #

# Prepare initializers
kernel_initializer = tf.keras.initializers.get(model_params['initializer']['kernel'])
bias_initializer = tf.keras.initializers.get(model_params['initializer']['bias'])

# Create preprocessing layer
model_input = tf.keras.layers.Input(input_shape, dtype='float32')
preprocessing = tf.keras.applications.vgg19.preprocess_input(model_input)

# Construct base model
vgg_imagenet = tf.keras.applications.vgg19.VGG19(
    weights='imagenet',
    include_top=False
)

# Remove required number of final layers from original vgg
vgg = tf.keras.Model(
    inputs=vgg_imagenet.input, 
    outputs=vgg_imagenet.layers[-(model_params['vgg_to_remove'] + 1)].output, 
    name='vgg19'
)

# Reinitialize original convolutional layers that will be trained
reinitialized = 0
for layer in reversed(vgg.layers):
    
    # Reinitialize required convolutional layers
    if model_params['vgg_conv_to_train'] is None or reinitialized < model_params['vgg_conv_to_train']:

        # Check if layer has wights to reinitialize
        if isinstance(layer, tf.keras.layers.Conv2D):

            # Save old weights and baises
            weights, biases = layer.get_weights()

            # Reinitialize
            weights = kernel_initializer(shape=weights.shape)
            biases = bias_initializer(shape=biases.shape)

            # Set new weights to the layer
            layer.set_weights([weights, biases])

            # Increment counter of reinitialized layers
            reinitialized += 1

            continue

    # Froze rest of layers
    layer.trainable = False

# Concatenate model and the preprocessing layer
model = vgg(preprocessing)

# Add Dense layers
model = Flatten()(model)
model = Dense(
    4096,
    activation='relu',
    kernel_initializer=kernel_initializer,
    bias_initializer=bias_initializer)(model)
model = Dense(
    4096,
    activation='relu',
    kernel_initializer=kernel_initializer,
    bias_initializer=bias_initializer)(model)
model = Dense(
    num_classes,
    activation='softmax',
    kernel_initializer=kernel_initializer,
    bias_initializer=bias_initializer)(model)

# Create model
model = Model(inputs=[model_input], outputs=[model])

# Load base model's weights
if model_params['base_model'] is not None:
    model.load_weights(os.path.join(PROJECT_HOME, model_params['base_model']))


# ========================================================== Run training ========================================================== #

# Initialize training API
trainer = NeuralTrainer(
    model=model,
    dirs=dirs,
    logging_params=logging_params,
    pipeline_params=pipeline_params,
    training_params=training_params
)

# Run training
trainer.initialize().run()

# ================================================================================================================================== #
