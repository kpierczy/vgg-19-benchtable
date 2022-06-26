# ====================================================================================================================================
# @file       augmentation.py
# @author     Krzysztof Pierczyk (krzysztof.pierczyk@gmail.com)
# @maintainer Krzysztof Pierczyk (krzysztof.pierczyk@gmail.com)
# @date       Wednesday, 6th January 2021 2:17:57 am
# @modified   Sunday, 26th June 2022 4:32:11 pm
# @project    vgg-19-testbench
# @brief      Test / Demonstration of implemented augmentation mechanism. Script loads the training dataset, takes
#             first example image and applies augmentation mechanisms to it as set in the config/pipe.py file.
#             Next it output original and modified versions of the image to the 'img.jpg' and 'imga.jpg' files
#             respectively.
#
# @copyright Krzysztof Pierczyk © 2022
# ====================================================================================================================================

# ============================================================= Imports ============================================================ #

import sys
import os

# ========================================================== Configuration ========================================================= #

# Expand pythonpath
sys.path.append(os.path.join(os.environ.get('PROJECT_HOME'), 'src'))

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
import numpy as np
from glob import glob
import copy


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


# ========================================================= Initialize API ========================================================= #

# Construct base model
model = tf.keras.applications.vgg19.VGG19(
    include_top=True
)

# Initialize training API for default data
pipeline_params['shuffle_buffer_size'] = 1
pipeline_params['augmentation']['on'] = False
trainer = NeuralTrainer(
    model=model,
    dirs=dirs,
    logging_params=logging_params,
    pipeline_params=pipeline_params,
    training_params=training_params
)

# Initialize training API for augmented data
pipeline_paramsa = copy.deepcopy(pipeline_params)
pipeline_paramsa['augmentation']['on'] = True
trainer_aug = NeuralTrainer(
    model=model,
    dirs=dirs,
    logging_params=logging_params,
    pipeline_params=pipeline_paramsa,
    training_params=training_params
)

# Initialize trainin API
trainer.initialize()
trainer_aug.initialize()

# Extract datasets
ds = trainer.pipe.training_set
dsa = trainer_aug.pipe.training_set


# ============================================================== Test ============================================================== #

def write_jpg(img, name):
    with tf.compat.v1.Session() as sess:
        enc = tf.image.encode_jpeg(img)
        fwrite = tf.io.write_file(tf.constant(name), enc)
        sess.run(fwrite)

img  = ds.unbatch().take(1).map(lambda x,y: x).as_numpy_iterator().next()
imga = dsa.unbatch().take(1).map(lambda x,y: x).as_numpy_iterator().next()

# Write example of augmented and non-augmented image to file
write_jpg(img, 'img.jpg')
write_jpg(imga, 'imga.jpg')

# Print additional info about images
print()
print('img => R: [{:f}, {:f}], G: [{:f}, {:f}], B: [{:f}, {:f}]'.format(
    np.min(img[:,:,0]), np.max(img[:,:,0]),
    np.min(img[:,:,1]), np.max(img[:,:,1]),
    np.min(img[:,:,2]), np.max(img[:,:,2])
))
print('imga => R: [{:f}, {:f}], G: [{:f}, {:f}], B: [{:f}, {:f}]'.format(
    np.min(imga[:,:,0]), np.max(imga[:,:,0]),
    np.min(imga[:,:,1]), np.max(imga[:,:,1]),
    np.min(imga[:,:,2]), np.max(imga[:,:,2])
))
print()

# ================================================================================================================================== #
