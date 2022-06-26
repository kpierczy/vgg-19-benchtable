# ====================================================================================================================================
# @file       ImageAugmentation.py
# @author     Krzysztof Pierczyk (krzysztof.pierczyk@gmail.com)
# @maintainer Krzysztof Pierczyk (krzysztof.pierczyk@gmail.com)
# @date       Wednesday, 6th January 2021 2:17:57 am
# @modified   Sunday, 26th June 2022 4:28:24 pm
# @project    vgg-19-testbench
# @brief      Implementation of the set of the most common image-data augmentation methods gather into the
#             single class designed to be used with tf.data.Dataset datasets.
# 
# @copyright Krzysztof Pierczyk © 2022
# ====================================================================================================================================

# ============================================================ Includes ============================================================ #

import tensorflow as tf
import tensorflow_addons as tfa
import math

# ============================================================== Class ============================================================= #

class ImageAugmentation:

    """
    ImageAugmentation class represents set of augmentation operations that can be applied to the 
    tf.data.Dataset object.
    """

    def __init__(self,
        brightness_range=(0, 0),
        contrast_range=(0, 0),
        vertical_flip=False,
        horizontal_flip=False,
        zoom_range=(0, 0),
        rotation_range=(0, 0),
        width_shift_range=(0, 0),
        height_shift_range=(0, 0),
        shear_x_range=(0, 0),
        shear_y_range=(0, 0),
        fill=0,
        dtype='uint8'
    ):

        """
        Initializes object representing set of the augmentation operations applied
        to the tf.data.Dataset at the self.__call__() call. Order of transformations is 
        as given in the parameters' list.

        Params | Attributes
        -------------------
        brightness_range : tuple or list of two floats
            [min; max) range for random brightness shift
        contrast_range : tuple or list of two floats
            [min; max) range for random contrast adjustement
        vertical_flip : bool
            if True, 50% of images are flipped up-down
        horizontal_flip : bool
            if True, 50% of images are flipped left-right
        zoom_range : tuple or list of two ints
            [min; max) range for random zooms (in [0, 1) range)
        rotation_range : tuple or list of two ints
            [min; max) range for random rotations             
        width_shift_range : tuple or list of two ints
            [min; max) range for random width-wise shifts
        height_shift_range : tuple or list of two ints
            [min; max) range for random height-wise shifts
        shear_x_range : tuple or list of two floats
            [min; max) range for random x-shear adjustement (@see tfa.image.shear_x())
        shear_y_range : tuple or list of two floats
            [min; max) range for random y-shear adjustement (@see tfa.image.shear_y())
        fill : int 
            value of the pixels needed to be filled after the shift, rotation & shear
        dtype : string or tf.dtype
            type of the images' representation
        
        [To Do]
        -------
        Parameters verification

        """

        self.brightness_range   = brightness_range
        self.contrast_range     = contrast_range
        self.vertical_flip      = vertical_flip
        self.horizontal_flip    = horizontal_flip
        self.zoom_range         = zoom_range
        self.rotation_range     = rotation_range
        self.width_shift_range  = width_shift_range
        self.height_shift_range = height_shift_range
        self.shear_x_range      = shear_x_range
        self.shear_y_range      = shear_y_range

        self.fill               = fill
        self.dtype              = dtype


    def __call__(self, dataset):

        """
        Applies augmentation operations to the dataset

        Params
        ------
        dataset : tf.data.Dataset
            the dataset to be augmented

        Returns
        -------
        aug_dataset : dataset: tf.data.Dataset
            the augmented dataset

        Note
        ----
        For shearing operation images need to be converted into 'uin8' format, so excessive
        partial values will be discarded.

        """

        def rand_range(limits, last=False):
            
            """
            Wrapper around tf.random.uniform() for shorter calls
            """

            return tf.random.uniform([], minval=limits[0], maxval=limits[1], dtype=self.dtype)

        def zoom_box(zoom):

            """
            Params
            ------
            zoom : float in [0, 1)
                percentage zoom for the cpiture
            Returns
            -------
            box : list
                crop box used as boxes[i] argument of the tf.image.crop_and_resize
            """

            return [zoom / 2, zoom / 2, 1 - zoom / 2, 1 - zoom / 2]

        # Random brightness disturbance
        if self.brightness_range[0] != 0 or self.brightness_range[1] != 0:
            dataset = dataset.map(lambda x, y: 
                ( tf.image.adjust_brightness(x, rand_range(self.brightness_range)), y )
            )
            dataset = dataset.map(lambda x, y: 
                ( tf.clip_by_value(x, clip_value_min=0, clip_value_max=255), y )
            )

        # Random contrast disturbance
        if self.contrast_range[0] != 0 or self.contrast_range[1] != 0:
            dataset = dataset.map(lambda x, y: 
                ( tf.image.adjust_contrast(x, rand_range(self.contrast_range)), y )
            )
            dataset = dataset.map(lambda x, y: 
                ( tf.clip_by_value(x, clip_value_min=0, clip_value_max=255), y )
            )
        
        # Random x-wise flip
        if self.vertical_flip:
            dataset = dataset.map(lambda x, y: 
                ( tf.image.random_flip_left_right(x), y )
            )

        # Random x-wise flip
        if self.horizontal_flip:
            dataset = dataset.map(lambda x, y: 
                ( tf.image.random_flip_up_down(x), y )
            )

        # Random zoom
        if self.zoom_range[0] != 0 or self.zoom_range[1] != 0:
            dataset = dataset.map(lambda x, y: 
                ( tf.image.crop_and_resize(
                    tf.expand_dims(x, axis=0), boxes=[zoom_box(rand_range(self.zoom_range))], box_indices=[0], crop_size=tf.shape(x)[:2]
                  )[0], y )
            )
            
        # Random rotations
        if self.rotation_range[0] != 0 or self.rotation_range[1] != 0:
            dataset = dataset.map(lambda x, y: 
                ( tfa.image.rotate(x, math.pi / 180 * rand_range(self.rotation_range), fill_mode='constant', fill_value=self.fill), y )
            )

        # Random width-shifts
        if self.width_shift_range[0] != 0 or self.width_shift_range[1] != 0:
            dataset = dataset.map(lambda x, y: 
                ( tfa.image.translate_xy(x, [rand_range(self.width_shift_range), 0.], self.fill), y )
            )

        # Random height-shifts
        if self.height_shift_range[0] != 0 or self.height_shift_range[1] != 0:
            dataset = dataset.map(lambda x, y: 
                ( tfa.image.translate_xy(x, [0, rand_range(self.height_shift_range), 0.], self.fill), y )
            )

        # Random shear-x
        if self.shear_x_range[0] != 0 or self.shear_x_range[1] != 0:
            dataset = dataset.map(lambda x, y: 
                ( tf.cast(tfa.image.shear_x(tf.cast(x, 'uint8'), rand_range(self.shear_x_range), self.fill), self.dtype), y )
            )

        # Random shear-y
        if self.shear_y_range[0] != 0 or self.shear_y_range[1] != 0:
            dataset = dataset.map(lambda x, y: 
                ( tf.cast(tfa.image.shear_y(tf.cast(x, 'uint8'), rand_range(self.shear_y_range), self.fill), self.dtype), y )
            )

        return dataset