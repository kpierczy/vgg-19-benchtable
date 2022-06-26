# ====================================================================================================================================
# @file       DataPipe.py
# @author     Krzysztof Pierczyk (krzysztof.pierczyk@gmail.com)
# @maintainer Krzysztof Pierczyk (krzysztof.pierczyk@gmail.com)
# @date       Wednesday, 6th January 2021 2:17:57 am
# @modified   Sunday, 26th June 2022 4:27:26 pm
# @project    vgg-19-testbench
# @brief      Implementation of the basic data pipeline used for tesnroflow models' training. The aim of the class
#             is to provide easy-to-use, performance-optimised pipeline for the medium-size image datasets stored
#             in the predefined folders' structure.
# 
# @copyright Krzysztof Pierczyk © 2022
# ====================================================================================================================================

# ============================================================ Includes ============================================================ #

import os
import sys
import numpy as np
import tensorflow as tf
from glob import glob
from functools import partial

# ============================================================== Class ============================================================= #

class DataPipe:

    """
    Data Pipe class implements basic set of methods used to load labeled image data as well as to produce data
    batches in the tensorflow-dataset-generator style. It also contains some static methods for inspecting and
    manipulating datasets. The image dataset to be loaded with the DataPipe class should be organised in the
    specific way:

        data_root_folder/
        |--- label_1/
        |    |--- image_1.jpg
        |    |--- image_2.jpg
        |    |--- ...
        |--- label_2/
        |    |--- image_1.jpg
        |    |--- image_2.jpg
        |    |--- ...
        |--- ...

    Training and validation datasets don't have to be hold in the same directory. The DataPipe class is able also
    to extract some portion of the validation dataset to create the third, test dataset.

    Note
    ----
    At the moment all images should be given in a JPG-decoded format (i.e. '*.jpg', '*.jpeg').
    """

    def __init__(self):

        """
        Initializes a new DataPipe. The self.initialize() call is required before the pipe can be used.

        Attributes
        ----------
        self.training_set : None or tf.data.Dataset
            training dataset initialized at self.initialize(...) call
        self.validation_set : None or tf.data.Dataset
            validation dataset initialized at self.initialize(...) call
        self.test_set : None or tf.data.Dataset
            test dataset initialized at self.initialize(...)
            if 'self.test_split' attribute is None, the self.test_set is None
        self.test_split : Unsigned Int
            ration of validation:test (like test_split:1) datasets sizes
            if 0, no test dataset is created
        self.initialized : bool
            True if self.initialize(...) method was already called
        self.batched : bool
            True if self.apply_batch() method was already called
        """

        # Internal datasets
        self.training_set = None
        self.validation_set = None
        self.test_set = None
        self.test_split = 0

        # Values hold to be applied on self.apply_batch() call
        self.batch_size = None
        self.prefetch_buffer_size = None

        # Object status
        self.initialized = False
        self.batched = False


    def initialize(self,
        training_dir, 
        validation_dir,
        val_split=1,
        test_split=0,
        dtype='uint8',
        ldtype='uint8',
        batch_size=64, 
        shuffle_buffer_size=None,
        prefetch_buffer_size=None,
        parallel_calls=None
    ):
        """
        Initializes new training, validation and test datasets hold by the object. Image data 
        is loaded from the given directories. The test dataset is created by taking out part of
        the validation dataset in ratio defined by val_split:test_split.

        Params
        ------
        training_dir : string
            directory of the training data (@see DataPipe.dataset_from_directory())
        validation_dir : string
            directory of the validation data (@see DataPipe.dataset_from_directory())
        val_split : Unsigned Int
            ratio of validation:test (like valid_split:test_split) datasets sizes
            if 0, no validation set is initialized
        test_split : Unsigned Int
            ratio of validation:test (like valid_split:test_split) datasets sizes
            if 0, no test set is initialized
        dtype : string or np.dtype
            type of the images' representation
        ldtype : string or np.dtype
            type of the images labels' representation
        batch_size : int 
            size of the batch
        shuffle_buffer_size : int or None
            size of the buffer used to shuffle the dataset (@see tf.data.Dataset.shuffle())
            if None, shuffling is not performed
        prefetch_buffer_size : int or None
            size of the buffer used to prefetch the data (@see tf.data.Dataset.prefetch())
            if None, buffer size is autotuned
            if 0, prefetching is not performed.
        parallel_calls : int or None
            number of parallel threads used to load dataset. tf.data.experimental.AUTOTUNE
            if None given

        Note
        ----
        Test dataset split is performed deterministically for the given 'validation_dir' and 'val_split'
        and 'test_split' ratios. Moreover object sorts the original validation set by classes/examples
        names before splitting and tries to choose test examples evenly.

        Note
        ----
        Batching operation is not applied until self.apply_batch() method is called. The reason for this
        is to give a mechanism to apply per-training-example data modifications before grouping them
        into batches.

        Note
        ----
        Prefetching of the data happens in the batch-wise fashion. For this reason prefetching will also
        be not-active until self.apply_batch() call.

        To Do
        -----
        When tf.data.Dataset is produced using tf.data.Dataset.filter, the overall size of the set is
        set to 'unkown'. Therefore progress bar showed during evaluation of the test set displays
        'X/Unknown' and does not give estimation of the time left. It's not really important when
        playing with small datasets, nonetheless annoying. Fix it!
        """

        # Create and shuffle a training set
        self.training_set = self.dataset_from_directory(
            [training_dir], dtype=dtype, ldtype=ldtype, parallel_calls=parallel_calls)[training_dir]
        if shuffle_buffer_size is not None:
            self.training_set = self.training_set.shuffle(shuffle_buffer_size)

        # Enumerate examples from the validation directory
        ds = self.dataset_from_directory(
            [validation_dir], dtype=dtype, ldtype=ldtype, parallel_calls=parallel_calls)[validation_dir]
        enum_ds = ds.enumerate()

        def test_filter(i, data):
            return tf.math.less(tf.math.floormod(i, val_split + test_split), test_split)

        def val_filter(i, data):
            return tf.math.greater_equal(tf.math.floormod(i, val_split + test_split), test_split)

        def extract(i, data):
            return data

        # Take out validation set
        if val_split != 0:
            self.validation_set = enum_ds.filter(val_filter).map(extract)

        # Take out test set
        if test_split != 0:
            self.test_set = enum_ds.filter(test_filter).map(extract)
                
        # Holding batching informations for self.apply_batch() call
        self.batch_size = batch_size
        if prefetch_buffer_size is None:
            self.prefetch_buffer_size = tf.data.experimental.AUTOTUNE
        elif prefetch_buffer_size == 0:
            self.prefetch_buffer_size = None
        else:
            self.prefetch_buffer_size = prefetch_buffer_size

        # Update object's state
        self.initialized = True
        self.batched = False
        
    
    def apply_batch(self):

        """
        Applies batching and sets prefetching buffers for both training and validation
        dataset in an initialized object.
        """

        if self.initialized and not self.batched:

            # Establish batch size and set prefetch buffer for training set
            self.training_set = self.training_set.batch(self.batch_size)
            if self.prefetch_buffer_size is not None:
                self.training_set = self.training_set.prefetch(self.prefetch_buffer_size)

            # Establish batch size and set prefetch buffer for validation set
            self.validation_set = self.validation_set.batch(self.batch_size)
            if self.prefetch_buffer_size is not None:
                self.validation_set = self.validation_set.prefetch(self.prefetch_buffer_size)
        
            # Establish batch size and set prefetch buffer for test set
            if self.test_set is not None:
                self.test_set = self.test_set.batch(self.batch_size)
                if self.prefetch_buffer_size is not None:
                    self.test_set = self.test_set.prefetch(self.prefetch_buffer_size)

    @staticmethod
    def dataset_size_from_dir(directory, dtype='float32', ldtype=None):

        """
        Computes size of the dataset as it was if converted to the particular dtype. The structure of the 
        directory should be as given in the DataPipe class' description.

        Params
        ------
        directory : str
            path to the directory to be searched
        dtype : str or np.dtype
            data type that image will be casted to when loaded to the memory
        l_dtype : None or str or np.dtype
            data type that image's label will be casted to when loaded to the memory

        Returns
        -------
        tuple
            tuple holding (images_size, labels_size)
            
        """

        folders = glob(os.path.join(directory, '*'))

        # Load images and labels from directories to RAM
        with tf.device('/CPU:0'):

            img_size = 0
            label_size = 0

            # Load the exampels
            for f in folders:
                images = glob(os.path.join(f, '*.jp*g'))
                for i in images:
                    
                    # Compute image's size
                    img = tf.keras.preprocessing.image.img_to_array(    
                            tf.keras.preprocessing.image.load_img(i),
                            dtype=dtype
                        ) 
                    img_size += img.nbytes

                    # Compute label's size
                    if ldtype is None:
                        label_size += sys.getsizeof(os.path.basename(f))
                    else:
                        label_size += np.dtype(ldtype).itemsize

        return img_size, label_size 


    @staticmethod
    def dataset_from_directory(
        directories, 
        size=None, 
        dtype='uint8', 
        ldtype='uint8', 
        channels=3,
        parallel_calls=None
    ):

        """
        Prepares a tf.data.Dataset from images data in the directories. The structure of the directory should
        be as given in the DataPipe class' description.

        Params
        ------
        directories : list of str
            list of paths to the directories holding datasets
        size : tuple of int or None
            if None, files' sizes are preserved when loading
            if tuple, every loaded image is resized to the size[0] x size[1]
        dtype : str or np.dtype or tf.dtypes
            type that the image data will be casted to when loaded
        ldtype : str or np.dtype or tf.dtypes
            type of the elements of the categorical vectors that labels will be casted to
        channels : int 
            number of image's channel
        parallel_calls : int or None
            number of parallel threads used to load dataset. tf.data.experimental.AUTOTUNE
            if None given

        Note
        ----
        At the moment all images should be given in a JPG-decoded format (i.e. '*.jpg', '*.jpeg').
        
        Note
        ----
        If size=None, sizes of all images should be equal

        Returns
        -------
        list of tf.data.Dataset
            list of dataset containing (image, label) tuples

        """

        def file_to_training_example(path):

            """
            Loads image with the given path and tranforms it to a tf.Tensor.

            Params
            ------
            path : str
                path to the image file

            Returns
            -------
            tuple
                (image, label) pair

            """

            # Load the raw image from the file
            img = tf.io.read_file(path)
            # Decode the JPEG file
            img = tf.image.decode_jpeg(img, channels=channels)
            # Convert to the required data type
            img = tf.cast(img, tf.dtypes.as_dtype(dtype))
            # Resize the image
            if size is not None:
                img = tf.image.resize(img, [size[0], size[1]])

            return img

        # Map of {'directory': tf.data.Dataset} pairs
        datasets = {}

        for d in directories:

            # Get list of image files in the directory
            files = glob(os.path.join(d, '*/*.jp*g'))
            files.sort()

            # Associate number values with the classes' subdirectories
            labels_dirs = glob(os.path.join(d, '*'))
            labels_dirs.sort()
            labels_dict = {}
            for i, l in enumerate(labels_dirs):
                labels_dict[l] = i

            # Create list of one-hot labels for the files
            labels = []
            for f in files:
                # Get a file's directory
                label_dir = os.path.dirname(f)
                # Create the one-hot label
                label = tf.one_hot(labels_dict[label_dir], depth=len(labels_dict), dtype=ldtype)
                # Add the pair to the dataset
                labels.append(label)
                
            # Transform list dataset to the tf.data.Dataset
            ds = tf.data.Dataset.from_tensor_slices((files, labels))

            # Convert filenames of images to the dataset
            parallel = parallel_calls if parallel_calls is not None else tf.data.experimental.AUTOTUNE
            ds = ds.map(
                lambda file, label: (file_to_training_example(file), label),
                num_parallel_calls=parallel
            )

            datasets[d] = ds

        return datasets
