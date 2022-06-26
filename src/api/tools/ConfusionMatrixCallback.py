# ====================================================================================================================================
# @file       ConfusionMatrixCallback.py
# @author     Krzysztof Pierczyk (krzysztof.pierczyk@gmail.com)
# @maintainer Krzysztof Pierczyk (krzysztof.pierczyk@gmail.com)
# @date       Wednesday, 6th January 2021 2:17:57 am
# @modified   Sunday, 26th June 2022 4:27:02 pm
# @project    vgg-19-testbench
# @brief      Implementation of the custom Keras callback periodically generating Confusion Matrix for the
#             Validation set.
# 
# @copyright Krzysztof Pierczyk © 2022
# ====================================================================================================================================

# ============================================================ Includes ============================================================ #

import io
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
import itertools

# ============================================================== Class ============================================================= #

class ConfusionMatrixCallback(tf.keras.callbacks.Callback):

    """
    Custom keras callback class writing a confussion matrix at the end of each X epochs.
    The matrix picture is saved to the raw (e.g. png, pdf) form in the logdir/raw folder
    as well as the tensorboard summary file in the logdir/tf foler. Subfolders are 
    created if needed.

    Params | Attributes
    -------------------
    logdir : string
        directory that created matrices will be held at
    validation_set : tf.data.Dataset
        validation dataset
    class_names : list of strings
        list of classes' names; index of the name defines the class numerical identifier
    freq : Int, optional (default: 1)
        frequency (in epochs) that the confusion matrix is created at
    fig_size : tuple or list of two Ints, optional (default: (20,20))
        size of the plot in cm [width, height]
    basename : string, optional (default: 'Confusion_Matrix')
        basename of the logs

    Params
    ------
    to_save : string, optional (default: 'both')
        if 'tf', only tensorboard image logs are saved
        if 'raw', only raw image logs are saved
        if 'both', both types of logs are saved

    Credits
    -------
    Original code: [https://www.tensorflow.org/tensorboard/image_summaries]
    """

    def __init__(self,
        logdir,
        validation_set,
        class_names,
        freq=1,
        fig_size=(20, 20),
        raw_fig_type='pdf',
        to_save='both',
        basename='Confusion_Matrix'
    ):

        # Dataset used to comput matrix 
        self.validation_set = validation_set
        self.class_names = class_names

        # Callback's frequency (in epochs)
        self.freq = freq

        # Figure's settings
        self.fig_size = np.array(fig_size) / 2.54
        self.raw_fig_type = raw_fig_type

        # Logs' basename
        self.basename = basename

        # Create folder for tf images
        self.tf_logdir = None
        if to_save != 'raw' and self.freq > 0:
            self.tf_logdir = os.path.join(logdir, 'tf')
            os.makedirs(self.tf_logdir, exist_ok=True)

        # Create folder for raw images
        self.raw_logdir = None
        if to_save != 'tf' and self.freq > 0:
            self.raw_logdir = os.path.join(logdir, 'raw')
            os.makedirs(self.raw_logdir, exist_ok=True)
        


    def on_epoch_end(self, tag, logs):

        """
        Writes the confusion matrix image to the log file at the end of the each 'self.freq' calls.
        Returns imidiately if 'self.freq' is equal to 0.

        Params
        ------
        tag : Int or string
            epoch's index or another kind of tag
        logs : Dict
            metrics results for this training tag
        """

        if self.freq <= 0:
            return

        # Parse epoch ndex
        epoch = tag if isinstance(tag, int) else 0

        # Update frequency counter
        if epoch % self.freq != 0:
            return

        # Use the model to predict the values from the validation dataset.
        predictions_softmax = self.model.predict(self.validation_set)
        predictions = tf.argmax(predictions_softmax, axis=1)

        # Calculate the confusion matrix.
        actual_categories_softmax = tf.concat([y for x, y in self.validation_set], axis=0)
        actual_categories = tf.argmax(actual_categories_softmax, axis=1)
        con_matrix = sklearn.metrics.confusion_matrix(actual_categories, predictions)

        # Log the confusion matrix as an image summary.
        figure = self.__plot_confusion_matrix(con_matrix, class_names=self.class_names, size=self.fig_size)

        # Save raw figure image
        if self.raw_logdir is not None:
            delimiter = '_' if str(tag) != '' else ''
            figname = self.basename + delimiter + str(tag) + '.' + self.raw_fig_type
            figure.savefig(os.path.join(self.raw_logdir, figname), bbox_inches='tight')

        # Log the confusion matrix as an image summary.
        if self.tf_logdir is not None:
            cm_image_tf = self.__plot_to_image(figure)
            file_writer_cm = tf.summary.create_file_writer(self.tf_logdir)
            with file_writer_cm.as_default():
                tf.summary.image(self.basename, cm_image_tf, step=epoch)

        plt.close(figure)
        return


    @staticmethod
    def __plot_confusion_matrix(con_matrix, class_names, size):

        """
        Returns a matplotlib figure containing the plotted confusion matrix.

        Params
        ------
        con_matrix : np.array of shape [n, n]
            a confusion matrix of integer classes
        class_names : np.array of shape [n]
            string names of the integer classes
        size : tuple or list of two Ints
            size of the plot in cm [width, height]
        """

        # Create the figure
        figure = plt.figure(figsize=size)

        # Print confusion matrix to the plot
        plt.imshow(con_matrix, interpolation='nearest', cmap=plt.cm.Blues)

        # Setup plto's environment
        plt.title("Confusion matrix")
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

        # Compute the labels from the normalized confusion matrix.
        labels = np.around(con_matrix.astype('float') / con_matrix.sum(axis=1)[:, np.newaxis], decimals=2)

        # Use white text if squares are dark; otherwise black.
        threshold = con_matrix.max() / 2.
        for i, j in itertools.product(range(con_matrix.shape[0]), range(con_matrix.shape[1])):
            color = "white" if con_matrix[i, j] > threshold else "black"
            plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

        # Format the figure
        plt.tight_layout()

        # Assign axes
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        return figure


    @staticmethod
    def __plot_to_image(figure):

        """
        Converts the matplotlib plot specified by 'figure' to a PNG-decoded tensorflow
        image tensor.

        Params
        ------
        figure : plt.figure
            figure to be printed

        Returns
        -------
        image : tf.Tensor
            printed image
        """

        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        plt.savefig(buf, format='png')

        # Close the figure
        buf.seek(0)

        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)

        # Add the batch dimension
        image = tf.expand_dims(image, 0)

        return image

# ================================================================================================================================== #
