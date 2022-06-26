# ====================================================================================================================================
# @file       LRTensorBoard.py
# @author     Krzysztof Pierczyk (krzysztof.pierczyk@gmail.com)
# @maintainer Krzysztof Pierczyk (krzysztof.pierczyk@gmail.com)
# @date       Wednesday, 6th January 2021 2:17:57 am
# @modified   Sunday, 26th June 2022 4:28:20 pm
# @project    vgg-19-testbench
# @brief      Wrapper around tf.keras.callbacks.TensoBoard that introduces learning rate's logging into the
#             tesnorboard
# 
# @see https://stackoverflow.com/questions/49127214/keras-how-to-output-learning-rate-onto-tensorboard
# @copyright Krzysztof Pierczyk © 2022
# ====================================================================================================================================

# ============================================================ Includes ============================================================ #

import tensorflow as tf

# ============================================================== Class ============================================================= #

class LRTensorBoard(tf.keras.callbacks.TensorBoard):

    """
    Wrapper around tf.keras.callbacks.TensoBoard that introduces learning rate's logging into the
    tesnorboard.
    """

    def __init__(self, **kwargs):

        """
        Initialized underlying tf.keras.callbacks.TensorBoard

        Params
        ------
        kwargs : keyword arguments
            @see tf.keras.callbacks.TensorBoard
        """

        super().__init__(**kwargs)


    def on_epoch_end(self, epoch, logs=None):

        """
        Adds learning rate to the logs and calls tf.keras.callbacks.TensorBoard.on_epoch_end()

        Params
        ------
        epoch: Int
            index of the current epoch
        logs : Dict
            Keras-internal logs
        """

        logs = logs or {}
        logs.update({'lr': tf.keras.backend.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)
