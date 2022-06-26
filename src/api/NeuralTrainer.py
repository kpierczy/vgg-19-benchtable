# ====================================================================================================================================
# @file       NeuralTrainer.py
# @author     Krzysztof Pierczyk (krzysztof.pierczyk@gmail.com)
# @maintainer Krzysztof Pierczyk (krzysztof.pierczyk@gmail.com)
# @date       Wednesday, 6th January 2021 2:17:57 am
# @modified   Sunday, 26th June 2022 4:28:46 pm
# @project    vgg-19-testbench
# @brief      Implementation of the complete image-recognision neural network's training flow.
# 
# @copyright Krzysztof Pierczyk © 2022
# ====================================================================================================================================

# ============================================================ Includes ============================================================ #

# Tensorflow
import tensorflow as tf
# Dedicated datapipe
from api.tools.ImageAugmentation       import ImageAugmentation
from api.tools.DataPipe                import DataPipe
from api.tools.ConfusionMatrixCallback import ConfusionMatrixCallback
from api.tools.LRTensorBoard           import LRTensorBoard
# Images manipulation
from PIL import Image
# Utilities
from glob import glob
import numpy as np
import pickle
import os

# ============================================================== Class ============================================================= #

class NeuralTrainer:

    """
    Implementation of the complete image-recognision neural network's training flow. The NeuralTrainer class
    provides ready to use set of procedures performing:
        
        * data pipeline preparation
        * training metrics logging
        * confusion matrix logging
        * result model testing 

    Workflow is configuration-driven, i.e. all required paramaters of the training process are passed to the
    object during construction in form of dictionaries (@see config/*.py files).
    """

    def __init__(self,
        model,
        dirs,
        logging_params,
        pipeline_params,
        training_params
    ):

        """
        Constructs training flow object.

        Pramas | Attributes
        -------------------
        model : tf.keras.Model
            model to be trained
        dirs : dictionary
            set of paths to input and output files directories
        logging_params : dictionary
            parameters of the logging system
        pipeline_params : dictionary
            parameters of the data pipeline
        training_params : dictionary
            parameters of the training process

        Attributes
        ----------
        pipe : DataPipe
            data pipeline constructed at the initialization phase
        callbacks : List of tf.keras.callbacks.Callback
            Set of callbacks applied during training (filled at the initialization phase)
        """

        # Model to be trained
        self.model = model

        # Data pipeline
        self.pipe = None

        # Set of the training callbacks
        self.callbacks = []

        # Configuration
        self.dirs = dirs
        self.logging_params = logging_params
        self.pipeline_params = pipeline_params
        self.training_params = training_params

        # Create output folder
        os.makedirs(self.dirs['output'], exist_ok=True)

        # Last training history
        self.__history = None


    def initialize(self):

        """
        Constructs data pipeline and default training callbacks 
        """

        # Prepare the pipeline
        self.__prepare_pipeline()

        # Compile model
        self.__compile_model()

        # Print models' statistics
        self.__show_model()

        # Prepare training callbacks
        self.__prepare_callbacks()

        return self


    def run(self):

        """
        Runs the training flow and tests the result model
        """

        # Train the model
        self.__train()

        # Test the model
        self.__test()

        pass


    def __prepare_pipeline(self):

        """
        Prepares data pipeline
        """

        # Create data pipe (contains training and validation sets)
        self.pipe = DataPipe()
        self.pipe.initialize(
            self.dirs['training'],
            self.dirs['validation'],
            val_split=self.pipeline_params['valid_test_split'][0],
            test_split=self.pipeline_params['valid_test_split'][1],
            dtype='float32',
            batch_size=self.training_params['batch_size'],
            shuffle_buffer_size=self.pipeline_params['shuffle_buffer_size'],
            prefetch_buffer_size=self.pipeline_params['prefetch_buffer_size'],
            parallel_calls=self.pipeline_params['parallel_calls']
        )

        # Augment the data pipe
        if self.pipeline_params['augmentation']['on']:
            self.pipe.training_set = ImageAugmentation(
                brightness_range=self.pipeline_params['augmentation']['brightness_range'],
                contrast_range=self.pipeline_params['augmentation']['contrast_range'],
                vertical_flip=self.pipeline_params['augmentation']['vertical_flip'],
                horizontal_flip=self.pipeline_params['augmentation']['horizontal_flip'],
                zoom_range=self.pipeline_params['augmentation']['zoom_range'],
                rotation_range=self.pipeline_params['augmentation']['rotation_range'],
                width_shift_range=self.pipeline_params['augmentation']['width_shift_range'],
                height_shift_range=self.pipeline_params['augmentation']['height_shift_range'],
                shear_x_range=self.pipeline_params['augmentation']['shear_x_range'],
                shear_y_range=self.pipeline_params['augmentation']['shear_y_range'],
                fill=self.pipeline_params['augmentation']['fill'],
                dtype='float32'
            )(self.pipe.training_set)

        # Apply batching to the data sets
        self.pipe.apply_batch()


    def __compile_model(self):

        """
        Compiles the model setting required optimizer and loss function
        """

        # Initialize optimizer
        optimizer = tf.keras.optimizers.get({
            "class_name": self.training_params['optimization']['optimizer'],
            "config": {"learning_rate": self.training_params['optimization']['learning_rate']['init']}}
        )

        # Compile the model
        self.model.compile(
            loss=self.training_params['optimization']['loss'],
            optimizer=optimizer,
            metrics=self.logging_params['metrics']
        )


    def __show_model(self):

        """
        Prints model statistics
        """

        print('\n\n')
        self.model.summary()
        print('\n\n')

    
    def __prepare_callbacks(self):

        """
        Prepares default training callbacks
        """

        # Create path to the logdir
        logdir = os.path.join(self.dirs['output'], 'logs')

        # Create output folder for weights saves
        modeldir  = os.path.join(self.dirs['output'], 'weights')
        os.makedirs(modeldir, exist_ok=True)

        # Create logging callbacks
        if self.logging_params['on']:

            #Create the logdir
            os.makedirs(logdir, exist_ok=True)

            # List names of images' classes
            class_folders = glob(os.path.join(self.dirs['validation'], '*'))
            class_names = [os.path.basename(folder) for folder in class_folders]
            class_names.sort()

            # Create tensorboard callback
            tensorboard_callback = LRTensorBoard(
                log_dir=logdir, 
                histogram_freq=self.logging_params['tensorboard']['histogram_freq'],
                write_graph=self.logging_params['tensorboard']['write_graph'],
                write_images=self.logging_params['tensorboard']['write_images'],
                update_freq=self.logging_params['tensorboard']['update_freq'],
                profile_batch=self.logging_params['tensorboard']['profile_batch']
            )
            self.callbacks.append(tensorboard_callback)

            # Create a confusion matrix callback
            cm_callback = ConfusionMatrixCallback(
                logdir=os.path.join(logdir, 'validation/cm'),
                validation_set=self.pipe.validation_set,
                class_names=class_names,
                freq=self.logging_params['confusion_matrix']['freq'],
                fig_size=self.logging_params['confusion_matrix']['size'],
                raw_fig_type=self.logging_params['confusion_matrix']['raw_ext'],
                to_save=self.logging_params['confusion_matrix']['to_save']
            )
            self.callbacks.append(cm_callback)

        # Create a checkpoint callback
        checkpoint_name = os.path.join(modeldir, 'weights-epoch_{epoch:02d}-val_loss_{val_loss:.2f}.hdf5')
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_name,
            save_weights_only=True,
            verbose=True,
            save_freq='epoch',
            save_best_only=self.logging_params['save_best_only']
        )
        self.callbacks.append(checkpoint_callback)

        # Create learning rate scheduler callback
        lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
            monitor=self.training_params['optimization']['learning_rate']['indicator'],
            factor=self.training_params['optimization']['learning_rate']['reduce_factor'],
            patience=self.training_params['optimization']['learning_rate']['patience'],
            verbose=self.training_params['optimization']['learning_rate']['verbosity'],
            min_delta=self.training_params['optimization']['learning_rate']['min_delta'],
            cooldown=self.training_params['optimization']['learning_rate']['cooldown'],
            min_lr=self.training_params['optimization']['learning_rate']['min']
        )
        self.callbacks.append(lr_callback)


    def __train(self):

        """
        Runs the training flow
        """

        # Start training
        self.__history = self.model.fit(
            x=self.pipe.training_set,
            validation_data=self.pipe.validation_set,
            epochs=self.training_params['epochs'],
            initial_epoch=self.training_params['initial_epoch'],
            steps_per_epoch=self.training_params['steps_per_epoch'],
            callbacks=self.callbacks,
            verbose=self.training_params['verbosity'],
            workers=self.training_params['workers'],
            use_multiprocessing=True if self.training_params['workers'] != 1 else False,
            shuffle=False
        )

        # Save training history
        if self.logging_params['on']:

            # Create path to the output folder
            historydir = os.path.join(self.dirs['output'], 'history')
            os.makedirs(historydir, exist_ok=True)

            # Compute index of the subrun
            subrun = len(glob(os.path.join(historydir, '*.pickle'))) + 1
            
            # Create path to the output file
            historyname = os.path.join(historydir, 'subrun_{:d}'.format(subrun))

            with open(historyname + '.pickle', 'wb') as history_file:
                pickle.dump(self.__history.history, history_file)


    def __test(self):

        """
        Tests the result model
        """

        if self.pipe.test_set is not None and self.logging_params['test_model'] is not None:

            # If the best models hould be evaluated, load appropriate weights
            if self.logging_params['test_model'] == 'best':

                # Find epoch's index of the best score
                best_score = np.nanmin(np.array(self.__history.history['val_loss']))

                # Find the weights file
                modeldir  = os.path.join(self.dirs['output'], 'weights')
                weights_file = glob(os.path.join(modeldir, '*val_loss_{:.2f}*'.format(best_score)))[0]

                # Load weights
                self.model.load_weights(weights_file)

            # Create path to the output folder
            testdir = os.path.join(self.dirs['output'], 'test')
            os.makedirs(testdir, exist_ok=True)

            # Compute index of the subrun
            subrun = len(glob(os.path.join(testdir, '*.pickle'))) + 1

            # Create basename for CM raw files (include type of the model that is tested: lates or best)
            testbasename = 'subrun_{:d}'.format(subrun) + '_' + self.logging_params['test_model']

            # Create path to the output file
            testname = os.path.join(testdir, testbasename)

            # List names of images' classes
            class_folders = glob(os.path.join(self.dirs['validation'], '*'))
            class_names = [os.path.basename(folder) for folder in class_folders]
            class_names.sort()

            # Prepare a new Confusion Matrix callback for the test set
            cm_callback = ConfusionMatrixCallback(
                logdir=os.path.join(testdir, 'cm'),
                validation_set=self.pipe.test_set,
                class_names=class_names,
                freq=self.logging_params['confusion_matrix']['freq'],
                fig_size=self.logging_params['confusion_matrix']['size'],
                raw_fig_type=self.logging_params['confusion_matrix']['raw_ext'],
                to_save=self.logging_params['confusion_matrix']['to_save'],
                basename=testbasename
            )

            # Wrap Confusion Matrix callback to be usable with tf.keras.Model.evaluate() method
            cm_callback.set_model(self.model)
            cm_callback_test_decorator = \
                tf.keras.callbacks.LambdaCallback(on_test_end=lambda logs: cm_callback.on_epoch_end('', logs))

            # Evaluate test score
            test_dict = self.model.evaluate(
                x=self.pipe.test_set,
                verbose=self.training_params['verbosity'],
                workers=self.training_params['workers'],
                use_multiprocessing=True if self.training_params['workers'] != 1 else False,
                return_dict=True,
                callbacks=[cm_callback_test_decorator]
            )

            # Save test score
            with open(testname + '.pickle', 'wb') as test_file:
                pickle.dump(test_dict, test_file)
