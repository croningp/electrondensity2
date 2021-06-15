##########################################################################################
#
# This model will aim to predict a QM9 property (a single property) from electron
# densities. To do so, it will do the usual 3D convs, then flattened, then single output.
#
# Author: Juan Manuel Parrilla
#
##########################################################################################

import os
import pickle

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.utils import transform_ed
from src.models.layers import identity_block, conv_block, drop_dimension
from src.utils.callbacks import CallbackSinglePrediction


class CNN3D_singleprediction():

    def __init__(self, cubeside, filters, strides, dense_size):
        """Stores the parameters and calls _build, which is the function that builds the
        model.

        Args:
            cubeside: Size of cube's size. It will assume the 3 axes have the same size.
            filters: List with the size of the filters to use in the convolutions.
            strides: List with the size of the strides to use in the convolutions.
            dense_size: Number of neurons in the last dense layer before outputs.
        """

        self.size = cubeside
        self.filters = filters
        self.strides = strides
        self.dense_size = dense_size
        self.model = self._build()

    def _build(self):
        """Build a 3D convolutional NN."""

        inputs = keras.Input((self.size, self.size, self.size, 1))
        x = inputs

        # First we do the pre-processing Jarek was doing
        x = tf.tanh(x)
        x = transform_ed(x)

        # Do the 3D convolutions
        for i, f_s in enumerate(zip(self.filters, self.strides)):
            x = conv_block(x, 3, f_s[0], stage=i, block='a', strides=f_s[1])
            x = identity_block(x, 3, f_s[0], stage=i, block='b')
            x = identity_block(x, 3, f_s[0], stage=i, block='c')
            if i > 1:
                x = drop_dimension(x)

        # Flatten it into 1D
        # x = layers.GlobalAveragePooling3D()(x)
        x = layers.Flatten()(x)
        x = layers.Dense(units=self.dense_size, activation="relu")(x)
        x = layers.Dropout(0.3)(x)

        # output layer
        outputs = layers.Dense(units=1)(x)
        outputs = layers.ReLU(max_value=1.0)(outputs)

        # create and return model
        model = keras.Model(inputs, outputs, name="3dcnn")
        return model

    def loss_function(self, real, pred):
        """This is needed because I want to normalize the real data. 
        I have calculated beforehand max and min alpha, to normalize it between 0 and 1.
        max is 196.62, min is 6.31."""

        # normalize real between 0 and 1
        real = real - 6.31
        real = real / (196.62-6.31)
        # calculate MSE and return
        return keras.losses.mean_squared_error(real, pred)

    def compile(self, learning_rate):
        # Compile model.
        initial_learning_rate = learning_rate
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate, decay_steps=10000, decay_rate=0.96, staircase=True
        )
        self.model.compile(
            loss=self.loss_function,
            optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        )

    def save(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
            os.makedirs(os.path.join(folder, 'weights'))

        with open(os.path.join(folder, 'params.pkl'), 'wb') as f:
            pickle.dump([
                self.size,
                self.filters,
                self.strides,
                self.dense_size,
            ], f)

    def load_weights(self, filepath):
        self.model.built = True
        self.model.load_weights(filepath)

    def train(self, train_dataset, valid_dataset, epochs, run_folder, initial_epoch=0):

        checkpoint_filepath = os.path.join(
            run_folder, "weights/weights-{epoch:03d}-{loss:.4f}-{val_loss:.4f}.h5")
        checkpoint1 = keras.callbacks.ModelCheckpoint(
            checkpoint_filepath, save_weights_only=True)
        checkpoint2 = keras.callbacks.ModelCheckpoint(
            os.path.join(run_folder, 'weights/weights.h5'),
            save_weights_only=True)

        custom_callback = CallbackSinglePrediction(
            next(valid_dataset.dataset_iter)
        )

        callbacks_list = [checkpoint1, checkpoint2, custom_callback]

        self.model.fit(
            train_dataset.dataset, validation_data=valid_dataset.dataset,
            epochs=epochs, initial_epoch=initial_epoch, callbacks=callbacks_list
        )
