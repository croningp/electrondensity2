##########################################################################################
#
# Variational AutoEncoder for 3D Electron Density Maps Code heavily based on:
# https://github.com/davidADSP/GDL_code/blob/tensorflow_2/models/VAE.py
# Which is from the book "Generative Deep Learning".
#
# I am not going to comment every line because I will mostly re-use the code
# from that repo. I suggest you get the book and check chapter 3, because the
# code is very well explained there.
#
# Author: Juan Manuel Parrilla Gutierrez (juanma@chem.gla.ac.uk)
#
##########################################################################################


from tensorflow.keras.layers import Input, Conv3D, Flatten, Dense, Conv3DTranspose
from tensorflow.keras.layers import Reshape, Activation, BatchNormalization
from tensorflow.keras.layers import LeakyReLU, Dropout, Layer
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler

import tensorflow as tf

import numpy as np
import os
import pickle

from src.utils import transform_ed, transform_back_ed


class Sampling(Layer):
    """Given mu and log_var, as provided by the VAE envoder, it will sample an
    individual. In the GDL book this was a function in VAE _build. No idea why
    he made it a class now.
    """

    def call(self, inputs):
        mu, log_var = inputs
        epsilon = K.random_normal(shape=K.shape(mu), mean=0., stddev=1.)
        return mu + K.exp(log_var / 2) * epsilon


class VAEModel(Model):
    """Init will receive encoder and descoder models. The original author (GDL) used this
    class mainly to add the train_step function which will calculate the KL loss.
    In the GDL book this was a function in _build. Now he made it a bit more complex
    being its own class and using GradientTape.
    """

    def __init__(self, encoder, decoder, r_loss_factor, **kwargs):
        super(VAEModel, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.r_loss_factor = r_loss_factor
        # to track the different losses
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    def preprocess_data(self, data):
        """ Jarek put the data through a tanh and then through a log """
        # from Jarek, data will be scaled between 0 (there are no negs) to 1
        data = tf.tanh(data)
        # from jarek. it applies a log
        return transform_ed(data)

    def losses(self, data):
        """ KL loss + reconstruction loss"""
        z_mean, z_log_var, z = self.encoder( self.preprocess_data(data) )
        reconstruction = self.decoder(z)
        reconstruction_loss = tf.reduce_mean(
            tf.square(data - reconstruction), axis=[1, 2, 3, 4]
        )
        reconstruction_loss *= self.r_loss_factor
        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = tf.reduce_sum(kl_loss, axis=1)
        kl_loss *= -0.5
        total_loss = reconstruction_loss + kl_loss
        return total_loss, reconstruction_loss, kl_loss

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            total_loss, reconstruction_loss, kl_loss = self.losses(data)
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        # update the trackers
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        total_loss, reconstruction_loss, kl_loss = self.losses(data)
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }

    def call(self, inputs):
        """ inputs must be as fetched from the TFRecordLoader """
        _, _, latent = self.encoder( self.preprocess_data(inputs) )
        return self.decoder(latent)


class VariationalAutoencoder():
    """Main VAE class that will get the specs and make the encoder and decoder.
    It will use the class above (VAEModel) to join them together and define
    train step. But training is done here.
    """

    def __init__(
        self, input_dim,
        encoder_conv_filters, encoder_conv_kernel_size, encoder_conv_strides,
        dec_conv_t_filters, dec_conv_t_kernel_size, dec_conv_t_strides,
        z_dim, r_loss_factor, use_batch_norm=False, use_dropout=False
    ):

        self.name = 'variational_autoencoder'

        self.input_dim = input_dim
        self.encoder_conv_filters = encoder_conv_filters
        self.encoder_conv_kernel_size = encoder_conv_kernel_size
        self.encoder_conv_strides = encoder_conv_strides
        self.decoder_conv_t_filters = dec_conv_t_filters
        self.decoder_conv_t_kernel_size = dec_conv_t_kernel_size
        self.decoder_conv_t_strides = dec_conv_t_strides
        self.z_dim = z_dim
        self.r_loss_factor = r_loss_factor

        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout

        self.n_layers_encoder = len(encoder_conv_filters)
        self.n_layers_decoder = len(dec_conv_t_filters)

        self._build()

    def _build(self):

        # THE ENCODER
        encoder_input = Input(shape=self.input_dim, name='encoder_input')

        x = encoder_input

        for i in range(self.n_layers_encoder):
            conv_layer = Conv3D(
                filters=self.encoder_conv_filters[i],
                kernel_size=self.encoder_conv_kernel_size[i],
                strides=self.encoder_conv_strides[i],
                padding='same', name='encoder_conv_' + str(i)
                )

            x = conv_layer(x)

            if self.use_batch_norm:
                x = BatchNormalization()(x)

            x = LeakyReLU()(x)

            if self.use_dropout:
                x = Dropout(rate=0.25)(x)

        shape_before_flattening = K.int_shape(x)[1:]

        x = Flatten()(x)
        x = Dense(self.z_dim*2, name='before_latent')(x)
        self.mu = Dense(self.z_dim, name='mu')(x)
        self.log_var = Dense(self.z_dim, name='log_var')(x)

        self.z = Sampling(name='encoder_output')([self.mu, self.log_var])

        self.encoder = Model(
            encoder_input, [self.mu, self.log_var, self.z], name='encoder'
            )

        # THE DECODER
        decoder_input = Input(shape=(self.z_dim,), name='decoder_input')

        x = Dense(self.z_dim*2, name='after_latent')(decoder_input)
        x = Dense(np.prod(shape_before_flattening))(x)
        x = Reshape(shape_before_flattening)(x)

        for i in range(self.n_layers_decoder):
            conv_t_layer = Conv3DTranspose(
                filters=self.decoder_conv_t_filters[i],
                kernel_size=self.decoder_conv_t_kernel_size[i],
                strides=self.decoder_conv_t_strides[i],
                padding='same', name='decoder_conv_t_' + str(i)
                )

            x = conv_t_layer(x)

            if i < self.n_layers_decoder - 1:
                if self.use_batch_norm:
                    x = BatchNormalization()(x)
                x = LeakyReLU()(x)
                if self.use_dropout:
                    x = Dropout(rate=0.25)(x)
            else:
                x = Activation('sigmoid')(x)

        decoder_output = x

        self.decoder = Model(decoder_input, decoder_output, name='decoder')

        # THE FULL VAE
        self.model = VAEModel(self.encoder, self.decoder, self.r_loss_factor)

    def compile(self, learning_rate):
        self.learning_rate = learning_rate
        optimizer = Adam(lr=learning_rate)
        self.model.compile(optimizer=optimizer)

    def save(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
            os.makedirs(os.path.join(folder, 'weights'))

        with open(os.path.join(folder, 'params.pkl'), 'wb') as f:
            pickle.dump([
                self.input_dim, self.encoder_conv_filters,
                self.encoder_conv_kernel_size, self.encoder_conv_strides,
                self.decoder_conv_t_filters, self.decoder_conv_t_kernel_size,
                self.decoder_conv_t_strides, self.z_dim,
                self.use_batch_norm, self.use_dropout
                ], f)

    def load_weights(self, filepath):
        self.model.built = True
        self.model.load_weights(filepath)

    def step_decay_schedule(self, initial_lr, decay_factor=0.5, step_size=1):
        '''
        Wrapper function to create a LearningRateScheduler with step decay schedule.
        '''
        def schedule(epoch):
            new_lr = initial_lr * (decay_factor ** np.floor(epoch/step_size))
            return new_lr

        return LearningRateScheduler(schedule)

    def train(
        self, train_dataset, valid_dataset, epochs, run_folder,
        initial_epoch=0, lr_decay=1
    ):

        lr_sched = self.step_decay_schedule(
            initial_lr=self.learning_rate, decay_factor=lr_decay, step_size=1
        )

        checkpoint_filepath = os.path.join(
                run_folder, "weights/weights-{epoch:03d}-{loss:.2f}.h5")
        checkpoint1 = ModelCheckpoint(
            checkpoint_filepath, save_weights_only=True)
        checkpoint2 = ModelCheckpoint(
            os.path.join(run_folder, 'weights/weights.h5'),
            save_weights_only=True)

        callbacks_list = [checkpoint1, checkpoint2, lr_sched]

        self.model.fit(
            train_dataset, validation_data=valid_dataset,
            #steps_per_epoch=1, validation_steps=1,
            epochs=epochs, initial_epoch=initial_epoch, callbacks=callbacks_list
        )

    def sample_model_validation(self, valid_dataset, savepath=None, num_batches=10):
        """
        Generates a given number of electron densities from the model and 
        saves them to disk if path is given.
        
        Args:
            valid_dataset: it must be a tfrecord, loaded with tfrecorloader
            savepath: str to path where to save the results
            num_batches: int how many batches to generate
        """
        
        original_cubes = []
        generated_cubes = []
        
        for i in range(num_batches):
            # cubes = self.generator(batch_size, training=False)
            next_batch = valid_dataset.next()[0]
            cubes = self.model(next_batch)
            cubes = transform_back_ed(cubes).numpy()
            generated_cubes.extend(cubes)
            original_cubes.extend(next_batch.numpy())
        
        generated_cubes = np.array(generated_cubes)
        original_cubes = np.array(original_cubes)
        
        if savepath is not None:
            print('Electron densities saved to {}'.format(savepath))
            with open(savepath, 'wb') as pfile:
                pickle.dump([original_cubes, generated_cubes], pfile)
            
        return original_cubes, generated_cubes

