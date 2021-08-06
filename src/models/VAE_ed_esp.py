##########################################################################################
#
# This code is very similar to VAEresnet, but instead of trying to reconstruct its input
# as a VAE does, it tries to convert electron densities to electro static potentials.
#
# Author: Juan Manuel Parrilla Gutierrez (juanma@chem.gla.ac.uk)
#
##########################################################################################


import tensorflow as tf
from tensorflow.keras.layers import Input, UpSampling3D
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

import numpy as np
import pickle

from src.models.VAE import VariationalAutoencoder, VAEModel
from src.models.layers import identity_block, conv_block
from src.utils import transform_ed


class ED2ESP(VAEModel):

    def __init__(self, encoder, decoder, r_loss_factor, **kwargs):
        super().__init__(encoder, decoder, r_loss_factor, **kwargs)

    def preprocess_data(self, data, usetanh=True):
        """ Jarek put the data through a tanh and then through a log """
        # from Jarek, data will be scaled between 0 (there are no negs) to 1

        if usetanh:
            data = tf.tanh(data)
        # from jarek. it applies a log
        return transform_ed(data)


    def preprocess_esp(self, data):
        """ Preprocesses esps by normalizing it between 0 and 1, and doing a dillation
        so that a data point uses a 5x5x5 area instead of a single cell"""

        # first we will do a dillation, which needs to be done for both + and -
        datap = tf.nn.max_pool3d(data, 5, 1, 'SAME')
        datan = tf.nn.max_pool3d(data*-1, 5, 1, 'SAME')
        data = datap + datan*-1

        # I have pre-calculated that data goes between -0.265 and 0.3213
        # with this division it will be roughly between -1 and 1
        data = data / 0.33
        # now we place it between 0 and 1
        data = (data+1) * 0.5
        return data

    def losses(self, data):
        """ KL loss + reconstruction loss
        data[0] has the ED, data[1] has the ESP"""

        x = self.preprocess_data(data[0])
        y = self.preprocess_esp(data[1])
        
        # z_mean, z_log_var, z = self.encoder(x)
        z = self.encoder(x)
        y_nn = self.decoder(z)
        reconstruction_loss = tf.reduce_mean(
            tf.square(y - y_nn), axis=[1, 2, 3, 4]
        )
        # reconstruction_loss *= self.r_loss_factor
        # kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        # kl_loss = tf.reduce_sum(kl_loss, axis=1)
        # kl_loss *= -0.5
        # total_loss = reconstruction_loss + kl_loss
        # return total_loss, reconstruction_loss, kl_loss
        return reconstruction_loss

    def train_step(self, data):
        with tf.GradientTape() as tape:
            reconstruction_loss = self.losses(data)
        grads = tape.gradient(reconstruction_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        # update the trackers
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        return {
            "loss": self.reconstruction_loss_tracker.result(),
        }

    def test_step(self, data):
        reconstruction_loss = self.losses(data)
        # update the trackers
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        return {
            "loss": self.reconstruction_loss_tracker.result(),
        }

    def call(self, inputs, usetanh=True):
        """ inputs must be as fetched from the TFRecordLoader """
        latent = self.encoder(self.preprocess_data(inputs, usetanh))
        return self.decoder(latent)

class VAE_ed_esp(VariationalAutoencoder):
    """This is exactly the same as in VAEresnet, the only different is the very last
    line where it uses ED2ESP instead of VAEmodel"""

    def __init__(
        self, input_dim,
        encoder_conv_filters, encoder_conv_kernel_size, encoder_conv_strides,
        dec_conv_t_filters, dec_conv_t_kernel_size, dec_conv_t_strides,
        z_dim, r_loss_factor, use_batch_norm=False, use_dropout=False,
    ):

        super().__init__(
            input_dim,
            encoder_conv_filters, encoder_conv_kernel_size, encoder_conv_strides,
            dec_conv_t_filters, dec_conv_t_kernel_size, dec_conv_t_strides,
            z_dim, r_loss_factor, use_batch_norm, use_dropout
        )

    def _build(self):

        # THE ENCODER
        encoder_input = Input(shape=self.input_dim, name='encoder_input')
        x = encoder_input

        for i in range(self.n_layers_encoder):
            # just fetch the parameters in a variable so it doesn't get super long
            filters = self.encoder_conv_filters[i]
            kernel_size = self.encoder_conv_kernel_size[i]
            strides = self.encoder_conv_strides[i]
            # and create the residual blocks. I follow how resnet50 does it.
            x = conv_block(x, kernel_size, filters, stage=i, block='a', strides=strides)
            x = identity_block(x, kernel_size, filters, stage=i, block='b')

        # shape_before_flattening = K.int_shape(x)[1:]

        # x = Flatten()(x)
        # self.mu = Dense(self.z_dim, name='mu')(x)
        # self.log_var = Dense(self.z_dim, name='log_var')(x)

        # self.z = Sampling(name='encoder_output')([self.mu, self.log_var])

        # self.encoder = Model(
        #     encoder_input, [self.mu, self.log_var, self.z], name='encoder'
        #     )
        self.encoder = Model (encoder_input, x, name='encoder')

        # THE DECODER
        # decoder_input = Input(shape=(self.z_dim,), name='decoder_input')
        decoder_input = Input(shape=K.int_shape(x)[1:], name='decoder_input')
        x = decoder_input

        # x = Dense(np.prod(shape_before_flattening))(decoder_input)
        # x = Reshape(shape_before_flattening)(x)

        for i in range(self.n_layers_decoder):
            # just fetch the parameters in a variable so it doesn't get super long
            filters = self.decoder_conv_t_filters[i]
            kernel_size = self.decoder_conv_t_kernel_size[i]
            strides = self.decoder_conv_t_strides[i]

            # in the decoder we will upsample instead of using conv strides to downsample
            for _ in range(strides-1):
                x = UpSampling3D()(x)

            stage = i+self.n_layers_encoder  # to get a number to continue naming
            # and create the residual blocks. I follow how resnet50 does it.
            x = conv_block(x, kernel_size, filters, stage=stage, block='a', strides=1)
            x = identity_block(x, kernel_size, filters, stage=stage, block='b')

        # last one with 1 feature map
        x = conv_block(x, kernel_size, 1, stage=stage+1, block='a', strides=1)

        decoder_output = x
        self.decoder = Model(decoder_input, decoder_output, name='decoder')
        # THE FULL VAE
        self.model = ED2ESP(self.encoder, self.decoder, self.r_loss_factor)

    def sample_model_validation(self, valid_dataset, savepath=None, num_batches=1):
        """
        Generates a given number of ESPs from the model and
        saves them to disk if path is given.

        Args:
            valid_dataset: it must be a tfrecord, loaded with tfrecorloader
            savepath: str to path where to save the results
            num_batches: int how many batches to generate
        """

        original_eds = []
        original_esps = []
        generated_esps = []

        for i in range(num_batches):
            next_batch = valid_dataset.next()
            # get batch of eds and transform into esps
            cubes = self.model(next_batch[0])
            # now transform them from 0..1 to -1 .. 1
            cubes = (cubes*2)-1
            # now between -0.33 and 0.33 which is the range of the orig data
            cubes = cubes * 0.33
            generated_esps.extend(cubes.numpy())
            original_eds.extend(next_batch[0].numpy())
            original_esps.extend(next_batch[1].numpy())


        generated_esps = np.array(generated_esps)[:10]
        original_esps = np.array(original_esps)[:10]
        original_eds = np.array(original_eds)[:10]

        if savepath is not None:
            print('Electron densities saved to {}'.format(savepath))
            with open(savepath, 'wb') as pfile:
                pickle.dump([original_eds, original_esps, generated_esps], pfile)

        return original_eds, original_esps, generated_esps
