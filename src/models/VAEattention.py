from tensorflow.keras.layers import Input, Flatten, Dense, Reshape, UpSampling3D, Conv3D
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

import numpy as np

from src.models.VAE import VariationalAutoencoder, Sampling, VAEModel
from src.models.layers import conv_block, identity_block, TransformerBlock
from src.models.layers import GoogleAttention as Attention


class VAEattention(VariationalAutoencoder):

    def __init__(
        self, input_dim,
        encoder_conv_filters, encoder_conv_kernel_size, encoder_conv_strides,
        dec_conv_t_filters, dec_conv_t_kernel_size, dec_conv_t_strides,
        z_dim, r_loss_factor, use_batch_norm=False, use_dropout=False
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
            fmaps = [filters//4, filters//4, filters]
            kernel_size = self.encoder_conv_kernel_size[i]
            strides = self.encoder_conv_strides[i]

            if i == 0: # first iteration are a sort of pseudo-embedding
                x = Conv3D(filters, kernel_size, strides=strides, padding='SAME')(x)
            if i < 2: # first 2 iterations add transformer block
                x = TransformerBlock(filters)(x)
            else: # for the others add convs
                # and create the residual blocks. I follow how resnet50 does it.
                x = conv_block(x, kernel_size, fmaps, stage=i, block='a', strides=strides)
                x = identity_block(x, kernel_size, fmaps, stage=i, block='b')

        shape_before_flattening = K.int_shape(x)[1:]

        x = Flatten()(x)
        self.mu = Dense(self.z_dim, name='mu')(x)
        self.log_var = Dense(self.z_dim, name='log_var')(x)

        self.z = Sampling(name='encoder_output')([self.mu, self.log_var])

        self.encoder = Model(
            encoder_input, [self.mu, self.log_var, self.z], name='encoder'
            )

        # # THE DECODER
        decoder_input = Input(shape=(self.z_dim,), name='decoder_input')

        x = Dense(np.prod(shape_before_flattening))(decoder_input)
        x = Reshape(shape_before_flattening)(x)

        for i in range(self.n_layers_decoder):
            # just fetch the parameters in a variable so it doesn't get super long
            filters = self.decoder_conv_t_filters[i]
            fmaps = [filters//4, filters//4, filters]
            kernel_size = self.decoder_conv_t_kernel_size[i]
            strides = self.decoder_conv_t_strides[i]
            stage = i+self.n_layers_encoder  # to get a number to continue naming

            if i < 3: # first 3 iterations just decode
                for j in range(strides-1):
                    x = UpSampling3D()(x)
                # create the residual block
                x = conv_block(x, kernel_size, fmaps, stage=stage, block='a', strides=1)
                x = identity_block(x, kernel_size, fmaps, stage=stage, block='b')
            if i == 3:
                x = conv_block(x, kernel_size, fmaps, stage=stage, block='a', strides=1)
            if i > 2: # last two iterations transformer blocks
                x = TransformerBlock(filters)(x)

            if i == (self.n_layers_decoder-1): # very last iteration last conv
                x = UpSampling3D()(x)
                x = Conv3D(filters, kernel_size, padding='SAME')(x)

        # last one with 1 feature map
        x = Conv3D(1, kernel_size, padding='SAME')(x)
        #x = conv_block(x, kernel_size, [1, 1, 1], stage=stage+1, block='a', strides=1)

        decoder_output = x
        self.decoder = Model(decoder_input, decoder_output, name='decoder')
        # THE FULL VAE
        self.model = VAEModel(self.encoder, self.decoder, self.r_loss_factor)
