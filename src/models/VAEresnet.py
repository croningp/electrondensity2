from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from src.models.VAE import VariationalAutoencoder
from src.models.layers import identity_block, conv_block


class VAEresnet(VariationalAutoencoder):

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

        x = conv_block(x, 3, [64, 64, 256], stage=2, block='a')   
        x = identity_block(x, 3, [64, 64, 256], stage=1, block='a')

        self.model = Model(encoder_input, x)
