##########################################################################################
#
# Variaton over ED2smiles.py, but using electrostatic potentials (ESP) instead of ED.
# Check that file first, the following script will just extend it a bit.
#
##########################################################################################


import tensorflow as tf
from tensorflow.keras import layers

from src.models.ED2smiles import E2S_Transformer
from src.models.ED2smiles import TransformerEncoder
from src.models.layers import ConvBlock, IdentityBlock


class ElectroStaticPotentialEmbedding(layers.Layer):
    """We need to transform a 4D into a 2D tensor. The drop of dimensionality in this 
    embedding will be achieved by using strides of 2 in 2 out of the 4 dimensions, until 
    they are 1,1,N,N and then we will squeeze them out.
    Same as V2 above but more convs
    """

    def __init__(self, num_hid=64):
        super().__init__()

        self.conv32 = ConvBlock(
            kernel_size=3, filters=num_hid, stage=0, block='a', strides=[2, 2, 2])
        self.conv16 = ConvBlock(kernel_size=3, filters=num_hid, stage=1, block='a',
                                strides=[2, 2, 1])
        self.conv8 = ConvBlock(kernel_size=3, filters=num_hid, stage=2, block='a',
                               strides=[2, 2, 1])
        self.conv4 = ConvBlock(kernel_size=3, filters=num_hid, stage=3, block='a',
                               strides=[2, 2, 1])
        self.conv2 = ConvBlock(kernel_size=3, filters=num_hid, stage=4, block='a',
                               strides=[2, 2, 1])
        self.conv1 = ConvBlock(kernel_size=3, filters=num_hid, stage=5, block='a',
                               strides=[2, 2, 1])

        self.id32 = IdentityBlock(
            kernel_size=3, filters=num_hid, stage=0, block='a')
        self.id16 = IdentityBlock(
            kernel_size=3, filters=num_hid, stage=1, block='a')
        self.id8 = IdentityBlock(
            kernel_size=3, filters=num_hid, stage=2, block='a')
        self.id4 = IdentityBlock(
            kernel_size=3, filters=num_hid, stage=3, block='a')
        self.id2 = IdentityBlock(
            kernel_size=3, filters=num_hid, stage=4, block='a')
        self.id1 = IdentityBlock(
            kernel_size=3, filters=num_hid, stage=5, block='a')

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

    def call(self, x):
        # First we do the pre-processing Jarek was doing
        x = self.preprocess_esp(x)
        # now from 64,64,64,1 to 32,32,32,num_hid
        x = self.conv32(x)
        x = self.id32(x)
        # now from 32,32,32,num_hid to 16,16,32,num_hid
        x = self.conv16(x)
        x = self.id16(x)
        # now from 16,16,32,1 to 8,8,32,num_hid
        x = self.conv8(x)
        x = self.id8(x)
        # now from 8,8,32,1 to 4,4,32,num_hid
        x = self.conv4(x)
        x = self.id4(x)
        # now from 4,4,32,num_hid to 2,2,32,num_hid
        x = self.conv2(x)
        x = self.id2(x)
        # now from 2,2,32,num_hid to 1,1,32,num_hid
        x = self.conv1(x)
        x = self.id1(x)

        return tf.squeeze(x, [1, 2])


class ESP2S_Transformer(E2S_Transformer):
    def __init__(
        self,
        num_hid=64,
        num_head=2,
        num_feed_forward=128,
        target_maxlen=24,  # max len of the smiles strings is 24, as set by Jarek
        num_layers_enc=4,
        num_layers_dec=1,
        num_classes=33,  # There are 33 different smiles tokens
    ):
        super().__init__(num_hid, num_head, num_feed_forward, target_maxlen,
                         num_layers_enc, num_layers_dec, num_classes)

        self.enc_input = ElectroStaticPotentialEmbedding(num_hid=num_hid)

        self.encoder = tf.keras.Sequential(
            [self.enc_input]
            + [
                TransformerEncoder(num_hid, num_head, num_feed_forward)
                for _ in range(num_layers_enc)
            ],
            name="transformer_encoder"
        )

        