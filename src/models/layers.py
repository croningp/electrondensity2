##########################################################################################
#
# Different layers to use in our models.
# - identity_block and conv_block are copied from keras resnet50, using conv3d instead.
# - attention is adapted from taki0112 SAGAN implementation
#
# Author: Juan Manuel Parrilla Gutierrez (juanma@chem.gla.ac.uk)
#
##########################################################################################


import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # decide type of convolutions depending on input tensor
    if len(input_tensor.shape) > 4:
        conv = layers.Conv3D
    else:
        conv = layers.Conv2D

    x = conv(filters1, kernel_size, padding='same', kernel_initializer='orthogonal',
             name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = conv(filters3, kernel_size, padding='same', kernel_initializer='orthogonal',
             name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(name=bn_name_base + '2b')(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=2):
    """A block that has a conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # decide type of convolutions depending on input tensor
    if len(input_tensor.shape) > 4:
        conv = layers.Conv3D
    else:
        conv = layers.Conv2D

    x = conv(filters1, kernel_size, strides=strides, padding='same',
             kernel_initializer='orthogonal', name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = conv(filters3, kernel_size, padding='same',
             kernel_initializer='orthogonal', name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(name=bn_name_base + '2b')(x)

    shortcut = conv(filters3, 1, strides=strides,
                    kernel_initializer='orthogonal',
                    name=conv_name_base + '1')(input_tensor)
    shortcut = layers.BatchNormalization(name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


class GoogleAttention(layers.Layer):

    def __init__(self, channels):
        super(GoogleAttention, self).__init__()
        self.ch = channels

    def hwd_flatten(self, x):
        """ From (h,w,d,c) to (h*w*d,c) """
        s = K.int_shape(x)[1:]
        return layers.Reshape((-1, s[-1]))(x)

    def build(self, input_shape):
        conv = layers.Conv3D  # just to simplify naming

        self.f = tf.keras.Sequential(
            [conv(self.ch // 8, 1, strides=1, kernel_initializer='orthogonal'),
             layers.MaxPooling3D(), ]
        )
        self.g = tf.keras.Sequential(
            [conv(self.ch // 8, 1, strides=1, kernel_initializer='orthogonal'),
             layers.MaxPooling3D(), ]
        )
        self.h = tf.keras.Sequential(
            [conv(self.ch // 2, 1, strides=1, kernel_initializer='orthogonal'),
             layers.MaxPooling3D(), ]
        )
        self.v = tf.keras.Sequential(
            [layers.UpSampling3D(),
             conv(self.ch, 1, strides=1, kernel_initializer='orthogonal'), ]
        )

        # self.f = conv(self.ch // 8, 1, strides=1, kernel_initializer='orthogonal')
        # self.g = conv(self.ch // 8, 1, strides=1, kernel_initializer='orthogonal')
        # self.h = conv(self.ch // 2, 1, strides=1, kernel_initializer='orthogonal')
        # self.v = conv(self.ch, 1, strides=1, kernel_initializer='orthogonal')

        # Create a trainable weight variable for this layer:
        self.gamma = self.add_weight(name='gamma', shape=[1],
                                     # initializer='uniform',
                                     initializer=tf.keras.initializers.Constant(0),
                                     trainable=True)

        super(GoogleAttention, self).build(input_shape)

    def call(self, x):

        f = self.f(x)
        g = self.g(x)
        h = self.h(x)

        # N = h * w * d -- [bs, N, N]
        s = tf.matmul(self.hwd_flatten(g), self.hwd_flatten(f), transpose_b=True)
        beta = layers.Softmax()(s)  # attention map
        o = tf.matmul(beta, self.hwd_flatten(h))  # [bs, N, C]

        # [bs, h, w, C]
        height, width, depth, num_channels = K.int_shape(x)[1:]
        o = layers.Reshape((height // 2, width // 2, depth // 2, num_channels // 2))(o)
        o = self.v(o)
        x = self.gamma * o + x

        return x


class TransformerBlock(layers.Layer):
    """ Adapted from https://keras.io/examples/nlp/text_classification_with_transformer/
    Using the attention defined above and Conv3D instead of dense layer.
    """

    def __init__(self,  channels, dense_factor=4, dropout=0.1):
        super(TransformerBlock, self).__init__()

        self.channels, self.df, self.dropout = channels, dense_factor, dropout

        self.attention = GoogleAttention(self.channels)
        self.dropout_attention = layers.Dropout(dropout)
        self.add_attention = layers.Add()
        self.layer_norm_attention = layers.LayerNormalization(epsilon=1e-6)

        self.fcn = tf.keras.Sequential(
            [layers.Conv3D(self.channels*self.df, 1, activation='relu'),
             layers.Conv3D(self.channels, 1), ]
        )
        self.dropout_fcn = layers.Dropout(dropout)
        self.layer_norm_fcn = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training=None):
        attn_output = self.attention(inputs)
        attn_output = self.dropout_attention(attn_output, training=training)
        out1 = self.layer_norm_attention(inputs + attn_output)

        # Feed Forward, that here we do with convs
        fcn_output = self.fcn(out1)
        fcn_output = self.dropout_fcn(fcn_output, training=training)
        x = self.layer_norm_fcn(out1 + fcn_output)

        return x
