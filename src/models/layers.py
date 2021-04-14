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

    def __init__(self, channels, name="gattention", **kwargs):
        super(GoogleAttention, self).__init__(name=name, **kwargs)
        self.ch = channels

    def hwd_flatten(self, x):
        """ From (h,w,d,c) to (h*w*d,c) """
        s = K.int_shape(x)[1:]
        return layers.Reshape((-1, s[-1]))(x)

    def build(self, input_shape):
        """ from https://stackoverflow.com/a/50820539/515028 """
        # Create a trainable weight variable for this layer:
        self.gamma = self.add_weight(name='gamma', shape=[1],
                                     initializer='uniform', trainable=True)
        # another possible init is tf.keras.initializers.Constant(0)
        # original paper uses initializer=tf.initializers.Zeros

        super(GoogleAttention, self).build(input_shape)

    def call(self, x):

        conv = layers.Conv3D  # just to simplify naming
        batch_size, height, width, depth, num_channels = x.get_shape().as_list()
        f = conv(self.ch // 8, 1, strides=1, kernel_initializer='orthogonal')(x)
        g = conv(self.ch // 8, 1, strides=1, kernel_initializer='orthogonal')(x)
        h = conv(self.ch // 2, 1, strides=1, kernel_initializer='orthogonal')(x)

        # N = h * w * d
        s = tf.matmul(self.hwd_flatten(g), self.hwd_flatten(f), transpose_b=True)  # [bs, N, N]

        beta = layers.Softmax()(s)  # attention map

        o = tf.matmul(beta, self.hwd_flatten(h))  # [bs, N, C]

        o = tf.reshape(o, shape=[batch_size, height, width, num_channels // 2])  # [bs, h, w, C]
        o = conv(self.ch, 1, strides=1, kernel_initializer='orthogonal')(o)
        x = self.gamma * o + x

        return x
