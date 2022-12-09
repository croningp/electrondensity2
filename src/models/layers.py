##########################################################################################
#
# Different layers to use in our models.
# - identity_block and conv_block are copied from keras resnet50, using conv3d instead.
# - attention is adapted from taki0112 SAGAN implementation
#
# Author: Juan Manuel Parrilla Gutierrez (juanma@chem.gla.ac.uk)
#
##########################################################################################


import math
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.layers import LeakyReLU


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: Integer describing the size of the last conv
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # decide type of convolutions depending on input tensor
    if len(input_tensor.shape) == 5:
        conv = layers.Conv3D
    elif len(input_tensor.shape) == 4:
        conv = layers.Conv2D
    else:
        conv = layers.Conv1D

    x = conv(filters // 4, kernel_size, padding='same', kernel_initializer='orthogonal',
             name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = conv(filters, kernel_size, padding='same', kernel_initializer='orthogonal',
             name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(name=bn_name_base + '2b')(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=2, activation='r'):
    """A block that has a conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: Integer describing the size of the last conv
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.
        activation: the type of activation
    # Returns
        Output tensor for the block.
    """

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # decide type of convolutions depending on input tensor
    if len(input_tensor.shape) == 5:
        conv = layers.Conv3D
    elif len(input_tensor.shape) == 4:
        conv = layers.Conv2D
    else:
        conv = layers.Conv1D

    # decide type of activation
    if activation == 'r':
        act_fun = layers.Activation('relu')
    elif activation == 'l':
        act_fun = LeakyReLU(alpha=0.2)

    # first conv is smaller
    filters1 = math.ceil(filters/4)

    x = conv(filters1, kernel_size, strides=strides, padding='same',
             kernel_initializer='orthogonal', name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(name=bn_name_base + '2a')(x)
    x = act_fun(x)

    x = conv(filters, kernel_size, padding='same',
             kernel_initializer='orthogonal', name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(name=bn_name_base + '2b')(x)

    shortcut = conv(filters, 1, strides=strides,
                    kernel_initializer='orthogonal',
                    name=conv_name_base + '1')(input_tensor)
    shortcut = layers.BatchNormalization(name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = act_fun(x)
    return x


def drop_dimension(input_tensor):
    """Given a 4D tensor it will return a 3D one, and given a 3D tensor it will return a
    2D one. I could just use reshape, but I decided to use conv to reduce channel to 1,
    and then squeeze.

    Args:
        input_tensor: 4D or 3D tensor.

    Returns:
        Tensor with one less dimension.
    """

    # decide type of convolutions depending on input tensor
    if len(input_tensor.shape) > 4:
        conv = layers.Conv3D
    else:
        conv = layers.Conv2D

    # do a convolution with only 1 channel
    channel1 = conv(1, kernel_size=3,
                    padding='same', kernel_initializer='orthogonal')(input_tensor)
    # and squeeze it out
    return tf.squeeze(channel1, -1)


class DropDimension(layers.Layer):
    """ Same as he function above "drop_dimension", but in a subclassed layer
    """
    def __init__(self):
        super(DropDimension, self).__init__()

    def build(self, input_shape):
        # decide type of convolutions depending on input tensor
        if len(input_shape) == 5:
            conv = layers.Conv3D
        else:
            conv = layers.Conv2D

        self.channel1 = conv(1, kernel_size=3, padding='same', kernel_initializer='orthogonal')

    def call(self, input_tensor):
        # do a convolution with only 1 channel
        channel1 = self.channel1(input_tensor)
        # and squeeze it out
        return tf.squeeze(channel1, -1)


def add_dimension(input_tensor, filters=64):
    """Given a 3D tensor it will return a 4D one, and given a 2D tensor it will return a
    3D one. As I above with drop_dimension, instead of just expand_dims here, I will also
    do a conv.

    Args:
        input_tensor: 3D or 2D tensor.
        filters: the number of filters used in the post convolution.

    Returns:
        Tensor with one more dimension.
    """

    # decide type of convolutions depending on input tensor
    if len(input_tensor.shape) > 3:
        conv = layers.Conv3D
    else:
        conv = layers.Conv2D

    # expand dimensions at the end
    it = tf.expand_dims(input_tensor, -1)
    # do a convolution with only 1 channel
    cf = conv(filters, kernel_size=3, padding='same', kernel_initializer='orthogonal')(it)
    return cf

class ConvBlock(layers.Layer):
    """ Same as the function convblock above, but in a subclassed layer.
    """
    def __init__(self, kernel_size, filters, stage, block, strides=2):
        super(ConvBlock, self).__init__()
        self.conv_name_base = 'res' + str(stage) + block + '_branch'
        self.bn_name_base = 'bn' + str(stage) + block + '_branch'
        self.kernel_size = kernel_size
        self.filters = filters
        self.strides = strides

    def build(self, input_shape):
        # decide type of convolutions depending on input tensor
        if len(input_shape) == 5:
            conv = layers.Conv3D
        elif len(input_shape) == 4:
            conv = layers.Conv2D
        else:
            conv = layers.Conv1D

        filters = self.filters
        strides = self.strides
        kernel_size = self.kernel_size
        
        self.conv1 = conv(filters // 4, kernel_size, strides=strides, padding='same',
             kernel_initializer='orthogonal', name=self.conv_name_base + '2a')
        self.bn_conv1 = layers.BatchNormalization(name=self.bn_name_base + '2a')
        self.act_conv1 = layers.Activation('relu')

        self.conv2 = conv(filters, kernel_size, padding='same',
             kernel_initializer='orthogonal', name=self.conv_name_base + '2b')
        self.bn_conv2 = layers.BatchNormalization(name=self.bn_name_base + '2b')

        self.convsc = conv(filters, 1, strides=strides, padding='same',
             kernel_initializer='orthogonal', name=self.conv_name_base + '1')
        self.bn_convsc = layers.BatchNormalization(name=self.bn_name_base + '1')
        self.act_final = layers.Activation('relu')

    def call(self, input_tensor):

        x = self.conv1(input_tensor)
        x = self.bn_conv1(x)
        x = self.act_conv1(x)

        x = self.conv2(x)
        x = self.bn_conv2(x)

        shortcut = self.convsc(input_tensor)
        shortcut = self.bn_convsc(shortcut)

        x = layers.add([x, shortcut])
        x = self.act_final(x)
        return x


class IdentityBlock(layers.Layer):
    """ Same as the function identity_block above, but in a subclassed layer.
    """
    def __init__(self, kernel_size, filters, stage, block):
        super(IdentityBlock, self).__init__()
        self.conv_name_base = 'res' + str(stage) + block + '_branch'
        self.bn_name_base = 'bn' + str(stage) + block + '_branch'
        self.kernel_size = kernel_size
        self.filters = filters

    def build(self, input_shape):
        # decide type of convolutions depending on input tensor
        if len(input_shape) == 5:
            conv = layers.Conv3D
        elif len(input_shape) == 4:
            conv = layers.Conv2D
        else:
            conv = layers.Conv1D

        filters = self.filters
        kernel_size = self.kernel_size
        
        self.conv1 = conv(filters // 4, kernel_size, padding='same',
             kernel_initializer='orthogonal', name=self.conv_name_base + '2a')
        self.bn_conv1 = layers.BatchNormalization(name=self.bn_name_base + '2a')
        self.act_conv1 = layers.Activation('relu')

        self.conv2 = conv(filters, kernel_size, padding='same',
             kernel_initializer='orthogonal', name=self.conv_name_base + '2b')
        self.bn_conv2 = layers.BatchNormalization(name=self.bn_name_base + '2b')

        self.act_final = layers.Activation('relu')

    def call(self, input_tensor):

        x = self.conv1(input_tensor)
        x = self.bn_conv1(x)
        x = self.act_conv1(x)

        x = self.conv2(x)
        x = self.bn_conv2(x)

        x = layers.add([x, input_tensor])
        x = self.act_final(x)
        return x


class Attention(layers.Layer):
    """ Adapted from SAGAN paper.
    Check this: https://github.com/taki0112/Self-Attention-GAN-Tensorflow/
    It will use 3D or 2D convs depending on the input_shape.
    """

    def __init__(self, channels):
        super(Attention, self).__init__()
        self.ch = channels

    def hwd_flatten(self, x):
        """ From (h,w,d,c) to (h*w*d,c) if 4D or (h,w,c) to (h*w,c) if 3D"""
        s = K.int_shape(x)[1:]
        return layers.Reshape((-1, s[-1]))(x)

    def build(self, input_shape):
        # decide type of convolutions depending on input tensor
        if len(input_shape) > 4:
            self.conv = layers.Conv3D
        else:
            self.conv = layers.Conv2D

        self.f = self.conv(self.ch // 8, 1, strides=2, kernel_initializer='orthogonal')
        self.g = self.conv(self.ch // 8, 1, strides=2, kernel_initializer='orthogonal')
        self.h = self.conv(self.ch // 2, 1, strides=2, kernel_initializer='orthogonal')
        self.v = tf.keras.Sequential(
            [layers.UpSampling3D(),
             self.conv(self.ch, 1, strides=1, kernel_initializer='orthogonal'), ]
        )

        # Create a trainable weight variable for this layer:
        self.gamma = self.add_weight(name='gamma', shape=[1],
                                     # initializer='uniform',
                                     initializer=tf.keras.initializers.Constant(0),
                                     trainable=True)

        super(Attention, self).build(input_shape)

    def call(self, x):

        f = self.f(x)
        g = self.g(x)
        h = self.h(x)

        # N = h * w * d -- [bs, N, N]
        s = tf.matmul(self.hwd_flatten(g), self.hwd_flatten(f), transpose_b=True)
        beta = layers.Softmax()(s)  # attention map
        o = tf.matmul(beta, self.hwd_flatten(h))  # [bs, N, C]

        if self.conv == layers.Conv3D:  # [bs, h, w, d, C]
            height, width, depth, num_channels = K.int_shape(x)[1:]
            o = layers.Reshape((height//2, width//2, depth//2, num_channels // 2))(o)
        else:  # self.conv == layers.Conv2D [bs, h, w, C]
            height, width, num_channels = K.int_shape(x)[1:]
            o = layers.Reshape((height//2, width//2, num_channels // 2))(o)

        o = self.v(o)
        x = self.gamma * o + x

        return x


class TransformerBlock(layers.Layer):
    """ Adapted from https://keras.io/examples/nlp/text_classification_with_transformer/
    Using the attention defined above and Conv instead of dense layer.
    """

    def __init__(self,  channels, dense_factor=4, dropout=0.1):
        super(TransformerBlock, self).__init__()

        self.channels, self.df, self.dropout = channels, dense_factor, dropout

        self.attention = Attention(self.channels)
        self.dropout_attention = layers.Dropout(dropout)
        self.add_attention = layers.Add()
        self.bn_attention = layers.BatchNormalization()
        self.dropout_fcn = layers.Dropout(dropout)
        self.bn_fcn = layers.BatchNormalization()

    def build(self, input_shape):
        # decide type of convolutions depending on input tensor
        if len(input_shape) > 4:
            self.conv = layers.Conv3D
        else:
            self.conv = layers.Conv2D

        self.fcn = tf.keras.Sequential(
            [self.conv(self.channels*self.df, 1, activation='relu', padding='SAME'),
             self.conv(self.channels, 1, padding='SAME'), ]
        )

    def call(self, inputs, training=None):
        attn_output = self.attention(inputs)
        attn_output = self.dropout_attention(attn_output, training=training)
        out1 = self.bn_attention(inputs + attn_output)

        # Feed Forward, that here we do with convs
        fcn_output = self.fcn(out1)
        fcn_output = self.dropout_fcn(fcn_output, training=training)
        x = self.bn_fcn(out1 + fcn_output)

        return x
