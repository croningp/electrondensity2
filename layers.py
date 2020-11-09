import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, AvgPool2D, Dense
from tensorflow.keras.layers import Conv2DTranspose, MaxPool2D, UpSampling2D, Layer, RNN, Bidirectional
from tensorflow.keras.layers import Conv3D, MaxPool3D, AvgPool3D, UpSampling3D, Conv3DTranspose, Activation, BatchNormalization
from tensorflow.keras import Model


class ResBlockDown3D(Model):
    """
    Single resnet like layer for downsampling 3D tensors.
    """
    def __init__(self,
                 num_channels,
                 pooling='MaxPool3D',
                 activation_fn='relu',
                 kernel_initializer=tf.keras.initializers.Orthogonal()):
        """
            Layer initializer
            Args:
                num_channels: int the number of output chanells from the layers
                pooling: string MaxPool3D or AvgPool3D to use to reduce dimensions
                activation_fn: str with a type of activation function to use 
                kernel_initializer: kernel initializer for 3DConv layers
        """
        
        super(ResBlockDown3D,self).__init__()
        self.num_channels = num_channels
        self.conv_1x1 = Conv3D(num_channels, 1, padding='same',  kernel_initializer=kernel_initializer)
        self.conv_3x3a = Conv3D(num_channels, 3, padding='same', kernel_initializer=kernel_initializer)
        self.conv_3x3b = Conv3D(num_channels, 3, padding='same', kernel_initializer=kernel_initializer)
        
        if pooling == 'MaxPool3D':
            self.pooling = MaxPool3D(2)
        elif pooling == 'AvgPool3D':
            self.pooling = AvgPool3D(2)
        else:
            raise ValueError('Unknowne type of pooling {}'.format(pooling))
        self.activation = Activation(activation_fn)
            
        
    def call(self, inputs):
        """
        Passes the input tensor through the layer
        Args:
            inputs: tensor of shape [batch_size, cube_dim, cube_dim, cube_dim, inp_chan]
        Returns:
            output: tensor of shape [batch_size, cube_dim//2, cube_dim//2, cube_dim//2, num_channels]
            
        """
        layer_1a = self.conv_1x1(inputs)
        layer_1a = self.pooling(layer_1a)
    
        layer_1b= self.activation(inputs)
        layer_1b = self.conv_3x3a(layer_1b)
        layer_1b = self.activation(layer_1b)
        layer_1b = self.conv_3x3b(layer_1b)
        layer_1b = self.pooling(layer_1b)

        output = layer_1a + layer_1b
        return output
    
class ResBlockUp3D(Model):
    """
    Single resnet like layer for upsampling 3D tensors.
    """
    
    def __init__(self,
                 num_channels,
                 use_batchnorm=False,
                 activation_fn='relu',
                 kernel_initializer=tf.keras.initializers.Orthogonal()):
        """
        Layer initializer
            Args:
                num_channels: int the number of output chanells from the layers
                use_batchnorm: bool if use batch normalization within layer
                activation_fn: str with a type of activation function to use 
                kernel_initializer: kernel initializer for 3DConv layers
        """
        
        
        
        super(ResBlockUp3D, self).__init__()
        self.num_channels = num_channels
        self.use_batchnorm = use_batchnorm
        self.conv_1x1 = Conv3D(num_channels, 1, padding='same',  kernel_initializer=kernel_initializer)
        self.conv_3x3a = Conv3D(num_channels, 3,  padding='same',  kernel_initializer=kernel_initializer)
        self.conv_3x3b = Conv3D(num_channels, 3,  padding='same',  kernel_initializer=kernel_initializer)
        self.activation = Activation(activation_fn)
        
        if use_batchnorm:
            self.batch_norm_1 = BatchNormalization(axis=-1)
            self.batch_norm_2 = BatchNormalization(axis=-1)
        
        self.upsampling = UpSampling3D(2)
        
    def call(self, inputs, training):
        
        """
        Passes the input tensor through the layer
        Args:
            inputs: tensor of shape [batch_size, cube_dim, cube_dim, cube_dim, inp_chan]
        Returns:
            output: tensor of shape [batch_size, cube_dim*2, cube_dim*2, cube_dim*2, num_channels]  
        """
        
        layer_1a = self.upsampling(inputs)
        layer_1a = self.conv_1x1(layer_1a)
        

        if self.use_batchnorm:
            layer_1b = self.batch_norm_1(inputs,training=training)
        else:
            layer_1b = inputs
        
        layer_1b = self.activation(layer_1b)
        layer_1b = self.upsampling(layer_1b)
        layer_1b = self.conv_3x3a(layer_1b)
        
        if self.use_batchnorm:
            layer_1b = self.batch_norm_2(layer_1b,training=training)
        layer_1b = self.activation(layer_1b)
        layer_1b = self.conv_3x3b(layer_1b)
        
        output = layer_1a + layer_1b
        
        return output
    
    
class ResBlockUp2D(Model):
    def __init__(self,
                 num_channels,
                 use_batchnorm=False,
                 activation_fn='relu',
                 kernel_initializer=tf.keras.initializers.Orthogonal()):
        super(ResBlockUp2D, self).__init__()
        self.num_channels = num_channels
        self.use_batchnorm = use_batchnorm
        self.conv_1x1 = Conv2D(num_channels, 1, padding='same',  kernel_initializer=kernel_initializer)
        self.conv_3x3a = Conv2D(num_channels, 3,  padding='same',  kernel_initializer=kernel_initializer)
        self.conv_3x3b = Conv2D(num_channels, 3,  padding='same',  kernel_initializer=kernel_initializer)
        self.activation = Activation(activation_fn)
        
        if use_batchnorm:
            self.batch_norm_1 = BatchNormalization(axis=-1)
            self.batch_norm_2 = BatchNormalization(axis=-1)
        
        self.upsampling = UpSampling2D(2)
        
    def call(self, inputs, training):
        
        layer_1a = self.upsampling(inputs)
        layer_1a = self.conv_1x1(layer_1a)
        

        if self.use_batchnorm:
            layer_1b = self.batch_norm_1(inputs,training=training)
        else:
            layer_1b = inputs
        
        layer_1b = self.activation(layer_1b)
        layer_1b = self.upsampling(layer_1b)
        layer_1b = self.conv_3x3a(layer_1b)
        
        if self.use_batchnorm:
            layer_1b = self.batch_norm_2(layer_1b,training=training)
        layer_1b = self.activation(layer_1b)
        layer_1b = self.conv_3x3b(layer_1b)
        
        output = layer_1a + layer_1b
        
        return output 
    
    
class ResBlockDown2D(Model):
    def __init__(self, num_channels,
                 pooling='MaxPool2D',
                 kernel_initializer='glorot_uniform',
                 activation_fn='relu'):
        super(ResBlockDown2D,self).__init__()
        self.num_channels = num_channels
        self.conv_1x1 = Conv2D(num_channels, 1, padding='same', kernel_initializer=kernel_initializer)
        self.conv_3x3a = Conv2D(num_channels, 3, padding='same', kernel_initializer=kernel_initializer)
        self.conv_3x3b = Conv2D(num_channels, 3, padding='same', kernel_initializer=kernel_initializer)
        
        if pooling == 'MaxPool2D':
            self.pooling = MaxPool2D(2)
        elif pooling == 'AvgPool2D':
            self.pooling = AvgPool2D(2)
        else:
            raise ValueError('Unknowne type of pooling {}'.format(pooling))
        self.activation = Activation(activation_fn)
            
        
    def call(self, inputs):
        
        layer_1a = self.conv_1x1(inputs)
        layer_1a = self.pooling(layer_1a)
    
        layer_1b= self.activation(inputs)
        layer_1b = self.conv_3x3a(layer_1b)
        layer_1b = self.activation(layer_1b)
        layer_1b = self.conv_3x3b(layer_1b)
        layer_1b = self.pooling(layer_1b)
        
        
        output = layer_1a + layer_1b
        
        return output
    
    
class ConvSelfAttn3D(Model):
    """Convodultional self-attention from https://arxiv.org/abs/1805.08318
    expanded to three dimensions.
    """
    def __init__(self,
                 attn_dim,
                 output_dim,
                 kernel_initializer=tf.keras.initializers.Orthogonal()):
        """
            Args:
                attn_dim: int for the attention dimension
                output_dim: feature_size for the attention transformed inputs
                kernel_initializer: str or initializer for the Conv3D used in attention
        """
        super(ConvSelfAttn3D, self).__init__()
        self.f_conv = Conv3D(attn_dim, 1,  kernel_initializer=kernel_initializer)
        self.g_conv = Conv3D(attn_dim, 1,  kernel_initializer=kernel_initializer)
        self.h_conv = Conv3D(attn_dim, 1,  kernel_initializer=kernel_initializer)
        self.v_conv = Conv3D(output_dim, 1,  kernel_initializer=kernel_initializer)
        self.scale = tf.Variable(0.0)
    
    def flatten(self, inputs):
        """Inner flattens the inputs
        Args:
            inputs: tensor of shape [batch_size, cube_dim, cube_dim, cube_dim, hidden_dim]
            output: tensor of [batch_size, cube_dim*cube_dim*cube_dim, hidden_dim]
        
        """
        inputs_shape = inputs.get_shape()
        batch_size = tf.TensorShape([inputs_shape[0]])
        hidden_dims = tf.TensorShape(
            [inputs_shape[1] * inputs_shape[2]* inputs_shape[3]])
        last_dim = tf.TensorShape([inputs_shape[-1]])
        new_shape = batch_size + hidden_dims + last_dim
        new_shape = [inputs_shape[0], tf.reduce_prod(inputs_shape[1:-1]), inputs_shape[-1]]
        flat = tf.reshape(inputs, new_shape)
        return flat
    
    
    def call(self, input):
        """Performs the attention
        Args:
            input: tensor of shape [batch_size, cube_dim, cube_dim, cube_dim, hidden_dim]
        Returns:
            output: tensor of shape [batch_size, cube_dim, cube_dim, cube_dim, output_dim]
            
        """
        fx = self.f_conv(input)
        gx = self.g_conv(input)
        hx = self.h_conv(input)
        
        fx_flat = self.flatten(fx)
        gx_flat = self.flatten(gx)
        hx_flat = self.flatten(hx)
        
        raw_attn_weights = tf.matmul(fx_flat, gx_flat, transpose_b=True)
        raw_attn_weights = tf.transpose(raw_attn_weights, perm=[0,2,1])
        attn_weights = tf.nn.softmax(raw_attn_weights, axis=-1)
        
        attn_flat = tf.matmul(attn_weights, hx_flat)
        attn = tf.reshape(attn_flat, hx.get_shape())
        output = self.v_conv(attn)
        
        output  = self.scale * output  + input
        return output
    
class ConvSelfAttn2D(Model):
    def __init__(self,  attn_dim, output_dim, kernel_initializer='glorot_uniform'):
        super(ConvSelfAttn2D, self).__init__()
        self.f_conv = Conv2D(attn_dim, 1, kernel_initializer=kernel_initializer)
        self.g_conv = Conv2D(attn_dim, 1, kernel_initializer=kernel_initializer)
        self.h_conv = Conv2D(attn_dim, 1, kernel_initializer=kernel_initializer)
        self.v_conv = Conv2D(output_dim, 1)
        
        self.scale = tf.Variable(0.0)
    
    def flatten(self, inputs):
        #tf.print(inputs.name)
    
        inputs_shape = inputs.get_shape()
        batch_size = tf.TensorShape([inputs_shape[0]])
        hidden_dims = tf.TensorShape([inputs_shape[1] * inputs_shape[2]])
        last_dim = tf.TensorShape([inputs_shape[-1]])
        new_shape = batch_size + hidden_dims + last_dim
        new_shape = [inputs_shape[0], tf.reduce_prod(inputs_shape[1:-1]), inputs_shape[-1]]
        return tf.reshape(inputs, new_shape)
    
    
    def call(self, input):
        fx = self.f_conv(input)
        gx = self.g_conv(input)
        hx = self.h_conv(input)
        
        fx_flat = self.flatten(fx)
        gx_flat = self.flatten(gx)
        hx_flat = self.flatten(hx)
        
        raw_attn_weights = tf.matmul(fx_flat, gx_flat, transpose_b=True)
        raw_attn_weights = tf.transpose(raw_attn_weights, perm=[0,2,1])
        attn_weights = tf.nn.softmax(raw_attn_weights, axis=-1)
        
        attn_flat = tf.matmul(attn_weights, hx_flat)
        attn = tf.reshape(attn_flat, hx.get_shape())
        output = self.v_conv(attn)
        
        output  = self.scale * output + input
        return output
    
    

# initial generator and discriminators
class Generator_v1(Model):
    def __init__(self, hidden_dim=1024):
        
        super(Generator_v1, self).__init__()
        self.hidden_dim = hidden_dim
        
        
        self.resblockup_1 = ResBlockUp3D(256, activation_fn='relu', kernel_initializer='glorot_uniform')
        self.resblockup_2 = ResBlockUp3D(128, activation_fn='relu', kernel_initializer='glorot_uniform')
        self.resblockup_3 = ResBlockUp3D(64, activation_fn='relu', kernel_initializer='glorot_uniform')
        self.resblockup_4 = ResBlockUp3D(32, activation_fn='relu', kernel_initializer='glorot_uniform')
    
        self.conv_4 = Conv3D(1, 3, activation='elu', padding='same')
        self.up_sampling_4 = UpSampling3D(2)

        self.dense = Dense(4096, activation='relu')
        
    

    def call(self, batch_size):
        
        z = tf.random.uniform([batch_size, self.hidden_dim], minval=-1.0, maxval=1.0)
        
        x = self.dense(z)
        x = tf.reshape(x, [-1, 2, 2, 2, 512])
        x = self.resblockup_1(x)
        x = self.resblockup_2(x)
        x = self.resblockup_3(x)
        x = self.resblockup_4(x)
        x = self.up_sampling_4(x)
        x = self.conv_4(x)
        
        return x
    
    
class Discriminator_v1(Model):
    def __init__(self, hidden_dim=1024):
        
        super(Discriminator_v1, self).__init__()
        self.hidden_dim = hidden_dim
        self.linear = Dense(1, kernel_initializer=tf.random_normal_initializer(stddev=.001))
        self.conv_1 = Conv3D(32, 3, activation='elu', padding='same')
        self.resblok_2 = ResBlockDown3D(64, activation_fn='relu', kernel_initializer='orthogonal')
        self.resblok_3 = ResBlockDown3D(128, activation_fn='relu', kernel_initializer='orthogonal')
        self.resblok_4 = ResBlockDown3D(256, activation_fn='relu', kernel_initializer='orthogonal')
        self.resblok_5 = ResBlockDown3D(512, activation_fn='relu', kernel_initializer='orthogonal')
        self.avg_pool_1 = MaxPool3D(2)
        self.activation = Activation('relu')
        #self.dense = Dense(1024, activation='relu', kernel_initializer=tf.random_normal_initializer(stddev=.001))
        self.flatten = Flatten()
        
        

    def call(self, inputs):
        
        x = self.conv_1(inputs)
        x= self.avg_pool_1(x)
        
    
        x= self.resblok_2(x)
        #x = self.attn(x)
        #x= self.attn(x)
        x = self.resblok_3(x)
        x = self.resblok_4(x)
        x = self.resblok_5(x)
        
        
        x = self.flatten(x)
        x = self.activation(x)
        output = self.linear(x)
        return output


# current generator and discriminator
class Generator_v3(Model):
    def __init__(self,
                 use_batchnorm,
                 hidden_dim=1024,
                 activation_fn='relu', 
                 kernel_initializer=tf.keras.initializers.Orthogonal(),
                 noise_distribution='uniform', 
                 use_attn=False):
        
        super(Generator_v3, self).__init__()
        self.hidden_dim = hidden_dim
        self.use_batchnorm = use_batchnorm
        self.noise_distribution = noise_distribution
        self.use_attn = use_attn
        
        self.resblockup_1 = ResBlockUp3D(256, use_batchnorm=use_batchnorm,
                                         activation_fn=activation_fn, 
                                         kernel_initializer=kernel_initializer)
        self.resblockup_2 = ResBlockUp3D(128, use_batchnorm=use_batchnorm,
                                         activation_fn=activation_fn,
                                         kernel_initializer=kernel_initializer)
        self.resblockup_3 = ResBlockUp3D(64, use_batchnorm=use_batchnorm,
                                         activation_fn=activation_fn,
                                         kernel_initializer=kernel_initializer)
        self.resblockup_4 = ResBlockUp3D(32, use_batchnorm=use_batchnorm,
                                         activation_fn=activation_fn, 
                                         kernel_initializer=kernel_initializer)
        
        self.conv_4 = Conv3D(1, 3, activation=activation_fn, padding='same',
                             kernel_initializer=kernel_initializer)
        self.up_sampling_4 = UpSampling3D(2)
    
        
        self.dense = Dense(4096, activation=activation_fn,
                           kernel_initializer=kernel_initializer)
        if self.use_attn:
            self.attn = ConvSelfAttn3D(attn_dim=32, output_dim=128)

    
    def generate(self, z_noise, training):
        
        x = self.dense(z_noise)
        x = tf.reshape(x, [-1, 2, 2, 2, 512])
        x = self.resblockup_1(x, training=training)
        x = self.resblockup_2(x, training=training)
        if self.use_attn:
            x = self.attn(x)
        x = self.resblockup_3(x, training=training)
        x = self.resblockup_4(x, training=training)
        x = self.up_sampling_4(x)
        x = self.conv_4(x)
               
        return x
        
    
    def sample_z(self, batch_size):
        if self.noise_distribution == 'uniform':
            z = tf.random.uniform([batch_size, self.hidden_dim], minval=-1.0, maxval=1.0)
        elif self.noise_distribution == 'normal':
            z = tf.random.normal([batch_size, self.hidden_dim], mean=0.0, stddev=1.0)
        return z
    
    def call(self, batch_size, training):
        z = self.sample_z(batch_size)
        x = self.generate(z, training)
        return x
            
       

        
class Discriminator_v3(Model):
    def __init__(self, hidden_dim=1024, activation_fn='relu', use_attn=True,
                  kernel_initializer=tf.keras.initializers.Orthogonal()):
        
        super(Discriminator_v3, self).__init__()
        self.hidden_dim = hidden_dim
        self.use_attn = use_attn
        self.conv_1 = Conv3D(32, 3, activation=activation_fn, padding='same')
        self.resblok_2 = ResBlockDown3D(64, activation_fn=activation_fn, kernel_initializer=kernel_initializer)
        self.resblok_3 = ResBlockDown3D(128, activation_fn=activation_fn, kernel_initializer=kernel_initializer)
        self.resblok_4 = ResBlockDown3D(256, activation_fn=activation_fn, kernel_initializer=kernel_initializer)
        self.resblok_5 = ResBlockDown3D(512, activation_fn=activation_fn, kernel_initializer=kernel_initializer)

        self.avg_pool_1 = MaxPool3D(2)
        

        self.activation = Activation(activation_fn)
        if self.use_attn:
            self.attn = ConvSelfAttn3D(attn_dim=32, output_dim=128)
        self.flatten = Flatten()
        self.linear = Dense(1)

    def call(self, inputs):
        x = self.conv_1(inputs)
        x = self.avg_pool_1(x)
        x = self.resblok_2(x)
        x = self.resblok_3(x)
        if self.use_attn:
            x = self.attn(x)
        x = self.resblok_4(x)
        x = self.resblok_5(x)
        x = self.flatten(x)
        x = self.activation(x)
        output = self.linear(x)
        return output
