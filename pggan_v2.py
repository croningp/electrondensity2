# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 22:54:15 2020

@author: jmg
"""
import os
import pickle
import numpy as np
import tqdm
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, AvgPool2D, LSTMCell, Dense
from tensorflow.keras.layers import Conv2DTranspose, MaxPool2D, UpSampling2D, Layer, RNN, Bidirectional
from tensorflow.keras.layers import LSTM, Conv3D, MaxPool3D, AvgPool3D, UpSampling3D, Conv3DTranspose, Activation, BatchNormalization
from tensorflow.keras import Model


physical_devices = tf.config.list_physical_devices('GPU') 
#try: 
tf.config.experimental.set_memory_growth(physical_devices[0], True) 
#except: 


from tfrecords import input_fn



class ResBlockDown(Model):
    def __init__(self, num_channels, pooling='MaxPool3D'):
        super(ResBlockDown,self).__init__()
        self.num_channels = num_channels
        self.conv_1x1 = Conv3D(num_channels, 1, padding='same',  kernel_initializer=tf.random_normal_initializer(stddev=.001))
        self.conv_3x3a = Conv3D(num_channels, 3, padding='same', kernel_initializer=tf.random_normal_initializer(stddev=.001))
        self.conv_3x3b = Conv3D(num_channels, 3, padding='same', kernel_initializer=tf.random_normal_initializer(stddev=.001))
        
        if pooling == 'MaxPool3D':
            self.pooling = MaxPool3D(2)
        elif pooling == 'AvgPool3D':
            self.pooling = AvgPool3D(2)
        else:
            raise ValueError('Unknowne type of pooling {}'.format(pooling))
        self.activation = Activation('relu')
            
        
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
    
class ResBlockUp(Model):
    def __init__(self, num_channels, use_batchnorm=False):
        super(ResBlockUp, self).__init__()
        self.num_channels = num_channels
        self.use_batchnorm = use_batchnorm
        self.conv_1x1 = Conv3D(num_channels, 1, padding='same')
        self.conv_3x3a = Conv3D(num_channels, 3,  padding='same')
        self.conv_3x3b = Conv3D(num_channels, 3,  padding='same')
        self.activation = Activation('relu')
        
        if use_batchnorm:
            self.batch_norm_1 = BatchNormalization(axis=-1)
            self.batch_norm_2 = BatchNormalization(axis=-1)
        
        self.upsampling = UpSampling3D(2)
        
    def call(self, inputs, training=False):
        
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














class Test(Model):
    def __init__(self):
        super(Test, self).__init__()
        self.conv_1 = Conv3D(8, 3, activation='relu', padding='same')
        self.conv_2 = Conv3D(1, 3, padding='same')
        self.upsampling = UpSampling3D(2)
        self.avg_pool_1 = MaxPool3D(2)
        
    def call(self, inputs):
        
        x= self.conv_1(inputs)
        x = self.avg_pool_1(x)
        x = self.upsampling(x)
        x = self.conv_2(x)
        
        
        return x
        





class ConvSelfAttn(Model):
    def __init__(self,  attn_dim, output_dim):
        super(ConvSelfAttn, self).__init__()
        self.f_conv = Conv3D(attn_dim, 1, kernel_initializer=tf.random_normal_initializer(stddev=.001))
        self.g_conv = Conv3D(attn_dim, 1, kernel_initializer=tf.random_normal_initializer(stddev=.001))
        self.h_conv = Conv3D(attn_dim, 1, kernel_initializer=tf.random_normal_initializer(stddev=.001))
        self.v_conv = Conv3D(output_dim, 1, kernel_initializer=tf.random_normal_initializer(stddev=.001))
        
        self.scale = tf.Variable(0.0)
    
    def flatten(self, inputs):
        #tf.print(inputs.name)
    
        inputs_shape = inputs.get_shape()
        batch_size = tf.TensorShape([inputs_shape[0]])
        hidden_dims = tf.TensorShape(
            [inputs_shape[1] * inputs_shape[2]* inputs_shape[3]])
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







class CNNEncoder(Model):
    
    def __init__(self):
        super(CNNEncoder, self).__init__()
        
        
        self.conv_1 = Conv3D(32, 3, activation='elu', padding='same')
        #self.conv_2 = Conv3D(64, 3, activation='elu', padding='same')
        #self.conv_3 = Conv3D(128, 3, activation='elu', padding='same')
        
        self.resblok_2 = ResBlockDown(64)
        self.resblok_3 = ResBlockDown(128)
        self.resblok_4 = ResBlockDown(256)
        self.resblok_5 = ResBlockDown(512)
        #self.resblok_6 = ResBlockDown(1024)
        #self.conv_4 = Conv3D(256, 3, activation='elu', padding='same')
        #self.conv_5 = Conv3D(512, 3, activation='elu', padding='same')
        
        self.avg_pool_1 = MaxPool3D(2)
        self.avg_pool_2 = MaxPool3D(2)
        self.avg_pool_3 = MaxPool3D(2)
        #self.avg_pool_4 = MaxPool3D(2)
        #self.avg_pool_5 = MaxPool3D(2)
        
        self.activation = Activation('relu')
        self.dense = Dense(1024, activation='relu', kernel_initializer=tf.random_normal_initializer(stddev=.001))
        #self.attn = ConvSelfAttn(attn_dim=16, output_dim=64)
        self.flatten = Flatten()
        
    def call(self, inputs):
        
        
        x = self.conv_1(inputs)
        x= self.avg_pool_1(x)
        
        
        #x = self.conv_2(x)
        #x= self.avg_pool_2(x)
        
        #x = self.conv_3(x)
        #x = self.avg_pool_3(x)
        
        
        x= self.resblok_2(x)
        #x = self.attn(x)
        #x= self.attn(x)
        x = self.resblok_3(x)
        x = self.resblok_4(x)
        x = self.resblok_5(x)
        #x = self.resblok_6(x)
        
        #x = self.conv_4(x)
        #x = self.avg_pool_4(x)
        
        
        #x = self.conv_5(x)
        #x = self.avg_pool_5(x)
        
        
        x = self.flatten(x)
        x = self.activation(x)
        return x
    
    
    
    
class CNNDecoder(Model):
    
    def __init__(self):
        super(CNNDecoder, self).__init__()
        
        
        
        #self.conv_1 = Conv3DTranspose(384, kernel_size=3, strides=(2, 2, 2), padding="same", activation='relu')
        #self.conv_2 = Conv3DTranspose(256, kernel_size=3, strides=(2, 2, 2), padding="same", activation='relu')
        #self.conv_3 = Conv3DTranspose(128, kernel_size=3, strides=(2, 2, 2), padding="same", activation='relu')
        #self.conv_4 = Conv3DTranspose(64, kernel_size=3, strides=(2, 2, 2), padding="same", activation='relu')
        #self.conv_5 = Conv3DTranspose(1, kernel_size=3, strides=(2, 2, 2), padding="same", activation='tanh')
        
        #self.resblockup_0 = ResBlockUp(512)
        self.resblockup_1 = ResBlockUp(256)
        self.resblockup_2 = ResBlockUp(128)
        self.resblockup_3 = ResBlockUp(64)
        self.resblockup_4 = ResBlockUp(32)
        
        #self.conv_0 = Conv3D(256, 3, activation='elu', padding='same')
        #self.conv_1 = Conv3D(128, 3, activation='elu', padding='same')
        #self.conv_2 = Conv3D(64, 3, activation='elu', padding='same')
        #self.conv_3 = Conv3D(32, 3, activation='elu', padding='same')
        self.conv_4 = Conv3D(1, 3, activation='elu', padding='same')
        #self.conv_5 = Conv3D(1, 3, padding='same')
        
       
        
        
       
        self.up_sampling_0 = UpSampling3D(2)
        self.up_sampling_1 = UpSampling3D(2)
        self.up_sampling_2 = UpSampling3D(2)
        self.up_sampling_3 = UpSampling3D(2)
        self.up_sampling_4 = UpSampling3D(2)
        self.up_sampling_5 = UpSampling3D(2)
        
        self.dense = Dense(4096, activation='relu')
        #self.attn = ConvSelfAttn(attn_dim=16, output_dim=64)
        
        
    def call(self, inputs):
        
        x = self.dense(inputs)
        x = tf.reshape(x, [-1, 2, 2, 2, 512])
        
        #x = self.resblockup_0(x)
        x = self.resblockup_1(x)
        x = self.resblockup_2(x)
        x = self.resblockup_3(x)
        #x = self.attn(x)
        
        x = self.resblockup_4(x)
        
        
       #x= self.up_sampling_0(x)
        #x= self.conv_0(x)
        
        #x = self.up_sampling_1(x)
        #x = self.conv_1(x)
        
        #x = self.up_sampling_2(x)
        #x = self.conv_2(x)
        
        
    
        #x = self.up_sampling_3(x)
        #x = self.conv_3(x)
        
        x = self.up_sampling_4(x)
        x = self.conv_4(x)
       
        #x = self.up_sampling_5(x)
       # x = self.conv_5(x)
       
    
 
        return x
    
    
class Generator(Model):
    def __init__(self, hidden_dim=1024):
        
        super(Generator, self).__init__()
        self.hidden_dim = hidden_dim
        self.decoder = CNNDecoder()


    def call(self, batch_size):
        
        z = tf.random.uniform([batch_size, self.hidden_dim], minval=-1.0, maxval=1.0)
        decoded = self.decoder(z)
        
        return decoded
        
class Discriminator(Model):
    def __init__(self, hidden_dim=1024):
        
        super(Discriminator, self).__init__()
        self.hidden_dim = hidden_dim
        self.encoder = CNNEncoder()
        self.linear = Dense(1, kernel_initializer=tf.random_normal_initializer(stddev=.001))

    def call(self, inputs):
        
        output = self.encoder(inputs)
        output = self.linear(output)
        return output
    


g_optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-4)
d_optimizer =   tf.keras.optimizers.RMSprop(learning_rate=1e-4)

generator = Generator()
discriminator = Discriminator()


g_checkpoint = tf.train.Checkpoint(optimizer=g_optimizer, model=generator)
d_checkpoint =  tf.train.Checkpoint(optimizer=d_optimizer, model=discriminator)
#g_checkpoint.restore('gen.ckpt-3')
#d_checkpoint.restore('dis.ckpt-3')
batch_density = iter(input_fn('data\\train.tfrecords',
                    train=True, batch_size=12, num_epochs=35))



@tf.function
def discrimator_train_step(density, batch_size=12, p_lambda=10):

    with tf.GradientTape() as tape:
        #loss = 0.5*tf.square(discriminator(density) - 1.0) + 0.5* tf.square(discriminator(generator(batch_size))+1)
        
        generated = generator(batch_size)
        with tf.GradientTape() as tape_2:
            epsilon = tf.random.uniform(density.shape, minval=0.0, maxval=1.0)
            x_hat = density * epsilon + (1 - epsilon)* generated
            tape_2.watch(x_hat)
            d_hat = discriminator(x_hat)
        
        w_hat_gradient = tape_2.gradient(d_hat, x_hat)
        penalty = p_lambda * tf.square(tf.norm(w_hat_gradient) - 1)
        loss = tf.reduce_mean(discriminator(density)) - tf.reduce_mean(discriminator(generated)) - penalty 
        
    gradients = tape.gradient(loss, discriminator.trainable_variables)
    gradients = [-g for g in gradients]
    d_optimizer.apply_gradients(zip(gradients, discriminator.trainable_variables))
    
    #clip = [v.assign(tf.clip_by_value(v, clip_value_min=-0.01, clip_value_max=0.01)) for v in discriminator.trainable_variables] 
        
    return loss



@tf.function
def generator_train_step(batch_size=12):

    with tf.GradientTape() as tape:
        #loss = tf.square(discriminator(generator(batch_size))-1)
        loss = discriminator(generator(batch_size))
        loss =  -tf.reduce_mean(loss)
        
    gradients = tape.gradient(loss, generator.trainable_variables)
    g_optimizer.apply_gradients(zip(gradients, generator.trainable_variables))
        
    return loss

    
def transorm_ed(density):
    
    density = density + 1e-4
    density = tf.math.log(density)
    density = density / tf.math.log(1e-4)
    
    return density

def transorm_back(density):
    
    density = density * tf.math.log(1e-4)
    density = tf.exp(density) - 1e-4
    
    return density
    

counter = 0
g_losses = []
d_losses = []
kl_losses = []

g_running_avg = 0.0
d_running_avg = 0.0
for j in tqdm.tqdm(range(100000)):
    counter+=1
    
    for i in range(5):
        density = next(batch_density)
        density = tf.tanh(density)
        density = transorm_ed(density)
        dloss = discrimator_train_step(density)
        d_losses.append(dloss)
        d_running_avg = 0.99 * d_running_avg + 0.01 * dloss
    
    for i in range(1):
        gloss = generator_train_step()
        g_losses.append(gloss)
        g_running_avg = 0.99* g_running_avg + 0.01 * gloss
    if counter % 10 == 0:
        print('\nD loss {}, G loss {}'.format(np.mean(d_losses), np.mean(g_losses)))
        print('Run avgs D loss {}, G loss {}\n'.format(d_running_avg, g_running_avg,))
        g_losses = []
        d_losses = []
        generated_cubes = generator(2)
        with open('generated.pkl', 'wb') as pfile:
            generated_cubes = transorm_back(generated_cubes)
            pickle.dump(generated_cubes.numpy(), pfile)
    if counter % 1000==0:
        gen_path = g_checkpoint.save('gen.ckpt')
        dis_path = d_checkpoint.save('dis.ckpt')
        print('Generator saved to', gen_path)
        print('Discriminator saved to', dis_path)
        
    
    
    #gloss, _ = pretrain_step(density)
    #glosses.append(gloss)
    #if counter %100 == 0:
        #schedule.assign(j/100000.)
     #   print('NLL loss: ', np.mean(glosses))
     #   glosses = []
      #  generted_cubes = sample_model(cell).numpy()
       # with open('generated.pkl', 'wb') as pfile:
        #    pickle.dump(generted_cubes, pfile)
        
        
        
        
        
        
        
        
        
        
        
        
        
