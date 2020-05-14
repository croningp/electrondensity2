# -*- coding: utf-8 -*-
"""
Created on Fri May  8 19:58:53 2020

@author: jmg
"""
import os
import numpy as np
import tqdm
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import LSTM, Conv3D, MaxPool3D, AvgPool3D, UpSampling3D, Conv3DTranspose, Activation, BatchNormalization
from tensorflow.keras.layers import Dense, Flatten, Conv2D, AvgPool2D, LSTMCell, Dense
from tfrecords import input_fn
import matplotlib.pyplot as plt

physical_devices = tf.config.list_physical_devices('GPU') 
#try: 
tf.config.experimental.set_memory_growth(physical_devices[0], True) 


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
        self.activation = Activation('elu')
            
        
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
    
    
class Inception(Model):
    
    def __init__(self, num_outputs=1):
        super(Inception, self).__init__()
        self.num_outputs = num_outputs
        
        
        self.conv_1 = Conv3D(32, 3, activation='elu',
                             padding='same',
                             kernel_initializer=tf.random_normal_initializer(stddev=.001))

        
        self.resblok_2 = ResBlockDown(64)
        self.resblok_3 = ResBlockDown(128)
        self.resblok_4 = ResBlockDown(256)
        self.resblok_5 = ResBlockDown(512)
        self.avg_pool_1 = MaxPool3D(2)

        
        self.activation = Activation('elu')
        self.dense = Dense(self.num_outputs, activation='elu',
                           kernel_initializer=tf.random_normal_initializer(stddev=.001))
        self.attn = ConvSelfAttn(attn_dim=16, output_dim=64)
        self.flatten = Flatten()
        
    def call(self, inputs):
        
        
        x = self.conv_1(inputs)
        x = self.avg_pool_1(x)
        x = self.activation(x)
        
        
        x= self.resblok_2(x)
        
        x= self.attn(x)
        x = self.resblok_3(x)
        x = self.resblok_4(x)
        x = self.resblok_5(x)
        
        
        #x = self.flatten(x)
        x = tf.reduce_sum(x, axis=[1,2,3])
        #x = self.activation(x)
        return self.dense(x)


    
def transorm_ed(density):
    
    density = density + 1e-4
    density = tf.math.log(density)
    density = density / tf.math.log(1e-4)
    
    return density

def transorm_back(density):
    
    density = density * tf.math.log(1e-4)
    density = tf.exp(density) - 1e-4
    
    return density
    


inception = Inception()
optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-5)
data_iterator = iter(input_fn('data\\train.tfrecords', properties=['homo', 'lumo'],
                              batch_size=20, num_epochs=100))

checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=inception)
#checkpoint.restore('models\\inception.ckpt-74')

@tf.function
def train_step(density, homo, lumo):
    
    homo_lumo = lumo - homo
    homo_lumo = homo_lumo[:, 0]
    
    density = tf.tanh(density)
    density = transorm_ed(density)
    with tf.GradientTape() as tape:
        predictions = inception(density)
        loss = tf.keras.losses.MSE(homo_lumo, predictions)
        loss = tf.reduce_mean(loss)
    
    gradients = tape.gradient(loss, inception.trainable_variables)
    optimizer.apply_gradients(zip(gradients, inception.trainable_variables))
    

    
    return loss
        
losses = []
accs = []
avg_loss = 0.0

for i in tqdm.tqdm(range(1, 100001)):
    densities, homo, lumo = data_iterator.__next__()
    loss = train_step(densities, homo, lumo)
    losses.append(loss.numpy())
    avg_loss = 0.995 * avg_loss + 0.005 * loss.numpy()
    if i % 100 == 0:
        print(np.mean(losses),)
        print(avg_loss)
        losses = []
        accs = []
    if i % 5000 == 0:
        checkpoint.save('models\\property.ckpt')
    
    
    