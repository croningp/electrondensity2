# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 01:24:26 2020

@author: jmg
"""
import numpy as np
import tqdm

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, AvgPool2D, LSTMCell, Dense
from tensorflow.keras.layers import Conv2DTranspose, MaxPool2D, UpSampling2D, Layer, RNN
from tensorflow.keras import Model
from tensorflow.distribute import MirroredStrategy 


from tfrecords import input_fn


class InputCNN(Model):
    def __init__(self):
        super(InputCNN, self).__init__()
        self.conv_1 = Conv2D(16, 3, activation='relu', padding='same')
        self.avgpool_1 = AvgPool2D(2)
        
        self.conv_2 = Conv2D(32, 3, activation='relu', padding='same')
        self.avgpool_2 = AvgPool2D(2)
        
        self.conv_3 = Conv2D(64, 3, activation='relu', padding='same')
        self.avgpool_3 = AvgPool2D(2)
        
        self.conv_4 = Conv2D(64, 3, activation='relu', padding='same')
        self.avgpool_4 = AvgPool2D(2)
        
        self.flatten = Flatten()
        
        
        
    
    def call(self, input):
        
        x = self.conv_1(input)
        x = self.avgpool_1(x)
        
        x = self.conv_2(x)
        x = self.avgpool_2(x)
        
        
        x = self.conv_3(x)
        x = self.avgpool_3(x)
        
        x = self.conv_4(x)
        x = self.avgpool_4(x)
        
        x = self.flatten(x)
        
        return x
    
    
    
class TransposeCNN(Model):
    def __init__(self):
        super(TransposeCNN, self).__init__()
      
        self.conv_1 = Conv2D(128, 3, activation='relu', padding='same')
        #self.avgpool_1 = AvgPool2D(2)
        
        self.conv_2 = Conv2D(64, 3, activation='relu', padding='same')
        #self.avgpool_2 = AvgPool2D(2)
        
        self.conv_3 = Conv2D(32, 3, activation='relu', padding='same')
        #self.avgpool_3 = AvgPool2D(2)
        
        self.conv_4 = Conv2D(16, 3, activation='relu', padding='same')
        #self.avgpool_4 = AvgPool2D(2)
        
        self.conv_5 = Conv2D(1, 3, padding='same')
        #self.avgpool_4 = AvgPool2D(2)
        
        self.upsampling = UpSampling2D(2)
        
    
    def call(self, input):
        
        x = tf.reshape(input, [-1, 4, 4, 64])
        x = self.conv_1(x)
        x= self.upsampling(x)
        
        x = self.conv_2(x)
        x= self.upsampling(x)
        
        x = self.conv_3(x)
        x= self.upsampling(x)
        
        x = self.conv_4(x)
        x= self.upsampling(x)
        
        x= self.conv_5(x)
        
        x = tf.tanh(x)
        #x = self.conv_5(x)
        #x= self.upsampling(x)
        return x
    

class ConvLSTM_Cell(Layer):
    def __init__(self, **kwargs):
        
        
        self.input_cnn = InputCNN()
        self.output_cnn = TransposeCNN()
        self.lstm = LSTMCell(256)
        self.mean = Dense(1024, activation='sigmoid')
        self.variance = Dense(1024, activation='sigmoid')
        self.linear = Dense(1024, activation='relu')
        super(ConvLSTM_Cell, self).__init__()
    
    def stochastic_layer(self, input):
        mean = self.mean(input)
        variance = self.variance(input)
        
        return mean + tf.random.uniform([16,1024]) * variance 
    
    
        
    def get_initial_state(self, inputs, batch_size, dtype):
        return self.lstm.get_initial_state(inputs, batch_size, dtype)
    
    def call(self, input, states):
        
        input_ = self.input_cnn(input)
        output, new_states = self.lstm(input_, states)
        #output = self.linear(output)
        output = self.stochastic_layer(output)
        output = self.output_cnn(output)
        
        return output, states
    
    @property
    def output_size(self):
        return tf.TensorShape([64,64,1])
    
    @property
    def state_size(self):
        return self.lstm.state_size
    
    


mirrored_strategy = MirroredStrategy()


optimizer = tf.keras.optimizers.Adam()

cell = ConvLSTM_Cell()
new_layer = RNN(cell, return_sequences=True)

@tf.function
def train_step(cubes):
    input_shape = cubes.get_shape()
    #new_shape = input_shape[0]*input_shape[1] + input_shape[2:]
    #input = tf.reshape(cubes, new_shape)
    zero_slice_shape = input_shape[0] + tf.TensorShape([1]) + input_shape[2:]
    zero_slice = tf.zeros(zero_slice_shape)    
    input_cubes = tf.concat([zero_slice, cubes[:, :-1]], axis=1)
    
    
    with tf.GradientTape() as tape:
        predicted = new_layer(input_cubes)
        loss = tf.keras.losses.MSE(predicted, cubes)
        loss = tf.reduce_mean(loss)
    
    gradients = tape.gradient(loss, new_layer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, new_layer.trainable_variables))
    
    
    return loss, predicted
    

from orbkit import grid, output
from cube import set_grid
set_grid(64, 0.5)
grid.init_grid()


batch_density = input_fn(['data/train.tfrecords','data/valid.tfrecords'],
                         train=True, batch_size=16, num_epochs=200)
counter = 1
losses = []
for i in tqdm.tqdm(iter(batch_density)):
    i = tf.tanh(i)
    #i = i_ + 1e-4
    #i = -tf.math.log(i)
    loss, generated = train_step(i)
    losses.append(loss)
    counter += 1
    if counter % 100 == 0:
        print(np.mean(losses))
        losses = []
    
    #generated = tf.math.exp(-generated)
    #generated -= 1e-4
        

output.view_with_mayavi(grid.x, grid.y, grid.z, generated[1, :, :, :, 0])
output.view_with_mayavi(grid.x, grid.y, grid.z, i[1, :, :, :, 0])
    
    
    
    