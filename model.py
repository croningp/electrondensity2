# -*- coding: utf-8 -*-
"""
One of the initial attempts to generate electron density. 
Created on Wed Apr  8 01:24:26 2020

@author: Jaroslaw Granda
"""
import os 
import pickle
import numpy as np
import tqdm


os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, AvgPool2D, LSTMCell, Dense
from tensorflow.keras.layers import Conv2DTranspose, MaxPool2D, UpSampling2D, Layer, RNN, Bidirectional
from tensorflow.keras.layers import LSTM
from tensorflow.keras import Model



from tfrecords import input_fn






class ConvSelfAttn(Model):
    def __init__(self,  attn_dim, output_dim):
        super(ConvSelfAttn, self).__init__()
        self.f_conv = Conv2D(attn_dim, 1)
        self.g_conv = Conv2D(attn_dim, 1)
        self.h_conv = Conv2D(attn_dim, 1)
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


class InputCNN(Model):
    def __init__(self):
        super(InputCNN, self).__init__()
        self.conv_1 = Conv2D(32, 4, activation='relu', padding='same')
        self.avgpool_1 = MaxPool2D(2)
        
        self.conv_2 = Conv2D(64, 4, activation='relu', padding='same')
        self.avgpool_2 = MaxPool2D(2)
        
        self.conv_3 = Conv2D(128, 3, activation='relu', padding='same')
        self.avgpool_3 = MaxPool2D(2)
        
        self.conv_4 = Conv2D(128, 3, activation='relu', padding='same')
        self.avgpool_4 = MaxPool2D(2)
        
        self.flatten = Flatten()
        self.attn = ConvSelfAttn(32, 128)
        
        
    
    def call(self, input):
        
        x = self.conv_1(input)
        x = self.avgpool_1(x)
        
        x = self.conv_2(x)
        x = self.avgpool_2(x)
        
        
        x = self.conv_3(x)
        x = self.avgpool_3(x)
        
        x = self.conv_4(x)
        x = self.attn(x)
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
        
        self.conv_4 = Conv2D(32, 4, activation='relu', padding='same')
        #self.avgpool_4 = AvgPool2D(2)
        
        self.conv_5 = Conv2D(2, 4, padding='same')
        #self.avgpool_4 = AvgPool2D(2)
        
        self.upsampling = UpSampling2D(2)
        self.attn = ConvSelfAttn(32,128)
    
    def call(self, input):
        
        x = tf.reshape(input, [-1, 4, 4, 64])
        x = self.conv_1(x)
        x = self.attn(x)
        x= self.upsampling(x)
        
        x = self.conv_2(x)
        x= self.upsampling(x)
        
        x = self.conv_3(x)
        x= self.upsampling(x)
        
        x = self.conv_4(x)
        x= self.upsampling(x)
        
        x= self.conv_5(x)
        
        mean, variance = tf.split(x, axis=-1, num_or_size_splits=2)
        
        #x = tf.tanh(x)
        #x = self.conv_5(x)
        #x= self.upsampling(x)
        return mean + variance * tf.random.normal(variance.get_shape())



        


class Discriminator(Model):
    
    def __init__(self):
        
        super(Discriminator, self).__init__()
        self.bi_lstm = Bidirectional(LSTM(128, return_sequences=True), )
        self.conv = InputCNN()
        self.linear = Dense(1, activation='sigmoid')
        
    def call(self, input):
        
        input = tf.transpose(input, perm=[1, 0, 2, 3, 4])
        input = tf.map_fn(self.conv, input)
        input = tf.transpose(input, perm=[1, 0, 2])
        #input = tf.tanh(input)
        output = self.bi_lstm(input)
        output_shape = output.get_shape()
        new_shape = output_shape[0]*output_shape[1] + output_shape[2:] 
        output = tf.reshape(output, new_shape)
        
        probabilty = self.linear(output)
        
        return tf.reduce_mean(probabilty)

class ConvLSTM_Cell(Layer):
    def __init__(self, **kwargs):
        
        
        self.input_cnn = InputCNN()
        self.output_cnn = TransposeCNN()
        self.lstms = [LSTMCell(1024), LSTMCell(1024)]
        self.mean = Dense(1024, activation='relu')
        self.variance = Dense(1024, activation='sigmoid',
                              bias_initializer=tf.constant_initializer(value=-20), 
                              )
        
        self.linear = Dense(1024, activation='relu')
        super(ConvLSTM_Cell, self).__init__()
    
    def stochastic_layer(self, input):
        mean = self.mean(input)
        variance = self.variance(input)
        
        return mean + tf.random.normal([16,1024]) * variance 
        
    
        
    def get_initial_state(self, inputs=None, batch_size=None, dtype=tf.float32):
        initial_state = [lstm.get_initial_state(inputs, batch_size, dtype) for lstm in self.lstms]  
        #initial_state =[[tf.random.normal(v.get_shape(), stddev=1.0) for v in lstm_state]
                        #for lstm_state in initial_state]
        #initial_state = [[v+1.0 for v in lstm_state] for lstm_state in initial_state]
        return initial_state
    def call(self, input, states):
        
        input_ = self.input_cnn(input)
        output, new_state_1 = self.lstms[0](input_, states[0])
        output, new_state_2 = self.lstms[1](output, states[1])
        output = self.linear(output)
        #output = self.stochastic_layer(output)
        output = self.output_cnn(output)
        
        output += input
        
        return output, [new_state_1, new_state_2]
    
    @property
    def output_size(self):
        return tf.TensorShape([64,64,1])
    
    @property
    def state_size(self):
        return [lstm.state_size for lstm in self.lstms]
    
    


#mirrored_strategy = MirroredStrategy()




optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

#discriminator_optimizer = tf.keras.optimizers.Adam()
cell = ConvLSTM_Cell()
#discriminator = Discriminator()
new_layer = RNN(cell, return_sequences=True)
schedule = tf.Variable(0, dtype=tf.float32, trainable=False)
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=new_layer)
#checkpoint.restore('model.ckpt-2')

@tf.function
def train_step(cubes):
    input_shape = cubes.get_shape()
    #new_shape = input_shape[0]*input_shape[1] + input_shape[2:]
    #input = tf.reshape(cubes, new_shape)
    zero_slice_shape = input_shape[0] + tf.TensorShape([1]) + input_shape[2:]
    zero_slice = tf.zeros(zero_slice_shape)   
    zero_slice += 1e-4
    zero_slice = - tf.math.log(zero_slice)
    
    input_cubes = tf.concat([zero_slice, cubes[:, :-1]], axis=1)
    #test_mse = tf.reduce_mean(tf.keras.losses.MSE(input_cubes, cubes))
    
    with tf.GradientTape() as tape:
        predicted = new_layer(input_cubes)
        sampled = sample_model(cell)
        real_proba = discriminator(sampled)
        gloss = tf.reduce_mean(-tf.math.log(real_proba))
        loss = tf.keras.losses.MSE(predicted, cubes)
        loss = tf.reduce_mean(loss)
        loss += 0.1 * gloss
    
    gradients = tape.gradient(loss, new_layer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, new_layer.trainable_variables))
        
    
    return loss


@tf.function
def pretrain_step(cubes):
    input_shape = cubes.get_shape()
    #new_shape = input_shape[0]*input_shape[1] + input_shape[2:]
    #input = tf.reshape(cubes, new_shape)
    zero_slice_shape = input_shape[0] + tf.TensorShape([1]) + input_shape[2:]
    zero_slice = tf.zeros(zero_slice_shape)   
    zero_slice += 1e-4
    zero_slice = - tf.math.log(zero_slice)
    
    input_cubes = tf.concat([zero_slice, cubes[:, :-1]], axis=1)
    #test_mse = tf.reduce_mean(tf.keras.losses.MSE(input_cubes, cubes))
    with tf.GradientTape() as tape:
        #predicted = new_layer(input_cubes)
        predicted = scheduled_sampling(cell, cubes)
        loss = tf.keras.losses.MSE(predicted, cubes)
        loss = tf.reduce_mean(loss)
    
    gradients = tape.gradient(loss, new_layer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, new_layer.trainable_variables))
        
    return loss, predicted






    
@tf.function
def train_discriminator(cubes):
    sampled = sample_model(cell)
    with tf.GradientTape(persistent=False) as tape:
        tf_proba = discriminator(cubes)       
        gen_proba = discriminator(sampled)
        sampled = sample_model(cell)
        discriminator_loss = tf.reduce_mean(-tf.math.log(tf_proba)) + tf.reduce_mean(-tf.math.log(1-gen_proba))
        
    discriminator_grads = tape.gradient(discriminator_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(discriminator_grads, discriminator.trainable_variables))
    
    return discriminator_loss


@tf.function
def sample_model(cell, initial_input=None):
    initial_state = cell.get_initial_state(batch_size=16)
    #if initial_input is None:
    initial_input = tf.zeros([16,64,64,1])
    initial_input = initial_input + 1e-4#+tf.random.normal(initial_input.get_shape(),
                                                        #  mean=0.0, stddev=1e-5)
    initial_input = -tf.math.log(initial_input)
    
    cell_outputs = []
    cell_input = initial_input
    cell_states = initial_state
    
    
    for i in range(64):
        cell_output, next_states = cell(cell_input, cell_states)
        cell_input = cell_output
        cell_states = next_states
        cell_outputs.append(tf.expand_dims(cell_output, axis=1))
    
    generated_cubes = tf.concat(cell_outputs, axis=1)
    
    return generated_cubes


@tf.function
def scheduled_sampling(cell, inputs):
    initial_state = cell.get_initial_state(batch_size=16)
    #if initial_input is None:
    initial_input = tf.zeros([16,64,64,1])
    initial_input = initial_input + 1e-4 #+ tf.random.normal(initial_input.get_shape(),
                                                       #   mean=0.0, stddev=1e-5)
    initial_input = -tf.math.log(initial_input)
    cell_outputs = []
    cell_input = initial_input
    cell_states = initial_state
    
    
    for i in range(64):
        cell_output, next_states = cell(cell_input, cell_states)
        random = tf.random.uniform([], minval=0.0, maxval=1.0)
        
        
        cell_input = (inputs[:, i])*(1-schedule) + schedule * cell_output
        #cell_input = tf.cond(tf.less(random, schedule), 
         #                    lambda:cell_output, lambda:inputs[:,i])
        
        cell_states = next_states
        cell_outputs.append(tf.expand_dims(cell_output, axis=1))
    
    generated_cubes = tf.concat(cell_outputs, axis=1)
    
    return generated_cubes





batch_density = iter(input_fn('data/train.tfrecords',
                         train=True, batch_size=16, num_epochs=35))

counter = 0
glosses = []

#checkpoint.restore('pretrained_generator.ckpt-4')
# pretraining generator
print('Pretraining generator...')
for j in tqdm.tqdm(range(100000)):
    counter+=1
    density = next(batch_density)
    #i = tf.tanh(i_)
    density = density + 1e-4
    density = -tf.math.log(density)
    gloss, _ = pretrain_step(density)
    glosses.append(gloss)
    if counter %100 == 0:
        schedule.assign(j/100000.)
        print('NLL loss: ', np.mean(glosses))
        glosses = []
        generted_cubes = sample_model(cell).numpy()
        with open('generated.pkl', 'wb') as pfile:
            pickle.dump(generted_cubes, pfile)

checkpoint.save('pretrained_generator.ckpt')
raise ValueError

print('Done')
print('Pretraining discriminator...')
counter = 0
dlosses = []
# pretraining discriminator
for j in tqdm.tqdm(range(1000)):
    counter+=1
    density = next(batch_density)
    #i = tf.tanh(i_)
    density = density + 1e-4
    density = -tf.math.log(density)
    dloss = train_discriminator(density)
    dlosses.append(dloss.numpy())
    if counter %100 == 0:
        print('discriminator loss: ', np.mean(dlosses))

print('Done')





for i in range(10000):
    
  
    
    glosses = []
    for j in tqdm.tqdm(range(100)):
        density = next(batch_density)
        density = density + 1e-4
        density = -tf.math.log(density)
        
        loss = train_step(density)
        glosses.append(loss)
        
    print('generator loss', np.mean(glosses))
    generted_cubes = sample_model(cell).numpy()
    with open('generated.pkl', 'wb') as pfile:
        pickle.dump(generted_cubes, pfile)
            
    dlosses = []
    for j in tqdm.tqdm(range(200)):
        counter+=1
        density = next(batch_density)
    #i = tf.tanh(i_)
        density = density + 1e-4
        density = -tf.math.log(density)
        dloss = train_discriminator(density)
        dlosses.append(dloss.numpy())
    print('discriminator loss: ', np.mean(dlosses))       
    
    #if counter % 50 == 0:
        #print(np.mean(losses), dloss)
       # losses = []
   # if counter % 10000 == 0:
       # checkpoint.save('model.ckpt')
        
    #if counter % 500 == 0:
        #generted_cubes = sample_model(cell).numpy()
       # with open('generated.pkl', 'wb') as pfile:
           # pickle.dump(generted_cubes, pfile)
            
            
            
        
        #generated = tf.exp(-generated) - 1e-4
        #output.view_with_mayavi(grid.x, grid.y, grid.z, generated[1, :, :, :, 0])
        #output.view_with_mayavi(grid.x, grid.y, grid.z, i[1, :, :, :, 0])
    
    
    
    