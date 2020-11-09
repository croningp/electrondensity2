# -*- coding: utf-8 -*-
"""
Created on Fri May  8 19:58:53 2020. Model to classify the number of nuclei in
the molecules. Used to calculate the inception score and Frechet inception distance.

@author: jmg
"""
import os
import numpy as np
import tqdm
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import LSTM, Conv3D, MaxPool3D, AvgPool3D, UpSampling3D, Conv3DTranspose, Activation, BatchNormalization
from tensorflow.keras.layers import Dense, Flatten, Conv2D, AvgPool2D, LSTMCell, Dense
from input.tfrecords import input_fn
try:
    import matplotlib.pyplot as plt
except:
    print('Matplotlib not found')

#physical_devices = tf.config.list_physical_devices('GPU') 
#try: 
#tf.config.experimental.set_memory_growth(physical_devices[0], True) 

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
    
    def __init__(self, num_outputs=9):
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
        return x, self.dense(x)


    
def transorm_ed(density):
    
    density = density + 1e-4
    density = tf.math.log(density)
    density = density / tf.math.log(1e-4)
    
    return density

def transorm_back(density):
    
    density = density * tf.math.log(1e-4)
    density = tf.exp(density) - 1e-4
    
    return density
    




@tf.function
def train_step(density, num_atoms, inception, optimizer):
    num_atoms = num_atoms - 1
    num_atoms = num_atoms[:, 0]
    density = tf.tanh(density)
    density = transorm_ed(density)
    with tf.GradientTape() as tape:
        _, predictions = inception(density)
        targets = tf.one_hot(num_atoms, depth=9)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=targets, logits=predictions)
        loss = tf.reduce_mean(loss)
    
    gradients = tape.gradient(loss, inception.trainable_variables)
    optimizer.apply_gradients(zip(gradients, inception.trainable_variables))
    
    pred_idxs = tf.argmax(predictions, axis=-1)
    accuracy = tf.cast(tf.equal(pred_idxs, num_atoms), tf.float32)
    accuracy = tf.reduce_mean(accuracy)
    
    return loss, accuracy, pred_idxs, tf.nn.softmax(predictions), targets
        
def train():
    losses = []
    accs = []
    avg_loss = 0.0
    avg_accs = 0.0
    
    inception = Inception()
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-5)
    data_iterator = iter(input_fn('/media/group/d22cc883-8622-4ecd-8e46-e3b0850bb89a/jarek/train.tfrecords',
                                  properties=['num_atoms'], batch_size=20, num_epochs=200, train=True))

    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=inception)
    checkpoint.restore('/media/group/d22cc883-8622-4ecd-8e46-e3b0850bb89a/jarek/inception_models/inception.ckpt-40')
    
    
    for i in tqdm.tqdm(range(1, 500001)):
        densities, num_atoms = data_iterator.__next__()
        loss, accuracy, pred, logits, targs = train_step(densities,
                                                         num_atoms,
                                                         inception=inception,
                                                         optimizer=optimizer)
        losses.append(loss.numpy())
        accs.append(accuracy.numpy())
        avg_loss = 0.999 * avg_loss + 0.001 * loss.numpy()
        avg_accs = 0.999 * avg_accs + 0.001 * accuracy.numpy()
        if i % 100 == 0:
            plt.imshow(targs[:])
            plt.savefig('targets.png')
            plt.imshow(logits)
            plt.savefig('logits.png')
            print(np.mean(losses), np.mean(accs))
            print(avg_loss, avg_accs)
            losses = []
            accs = []
        if i % 5000 == 0:
            checkpoint.save('/media/group/d22cc883-8622-4ecd-8e46-e3b0850bb89a/jarek/inception_models/inception.ckpt')
            

def kl_divergence(proba, marginal):
    
    kl_div = proba *np.log(1e-10+(proba/(marginal)))
    kl_div = np.sum(kl_div, axis=-1)
    return kl_div


def entropy(proba):
    entropy = proba * np.log(1e-10+proba)
    entropy = np.sum(entropy, axis=-1)
    return -entropy
          
def calculate_inception_score(data_iterator):
    
    
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-5)
    
    inception = Inception()
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=inception)
    checkpoint.restore('/media/group/d22cc883-8622-4ecd-8e46-e3b0850bb89a/jarek/inception_models/inception.ckpt-40')
    
    proba = []
    
    
    for density in tqdm.tqdm(data_iterator):
        if type(density) == tuple:
            density = density[0]
        density = tf.tanh(density)
        density = transorm_ed(density)
        
        _, predictions = inception(density)
        predictions = tf.nn.softmax(predictions, axis=-1)
        proba.append(predictions.numpy())
        
            
    
    proba = proba[:-1]
    proba = np.array(proba)
    proba = proba.reshape([-1, 9])
        
    cum_proba = np.mean(proba, axis=0)
    
    kl_div = kl_divergence(proba, cum_proba)
    inception = np.exp(np.mean(kl_div))
    
    
    dentropy = entropy(proba)
    dentropy = np.mean(dentropy)
        
    return proba, cum_proba, inception, dentropy


def score(file_name):
    data = np.load(file_name)
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.map(transorm_back)
    dataset = dataset.batch(20)
    dataset = dataset.repeat(1)
    
    dataset_iterator = iter(dataset)
    proba, marginal, inception, ent = calculate_inception_score(dataset_iterator)
    
    return ent

if __name__=='__main__':
    
    train()
    
    #dataset_iterator = iter(input_fn('/media/group/d22cc883-8622-4ecd-8e46-e3b0850bb89a/jarek/train.tfrecords',
     #                         batch_size=32, num_epochs=1))
    
    
    
    
    
    #data = np.load('/media/group/d22cc883-8622-4ecd-8e46-e3b0850bb89a/jarek/gan_samples.pkl', allow_pickle=True)
    #dataset = tf.data.Dataset.from_tensor_slices(data)
    #dataset = dataset.map(transorm_back)
    #dataset = dataset.batch(20)
    #dataset = dataset.repeat(1)
    
    #dataset_iterator = iter(dataset)
    #proba, marginal, inception, ent = calculate_inception_score(dataset_iterator)
    #print('Entropy', ent)
        

        
        
    
    
    
    
    
