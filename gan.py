# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 10:35:04 2020

@author: group
"""

import os
import pickle
import numpy as np
import tqdm
import time
import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.keras.layers import Dense, Flatten, Conv2D, AvgPool2D, LSTMCell, Dense
from tensorflow.keras.layers import Conv2DTranspose, MaxPool2D, UpSampling2D, Layer, RNN, Bidirectional
from tensorflow.keras.layers import LSTM, Conv3D, MaxPool3D, AvgPool3D, UpSampling3D, Conv3DTranspose, Activation, BatchNormalization
from tensorflow.keras import Model

os.chdir('/home/jarek/electrondensity2')
#os.chdir('Y:\\')

from input.tfrecords import input_fn
from layers import ResBlockDown3D, ResBlockUp3D, ConvSelfAttn3D, Generator_v3, Discriminator_v3
from layers import SpatialDiscriminator, TemporalDiscriminator
from utils import  transorm_ed, transorm_back_ed




class GAN_Base():
    
    def __init__(self, 
                 generator,
                 discriminator,
                 generator_config, 
                 discriminator_config, 
                 g_learning_rate=1e-4,
                 d_learning_rate=1e-4,
                 g_optimizer=tf.keras.optimizers.RMSprop,
                 d_optimizer=tf.keras.optimizers.RMSprop,
                 distributed_training=True
                 ):
    
        super(GAN_Base, self).__init__()
    

        self.distributed_training=distributed_training
        
        if distributed_training:
            self.strategy = tf.distribute.MirroredStrategy(
                    cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
        
        if hasattr(self, 'strategy'):
            with self.strategy.scope():
                 self.generator = generator(**generator_config)
                 self.discriminator = discriminator(**discriminator_config)
                 self.g_optimizer = g_optimizer(g_learning_rate)
                 self.d_optimizer = d_optimizer(d_learning_rate)
                 
                 self.g_checkpoint = tf.train.Checkpoint(optimizer=self.g_optimizer,
                                            model=self.generator)
                 self.d_checkpoint =  tf.train.Checkpoint(optimizer=self.d_optimizer,
                                            model=self.discriminator)
        else:
            self.generator = generator(**generator_config)
            self.discriminator = discriminator(**discriminator_config)
            self.g_optimizer = g_optimizer(g_learning_rate)
            self.d_optimizer = d_optimizer(d_learning_rate)
            
        
            self.g_checkpoint = tf.train.Checkpoint(optimizer=self.g_optimizer,
                                           model=self.generator)
            self.d_checkpoint =  tf.train.Checkpoint(optimizer=self.d_optimizer,
                                            model=self.discriminator)
        

    def discriminator_step_fn(self, *args):
        pass
    
    def generator_step_fn(self, *args):
        pass
    
    
    def restore(self, dis_path, gen_path):
        if hasattr(self, 'strategy'):
            with self.strategy.scope():
                self.g_checkpoint.restore(gen_path)
                self.d_checkpoint.restore(dis_path)
        else:
            self.g_checkpoint.restore(gen_path)
            self.d_checkpoint.restore(dis_path)
    
    @tf.function
    def discrimator_train_step(self, *args, **kwargs):
        if hasattr(self, 'strategy'):
            per_replica_losses = self.strategy.experimental_run_v2(self.discriminator_step_fn,
                                                                   args=args,
                                                                   kwargs=kwargs)
            loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN,
                                        per_replica_losses, axis=None)
        else:
            loss = self.discriminator_step_fn(*args, **kwargs)
        return loss
            
    @tf.function
    def generator_train_step(self, *args, **kwargs):
        if hasattr(self, 'strategy'):
            per_replica_losses = self.strategy.experimental_run_v2(self.generator_step_fn,
                                                                   args=args, kwargs=kwargs)
            loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)
        else:
            loss = self.generator_step_fn(*args, **kwargs)
        return loss
    
    
    def sample_model(self, path=None, num_samples=1000, batch_size=32):
        print('Electron density saved to {}'.format(path))
        num_iter = num_samples // batch_size + 1
        generated_cubes = []
        
        for i in tqdm.tqdm(range(num_iter)):
            cubes = self.generator(batch_size, training=False)
            cubes = transorm_back_ed(cubes).numpy()
            generated_cubes.extend(cubes)
        
        generated_cubes = np.array(generated_cubes)[:num_samples]
        
        if path is not None:
            with open(path, 'wb') as pfile:
                pickle.dump(generated_cubes, pfile)
            
        return generated_cubes
    
    
    
    def explore_latent_space(self, num_steps=10, batch_size=32):
        
        noise_z1 = self.generator.sample_z(batch_size)
        noise_z2 = self.generator.sample_z(batch_size)
        
        noise_diff = noise_z2 - noise_z1
        noise_step = noise_diff / num_steps
        
        
        for i in range(num_steps+1):
            noise = noise_z1 + i * noise_step
            cubes = self.generator.generate(noise, training=False)
            cubes = transorm_back_ed(cubes).numpy()
            with open('step{}.pkl'.format(i), 'wb') as pfile:
                pickle.dump(cubes, pfile)
                
                
    def explore_latent_space_v2(self, num_steps=10, batch_size=32):
        
        noise_z1 = self.generator.sample_z(batch_size)
        noise_z2 = self.generator.sample_z(batch_size)
        noise_z3 = self.generator.sample_z(batch_size)
        
        noise_diff = noise_z2 - noise_z1
        noise_step = noise_diff / num_steps
        
        noise_diff2 = noise_z3 - noise_z2
        noise_step2 = noise_diff2 / num_steps
        
        for i in range(num_steps+1):
            noise = noise_z1 + i * noise_step
            cubes = self.generator.generate(noise, training=False)
            cubes = transorm_back_ed(cubes).numpy()
            with open('step{}.pkl'.format(i), 'wb') as pfile:
                pickle.dump(cubes, pfile)
        
        for i in range(num_steps+1):
            noise = noise_z2 + i * noise_step2
            cubes = self.generator.generate(noise, training=False)
            cubes = transorm_back_ed(cubes).numpy()
            with open('step{}.pkl'.format(i+num_steps), 'wb') as pfile:
                pickle.dump(cubes, pfile)
            
    
class GP_WGAN(GAN_Base):
    def __init__(self, *args, **kwargs):
        super(GP_WGAN, self).__init__(*args, **kwargs)
        
    def discriminator_step_fn(self, density, p_lambda=10):
        density = tf.tanh(density)
        density = transorm_ed(density)
        
        with tf.GradientTape() as tape:
        #loss = 0.5*tf.square(discriminator(density) - 1.0) + 0.5* tf.square(discriminator(generator(batch_size))+1)
            batch_size = tf.shape(density)[0]
            generated = self.generator(batch_size, training=False)
            with tf.GradientTape() as tape_2:
                epsilon = tf.random.uniform(density.shape, minval=0.0, maxval=1.0)
                x_hat = density * epsilon + (1 - epsilon)* generated
                tape_2.watch(x_hat)
                d_hat = self.discriminator(x_hat)
        
            w_hat_gradient = tape_2.gradient(d_hat, x_hat)
            penalty = p_lambda * tf.square(tf.norm(w_hat_gradient) - 1)
            loss = tf.reduce_mean(self.discriminator(density)) - tf.reduce_mean(self.discriminator(generated)) - penalty 
        
        gradients = tape.gradient(loss, self.discriminator.trainable_variables)
        gradients = [-g for g in gradients]
        self.d_optimizer.apply_gradients(zip(gradients, self.discriminator.trainable_variables))
        return loss
    
    def generator_step_fn(self, batch_size):
        with tf.GradientTape() as tape:
            loss = self.discriminator(self.generator(batch_size, training=True))
            loss =  -tf.reduce_mean(loss)
        gradients = tape.gradient(loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(gradients, self.generator.trainable_variables))
        return loss
    
    

class LS_GAN(GAN_Base):
    def __init__(self, *args, **kwargs):
        super(LS_GAN, self).__init__(*args, **kwargs)
        
    def discriminator_step_fn(self, density, p_lambda=10):
        density = tf.tanh(density)
        density = transorm_ed(density)
        
        with tf.GradientTape() as tape:
        #loss = 0.5*tf.square(discriminator(density) - 1.0) + 0.5* tf.square(discriminator(generator(batch_size))+1)
            batch_size = tf.shape(density)[0]
            generated = self.generator(batch_size, training=False)
            loss = 0.5*tf.square(self.discriminator(density) - 1.0) + 0.5*tf.square(self.discriminator(generated))
            loss = tf.reduce_mean(loss)
            
        gradients = tape.gradient(loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(gradients, self.discriminator.trainable_variables))
        return loss
    
    def generator_step_fn(self, batch_size):
        with tf.GradientTape() as tape:
            loss = 0.5* tf.square(self.discriminator(self.generator(batch_size, training=True))-1.0)
            loss =  tf.reduce_mean(loss)
        gradients = tape.gradient(loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(gradients, self.generator.trainable_variables))
        return loss
    

    
class DoubleDiscriminatorGAN(GAN_Base):
    def __init__(self, spatial_discriminator, *args,  **kwargs):
        super(DoubleDiscriminatorGAN, self).__init__(*args, **kwargs)
        self.spatial_discriminator = spatial_discriminator(kernel_initializer='orthogonal')
        
    def discriminator_step_fn(self, density, p_lambda=10):
        density = tf.tanh(density)
        density = transorm_ed(density)
        
        with tf.GradientTape() as tape:
        #loss = 0.5*tf.square(discriminator(density) - 1.0) + 0.5* tf.square(discriminator(generator(batch_size))+1)
            batch_size = tf.shape(density)[0]
            generated = self.generator(batch_size, training=False)
            with tf.GradientTape() as tape_2:
                epsilon = tf.random.uniform(density.shape, minval=0.0, maxval=1.0)
                x_hat = density * epsilon + (1 - epsilon)* generated
                tape_2.watch(x_hat)
                d_hat = self.discriminator(x_hat) + 0.1 * self.spatial_discriminator(x_hat)
        
            w_hat_gradient = tape_2.gradient(d_hat, x_hat)
            penalty = p_lambda * tf.square(tf.norm(w_hat_gradient) - 1)
            loss = tf.reduce_mean(self.discriminator(density)) - tf.reduce_mean(self.discriminator(generated)) - penalty 
            loss = loss + 0.1 * tf.reduce_mean(self.spatial_discriminator(density)) - 0.1* tf.reduce_mean(self.spatial_discriminator(generated))
        
        gradients = tape.gradient(loss,
                                  self.discriminator.trainable_variables+self.spatial_discriminator.trainable_variables)
        gradients = [-g for g in gradients]
        self.d_optimizer.apply_gradients(zip(gradients, self.discriminator.trainable_variables))
        return loss
    
    def generator_step_fn(self, batch_size):
        with tf.GradientTape() as tape:
            generated_ed = self.generator(batch_size, training=True)
            loss = self.discriminator(generated_ed) + 0.1*self.spatial_discriminator(generated_ed)
            loss =  -tf.reduce_mean(loss)
        gradients = tape.gradient(loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(gradients, self.generator.trainable_variables))
        return loss
    

    
    
    
    
    

class GANTrainer():
    
    
    def __init__(self,
                 gan,
                 batch_size=16,
                 num_epochs=100,
                 num_training_steps=100000, 
                 steps_train_generator=1,
                 steps_train_discriminator=5,
                 write_summary=True,
                 save_model_every_steps=1000,
                 path='/media/group/d22cc883-8622-4ecd-8e46-e3b0850bb89a2/jarek'
                 ):
        
        self.gan = gan
        self.save_model_every_steps = save_model_every_steps
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_training_steps = num_training_steps
        self.steps_train_generator = steps_train_generator
        self.steps_train_discriminator = steps_train_discriminator
        self.path = path
        
        local_device_protos = device_lib.list_local_devices()
        self.num_gpus = len([x.name for x in local_device_protos if x.device_type == 'GPU'])
        
        
        
        data_paths = [os.path.join(self.path, 'train.tfrecords'),
                      os.path.join(self.path, 'valid.tfrecords')]
        
        
        if self.gan.distributed_training == True:
            with self.gan.strategy.scope():
                self.densities = input_fn(data_paths,
                                          train=True,
                                          batch_size=self.batch_size*self.num_gpus,
                                          num_epochs=self.num_epochs)
                
                self.densities = gan.strategy.experimental_distribute_dataset(self.densities)
        else:   
            
            self.densities = input_fn(data_paths,
                                      train=True,
                                      batch_size=self.batch_size,
                                      num_epochs=self.num_epochs)
        
        self.density_generator = iter(self.densities)
        
        self.counter = 0
        self.g_losses = []
        self.d_losses = []
        self.g_running_avg = 0.0
        self.d_running_avg = 0.0
        
        if write_summary:
            time_stamp = str(int(time.time()))
            
            path = os.path.join(self.path, 'logs', time_stamp, 'train')
            self.summary_writer = tf.summary.create_file_writer(path)
        
    def train_discriminator(self):
        for i in range(self.steps_train_discriminator):
            batch_density = next(self.density_generator)[0]
            self.d_loss = self.gan.discrimator_train_step(batch_density)
            self.d_losses.append(self.d_loss)
            self.d_running_avg = 0.99 * self.d_running_avg + 0.01 * self.d_loss
    
    
    def train_generator(self):
        for i in range(self.steps_train_generator):
            self.g_loss = self.gan.generator_train_step(self.batch_size)
            self.g_losses.append(self.g_loss)
            self.g_running_avg = 0.99 * self.g_running_avg + 0.01 * self.g_loss
    
    def write_summary(self):
        if not hasattr(self, 'summary_writer'):
            return
        with self.summary_writer.as_default():
            tf.summary.scalar('g_loss', self.g_loss, step=counter)
            tf.summary.scalar('d_loss', self.d_loss, step=counter)
            self.summary_writer.flush()
            
    def print_stats(self):
        print('\nD loss {}, G loss {}'.format(np.mean(self.d_losses), np.mean(self.g_losses)))
        print('Run avgs D loss {}, G loss {}\n'.format(self.d_running_avg, self.g_running_avg,))
        self.g_losses = []
        self.d_losses = []
        
    def save_model(self):
            model_path = os.path.join(self.path, 'models')
            gen_path = os.path.join(model_path, 'gen.ckpt')
            dis_path = os.path.join(model_path, 'dis.ckpt')
            gen_path = self.gan.g_checkpoint.save(gen_path)
            dis_path = self.gan.d_checkpoint.save(dis_path)
            print('Generator saved to', gen_path)
            print('Discriminator saved to', dis_path)
            
    def save_ed(self):
        
        curr_dir = os.getcwd()
        ed_path = os.path.join(curr_dir, 'generated_ed.pkl')
        print('Electron density saved to {}'.format(ed_path))
        generated_cubes = self.gan.generator(8, training=False)
        with open(ed_path, 'wb') as pfile:
                generated_cubes = transorm_back_ed(generated_cubes)
                pickle.dump(generated_cubes.numpy(), pfile)
        
    def train(self):
        if self.gan.distributed_training==True:
            with self.gan.strategy.scope() as s:
                for i in tqdm.tqdm(range(self.num_training_steps)):
                    self.counter+=1
                    self.train_discriminator()
                    self.train_generator()
                    if i % 10 == 0:
                        self.print_stats()
                        self.save_ed()
                    if i % self.save_model_every_steps == 0:
                        self.save_model()
        else:
                for i in tqdm.tqdm(range(self.num_training_steps)):
                    self.counter+=1
                    self.train_discriminator()
                    self.train_generator()
                    if i % 10 == 0:
                        self.print_stats()
                        self.save_ed()
                    if i % self.save_model_every_steps == 0:
                        self.save_model()
                        