# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 16:49:46 2020

@author: group
"""
import os 
import tqdm
import numpy as np
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import tensorflow as tf
os.chdir('/home/jarek/electrondensity2')
from layers import Generator_v3, Discriminator_v3
from utils import  transorm_ed, transorm_back_ed
from gan import GP_WGAN
import pickle





generator_config = {'use_batchnorm':True, 'activation_fn':'elu',
                    'kernel_initializer':'glorot_uniform',
                    'noise_distribution':'normal', 
                    'use_attn':True}

discrimator_config = {'activation_fn':'relu', 'use_attn':False, 
                     'kernel_initializer':'orthogonal'}


gan = GP_WGAN(Generator_v3, Discriminator_v3, generator_config, discrimator_config,
              distributed_training=True)


gan.restore('/media/group/d22cc883-8622-4ecd-8e46-e3b0850bb89a/jarek/model_4/dis.ckpt-40',
            '/media/group/d22cc883-8622-4ecd-8e46-e3b0850bb89a/jarek/model_4/gen.ckpt-240')



noise_t = gan.generator.sample_z(32)

@tf.function
def grad(noise):
    
    output = gan.generator.generate(noise, training=False)
    output = transorm_back_ed(output)

    fitness = tf.reduce_sum(output, axis=[1,2,3,4,])
    gradients = tf.gradients(fitness, noise)
    
    return fitness, gradients, output


noise_initial = gan.generator.sample_z(32)
_, _, initial_output = grad(noise_t)
fitness = []

noise_t = noise_initial
for i in tqdm.tqdm(range(3000)):
    f, grads, output = grad(noise_t)
    print(np.mean(f.numpy()))
    fitness.append(np.mean(f.numpy()))
    noise_t += 0.01 * grads[0].numpy()
    noise_t = np.clip(noise_t, a_min=-1.0, a_max=1.0)
    
with open('optimized.pkl', 'wb') as file:
    pickle.dump(output, file)
    
with open('initial.pkl', 'wb') as file:
    pickle.dump(initial_output, file)
    

plt.plot(fitness)
    

