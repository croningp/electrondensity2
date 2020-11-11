# -*- coding: utf-8 -*-
"""
Proof of principle optimization in the latent space by maximizing number of electrons. 
Created on Tue Oct 20 16:49:46 2020

@author: Jaroslaw Granda
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


gan.restore('/media/group/d22cc883-8622-4ecd-8e46-e3b0850bb89a2/jarek/model_4/dis.ckpt-40',
            '/media/group/d22cc883-8622-4ecd-8e46-e3b0850bb89a2/jarek/model_4/gen.ckpt-240')


# sample the noise with batch size 32
noise_t = gan.generator.sample_z(32)

@tf.function
def grad(noise):
    """Computes the gradient of fintess function with respect to latent z
    Args:
        noise: a tensor with hidden noise z of shape [batch_size, noise_size]
    Returns:
        fitness: tensor with current fitness
        gradients: a tensor with shape[batch_size, noise_size]
        output: a tensor with generated electron densities with shape
                [batch_size, 64, 64, 64, 1]
    """
    output = gan.generator.generate(noise, training=False)
    output = transorm_back_ed(output)
    fitness = tf.reduce_sum(output, axis=[1,2,3,4,])
    gradients = tf.gradients(fitness, noise)
    return fitness, gradients, output


noise_initial = gan.generator.sample_z(32)
_, _, initial_output = grad(noise_t)
fitness = []
# gradient ascent training loop
noise_t = noise_initial
for i in tqdm.tqdm(range(3000)):
    f, grads, output = grad(noise_t)
    print(np.mean(f.numpy()))
    fitness.append(np.mean(f.numpy()))
    noise_t += 0.01 * grads[0].numpy()
    # clip the noise to -1. 1. to prevent from driffting from normal distribution
    noise_t = np.clip(noise_t, a_min=-1.0, a_max=1.0)
   
# pickle results
with open('optimized.pkl', 'wb') as file:
    pickle.dump(output, file)
    
with open('initial.pkl', 'wb') as file:
    pickle.dump(initial_output, file)
    
# plot fitness
plt.plot(fitness)
    

