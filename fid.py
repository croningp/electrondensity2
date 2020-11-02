# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 19:10:52 2020

@author: group
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 12:29:11 2020

@author: group
"""


import os
import pickle
import numpy as np
import scipy
import tqdm
import time
import matplotlib.pyplot as plt


os.environ["CUDA_VISIBLE_DEVICES"]="1"
import tensorflow as tf

os.chdir('/home/jarek/electrondensity2')

from input.tfrecords import input_fn
from layers import ResBlockDown3D, ResBlockUp3D, ConvSelfAttn3D, Generator_v3, Discriminator_v3
from layers import SpatialDiscriminator, TemporalDiscriminator
from utils import  transorm_ed, transorm_back_ed
from gan import GP_WGAN, GANTrainer

from inception import Inception


generator_config = {'use_batchnorm':True, 'activation_fn':'elu',
                    'kernel_initializer':'glorot_uniform',
                    'noise_distribution':'normal', 
                    'use_attn':True, 
                    'use_layernorm':False}

discrimator_config = {'activation_fn':'relu', 'use_attn':False, 
                     'kernel_initializer':'orthogonal'}


gan = GP_WGAN(Generator_v3, Discriminator_v3, generator_config, discrimator_config,
              distributed_training=True)





optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-5)
    
inception = Inception()
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=inception)
checkpoint.restore('/media/group/d22cc883-8622-4ecd-8e46-e3b0850bb89a/jarek/inception_models/inception.ckpt-40')





scores = []

for ckpt in range(1, 190, 10):
    
    
    real_data_iterator = iter(input_fn('/media/group/d22cc883-8622-4ecd-8e46-e3b0850bb89a/jarek/train.tfrecords',
                              batch_size=32, num_epochs=1))
    
    
    gan.restore('/media/group/d22cc883-8622-4ecd-8e46-e3b0850bb89a/jarek/model_4/dis.ckpt-{}'.format(ckpt),
            '/media/group/d22cc883-8622-4ecd-8e46-e3b0850bb89a/jarek/model_4/gen.ckpt-{}'.format(ckpt))


    f_a = []
    r_a = []
    
    for i in tqdm.tqdm(range(500)):
    
        real_cubes = real_data_iterator.__next__()[0]
        real_cubes = tf.tanh(real_cubes)
        real_cubes = transorm_ed(real_cubes)
    
        real_activations, _ = inception(real_cubes)
        real_activations = real_activations.numpy()
        r_a.extend(real_activations)
    
        fake_cubes = gan.generator(32, training=False)
        
    
    
        fake_activations, _ = inception(fake_cubes)
        fake_activations = fake_activations.numpy()
        f_a.extend(fake_activations)
        
    
    u1 = np.mean(f_a, axis=0)
    u2 = np.mean(r_a, axis=0)
    diff = u1 - u2
    diff_squared = diff.dot(diff)
    
    
    s1 = np.cov(f_a, rowvar=False)
    s2 = np.cov(r_a, rowvar=False)
    prod = s1.dot(s2)
    sqrt_prod, _ = scipy.linalg.sqrtm(prod, disp=False)
    
    if np.iscomplexobj(sqrt_prod):
        sqrt_prod = sqrt_prod.real
    
    prod_trace = np.trace(sqrt_prod)
    
    final_score = diff_squared + np.trace(s1) + np.trace(s2) - 2* prod_trace
    scores.append(final_score)
    
    print('Checkpoint {} Score {}'.format(ckpt, final_score))
    
plt.plot(scores)
print(scores)


# entropies = []
# incpetion_scores = []


# for ckpt in tqdm.tqdm(range(10, 130, 10)):
    
#     try:
#         gan.restore('/media/group/d22cc883-8622-4ecd-8e46-e3b0850bb89a/jarek/model_3/dis.ckpt-{}'.format(ckpt),
#             '/media/group/d22cc883-8622-4ecd-8e46-e3b0850bb89a/jarek/model_3/gen.ckpt-{}'.format(ckpt))
#     except:
#         continue
#     samples = gan.sample_model(num_samples=10000)
    
    
#     dataset = tf.data.Dataset.from_tensor_slices(samples)
#     #dataset = dataset.map(transorm_back)
#     dataset = dataset.batch(20)
#     dataset = dataset.repeat(1)
    
#     dataset_iterator = iter(dataset)
#     proba, marginal, inception, ent = calculate_inception_score(dataset_iterator)
#     print('Entropy', ent)
        
#     entropies.append(ent)
#     incpetion_scores.append(inception)

# print(entropies)
# print('----------------------------------')
# print(incpetion_scores)
