# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 12:29:11 2020

@author: group
"""


import os
import pickle
import numpy as np
import tqdm
import time
os.environ["CUDA_VISIBLE_DEVICES"]="1,2"
import tensorflow as tf

os.chdir('/home/jarek/electrondensity2')

from electrondensity2.input.tfrecords import input_fn
from electrondensity2.layers import ResBlockDown3D, ResBlockUp3D, ConvSelfAttn3D, Generator_v3, Discriminator_v3
from electrondensity2.layers import SpatialDiscriminator, TemporalDiscriminator
from electrondensity2.utils import  transorm_ed, transorm_back_ed
from electrondensity2.gan import GP_WGAN, GANTrainer

from inception import calculate_inception_score


generator_config = {'use_batchnorm':False, 'activation_fn':'elu',
                    'kernel_initializer':'glorot_uniform',
                    'noise_distribution':'normal'}

discrimator_config = {'activation_fn':'relu', 'use_attn':False, 
                     'kernel_initializer':'orthogonal'}


gan = GP_WGAN(Generator_v3, Discriminator_v3, generator_config, discrimator_config,
              distributed_training=True)

entropies = []
incpetion_scores = []


for ckpt in tqdm.tqdm(range(10, 130, 10)):
    
    try:
        gan.restore('/media/group/d22cc883-8622-4ecd-8e46-e3b0850bb89a/jarek/model_3/dis.ckpt-{}'.format(ckpt),
            '/media/group/d22cc883-8622-4ecd-8e46-e3b0850bb89a/jarek/model_3/gen.ckpt-{}'.format(ckpt))
    except:
        continue
    samples = gan.sample_model(num_samples=10000)
    
    
    dataset = tf.data.Dataset.from_tensor_slices(samples)
    #dataset = dataset.map(transorm_back)
    dataset = dataset.batch(20)
    dataset = dataset.repeat(1)
    
    dataset_iterator = iter(dataset)
    proba, marginal, inception, ent = calculate_inception_score(dataset_iterator)
    print('Entropy', ent)
        
    entropies.append(ent)
    incpetion_scores.append(inception)

print(entropies)
print('----------------------------------')
print(incpetion_scores)
