# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 22:54:15 2020

@author: jmg
"""
import os
import pickle
import numpy as np
import tqdm
import time
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
import tensorflow as tf

os.chdir('/home/jarek/electrondensity2')

from input.tfrecords import input_fn
from layers import ResBlockDown3D, ResBlockUp3D, ConvSelfAttn3D, Generator_v3, Discriminator_v3
from layers import SpatialDiscriminator, TemporalDiscriminator
from utils import  transorm_ed, transorm_back_ed
from gan import GP_WGAN, GANTrainer, LS_GAN







#generator_config = {}
#discrimator_config = {}
#                         discrimator_config)







generator_config = {'use_batchnorm':True, 'activation_fn':'elu',
                    'kernel_initializer':'glorot_uniform',
                    'noise_distribution':'normal', 
                    'use_attn':True}

discrimator_config = {'activation_fn':'relu', 'use_attn':True, 
                     'kernel_initializer':'orthogonal'}


gan = GP_WGAN(Generator_v3, Discriminator_v3, generator_config, discrimator_config,
              distributed_training=True)


#gan = LS_GAN(Generator_v3, Discriminator_v3, generator_config, discrimator_config,
 #             distributed_training=False, d_learning_rate=1e-5)


#gan.restore('/media/group/d22cc883-8622-4ecd-8e46-e3b0850bb89a/jarek/models/dis.ckpt-69',
#            '/media/group/d22cc883-8622-4ecd-8e46-e3b0850bb89a/jarek/models/gen.ckpt-69')

#gan.sample_model('/home/jarek/samples.pkl', num_samples=100)
trainer = GANTrainer(gan, num_training_steps=500000, 
                      steps_train_discriminator=1)

trainer.train()
    
    

    

    



