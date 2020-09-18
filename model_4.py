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
os.environ["CUDA_VISIBLE_DEVICES"]="1,2"
import tensorflow as tf

os.chdir('/home/jarek/electrondensity2')

from electrondensity2.input.tfrecords import input_fn
from electrondensity2.layers import ResBlockDown3D, ResBlockUp3D, ConvSelfAttn3D, Generator_v3, Discriminator_v3
from electrondensity2.layers import SpatialDiscriminator, TemporalDiscriminator
from electrondensity2.utils import  transorm_ed, transorm_back_ed
from electrondensity2.gan import GP_WGAN, GANTrainer







#generator_config = {}
#discrimator_config = {}
#                         discrimator_config)


#gan = LS_GAN(Generator_v3, Discriminator_v3, generator_config, discrimator_config,
        #      distributed_training=False, d_learning_rate=1e-5)



generator_config = {'use_batchnorm':False, 'activation_fn':'elu',
                    'kernel_initializer':'glorot_uniform',
                    'noise_distribution':'normal'}

discrimator_config = {'activation_fn':tf.nn.swish, 'use_attn':False, 
                     'kernel_initializer':'orthogonal'}


gan = GP_WGAN(Generator_v3, Discriminator_v3, generator_config, discrimator_config,
              distributed_training=True)


gan.restore('/media/group/d22cc883-8622-4ecd-8e46-e3b0850bb89a/jarek/model_3/dis.ckpt-132',
            '/media/group/d22cc883-8622-4ecd-8e46-e3b0850bb89a/jarek/model_3/gen.ckpt-132')


trainer = GANTrainer(gan, num_training_steps=100000, 
                      steps_train_discriminator=5)

trainer.train()
    
    

    

    



