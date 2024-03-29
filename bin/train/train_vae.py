##########################################################################################
#
# This script is a sort of main file for the model in src.models.VAE.py, and other VAE
# variants. This script will aim to show to train it, and also to save and load the model.
# This file is modified to train in Dragonsoop
#
# Author: Juan Manuel Parrilla Gutierrez (juanma@chem.gla.ac.uk)
#
##########################################################################################

import os
from datetime import datetime
import tensorflow as tf

from src.utils.TFRecordLoader import TFRecordLoader
from src.models.VAEresnet import VAEresnet

# RUN PARAMS #############################################################################
os.environ["CUDA_VISIBLE_DEVICES"] = '2,3,4,5'
RUN_FOLDER = 'logs/vae/'
mode = 'load'  # use 'build' to start train, 'load' to continue an old train

if mode == 'build':
    startdate = datetime.now().strftime('%Y-%m-%d')
    RUN_FOLDER += startdate + '/'

    if not os.path.exists(RUN_FOLDER):
        os.mkdir(RUN_FOLDER)
        os.mkdir(os.path.join(RUN_FOLDER, 'weights'))
        os.mkdir(os.path.join(RUN_FOLDER, 'edms'))

else:  # mode == 'load'
    RUN_FOLDER += '2021-05-25/'  # fill with the right date

DATA_FOLDER = '/media/group/d22cc883-8622-4ecd-8e46-e3b0850bb89a2/jarek/'  # in DS
# DATA_FOLDER = '/home/nvme/juanma/Data/Jarek/'  # in auchentoshan

# DATA ###################################################################################
# paths to the train and validation sets
path2tf = DATA_FOLDER + 'train.tfrecords'
path2va = DATA_FOLDER + 'valid.tfrecords'
# load train and validation sets
tfr = TFRecordLoader(path2tf, batch_size=64)
tfr_va = TFRecordLoader(path2va, batch_size=32)

# ARCHITECTURE ###########################################################################
# create VAE model

# Create a MirroredStrategy.
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    vae = VAEresnet(
        input_dim=tfr.ED_SHAPE,
        encoder_conv_filters=[16, 32, 64, 128],
        encoder_conv_kernel_size=[3, 3, 3, 3],
        encoder_conv_strides=[2, 2, 2, 2],
        dec_conv_t_filters=[128, 64, 32, 16],
        dec_conv_t_kernel_size=[3, 3, 3, 3],
        dec_conv_t_strides=[2, 2, 2, 2],
        z_dim=400,
        use_batch_norm=True,
        use_dropout=True,
        r_loss_factor=50000
        )
    
    LEARNING_RATE = 0.0005
    vae.compile(LEARNING_RATE)

print(vae.encoder.summary())
print(vae.decoder.summary())

if mode == 'build':
    vae.save(RUN_FOLDER)
else:
    vae.load_weights(os.path.join(RUN_FOLDER, 'weights/weights.h5'))

# TRAINING ###############################################################################
EPOCHS = 1000
INITIAL_EPOCH = 31
EPOCHS_PRINT = 5

vae.train(tfr, tfr_va, EPOCHS, RUN_FOLDER, INITIAL_EPOCH, EPOCHS_PRINT)

