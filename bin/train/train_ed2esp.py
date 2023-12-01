##########################################################################################
#
# This script is a sort of main file for ED2ESP (FCN) model. This script will aim to 
# show to train it, and also to save and load the model.
#
# Author: Juan Manuel Parrilla Gutierrez (juanma@chem.gla.ac.uk)
#
##########################################################################################

import os
from datetime import datetime
import tensorflow as tf

from src.utils.TFRecordLoader import TFRecordLoader
from src.models.ED2ESP import VAE_ed_esp

# RUN PARAMS #############################################################################
os.environ["CUDA_VISIBLE_DEVICES"] = '2,3,4,5'
RUN_FOLDER = 'logs/vae_ed_esp/'
mode = 'build'  # use 'build' to start train, 'load' to continue an old train

if mode == 'build':
    startdate = datetime.now().strftime('%Y-%m-%d')
    RUN_FOLDER += startdate + '/'

    if not os.path.exists(RUN_FOLDER):
        os.mkdir(RUN_FOLDER)
        os.mkdir(os.path.join(RUN_FOLDER, 'weights'))
        os.mkdir(os.path.join(RUN_FOLDER, 'edms'))

else:  # mode == 'load'
    RUN_FOLDER += '2021-05-25/'  # fill with the right date

DATA_FOLDER = '/media/group/d22cc883-8622-4ecd-8e46-e3b0850bb89a2/juanma/'  # in DS
# DATA_FOLDER = '/home/nvme/juanma/Data/Jarek/'  # in auchentoshan

# DATA ###################################################################################
# paths to the train and validation sets
path2tf = DATA_FOLDER + 'train.tfrecords'
path2va = DATA_FOLDER + 'valid.tfrecords'
# load train and validation sets
tfr = TFRecordLoader(path2tf, batch_size=64, properties=['electrostatic_potential'])
tfr_va = TFRecordLoader(path2va, batch_size=32, properties=['electrostatic_potential'])

# ARCHITECTURE ###########################################################################
# create VAE model

# Create a MirroredStrategy.
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    vae = VAE_ed_esp(
        input_dim=tfr.ED_SHAPE,
        encoder_conv_filters=[16, 32, 64, 64],
        encoder_conv_kernel_size=[3, 3, 3, 3],
        encoder_conv_strides=[2, 2, 2, 2],
        dec_conv_t_filters=[64, 64, 32, 16],
        dec_conv_t_kernel_size=[3, 3, 3, 3],
        dec_conv_t_strides=[2, 2, 2, 2],
        z_dim=400,
        use_batch_norm=True,
        use_dropout=True,
        r_loss_factor=50000,
        )
    
    LEARNING_RATE = 0.0001
    vae.compile(LEARNING_RATE)

print(vae.encoder.summary())
print(vae.decoder.summary())

if mode == 'build':
    vae.save(RUN_FOLDER)
else:
    vae.load_weights(os.path.join(RUN_FOLDER, 'weights/weights.h5'))

# TRAINING ###############################################################################
EPOCHS = 1000
INITIAL_EPOCH = 0
EPOCHS_PRINT = 5

vae.train(tfr, tfr_va, EPOCHS, RUN_FOLDER, INITIAL_EPOCH, EPOCHS_PRINT, lr_decay=0.99)

