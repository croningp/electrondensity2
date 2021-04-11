##########################################################################################
#
# This script is a sort of main file for the model in src.models.VAE.py, which is a
# variational autoencoder. This script will aim to show to train it, and also to
# save and load the model.
#
# Author: Juan Manuel Parrilla Gutierrez (juanma@chem.gla.ac.uk)
#
##########################################################################################

import os
from datetime import datetime

from src.utils.TFRecordLoader import TFRecordLoader
from src.models.VAE import VariationalAutoencoder

# RUN PARAMS #############################################################################
RUN_FOLDER = 'logs/vae/'
mode = 'build'  # use 'build' to start train, 'load' to continue an old train

if mode == 'build':
    startdate = datetime.now().strftime('%Y-%m-%d')
    RUN_FOLDER += startdate + '/'

    if not os.path.exists(RUN_FOLDER):
        os.mkdir(RUN_FOLDER)
        os.mkdir(os.path.join(RUN_FOLDER, 'weights'))

else:  # mode == 'load'
    RUN_FOLDER += '2021-04-10/'  # fill with the right date

DATA_FOLDER = '/home/nvme/juanma/Data/Jarek/'

# DATA ###################################################################################
# paths to the train and validation sets
path2tf = DATA_FOLDER + 'train.tfrecords'
path2va = DATA_FOLDER + 'valid.tfrecords'
# load train and validation sets
tfr = TFRecordLoader(path2tf)
tfr_va = TFRecordLoader(path2va)

# ARCHITECTURE ###########################################################################
# create VAE model
vae = VariationalAutoencoder(
    input_dim=tfr.ED_SHAPE,
    encoder_conv_filters=[16, 64, 128, 256],
    encoder_conv_kernel_size=[3, 3, 3, 3],
    encoder_conv_strides=[2, 2, 2, 2],
    dec_conv_t_filters=[256, 128, 64, 16],
    dec_conv_t_kernel_size=[3, 3, 3, 3],
    dec_conv_t_strides=[2, 2, 2, 2],
    z_dim=256,
    use_batch_norm=True,
    use_dropout=True,
    r_loss_factor=50000
    )

if mode == 'build':
    vae.save(RUN_FOLDER)
else:
    vae.load_weights(os.path.join(RUN_FOLDER, 'weights/weights.h5'))

# TRAINING ###############################################################################
LEARNING_RATE = 0.0005
EPOCHS = 1000
INITIAL_EPOCH = 0

vae.compile(LEARNING_RATE)

vae.train(tfr.dataset, tfr_va.dataset, EPOCHS, RUN_FOLDER, INITIAL_EPOCH)

