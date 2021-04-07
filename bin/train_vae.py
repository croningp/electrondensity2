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
RUN_FOLDER = "logs/vae/"
startdate = datetime.now().strftime('%Y-%m-%d')
RUN_FOLDER += startdate + "/"

if not os.path.exists(RUN_FOLDER):
    os.mkdir(RUN_FOLDER)
    os.mkdir(os.path.join(RUN_FOLDER, 'weights'))

mode = 'build'  # 'load' #  use build to train, load to continue an old train
DATA_FOLDER = "/home/nvme/juanma/Data/Jarek/"

# DATA ###################################################################################
# paths to the train and validation sets
path2tf = DATA_FOLDER + "train.tfrecords"
path2va = DATA_FOLDER + "valid.tfrecords"
# load train and validation sets
tfr = TFRecordLoader(path2tf)
tfr_va = TFRecordLoader(path2va)

# ARCHITECTURE ###########################################################################
# create VAE model
vae = VariationalAutoencoder(
    input_dim=tfr.ED_SHAPE,
    encoder_conv_filters=[32, 64, 64, 64],
    encoder_conv_kernel_size=[3, 3, 3, 3],
    encoder_conv_strides=[2, 2, 2, 2],
    decoder_conv_t_filters=[64, 64, 32, 3],
    decoder_conv_t_kernel_size=[3, 3, 3, 3],
    decoder_conv_t_strides=[2, 2, 2, 2],
    z_dim=200,
    use_batch_norm=True,
    use_dropout=True
    )

if mode == 'build':
    vae.save(RUN_FOLDER)
else:
    vae.load_weights(os.path.join(RUN_FOLDER, 'weights/weights.h5'))

# TRAINING ###############################################################################
LEARNING_RATE = 0.0005
R_LOSS_FACTOR = 10000
EPOCHS = 200
PRINT_EVERY_N_BATCHES = 100
INITIAL_EPOCH = 0

vae.compile(LEARNING_RATE, R_LOSS_FACTOR)

vae.train(tfr_va.dataset, tfr_va.dataset, EPOCHS, RUN_FOLDER, INITIAL_EPOCH)

