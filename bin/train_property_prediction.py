##########################################################################################
#
# This model will aim to predict a QM9 property (a single property) from electron
# densities. To do so, it will do the usual 3D convs, then flattened, then single output.
#
# Author: Juan Manuel Parrilla
#
##########################################################################################

import os
from datetime import datetime
import tensorflow as tf

from src.utils.TFRecordLoader import TFRecordLoader
from src.models.CNN3D_featureprediction import CNN3D_singleprediction


# RUN PARAMS #############################################################################
# os.environ["CUDA_VISIBLE_DEVICES"] = '2,3,4,5'
RUN_FOLDER = 'logs/feature_prediction/'
mode = 'build'  # use 'build' to start train, 'load' to continue an old train

if mode == 'build':
    startdate = datetime.now().strftime('%Y-%m-%d')
    RUN_FOLDER += startdate + '/'

    if not os.path.exists(RUN_FOLDER):
        os.mkdir(RUN_FOLDER)
        os.mkdir(os.path.join(RUN_FOLDER, 'weights'))

else:  # mode == 'load'
    RUN_FOLDER += '2021-05-25/'  # fill with the right date

# DATA_FOLDER = '/media/group/d22cc883-8622-4ecd-8e46-e3b0850bb89a2/jarek/'  # in DS
DATA_FOLDER = '/home/nvme/juanma/Data/Jarek/'  # in auchentoshan


# DATA ###################################################################################
# paths to the train and validation sets
path2tf = DATA_FOLDER + 'train.tfrecords'
path2va = DATA_FOLDER + 'valid.tfrecords'
# load train and validation sets
tfr = TFRecordLoader(path2tf, batch_size=64, train=True, properties=['alpha'])
tfr_va = TFRecordLoader(path2va, batch_size=32, properties=['alpha'])


# ARCHITECTURE ###########################################################################
# create 3D CNN with a single output neuron for property prediction

# Create a MirroredStrategy.
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    cnn3D = CNN3D_singleprediction(
                cubeside = 64, 
                filters = [64,64,128,256], 
                strides= [2,2,2,2], 
                dense_size = 512,
        )
    
    LEARNING_RATE = 0.0005
    cnn3D.compile(LEARNING_RATE)

print(cnn3D.model.summary())

if mode == 'build':
    cnn3D.save(RUN_FOLDER)
else:
    cnn3D.load_weights(os.path.join(RUN_FOLDER, 'weights/weights.h5'))

# TRAINING ###############################################################################
EPOCHS = 1000
INITIAL_EPOCH = 0

cnn3D.train(tfr, tfr_va, EPOCHS, RUN_FOLDER, INITIAL_EPOCH)