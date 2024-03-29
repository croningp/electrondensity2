##########################################################################################
#
# Trains the ED decorated with ESP to smiles transformer
#
##########################################################################################

import os
from datetime import datetime
import tensorflow as tf

from src.utils.TFRecordLoader import TFRecordLoader
from src.models.ED_ESP2smiles import ED_ESP2S_Transformer
from src.datasets.utils.tokenizer import Tokenizer

# RUN PARAMS #############################################################################
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
RUN_FOLDER = 'logs/ed_esp2smiles/'
mode = 'load'  # use 'build' to start train, 'load' to continue an old train

if mode == 'build':
    startdate = datetime.now().strftime('%Y-%m-%d')
    RUN_FOLDER += startdate + '/'

    if not os.path.exists(RUN_FOLDER):
        os.mkdir(RUN_FOLDER)
        os.mkdir(os.path.join(RUN_FOLDER, 'weights'))
        os.mkdir(os.path.join(RUN_FOLDER, 'smiles'))

else:  # mode == 'load'
    RUN_FOLDER += '2021-12-09/'  # fill with the right date

# DATA_FOLDER = '/media/group/d22cc883-8622-4ecd-8e46-e3b0850bb89a2/juanma/'  # in dragonsoop
DATA_FOLDER = '/home/juanma/Data/'

# DATA ###################################################################################
# paths to the train and validation sets
path2tf = DATA_FOLDER + 'train.tfrecords'
path2va = DATA_FOLDER + 'valid.tfrecords'
# load train and validation sets
tfr = TFRecordLoader(path2tf, batch_size=64, train=True, properties=['electrostatic_potential', 'smiles'])
tfr_va = TFRecordLoader(path2va, batch_size=64, properties=['electrostatic_potential', 'smiles'])

# path to smiles tokenizer
path2to = DATA_FOLDER + 'tokenizer.json'
# load tokenizer
tokenizer = Tokenizer()
tokenizer.load_from_config(path2to)

# ARCHITECTURE ###########################################################################
# create Transformer model
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    e2s = ED_ESP2S_Transformer(
            num_hid=64,
            num_head=4,
            num_feed_forward=512,
            num_layers_enc=3,
            num_layers_dec=3,
            )
    e2s.compile_model()

    batch = next(tfr_va.dataset_iter)
    e2s([ [batch[0], batch[1]], batch[2]])
    e2s.summary()
    # reload validation set after we used 1 batch to build the model
    tfr_va = TFRecordLoader(path2va, batch_size=64, properties=['electrostatic_potential', 'smiles'])

if mode == 'build':
    e2s.save_build(RUN_FOLDER)
else:
    e2s.load_weights(os.path.join(RUN_FOLDER, 'weights/weights.h5'))

# TRAINING ###############################################################################
EPOCHS = 1000
INITIAL_EPOCH = 111
EPOCHS_PRINT = 1

e2s.train(tfr, tfr_va, EPOCHS, RUN_FOLDER, tokenizer, INITIAL_EPOCH, EPOCHS_PRINT)
