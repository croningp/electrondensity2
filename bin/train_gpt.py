import os
from datetime import datetime
import tensorflow as tf

from src.utils.TFRecordLoader import TFRecordLoader
from src.models.gpt import GPT
from src.datasets.utils.tokenizer import Tokenizer

# RUN PARAMS #############################################################################
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
RUN_FOLDER = 'logs/gpt/'
mode = 'build'  # use 'build' to start train, 'load' to continue an old train

if mode == 'build':
    startdate = datetime.now().strftime('%Y-%m-%d')
    RUN_FOLDER += startdate + '/'

    if not os.path.exists(RUN_FOLDER):
        os.mkdir(RUN_FOLDER)
        os.mkdir(os.path.join(RUN_FOLDER, 'weights'))

else:  # mode == 'load'
    RUN_FOLDER += '2021-05-07/'  # fill with the right date

DATA_FOLDER = '/home/nvme/juanma/Data/Jarek/'

# DATA ###################################################################################
# paths to the train and validation sets
path2tf = DATA_FOLDER + 'train.tfrecords'
path2va = DATA_FOLDER + 'valid.tfrecords'
# load train and validation sets
tfr = TFRecordLoader(path2tf, batch_size=64, properties=['smiles'])
tfr_va = TFRecordLoader(path2va, batch_size=64, properties=['smiles'])

# path to smiles tokenizer
path2to = DATA_FOLDER + 'tokenizer.json'
# load tokenizer
tokenizer = Tokenizer()
tokenizer.load_from_config(path2to)

# ARCHITECTURE ###########################################################################
# create GPT model
gpt = GPT()

if mode == 'build':
    gpt.save_build(RUN_FOLDER)
else:
    gpt.load_weights(os.path.join(RUN_FOLDER, 'weights/weights.h5'))

# TRAINING ###############################################################################
LEARNING_RATE = 0.001
EPOCHS = 1000
INITIAL_EPOCH = 0
EPOCHS_PRINT = 5

gpt.compile_model(LEARNING_RATE)

gpt.train(tfr_va, tfr_va, EPOCHS, RUN_FOLDER, tokenizer, INITIAL_EPOCH, EPOCHS_PRINT)