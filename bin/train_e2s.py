import os
from datetime import datetime
import tensorflow as tf

from src.utils.TFRecordLoader import TFRecordLoader
from src.models.ED2smiles import E2S_Transformer
from src.datasets.utils.tokenizer import Tokenizer

# RUN PARAMS #############################################################################
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
RUN_FOLDER = 'logs/e2s/'
mode = 'build'  # use 'build' to start train, 'load' to continue an old train

if mode == 'build':
    startdate = datetime.now().strftime('%Y-%m-%d')
    RUN_FOLDER += startdate + '/'

    if not os.path.exists(RUN_FOLDER):
        os.mkdir(RUN_FOLDER)
        os.mkdir(os.path.join(RUN_FOLDER, 'weights'))
        os.mkdir(os.path.join(RUN_FOLDER, 'smiles'))

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
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    e2s = E2S_Transformer(
            num_hid=64,
            num_head=2,
            num_feed_forward=128,
            num_layers_enc=2,
            num_layers_dec=2,
            )

batch = next(tfr_va.dataset_iter)
e2s.build([batch[0].shape, batch[1].shape])
e2s.summary()

if mode == 'build':
    e2s.save_build(RUN_FOLDER)
else:
    e2s.load_weights(os.path.join(RUN_FOLDER, 'weights/weights.h5'))

# TRAINING ###############################################################################
EPOCHS = 1000
INITIAL_EPOCH = 0
EPOCHS_PRINT = 5

e2s.compile_model()

e2s.train(tfr_va, tfr_va, EPOCHS, RUN_FOLDER, tokenizer, INITIAL_EPOCH, EPOCHS_PRINT)
