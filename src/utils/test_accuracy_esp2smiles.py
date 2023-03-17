##########################################################################################
#
# Tests the accuracy of the Transformer model to generate smiles from electron density.
# For it to match, all the tokens (from start to first stop) need to match perfectly.
# Given a folder with weights, it will load them into the model, and test the accuracy
# against the test set.
# Author: juanma.parrilla@gcu.ac.uk
#
#########################################################################################

import glob
import argparse
import numpy as np
import tensorflow as tf
from datetime import datetime

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from src.models.ESP2smiles import ESP2S_Transformer
from src.utils.TFRecordLoader import TFRecordLoader

def clean_predictions(predictions):
    """This function will set as 31 everything after the first 31 (stop)

    Args:
        predictions ([type]): batch of lists of tokens as predicted by the NN
    """

    # for each prediction in the batch
    for i,pred in enumerate(predictions):
        found31 = False
        # for each token in this prediction
        for j,token in enumerate(pred):
            # if the current token is 31 then mark found
            if token == 31:
                found31 = True
            # make sure every other token is also 31 until the end
            if found31 == True:
                predictions[i][j] = 31


if __name__ == "__main__":

    # process arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="file to open", type=str)
    args = parser.parse_args()
    print('Opening {}'.format(args.folder))

    # log file where to save the results
    logfile = args.folder + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ".txt"
    with open(logfile, 'w') as f:
        print('TESTING ACCURACY ' + args.folder, file=f)

    # get list of weights to test
    allweights = sorted(glob.glob(args.folder+'*.h5'))

    # load validation data
    DATA_FOLDER = '/home/nvme/juanma/Data/ED/'
    path2va = DATA_FOLDER + 'valid.tfrecords'
    tfr_va = TFRecordLoader(path2va, batch_size=64, properties=['electrostatic_potential', 'smiles'])

    # create model
    e2s = ESP2S_Transformer(
            num_hid=64,
            num_head=4,
            num_feed_forward=512,
            num_layers_enc=2,
            num_layers_dec=2,
            )

    batch = next(tfr_va.dataset_iter)
    e2s.build([batch[1].shape, batch[2].shape])

    m = tf.keras.metrics.Accuracy()
    numbatches = 10  # number of batches to test

    for weight in allweights:
        # load the current weight
        e2s.load_weights(weight)
        good_sequences = 0  # to keep track of the number of good sequences predicted
        # load the record again so the batches are always the same
        tfr_va = TFRecordLoader(path2va, batch_size=32, properties=['electrostatic_potential', 'smiles'])

        for i in range(numbatches):
            # get the next batch
            batch = next(tfr_va.dataset_iter)
            # calculate the predictions
            predictions = e2s.generate(batch, startid=0)
            # transform in the source batch every 32 (null) to 31 (stop)
            batch_31s = batch[2].numpy()
            batch_31s[batch_31s==32] = 31
            # in the predictions make sure after a stop everything else is also stop
            preds_numpy = predictions.numpy()
            clean_predictions(preds_numpy)
            # just substract one from the other, 0 means they are the same
            comp = preds_numpy-batch_31s
            good_sequences += np.sum(~comp.any(1))
            # calculate the accuracy also as token by token comparison
            m.update_state(predictions, batch_31s)
        
        # print weight file, number of good smiles, accuracy token by token
        print(weight, good_sequences/(32*numbatches), m.result().numpy())
        # save it also into a file
        with open(logfile, 'a') as f:
            print(weight, good_sequences/(32*numbatches), m.result().numpy(), file=f)
        # need to reset metric now to start fresh next iteration
        m.reset_states()
