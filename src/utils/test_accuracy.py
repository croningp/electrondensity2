##########################################################################################
#
# Given a folder with weights, it will load them into the model, and test the accuracy
# against the test set.
#
#########################################################################################

import glob
import argparse
import numpy as np
import tensorflow as tf

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from src.models.ED2smiles import E2S_Transformer
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

    # get list of weights to test
    allweights = sorted(glob.glob(args.folder+'*.h5'))

    # load validation data
    DATA_FOLDER = '/media/group/d22cc883-8622-4ecd-8e46-e3b0850bb89a2/jarek/tfrecords/'
    path2va = DATA_FOLDER + 'valid.tfrecords'
    tfr_va = TFRecordLoader(path2va, batch_size=64, properties=['smiles'])

    # create model
    e2s = E2S_Transformer(
        num_hid=64,
        num_head=2,
        num_feed_forward=128,
        num_layers_enc=2,
        num_layers_dec=2,
        )
    # instead of doing the whole thing, just take a batch for faster calculations
    for i in range(15):
        batch = next(tfr_va.dataset_iter)
    e2s.build([batch[0].shape, batch[1].shape])

    m = tf.keras.metrics.Accuracy()

    for weight in allweights:
        # load the current weight
        e2s.load_weights(weight)
        # calculate the predictions
        predictions = e2s.generate(batch, startid=0)
        # transform in the source batch every 32 (null) to 31 (stop)
        batch_31s = batch[1].numpy()
        batch_31s[batch_31s==32] = 31
        # in the predictions make sure after a stop everything else is also stop
        preds_numpy = predictions.numpy()
        clean_predictions(preds_numpy)
        # just substract one from the other, 0 means they are the same
        comp = preds_numpy-batch_31s
        # calculate the accuracy also as token by token comparison
        m.update_state(predictions, batch[1])
        # print weight file, number of good smiles, accuracy token by token
        print(weight, np.sum(~comp.any(1))/64., m.result().numpy())
        # need to reset metric now to start fresh next iteration
        m.reset_states()
