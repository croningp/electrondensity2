##########################################################################################
#
# This script is used to collect the smiles from the test and train sets, which are stored
# as tfrecords. So basically it just loads a tfrecord and iterated through it, getting
# the smiles, and then it stores the list of smiles as a pickle file.
#
##########################################################################################

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import pickle
import argparse
import tensorflow as tf

from src.utils.TFRecordLoader import TFRecordLoader


if __name__ == "__main__":

    # read file from user
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="file to open", type=str)
    parser.add_argument("savename", help="file to save as pickle", type=str)
    args = parser.parse_args()

    # load the tf record
    tfr = TFRecordLoader(args.file, batch_size=64, properties=['smiles_string'])

    # where to store all the smiles
    all_smiles = []

    # iterate through all the entries in the tfrecord
    for batch in tfr.dataset_iter:
        # iterate through entries in the batch. smiles in [1]. We also convert it to numpy
        for entry in batch[1].values.numpy():
            # get the smiles, decode it and save it into smiles
            all_smiles.append( entry.decode('UTF-8') )

    # save it into a pickle 
    with open(args.savename, 'wb') as handle:
        pickle.dump(all_smiles, handle)






