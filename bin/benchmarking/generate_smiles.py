##########################################################################################
#
# This script is used to benchmark the models. It takes a random latent vector, it uses
# the VAE to generate the 3D molecules, and then it uses the transformer to get the smiles
# and it saves the results in a pickle file
#
# Author: Juanma juanma@chem.gla.ac.uk
#
##########################################################################################

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import pickle
import numpy as np
from datetime import datetime
import tensorflow as tf
from tensorflow.keras import backend as K

from src.utils.optimiser_utils import load_vae_model, load_ED_to_ESP
from src.utils.optimiser_utils import grad_size, load_cage_host_ed_esp


def latent_vector_to_ed_esp(latent_vector, vae, ed2esp):
    """ given a latent vector, it will generate its electron densities and electro
    static potentials """.

    # using VAE convert the latent_vector into 3D electron densities
    eds = vae.decoder(noise)
    eds = transform_back_ed(output)

    # get the ESP from the generated guests ED. It will be between 0s and 1s
    esps = ed2esp.model(eds, usetanh=False)
    # now transform them from 0..1 to -1 .. 1
    cubes = (esps*2)-1
    # now between -0.33 and 0.33 which is the range of the orig data
    esps = cubes * 0.33

    return eds, esps


def ed_esp_to_smiles(eds, esps, e2s):
    """ it uses the electron densities, and the electrostatic potentials, to generate
    smiles using the transformer model"""

    # use model to generate token predictions based on the electron densities
    preds = e2s.generate([eds, esps, []], startid=0, greedy=True)
    preds = preds.numpy()

    smiles = []  # where to store the generated smiles
    target_end_token_idx = 31  # 31 means END

    # now we will transform the tokens into smiles letter
    for i in range(len(preds)):
        prediction = ""
        for idx in preds[i, 1:]:
            prediction += toks.num2token[str(idx)]
            if idx == target_end_token_idx:
                break

        # clean out the tokens
        prediction = prediction.replace('STOP', '')
        # add to results
        smiles.append(prediction)

    return smiles


def latent_vector_to_smiles(latent_vector, vae, ed2esp, e2s):

    eds, esps = latent_vector_to_ed_esp(latent_vector, vae, ed2esp)
    return ed_esp_to_smiles(eds, esps, e2s)


def load_tokenizer(data_folder):
    """Just loads the tokenizer to convert tokens such as '30' into letters such as 'C'

    Args:
        data_folder: Path to the data folder containing the file tokenizer.json

    Returns:
        tokenizer object
    """
    # path to smiles tokenizer
    path2to = data_folder + 'tokenizer.json'
    # load tokenizer
    tokenizer = Tokenizer()
    tokenizer.load_from_config(path2to)
    return tokenizer


def load_transformer(modelpath, datapath):
    """Create model from config file, and load the weights

    Args:
        modelpath: path to the log of the model. should be something like:
                   "logs/vae/2021-05-11"
        datapath: path to the TFRecord. We only need 1 batch to properly build model.

    Returns:
        model: returns the model with loaded weights
    """

    # load validation data. We just need a batch to properly build the model
    path2va = datapath + 'valid.tfrecords'
    tfr_va = TFRecordLoader(path2va, batch_size=64, properties=['electrostatic_potential', 'smiles'])
    batch = next(tfr_va.dataset_iter)

    # load the model configuration from the params.pkl file
    with open(os.path.join(modelpath, 'params.pkl'), 'rb') as handle:
        config = pickle.load(handle)

    # create the model
    e2s = ED_ESP2S_Transformer(
        num_hid=config[0],
        num_head=config[1],
        num_feed_forward=config[2],
        num_layers_enc=config[3],
        num_layers_dec=config[4],
    )
    batch = next(tfr_va.dataset_iter)
    e2s([ [batch[0], batch[1]], batch[2]])

    # load the weights and return it
    e2s.load_weights(os.path.join(modelpath, 'weights/weights.h5'))
    return e2s, batch


if __name__ == "__main__":

    # folder where to save the logs of this run
    startdate = datetime.now().strftime('%Y-%m-%d')
    RUN_FOLDER = startdate +'_'+ str(ed_factor)

    # just do a while loop to make sure the folder doesnt exist
    n = 0
    while os.path.exists(RUN_FOLDER+'_'+str(n)+'/'):
        n += 1
    RUN_FOLDER += '_'+str(n)+'/'
    os.mkdir(RUN_FOLDER)

    BATCH_SIZE = 48

    # load models
    vae, z_dim = load_vae_model('logs/vae/2021-05-25/')
    ed_to_esp = load_ED_to_ESP('logs/vae_ed_esp/2021-07-18')
    e2s, _ = load_model('logs/ed_esp2smiles/2021-12-09/', DATA_FOLDER)
    # load tokenizer
    toks = load_tokenizer(DATA_FOLDER)

    # generate random latent vector
    latent_vector = K.random_uniform(shape = (BATCH_SIZE, z_dim), minval = -3.0, maxval = 3.0)

    print( latent_vector_to_smiles(latent_vector, vae, ed_to_esp, e2s) )





