##########################################################################################
#
# This script is used to benchmark the models. It takes a random latent vector, it uses
# the VAE to generate the 3D molecules, and then it uses the transformer to get the smiles
# and it saves the results in a pickle file
#
# To run it, execute for example
# python bin/benchmarking/generate_smiles.py --std_dev 5.0 --Nsmiles 1000
# where that 5.0 after std_dev is the standard dev of the random latent vector and
# that 1000 after Nsmiles is the number of smiles to generate
#
# Author: Juanma juanma@chem.gla.ac.uk
#
##########################################################################################

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import pickle
from datetime import datetime
import pickle
import argparse

from tensorflow.keras import backend as K

from src.utils import transform_back_ed
from src.utils.optimiser_utils import load_vae_model, load_ED_to_ESP
from src.utils.edesp_to_smiles import load_tokenizer, load_transformer_model


def latent_vector_to_ed_esp(latent_vector, vae, ed2esp):
    """ given a latent vector, it will generate its electron densities and electro
    static potentials """

    # using VAE convert the latent_vector into 3D electron densities
    eds = vae.decoder(latent_vector)
    eds = transform_back_ed(eds)

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


if __name__ == "__main__":

    # first read and parse from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--std_dev", help="latent vector std dev", type=float)
    parser.add_argument("--Nsmiles", help="smiles to generate", type=int)
    args = parser.parse_args()
    std_dev = args.std_dev
    number_to_generate = args.Nsmiles

    # name of the pickle file where to save the smiles
    startdate = datetime.now().strftime('%Y-%m-%d')
    pickle_file = str(std_dev) +'_'+ str(number_to_generate) + '_' + startdate +'.p'

    # DATA_FOLDER = '/home/nvme/juanma/Data/ED/'  # in auchentoshan
    DATA_FOLDER = '/media/extssd/juanma/'

    # load models
    vae, z_dim = load_vae_model('logs/vae/2021-05-25/')
    ed_to_esp = load_ED_to_ESP('logs/vae_ed_esp/2021-07-18')
    e2s, _ = load_transformer_model('logs/ed_esp2smiles/2021-12-09/', DATA_FOLDER)
    # load tokenizer
    toks = load_tokenizer(DATA_FOLDER)

    # generate random latent vector
    BATCH_SIZE = 48
    latent_vector = K.random_uniform(shape = (BATCH_SIZE, z_dim), minval = -std_dev, maxval = std_dev)

    # now just itereate and generate the smiles
    counter = 0
    final_smiles = []
    while counter < number_to_generate:
        print(str(counter) + "/" + str(number_to_generate))
        final_smiles += latent_vector_to_smiles(latent_vector, vae, ed_to_esp, e2s)
        counter += BATCH_SIZE
    
    # save in pickle file
    with open(pickle_file, 'wb') as handle:
        pickle.dump(final_smiles, handle)





