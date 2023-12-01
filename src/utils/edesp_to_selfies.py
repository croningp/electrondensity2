##########################################################################################
#
# Given a pickle file with electron densities, it will use the transformer model to
# generate the smiles, save them as a pickle file, and then it will use rdkit to generate
# molecule visualisations.
#
# @author: Juan Manuel Parrilla (juanma@chem.gla.ac.uk)
#
##########################################################################################


from src.utils.TFRecordLoader import TFRecordLoader
from src.datasets.utils.tokenizer import SelfiesTokenizer
from src.models.ED_ESP2selfies import ED_ESP2SF_Transformer
from rdkit.Chem import Draw
from rdkit import Chem
import pandas as pd
import argparse
import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import selfies as sf
# imports to get access to sascorer
from rdkit.Chem import RDConfig
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer


def load_tokenizer(data_folder):
    """Just loads the tokenizer to convert tokens such as '30' into letters such as 'C'

    Args:
        data_folder: Path to the data folder containing the file tokenizer.json

    Returns:
        tokenizer object
    """
    # path to smiles tokenizer
    path2to = data_folder + 'sf_token_config.json'
    # load tokenizer
    tokenizer = SelfiesTokenizer()
    tokenizer.load_from_config(path2to)
    return tokenizer


def load_transformer_model(modelpath, datapath):
    """Create model from config file, and load the weights

    Args:
        modelpath: path to the log of the model. should be something like:
                   "logs/ed_esp2selfies/2021-05-11"
        datapath: path to the TFRecord. We only need 1 batch to properly build model.

    Returns:
        model: returns the model with loaded weights
    """

    # load validation data. We just need a batch to properly build the model
    path2va = datapath + 'valid.tfrecords'
    tfr_va = TFRecordLoader(path2va, batch_size=64,
                            properties=['electrostatic_potential', 'selfies'])
    batch = next(tfr_va.dataset_iter)

    # load the model configuration from the params.pkl file
    with open(os.path.join(modelpath, 'params.pkl'), 'rb') as handle:
        config = pickle.load(handle)

    # create the model
    e2s = ED_ESP2SF_Transformer(
        num_hid=config[0],
        num_head=config[1],
        num_feed_forward=config[2],
        num_layers_enc=config[3],
        num_layers_dec=config[4],
    )
    batch = next(tfr_va.dataset_iter)
    e2s([[batch[0], batch[1]], batch[2]])

    # load the weights and return it
    e2s.load_weights(os.path.join(modelpath, 'weights/weights.h5'))
    return e2s, batch


if __name__ == "__main__":

    DATA_FOLDER = '/home/nvme/juanma/Data/ED/'

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="folder to open", type=str)
    args = parser.parse_args()
    print('Opening {}'.format(args.folder))

    # load the model
    e2s, batch = load_transformer_model('logs/ed_esp2selfies/2023-09-04/', DATA_FOLDER)

    # load tokenizer
    toks = load_tokenizer(DATA_FOLDER)

    # load Dario's db of commercially available molecules
    commercialDB = pd.read_csv(DATA_FOLDER+'merged_db_MA.csv', usecols=['smiles'])

    # load the eds. I screwed when naming the files, ESP is actually ED.
    with open(args.folder+"/cb6_esped_optimised_final.p", 'rb') as pfile:
        ed_cubes = pickle.load(pfile)

    # load the esps
    with open(args.folder+"/cb6_espesp_optimised_final.p", 'rb') as pfile:
        esp_cubes = pickle.load(pfile)

    # use model to generate token predictions based on the electron densities
    preds = e2s.generate([ed_cubes, esp_cubes, []], startid=0, greedy=True)
    preds = preds.numpy()

    smiles = []  # where to store the generated smiles
    target_end_token_idx = 0  # 0 means [STOP]

    # now we will transform the tokens into selfies letter
    for i in range(len(preds)):
        prediction = ""
        for idx in preds[i, 1:]:
            prediction += toks.num2token[str(idx)]
            if idx == target_end_token_idx:
                break

        # clean out the tokens
        prediction = prediction.replace('[STOP]', '')
        prediction = prediction.replace('STOP', '')

        # Convert to smiles and add to results
        smiles.append(sf.decoder(prediction))

    print(smiles)

    # check if these molecules are commercially avaiable
    # i want to print the commercial the last, for easier visualisation
    commercial_smiles = [s for s in smiles if s in commercialDB.values]
    noncommercial_smiles = [s for s in smiles if s not in commercialDB.values]

    # now smiles to molecules
    comm_mols = [Chem.MolFromSmiles(m) for m in commercial_smiles]
    noncomm_mols = [Chem.MolFromSmiles(m) for m in noncommercial_smiles]
    mols = noncomm_mols + comm_mols

    # calculate synthetic scores
    comm_scores = [sascorer.calculateScore(m) if m is not None else 10 for m in comm_mols]
    noncomm_scores = [sascorer.calculateScore(m) if m is not None else 10
                    for m in noncomm_mols]
    # transform them to strings
    comm_scores = [f"{s:.1f}" for s in comm_scores]
    noncomm_scores = [f"{s:.1f}" for s in noncomm_scores]

    # create legend with commercial availability
    comm_legend = [s+" Y" for s in commercial_smiles]
    noncomm_legend = [s+" N" for s in noncommercial_smiles]

    # add the scores
    comm_legend = [t[0]+" "+t[1] for t in zip(comm_legend, comm_scores)]
    noncomm_legend = [t[0]+" "+t[1] for t in zip(noncomm_legend, noncomm_scores)]

    legend = noncomm_legend + comm_legend

    # molecules to image
    img = Draw.MolsToGridImage(mols, molsPerRow=8, subImgSize=(200, 200),
                            legends=legend)
    img.save('smiles'+'.png')
