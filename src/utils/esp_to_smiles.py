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
from src.datasets.utils.tokenizer import Tokenizer
from src.models.ESP2smiles import ESP2S_Transformer
from rdkit.Chem import Draw
from rdkit import Chem
import pandas as pd
import argparse
import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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
    path2to = data_folder + 'tokenizer.json'
    # load tokenizer
    tokenizer = Tokenizer()
    tokenizer.load_from_config(path2to)
    return tokenizer


def load_model(modelpath, datapath):
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
    e2s = ESP2S_Transformer(
        num_hid=config[0],
        num_head=config[1],
        num_feed_forward=config[2],
        num_layers_enc=config[3],
        num_layers_dec=config[4],
    )
    e2s.build([batch[1].shape, batch[2].shape])

    # load the weights and return it
    e2s.load_weights(os.path.join(modelpath, 'weights/weights.h5'))
    return e2s, batch


if __name__ == "__main__":

    DATA_FOLDER = '/home/nvme/juanma/Data/ED/'

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="file to open", type=str)
    args = parser.parse_args()
    print('Opening {}'.format(args.file))

    # load the model
    e2s, batch = load_model('logs/esp2s/2021-08-17/', DATA_FOLDER)

    # load tokenizer
    toks = load_tokenizer(DATA_FOLDER)

    # load Dario's db of commercially available molecules
    commercialDB = pd.read_csv(DATA_FOLDER+'merged_db_MA.csv', usecols=['smiles'])

    # load the electrostatic potentials
    with open(args.file, 'rb') as pfile:
        cubes = pickle.load(pfile)

    # use model to generate token predictions based on the electron densities
    preds = e2s.generate([[], cubes, []], startid=0, greedy=True)
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

    print(smiles)

    # now smiles to molecules
    mols = [Chem.MolFromSmiles(m) for m in smiles]

    # calculate synthetic scores
    scores = [sascorer.calculateScore(m) if m is not None else 10 for m in mols]
    # transform them to strings
    scores = [ f"{s:.1f}" for s in scores]

    # check if these molecules are commercially avaiable
    legend = [ s+" Y" if s in commercialDB.values else s+" N" for s in smiles ]

    # add the scores
    legend = [ t[0]+" "+t[1]  for t in zip(legend, scores)]

    # molecules to image
    img = Draw.MolsToGridImage(mols, molsPerRow=8, subImgSize=(200, 200),
                               legends=legend)
    img.save('smiles'+'.png')
