##########################################################################################
#
# Given a list of smiles (in a pickle file), it will generate a PNG with the molecules 
# visualised. Smiles to molecules is done using orbkit.
#
##########################################################################################

from rdkit import Chem
from rdkit.Chem import Draw
import pickle
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="file to open", type=str)
    args = parser.parse_args()
    print('Opening {}'.format(args.file))

    with open(args.file, 'rb') as pfile:
        smiles = pickle.load(pfile)

    # [0] contains originals, [1] contains predictions
    zipped = [val for pair in zip(smiles[0], smiles[1]) for val in pair]

    mols = [Chem.MolFromSmiles(m) for m in zipped]
    # mols = [mol for mol in mols if mol is not None]
    img = Draw.MolsToGridImage(mols, molsPerRow=6, subImgSize=(1000, 1000),
                               legends=zipped)
    img.save('grid'+'.png')
