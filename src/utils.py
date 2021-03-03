from rdkit import Chem

def canonical_smiles(smiles):
    """Makes smiles string canonical using rdkit algoritm.
    
    Args:
        smiles: string with molecular smiles
    Returns:
        smiles: string with canonical smiles
    """
    mol = Chem.MolFromSmiles(smiles)
    smiles = Chem.MolToSmiles(mol)
    return smiles
