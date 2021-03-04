import sys
import logging

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
	
def configure_logger(
                    logs_dir,
                    name=__name__,
                    console_logging_level=logging.INFO,
                    ):
    logger = logging.getLogger(name)
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    
    log_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(log_format)
    ch.setLevel(console_logging_level)
    logger.addHandler(ch)
