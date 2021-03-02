# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 11:16:16 2021

@author: jmg
"""
import os
import logging

from tqdm import tqdm

from src.datasets import Dataset
from src.datasets.utils import download_and_unpack
from src.datasets.utils.xtb import prepare_xtb_input, 



def read_qm9_file(path):
    
    """ Extract data from single file in qm9 dataset
    
        I.  Property  Unit         Description
        --  --------  -----------  --------------
        1  tag       -            "gdb9"; string constant to ease extraction via grep
        2  index     -            Consecutive, 1-based integer identifier of molecule
        3  A         GHz          Rotational constant A
        4  B         GHz          Rotational constant B
        5  C         GHz          Rotational constant C
        6  mu        Debye        Dipole moment
        7  alpha     Bohr^3       Isotropic polarizability
        8  homo      Hartree      Energy of Highest occupied molecular orbital (HOMO)
        9  lumo      Hartree      Energy of Lowest occupied molecular orbital (LUMO)
        10  gap       Hartree      Gap, difference between LUMO and HOMO
        11  r2        Bohr^2       Electronic spatial extent
        12  zpve      Hartree      Zero point vibrational energy
        13  U0        Hartree      Internal energy at 0 K
        14  U         Hartree      Internal energy at 298.15 K
        15  H         Hartree      Enthalpy at 298.15 K
        16  G         Hartree      Free energy at 298.15 K
        17  Cv        cal/(mol K)  Heat capacity at 298.15 K
        
        Returns:
            num_atoms: int number of atoms in molecule
            prop_dict: dict with properties extracted from files
            coordinates: array with atoms and their coordiantes strings
            frequencies: array with molecular frequencies
            smiles: string with the smiles of molecule
        
    """
    with open(path) as qm9_file:
        data = qm9_file.readlines()
    data = [line.strip('\n') for line in data]
    num_atoms = int(data[0])
    properties = data[1].split('\t')[:-1]
    dict_keys = ['tag_index', 'A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo',
                 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv']
    prop_dict = {}
    
    for idx, prop_name in enumerate(dict_keys):
        prop_dict[prop_name] = properties[idx]
    
    coordinates = data[2:2+num_atoms]
    coordinates = [c.replace('*^', 'e') for c in coordinates]
    coordinates = [c.split('\t')[:-1] for c in coordinates]
    frequencies = data[2+num_atoms].split('\t')
    smiles = ''.join(data[3+num_atoms].split('\t')[0])
    inchi = data[4+num_atoms]
     
    return num_atoms, prop_dict, coordinates, frequencies, smiles

class QM9Dataset(Dataset):
    def __init__(self):
        super(QM9Dataset, self).__init__()
        
    @property
    def name(self):
        return 'QM9Dataset'
    
    @property
    def sourcedir(self):
        return os.path.join(self.dir, 'source_data')
    
    @property
    def output_dir(self):
        return os.path.join(self.dir, 'output')
    
    
    def create_dataset_dirs(self):
        if not os.path.exists(self.sourcedir):
            os.mkdir(self.sourcedir)
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
            
    
    def _parse_single_qm9_file(self, qm9_file: str):
        
        """Computes electron desnity, extracts properties and SMILES string 
        from a qm9 xyz file. Results are stores in the the pickle file.
       
            Args:
                qm9_file (str): a path to qm9 xyz file to process
            Returns:
                    None
        """
                
        qm9_file_path = os.path.join(self.sourcedir, qm9_file)
        file_id = qm9_file.strip('.xyz').split('_')[1]
        output_dir = os.path.join(self.output_dir, file_id)
        os.makedirs(output_dir, exist_ok=True)
        
        num_atoms, properties, coords, _, smiles = read_qm9_file(qm9_file_path)
        xtb_input_file_path = os.path.join(output_dir, 'input.xtb')
        prepare_xtb_input(coords, xtb_input_file_path)
        output_dir = os.path.abspath(output_dir)
        
        return output_dir
        
    
    
    
    #xyz_file = os.path.join(output_dir,'input.xtb' )
    #prepare_xtb_input(xyz_file, coords)
    #output_dir = os.path.abspath(output_dir)
    #xtb_path = '/home/jarek/miniconda3/bin/xtb'
    #run_xtb(xtb_path, xyz_file, output_dir, molden=True)
    #molden_input = os.path.join(output_dir, 'molden.input')
    #rho = parse_molden_file(molden_input, step_size=0.5 )
    #output_dict = {}
    #output_dict['electron_density'] = rho
    #output_dict['properties'] = properties
    #output_dict['smiles'] = smiles
    #output_file = os.path.join(output_dir, 'output.pkl')
    #with open(output_file, 'wb+') as ofile:
    #    pickle.dump(output_dict, ofile)

    
    def _compute_electron_density(self):
        """
        Computes electron density for each molecules in the dataset
        using Semiempirical Extended Tight-Binding method and orbkit.
        
        """
    
        qm9_files = os.listdir(self.sourcedir)
        for qm9_file in tqdm(qm9_files, desc='Parsing qm9 files'):
            return self._parse_single_qm9_file(qm9_file)
            
    def _compile_tfrecord(self):
        pass
    
    
        
    def generate_dataset(self):
        """
        Generate QM9 electron density dataset:
            
        1. Download and unpack QM9 dataset
        2. Generate electron density for each molecule in the dataset
        3. Create tensorflow's tfrecords 

        """
        self.create_dataset_dirs()
        url = "https://ndownloader.figshare.com/files/3195389"
        download_and_unpack(url, self.sourcedir)
        
        
    def is_generated(self):
        return os.path.exists()
        
        
        
        
        
        
        
        
        
    
        
    