# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 11:16:16 2021

@author: jmg
"""
import os
import logging
import pickle
import shutil
from typing import List

import numpy as np
from tqdm import tqdm

from src import CPU_COUNT
from src.utils import canonical_smiles
from src.datasets import Dataset
from src.datasets.utils import download_and_unpack
from src.datasets.utils.esp import ESP
from src.datasets.utils.xtb import prepare_xtb_input, run_xtb
from src.datasets.utils.orbkit import electron_density_from_molden
from src.datasets.utils.tokenizer import Tokenizer, SelfiesTokenizer
from src.datasets.utils.tfrecords import parellel_serialize_to_tfrecords, tfrecord_reader, serialize_to_tfrecords

logger = logging.getLogger(__name__)

def read_qm9_file(path):
    
    """ Extract data from single xyz file in qm9 dataset
    
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
    def __init__(self, n_points: int, step_size: float):
        """
            Class for generating, loading and handling electron densities
             QM9 dataset.
            Args:
                n_points: int electron_density will have n_points^3 voxels
                step_size: float resolution of electron density in Bohr atomic units
    	"""
        super(QM9Dataset, self).__init__()
        self.n_points = n_points
        self.step_size = step_size
        self.url = "https://ndownloader.figshare.com/files/3195389"

        self.esp = ESP(n_points, step_size)  # class to calculate the electrostatic potentials
        
    @property
    def name(self):
        return 'QM9Dataset'
    
    @property
    def sourcedir(self):
        return os.path.join(self.dir, 'source_data')
    
    @property
    def tokenizer_config_path(self):
        return os.path.join(self.dir, 'tokenizer_config.json')
    
    @property
    def sf_token_config_path(self):
        return os.path.join(self.dir, 'sf_token_config.json')
    
    @property
    def output_dir(self):
        return os.path.join(self.dir, 'output')
    
    
    def create_dataset_dirs(self):
        if not os.path.exists(self.sourcedir):
            os.mkdir(self.sourcedir)
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
            
    
    def _parse_single_qm9_file(self, qm9_file: str, rewrite: bool = True, esp: bool=True):
        
        """Computes electron desnity, extracts properties and SMILES string 
        from a qm9 xyz file. Results are stores in the the pickle file.
       
            Args:
                qm9_file (str): a path to qm9 xyz file to process
                rewrite (bool): If true (default) it will redo the folder. If false, if
                            the folder exists, it won't do anything and leave it as it is.
                esp (bool): If using xtb to calculate electrostatic potentials
            Returns:
                    None
        """

        # First prepare the path to the qm9 file, and prepare the output folder         
        qm9_file_path = os.path.join(self.sourcedir, qm9_file)
        file_id = qm9_file.strip('.xyz').split('_')[1]
        output_dir = os.path.join(self.output_dir, file_id)
        if os.path.exists(output_dir) and not rewrite:
            logger.info(f'Folder {output_dir} exists. Not re-doing it.')
            return
        os.makedirs(output_dir, exist_ok=True)

        # read qm9 file, prepare and execute xtb
        num_atoms, properties, coords, _, smiles = read_qm9_file(qm9_file_path)
        xtb_input_file_path = os.path.join(output_dir, 'input.xtb')
        prepare_xtb_input(coords, xtb_input_file_path)
        output_dir = os.path.abspath(output_dir)
        xtb_exec_path = shutil.which('xtb')
        run_xtb(xtb_exec_path, xtb_input_file_path, output_dir, molden=True, esp=esp)

        # calculate electron density using orbkit, from xtb results
        molden_input = os.path.join(output_dir, 'molden.input')
        rho = electron_density_from_molden(molden_input, n_points=self.n_points,
                                           step_size=self.step_size)
        espxtb_input = os.path.join(output_dir, 'xtb_esp.dat')
        # calculate esp cube from xtb
        if esp:
            molecule_esp = self.esp.calculate_espcube_from_xtb(espxtb_input)

        output_dict = {}
        output_dict['electron_density'] = rho
        if esp:
            output_dict['electrostatic_potential'] = molecule_esp
        output_dict['properties'] = properties
        output_dict['smiles'] = smiles
        output_dict['num_atoms'] = int(num_atoms)
        output_file = os.path.join(output_dir, 'output.pkl')
        with open(output_file, 'wb+') as ofile:
            pickle.dump(output_dict, ofile)

    def _compute_electron_density(self, rewrite: bool = True, esp: bool=True):
        """
        Computes electron density for each molecules in the dataset
        using Semiempirical Extended Tight-Binding method and orbkit.

            Args:
                rewrite (bool): If true (default) it will redo the folder. If false, if
                    the folder exists, it won't do anything and leave it as it is.
                esp (bool): If using xtb to calculate electrostatic potentials
        
        """
    
        qm9_files = os.listdir(self.sourcedir)#[:500]
        for qm9_file in tqdm(qm9_files, desc='Generating electron density'):
            self._parse_single_qm9_file(qm9_file, rewrite=rewrite, esp=esp)
            
    def _initialize_tokenizer(self):
        """
        This function initializes tokenizer for QM9 dataset SMILES and SELFIES.
        """

        # if from scratch, uncomment next line and comment the other two
        # if the smiles dataset pickle was already generated, comment next one and uncomment next 2
        dataset_smiles = self.get_dataset_smiles()
        # with open("/home/nvme/juanma/Data/ED/dataset_smiles_qm9.pkl", 'rb') as file:
        #    dataset_smiles = pickle.load(file)

        self.tokenizer = Tokenizer()
        self.tokenizer.initilize_from_dataset(dataset_smiles)
        self.tokenizer.save_config(self.tokenizer_config_path)

        self.sf_tokenizer = SelfiesTokenizer()
        self.sf_tokenizer.initialise_from_dataset(dataset_smiles)
        self.sf_tokenizer.save_config(self.sf_token_config_path)
            
    def _split_dataset(self, train_ratio=0.9, valid_ratio=0.1):
        
        ed_paths = self.get_eds_paths()
        # random shuffle
        ed_paths = np.array(ed_paths)
        len_data = len(ed_paths)
        idxs = np.arange(len_data)
        np.random.shuffle(idxs)
        ed_paths = ed_paths[idxs]

        train_idx = int(len_data * train_ratio)
        valid_idx = int(len_data * (train_ratio+valid_ratio))

        train_set = ed_paths[:train_idx]
        valid_set = ed_paths[train_idx:valid_idx]
        test_set = ed_paths[valid_idx:]
        
        return train_set, valid_set # , test_set
        
    
    def get_eds_paths(self) -> List[str]:
        dirs = os.listdir(self.output_dir)
        eds_paths = []
        for compound in tqdm(dirs):
            cube_file = 'output.pkl'
            cube_path = os.path.join(self.output_dir, compound, cube_file)
            eds_paths.append(cube_path)
        return eds_paths
    
    
    def get_dataset_smiles(self) -> List[str]:
        """
        Create list of smiles from the dataset
        
        Returns:
            smiles_list: a list with smiles for dataset
        """
            
        ed_paths = self.get_eds_paths()
        smiles_list = []
        for compound_path in tqdm(ed_paths, 
                                  desc='Reading files'):
            with open(compound_path, 'rb') as file:
                data = pickle.load(file)
                smiles = data['smiles']
                smiles = canonical_smiles(smiles)
                smiles_list.append(smiles)       
        return smiles_list
    
    def generate(self):
        """
        Generate QM9 electron density dataset:
            
        1. Download and unpack QM9 dataset
        2. Generate electron density for each molecule in the dataset
        3. Create dataset tensorflow's tfrecords format

        """
        self.create_dataset_dirs()
        logger.info('Downloading and extracting QM9 dataset')
        download_and_unpack(self.url, self.sourcedir)
        logger.info('Computing electron densities')
        self._compute_electron_density(rewrite=False, esp=True)
        logger.info('Creating dataset SMILES tokenizer')
        self._initialize_tokenizer()
        logger.info('Splitting and serializing dataset into tfrecords')
        splits = self._split_dataset()
        for key, split in zip(['train', 'valid', 'test'], splits):
            logger.info('Creating {} set'.format(key))
            split_output_path = os.path.join(self.dir, '{}.tfrecords'.format(key))
            parellel_serialize_to_tfrecords(split, split_output_path,
                                            self.tokenizer_config_path, self.sf_token_config_path,
                                            num_processes=CPU_COUNT)
            # serialize_to_tfrecords(split, split_output_path,
            #                        self.tokenizer_config_path, self.sf_token_config_path)
            
            
    def load(self,
             splits: List[str],
             properties: List[str] = [],
             train: bool = True, 
             num_epochs: int = 1,
             batch_size: int = 16,
             buffer_size: int = 1000):
        
        
        record_names = [name+'.tfrecords' for name in splits]
        self.tokenizer = Tokenizer()
        self.tokenizer.load_from_config(self.tokenizer_config_path)
        
        smiles_max_length = self.tokenizer.max_length
        
        records_paths = [os.path.join(self.dir, name) for name in record_names]
        data_iterator = tfrecord_reader(records_paths, smiles_max_length=smiles_max_length,
                               properties=properties,  train=train, 
                               num_epochs=num_epochs, batch_size=batch_size,
                               buffer_size=buffer_size)
        
        return data_iterator
        
    def is_generated(self):
        return os.path.exists()

if __name__ == '__main__':
    d = QM9Dataset(n_points=64, step_size=0.5)
    d.generate_dataset()

        
        
        
        
        
        
        
        
    
        
    
