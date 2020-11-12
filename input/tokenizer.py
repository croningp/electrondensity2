# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 17:06:57 2019

@author: Jaroslaw Granda
"""
import os
import re
import pickle
import json
import rdkit
import tqdm


def mol_to_smiles(path, mol_format='mol'):
    """Converts the mol file to canonical smiles
    Args:
        path: string to location of mol file
    Returns:
        smiles: strings with smiles of molecule
    """
    
    obmol = list(pybel.readfile(mol_format, path))[0]
    smiles = obmol.write(format='smi')
    smiles = smiles.split('\t')[0]
    return smiles

def canonical_smiles(smiles):
    """Makes smiles string canonical using rdkit algoritm.
    
    Args:
        smiles: string with molecular smiles
    Returns:
        smiles: string with canonical smiles
    """
    mol = rdkit.Chem.MolFromSmiles(smiles)
    smiles = rdkit.Chem.MolToSmiles(mol)
    return smiles

def tokenize_smiles(smiles):
    """Tokenize smiles using regular expressions
    Args:
        smiles: a string with smiles
    Returns:
        tokens: a list with tokens
    """
    
    token_regex = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|"
    token_regex += "-|\+|\\\\\/|:|∼|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    tokens = re.findall(token_regex, smiles)
    return tokens

def get_dataset_smiles(path):
    """
    Pickles the list of smiles from the whole dataset
    Args:
        path: string to the database path
    Returns:
        smiles_list: a list with smiles for each compound
    """
    
    
    if os.path.exists('dataset_smiles.pkl'):
        with open('dataset_smiles.pkl', 'rb') as file:
            dataset_smiles = pickle.load(file)
            return dataset_smiles
    
    dirs = os.listdir(path)
    smiles_list = []
    for compound in tqdm.tqdm(dirs):
        compound_path = os.path.join(path, compound, 'output.pkl')
        
        with open(compound_path, 'rb') as file:
            data = pickle.load(file)
            smiles = data['smiles']
            smiles = canonical_smiles(smiles)
            smiles_list.append(smiles)
            
    with open('dataset_smiles.pkl', 'wb') as file:
        pickle.dump(smiles_list, file)
        
    return smiles_list
        
        
def get_unique_tokens(smiles_list):
    """Compiles a list of unique tokens for the whole dataset 
        of smiles.
        Args:
            smiles_list: list with smiles string
        Returns:
            unique_tokens: list with unique tokens for the dataset
    """
    
    unique_tokens = []
    for smiles in smiles_list:
        tokens = tokenize_smiles(smiles)
        for t in tokens:
            if t not in unique_tokens:
                unique_tokens.append(t)
    print('Unique tokens: \n {}\n'.format(unique_tokens))
    return unique_tokens


def get_dictionaries(dataset_smiles):
    """
    Create dictionaries allowing for conversion of smiles string into one-hot 
    encoded representation.
    
    Args:
        dataset_smiles: a list with smiles string of the dataset
    Returns:
        num2token: a dict for converting a int index into its string
        representation
        token2num: a dict for converting a token into int value 
    """
    stats = get_dictionary_stats(dataset_smiles)
    print(stats)
    unique_tokens = get_unique_tokens(dataset_smiles)
    unique_tokens.append('START')
    unique_tokens.append('STOP')
    unique_tokens.append('NULL')
    num2token = {}
    token2num = {}
    for idx, token in enumerate(unique_tokens):
        num2token[idx]  = token
        token2num[token] = idx
    print(num2token)
    print(token2num)
    return num2token, token2num

def encode_dataset(smiles_list, token2num):
    """
    Encodes the dataset using token2num dict.
    
    
    """


    encoded = []
    for sm in smiles:
        encoded_molecule = encode_smiles(sm, token2num)
        encoded.append(encoded_molecule)
    return encoded


def tokenize_dataset(smiles_list):
    """
    Tokenizes the list of smiles strings
    Args:
        smiles_list: a list with smiles strings
    Returns:
        tokenized_smiles: a list  of list with tokens for each smiles
    
    """
    tokenized_smiles = []
    for smiles in smiles_list:
        tokenized = tokenize_smiles(smiles)
        tokenized_smiles.append(tokenized)
    
    return tokenized_smiles


def get_max_length(smiles_list):
    """
    Computes the max lenght of smiles in the dataset to which shorter strings
    should be padded. 
    Args:
        smiles_list: a list with smiles strings 
    Returns:
        max_len: int for to which pad shorter smiles
    """
    lengths = list(map(len, smiles_list))    
    max_len = max(lengths) + 2
    return max_len

def get_dictionary_stats(smiles_list):
    """
    Computes the tokens stats in the dataset
    Args:
        smiles_list: a list with smiles strings
    Returns:
        sorted_dict: a list with tokens sorted according to popularity
    """
    stats = {}
    for sm in smiles_list:
        tokens = tokenize_smiles(sm)
        for t in tokens:
            if t not in stats:
                stats[t] = 1
            else:
                stats[t] += 1
    
    sorted_dict = sorted(stats.items(), key=lambda stats: stats[1],
                         reverse=True)
    return  sorted_dict
    

class Tokenizer():
    """
    Class handling encoding and decoding of smiles strings
    """
    
    def __init__(self, config_path, initialize_from_dataset=False, dataset_path=None):
        """Initializer for the class, by default it will look for json with 
        its configuration in the path folder. If the config doesn't
        exist it needs to be initialized from the processed dataset.
        Args:
            path: a path to the config file or path to save config file if
                initialize_from_dataset=True
            initialize_from_dataset: a bool if to initialize config from the
            datast 
            dataset_path: a string with path to processed dataset
        """
        
        if not initialize_from_dataset:
            
            if not os.path.exists(config_path):
                raise ValueError('Please initialize tokenizer by running\
                                 initialize_from_dataset=True')
            
            with open(config_path, 'r') as file:
                config = json.load(file)
            self.num2token = config['num2token']
            self.token2num = config['token2num']
            self.max_length = config['max_length']
            
        else:
            # initialization procedure
            dataset_smiles = get_dataset_smiles(dataset_path)
            self.num2token, self.token2num = get_dictionaries(dataset_smiles)
            
            self.tokenized_dataset = tokenize_dataset(dataset_smiles)
            self.max_length = get_max_length(self.tokenized_dataset)
            
            print('Max length of smiles {}'.format(self.max_length))
            
            config  = {}
            
            config['num2token'] = self.num2token
            config['token2num'] = self.token2num
            config['max_length'] = self.max_length
            
            with open(config_path, 'w') as file:
                json.dump(config, file)
            
            
            
    def encode_smiles(self, smiles):
        """
        Encodes the smiles string into vector representation using token2num
        dict.
        
        Args:
            smiles: string with smiles
        Returns:
            encoded_smiles: an array of ints with encoded smiles
        
        """

        smiles = canonical_smiles(smiles)
        tokens = ['START'] + tokenize_smiles(smiles) + ['STOP']
        num_tokens = len(tokens)
        if self.max_length - num_tokens < 0:
            raise ValueError('The lenght of smiles {} is larger then max {}'.format(smiles, num_tokens))
        append_tokens = ['NULL'] * (self.max_length - num_tokens)
        tokens += append_tokens
        encoded_smiles = list(map(self.token2num.get, tokens))
    
        if None in encoded_smiles:
            raise ValueError('One or more tokens are not present in the dictionary')
        return encoded_smiles
    
    def decode_smiles(self, encoded_smiles):
        """
        Decodes a single smiles string
        Args:
            encoded_smiles: an array of size [seq_len]
        Returns:
               smiles: string with smiles 
        """
        avoid_tokens = ['START', 'STOP', 'NULL']
        encoded_smiles = [str(t) for t in encoded_smiles]
        tokens = list(map(self.num2token.get, encoded_smiles))
        tokens = [t for t in tokens if t not in avoid_tokens]
        smiles = ''.join(tokens)
        return smiles
    
    def batch_decode_smiles(self, encoded_smiles):
        """
        Decode smiles in the batch
        Args:
            encoded_smiles: an array of ints with dim [batch_size, seq_len]
        Return:
            decoded_smiles: an array with of len batch_size with decoded smiles
            strings
            
        """
        decoded_smiles = [self.decode_smiles(s) for s in encoded_smiles]
        return decoded_smiles
    
    
            
  

if __name__ == '__main__':
    t = Tokenizer('data\\')
    #create_smiles_pickle('C:\\Users\\group\\Desktop\\test')
    #create_smiles_pickle('/mnt/orkney1/electron_density/pm6')
    #create_dictionaries('smiles.pkl')
    ##with open('token2num.pkl', 'rb') as file:
      ##  token2num = pickle.load(file)
        
    #with open('num2token.pkl', 'rb') as file:
     #   num2token = pickle.load(file)
        
