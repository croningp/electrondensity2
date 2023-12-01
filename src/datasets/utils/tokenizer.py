import re
import json
import pickle
import selfies as sf
from typing import List, Tuple

def tokenize_smiles(smiles: str) -> List[str]:
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

def get_unique_tokens(smiles_list: List[str]) -> List[str]:
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
    #print('Unique tokens: \n {}\n'.format(unique_tokens))
    return unique_tokens


def get_dictionaries(dataset_smiles: List[str]) -> Tuple[dict , dict]:
    """
    Create dictionaries allowing for conversion of smiles string into one-hot 
    encoded representation and in reverse.
    
    Args:
        dataset_smiles: a list with smiles string of the dataset
    Returns:
        num2token: a dict for converting a int index into its string
        representation
        token2num: a dict for converting a token into int value 
    """
    #stats = get_dictionary_stats(dataset_smiles)
    unique_tokens = get_unique_tokens(dataset_smiles)
    unique_tokens.append('START')
    unique_tokens.append('STOP')
    unique_tokens.append('NULL')
    num2token = {}
    token2num = {}
    for idx, token in enumerate(unique_tokens):
        num2token[idx]  = token
        token2num[token] = idx
    return num2token, token2num

def tokenize_dataset(smiles_list: List[str]) -> List[List[str]]:
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


def get_max_length(smiles_list: List[str]) -> int:
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
    

class Tokenizer(object):
    """
    Class handling encoding and decoding of smiles strings
    """
    
    def __init__(self):
        pass
            
    def initilize_from_dataset(self, dataset_smiles: List[str]):
        self.num2token, self.token2num = get_dictionaries(dataset_smiles)
        self.tokenized_dataset = tokenize_dataset(dataset_smiles)
        self.max_length = get_max_length(self.tokenized_dataset)
    
    def save_config(self, config_path: str):
        config  = {}
        config['num2token'] = self.num2token
        config['token2num'] = self.token2num
        config['max_length'] = self.max_length
        with open(config_path, 'w') as file:
            json.dump(config, file)
        
    def load_from_config(self, config_path: str):
        with open(config_path, 'r') as file:
            config = json.load(file)
            self.num2token = config['num2token']
            self.token2num = config['token2num']
            self.max_length = config['max_length']
        
    def encode_smiles(self, smiles: str) -> List[int]:
        """
        Encodes smiles string into vector representation using token2num
        dict.
        
        Args:
            smiles: string with smiles
        Returns:
            encoded_smiles: an array of ints with encoded smiles
        
        """

        tokens = ['START'] + tokenize_smiles(smiles) + ['STOP']
        num_tokens = len(tokens)
        # if self.max_length - num_tokens < 0:
        #     raise ValueError('The lenght of smiles {} is larger then max {}'.format(smiles, num_tokens))
        #     return
        append_tokens = ['NULL'] * (self.max_length - num_tokens)
        tokens += append_tokens
        encoded_smiles = list(map(self.token2num.get, tokens))
    
        if None in encoded_smiles:
            raise ValueError('One or more tokens are not present in the dictionary')
        return encoded_smiles
    
    def decode_smiles(self, encoded_smiles: List[str]) -> str:
        """
        Decodes a single smiles string
        Args:
            encoded_smiles: an array of size [seq_len]
        Returns:
               smiles: string with smiles 
        """
        avoid_tokens = ['START', 'STOP', 'NULL']
        tokens = list(map(self.num2token.get, encoded_smiles))
        tokens = [t for t in tokens if t not in avoid_tokens]
        smiles = ''.join(tokens)
        return smiles
    
    def batch_decode_smiles(self, encoded_smiles: List[List[int]]) -> List[str]:
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

class SelfiesTokenizer(object):
    """
    Class handling encoding and decoding of SELFIES strings
    Heavily uses and relies on https://github.com/aspuru-guzik-group/selfies
    """
    
    def __init__(self):
        pass


    def initialise_from_dataset(self, dataset_smiles):
        """Given a smiles dataset, it will generate a SELFIES tokenizer

        Args:
            dataset_smiles: List containing list of smiles in the dataset. 
            Generated when creating the TFRecords.
        """
        
        # Create list of selfies, with start and stop tokens
        dataset = ["[START]" + sf.encoder(smile) + "[STOP]" for smile in dataset_smiles]
        # obtain alphabet
        alphabet = sf.get_alphabet_from_selfies(dataset)
        # [nop] is the null (padding) for selfies
        alphabet.add("[nop]")
        # obtain max length of selfies
        self.max_length = max(sf.len_selfies(s) for s in dataset)
        # generate dict to label the selfies
        self.token2num = {s: i for i, s in enumerate(alphabet)}
        self.num2token = {i: c for c, i in self.token2num.items()}


    def initialise_from_dataset_pickle(self, dataset_smiles):
        """Given a smiles dataset, it will generate a SELFIES tokenizer

        Args:
            dataset_smiles: File (pickle) containing list of smiles in the dataset. 
            Generated when creating the TFRecords.
        """

        # load list from pickle
        with open(dataset_smiles, 'rb') as file:
            smiles = pickle.load(file)
        
        self.initialise_from_dataset(smiles)


    def save_config(self, config_path: str):
        config  = {}
        config['num2token'] = self.num2token
        config['token2num'] = self.token2num
        config['max_length'] = self.max_length
        with open(config_path, 'w') as file:
            json.dump(config, file)


    def load_from_config(self, config_path: str):
        with open(config_path, 'r') as file:
            config = json.load(file)
            num2token = config['num2token']
            # self.num2token = {int(k):v for k,v in num2token.items()}
            self.num2token = {k:v for k,v in num2token.items()}
            self.token2num = config['token2num']
            self.max_length = config['max_length']


    def encode_selfie(self, selfie: str) -> List[int]:
        """
        Encodes selfie string into vector representation using token2num
        dict.
        
        Args:
            selfie: string with selfie
        Returns:
            encoded_selfie: an array of ints with encoded selfie
        
        """

        tokens = "[START]" + selfie + "[STOP]"

        label = sf.selfies_to_encoding(
            selfies=tokens,
            vocab_stoi=self.token2num,
            pad_to_len=self.max_length,
            enc_type="label"
        )

        return label
    

    def decode_selfie(self, encoded_selfie: List[str]) -> str:
        """
        Decodes a single selfie string
        Args:
            encoded_smiles: an array of size [seq_len]
        Returns:
               selfies: string with selfie
        """

        tokens = sf.encoding_to_selfies(
            encoding=encoded_selfie,
            vocab_itos=self.num2token,
            enc_type="label"
        )
        tokens = sf.split_selfies(tokens)

        avoid_tokens = ['[START]', '[STOP]', '[nop]']

        tokens = [t for t in tokens if t not in avoid_tokens]
        selfies = ''.join(tokens)
        return selfies