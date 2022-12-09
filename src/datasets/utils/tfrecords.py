"""
Created on Tue May 21 11:25:04 2019

@author: Jaroslaw Granda
"""

import pickle
from tqdm import tqdm
from functools import partial
from typing import List

import tensorflow as tf

from src.datasets.utils.tokenizer import Tokenizer
    
def wrap_float(value):
    """Wraps a single float into tf FloatList"""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def wrap_float_list(value):
    """Wraps a  float list into tf FloatList"""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def wrap_int_list(value):
    """Wraps a int list into IntList"""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def wrap_string(value):
    """Wraps a string into bytes list"""
    value = bytes(value, 'utf-8')
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def prepare_TFRecord(data, tokenizer, esp):
    """Builds a tfrecod from data"""
    density = wrap_float_list(data['electron_density'].reshape(-1))
    record_dict = {'density': density,}

    if esp:
        esp = wrap_float_list(data['electrostatic_potential'].reshape(-1))
        record_dict['electrostatic_potential'] = esp
    
    for key in data['properties']:
        key_data = data['properties'][key]
        
        try:
            key_data = float(key_data)
            key_data = wrap_float(key_data)
            record_dict[key] = key_data
        except:
            key_data = wrap_string(key_data)
            record_dict[key] = key_data
    
    smiles = data['smiles']
    encoded_smiles = tokenizer.encode_smiles(smiles)
    record_dict['smiles_string'] = wrap_string(smiles)
    record_dict['smiles'] = wrap_int_list(encoded_smiles)
    record_dict['num_atoms'] = wrap_int_list([data['num_atoms']])
    record = tf.train.Features(feature=record_dict)
    tfrecord = tf.train.Example(features=record)
    return tfrecord


def parse_fn(serialized, smiles_max_length: int, properties:List[str]=[]):
    """Parse the serialized object."""
    features =\
                {'density': tf.io.FixedLenFeature([64, 64, 64], tf.float32),}
    for prop in properties:
        if prop == 'num_atoms':
            features[prop] = tf.io.FixedLenFeature([1], tf.int64)
        elif prop == 'smiles':
            features[prop] = tf.io.FixedLenFeature([smiles_max_length], tf.int64)
        elif prop == 'smiles_string':
            features[prop] = tf.io.VarLenFeature(tf.string)
        else:
            features[prop] = tf.io.FixedLenFeature([1], tf.float32)
    
    parsed_example = tf.io.parse_single_example(serialized, features=features)
    density = parsed_example['density']
    density = tf.expand_dims(density, axis=-1)
    properties = [parsed_example[p] for p in properties]
    return (density, *properties)


def train_preprocess(electron_density, *args):
    """Augments the dataset by flipping the electron density
    along the x, y, z axes.

    Args:
        electron_density: An array of shape [cube_x, cube_y, cube_z]
        *args: float a value of the homo-lumo_gap

    Return:
        electron_density: a new electron denisty flipped either along
        x, y, or z axis
       
    """
    input_shape = electron_density.get_shape().as_list()
    # flip each along each axes
    for flip_index in range(3):
        uniform_random =  tf.random.uniform([], 0, 1.0)
        mirror_cond = tf.less(uniform_random, 0.5)

        electron_density = tf.cond(mirror_cond,
                                   lambda: tf.reverse(electron_density,
                                                      [flip_index]),
                                   lambda: electron_density)

    # random rotation
    permutations = tf.range(3, dtype=tf.int32)
    permutations = tf.random.shuffle(permutations)
    last_dim = tf.constant([3], tf.int32)
    permutations = tf.concat([permutations, last_dim], axis=0)
    electron_density = tf.transpose(electron_density, perm=permutations)
    electron_density.set_shape(input_shape)

    return (electron_density, *args)

def serialize_to_tfrecords(paths, out_path, tokenizer_config_path):
    """ Serializes all the data into one file with TFRecords.
        Args:
            path: string the main folder with cube files
            outpath: path where to save the tfrecods
        Returns:
            None
    """

    writer = tf.io.TFRecordWriter(out_path)
    tokenizer = Tokenizer()
    tokenizer.load_from_config(tokenizer_config_path)

    for cube_path in tqdm.tqdm(paths):
        with open(cube_path, 'rb') as f:
            data = pickle.load(f)
        example = prepare_TFRecord(data, tokenizer)
        serialized = example.SerializeToString()
        writer.write(serialized)
    writer.close()

def worker(input_queue, serialized_queue, tokenizer_config, esp=True):
    """
    Worker process for parallel serialization
    Args:
        input_queue: Queue for input data
        serialized_queue: Queue to put processed data
        tokenizer_config: Tokenizer created before
        esp: if data contains also electrostatic potentials
    Return:
        None

    """
    tokenizer = Tokenizer()
    tokenizer.load_from_config(tokenizer_config)
    while True:
            data = input_queue.get()
            example = prepare_TFRecord(data, tokenizer, esp)
            serialized = example.SerializeToString()
            serialized_queue.put(serialized)
            
def parellel_serialize_to_tfrecords(paths,
                                    out_path,
                                    tokenizer_config_path,
                                    esp=True,
                                    num_processes=12):
    """ Serializes all the data into one file with TFRecords.
        Args:
            path: string the main folder with cube files
            outpath: string path where to save the tfrecods
            esp: if data contains also electrostatic potentials
            num_processes: int how many cpus use for serialization
        Returns:
            None
    """
    import multiprocessing as mp
    input_queue = mp.Queue(maxsize=100)
    serialized_queue = mp.Queue(maxsize=100)
    processes = [mp.Process(target=worker, args=(input_queue, serialized_queue, tokenizer_config_path))
                 for i in range(num_processes)]
    [p.start() for p in processes]
    writer = tf.io.TFRecordWriter(out_path)
    
    data_len = len(paths)
    n_parsed = 0
    n_read = 0
    
    with tqdm(total=data_len, unit='F', desc='Parsed file', initial=0) as pbar:
        while n_parsed < data_len:
            if n_read < data_len:
                with open(paths[n_read], 'rb') as f:
                    data = pickle.load(f)
                    input_queue.put(data)
                    n_read += 1
                    
            while not serialized_queue.empty():
                serialized = serialized_queue.get()
                writer.write(serialized)
                n_parsed += 1
                pbar.update(1)
    writer.close()
    [p.terminate() for p in processes]

def tfrecord_reader(filenames, smiles_max_length, properties=[],
                    train=True, num_epochs=1, batch_size=16, buffer_size=1000):
    """ Create tensorflow dataset iterator which has functionality for reading and
        shuffling the data from tfrecods files.
        Args:
            filenames: an array or string with filename/s
            train: bool True if train set
            batch_size: int the size of batch
            buffer_size: int the size of the buffer for shuffling the data.
        Returns:
            batch_density, batch_homo_lumo: batch tensors sampled from tfrecords.
    """
    
    parse = partial(parse_fn, smiles_max_length=smiles_max_length, properties=properties)
    dataset = tf.data.TFRecordDataset(filenames=filenames, num_parallel_reads=10).prefetch(tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(map_func=parse, num_parallel_calls=12)
    if train:
        dataset = dataset.shuffle(buffer_size=buffer_size)
        dataset = dataset.map(map_func=train_preprocess, num_parallel_calls=12)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)    
    return iter(dataset)
