# -*- coding: utf-8 -*-
"""
Created on Tue May 21 11:25:04 2019

@author: Jaroslaw Granda
"""
import os
import time
import sys
import pickle
import numpy as np
import tqdm
from functools import partial
from collections import namedtuple
try:
    from rdkit import Chem
except:
    print('Rdkit not found')
import tensorflow as tf
from .tokenizer import Tokenizer

    
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


def prepare_TFRecord(data):
    """Builds a tfrecod from data"""
    tokenizer = Tokenizer('data\\')
    density = wrap_float_list(data['electron_density'].reshape(-1))
    record_dict = {'density': density,}
    
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
    mol = Chem.MolFromSmiles(smiles)
    
    num_atoms = mol.GetNumAtoms()
    record_dict['num_atoms'] = wrap_int_list([num_atoms])
    #print(smiles)
    #print(num_atoms)
    #print()
    record = tf.train.Features(feature=record_dict)
    tfrecord = tf.train.Example(features=record)
    return tfrecord


def parse_fn(serialized, properties=[]):
    """Parse the serialized object."""
    features =\
                {
                'density': tf.io.FixedLenFeature([64, 64, 64], tf.float32),
                #'homo_lumo_gap':tf.FixedLenFeature([1], tf.float32),
                #'smiles':tf.FixedLenFeature([35], tf.int64)
                    }
    for prop in properties:
        if prop == 'num_atoms':
            features[prop] = tf.io.FixedLenFeature([1], tf.int64)
        elif prop == 'smiles':
            features[prop] = tf.io.FixedLenFeature([24], tf.int64)
        elif prop == 'smiles_string':
            features[prop] = tf.io.VarLenFeature(tf.string)
        else:
            features[prop] = tf.io.FixedLenFeature([1], tf.float32)
    
            
    parsed_example = tf.io.parse_single_example(serialized, features=features)
    density = parsed_example['density']
    density = tf.expand_dims(density, axis=-1)
    #homo_lumo_gap = parsed_example['homo_lumo_gap'][0]
    properties = [parsed_example[p] for p in properties]
    return (density, *properties)


def train_preprocess(electron_density, *args):
    """Augments the dataset by flipping the electron density
    along the x, y, z axes.

    Args:
        electron_density: An array of shape [cube_x, cube_y, cube_z]
        homo_lumo_gap: float a value of the homo-lumo_gap

    Return:
        electron_density: a new electron denisty flipped either along
        x, y, or z axis
        homo_lumo_gap: original homo lumo
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

def convert_to_tfrecords(paths, out_path):
    """ Serializes all the data into one file with TFRecords.

        Args:
            path: string the main folder with cube files
            outpath: path where to save the tfrecods
        Returns:
            None
    """

    writer = tf.io.TFRecordWriter(out_path)

    for cube_path in tqdm.tqdm(paths):
        time_1 = time.time()
        with open(cube_path, 'rb') as f:
            data = pickle.load(f)
        try:
            time_2 = time.time()
            example = prepare_TFRecord(data)
            serialized = example.SerializeToString()
            time_3 = time.time()
            writer.write(serialized)
            end_time = time.time()
            
        except ValueError:
            pass
        #print('Reading time from disc', time_2- time_1)
        #print('Serialization time', time_3-time_2)
        #print('Writing time', end_time-time_3)
        #print('Overall time', end_time - time_1)
    writer.close()

def worker(input_queue, serialized_queue):
    while True:
            data = input_queue.get()
            example = prepare_TFRecord(data)
            serialized = example.SerializeToString()
            serialized_queue.put(serialized)
            
def parellel_convert_to_tfrecords(paths, out_path, num_processes=12):
    """ Serializes all the data into one file with TFRecords.

        Args:
            path: string the main folder with cube files
            outpath: path where to save the tfrecods
            num_processes: int how many processes use for serialization
        Returns:
            None
    """
    print('-----------------',len(paths))
    import multiprocessing as mp
    input_queue = mp.Queue(maxsize=1000)
    serialized_queue = mp.Queue(maxsize=1000)
    processes = [mp.Process(target=worker, args=(input_queue, serialized_queue)) for i in range(num_processes)]
    [p.start() for p in processes]
    writer = tf.io.TFRecordWriter(out_path)

    for cube_path in tqdm.tqdm(paths):
        
        with open(cube_path, 'rb') as f:
            #print(cube_path)
            data = pickle.load(f)
            time.sleep(0.005)
        input_queue.put(data)
            
        while not serialized_queue.empty():
            serialized = serialized_queue.get()
            writer.write(serialized)
            time.sleep(0.005)
    writer.close()
    [p.terminate() for p in processes]


def train_validation_test_split(path, output_dir,
                                train_size=0.9,
                                valid_size=0.1,
                                parallel=False):
    """
    Creates tfrecords for train, validation and test set.

    """

    dirs = os.listdir(path)[:5000]
    compounds_path = []
    print('Reading files')
    for compound in tqdm.tqdm(dirs):
        cube_file = 'output.pkl'
        cube_path = os.path.join(path, compound, cube_file)
        compounds_path.append(cube_path)

    # random shuffle
    compounds_path = np.array(compounds_path)
    len_data = len(compounds_path)
    idxs = np.arange(len_data)
    np.random.shuffle(idxs)
    compounds_path = compounds_path[idxs]
    compounds_path = compounds_path

    train_idx = int(len_data * train_size)
    valid_idx = int(len_data * (train_size+valid_size))

    train_set = compounds_path[:train_idx]
    valid_set = compounds_path[train_idx:valid_idx]
    test_set = compounds_path[valid_idx:]

    # generate tfrecords
    print('Preparing train set {} cubes'.format(len(train_set)))
    train_path = os.path.join(output_dir, 'train.tfrecords')
    if parallel:
        parellel_convert_to_tfrecords(train_set, train_path)
    else:
        convert_to_tfrecords(train_set, train_path)
    
    print('Preparing valid set {}'.format(len(valid_set)))
    valid_path = os.path.join(output_dir, 'valid.tfrecords')
    if parallel:
        parellel_convert_to_tfrecords(valid_set, valid_path)
    else:
        convert_to_tfrecords(valid_set, valid_path)
    print('Preparing test set {}'.format(len(test_set)))
    test_path = os.path.join(output_dir, 'test.tfrecords')
    if parallel:
        parellel_convert_to_tfrecords(test_set, test_path)
    else:
        convert_to_tfrecords(test_set, test_path)
    
def input_fn(filenames, properties=[], train=True, num_epochs=1, batch_size=16, buffer_size=1000):
    """ Create tensorflow dataset which has functionality for reading and
        shuffling the data from tfrecods files.
        Args:
            filenames: an array or string with filename/s
            train: bool True if train set
            batch_size: int the size of batch
            buffer_size: int the size of the buffer for shuffling the data.
        Returns:
            batch_density, batch_homo_lumo: batch tensors sampled from tfrecords.
    """
    
    parse = partial(parse_fn, properties=properties)
    dataset = tf.data.TFRecordDataset(filenames=filenames, num_parallel_reads=10).prefetch(tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(map_func=parse, num_parallel_calls=12)

    if train:
        dataset = dataset.shuffle(buffer_size=buffer_size)
        dataset = dataset.map(map_func=train_preprocess, num_parallel_calls=12)

    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    
    #dataset = dataset.cache()
    
   # iterato = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
    #batch_density = iterator.get_next()

    return dataset


if __name__ == '__main__':
    pass
    #with open('D:\\qm9\\080684\\output.pkl', 'rb') as df:
        #data = pickle.load(df)
#<<<<<<< HEAD
#    train_validation_test_split('/media/extssd/jarek/qm9_cubes', '/media/extssd/jarek/', train_size=0.9, valid_size=0.1)
#    #a = input_fn('C:\\Users\\jmg\\Desktop\\programming\data\\train.tfrecords', properties=['num_atoms', 'smiles'])
#    #i = iter(a)
#    #d, n, s = i.__next__() 
#=======
#    train_validation_test_split('D:\qm9', 'C:\\Users\\jmg\\Desktop\\programming\\', train_size=0.9, parallel=False)
#    
#    a = input_fn('C:\\Users\\jmg\\Desktop\\programming\\train.tfrecords',
#                 properties=['num_atoms', 'smiles'])
#    i = iter(a)
#    d, n, s = i.__next__() 
#>>>>>>> d20c99e826c1aec1b65e141d63ffd80fdf17308c
    #for i in iter(batch_density):
     #   print(tf.reduce_mean(i))
    #sys.path.append('C:\\Users\\group\\Desktop\\ElectronDensityML\\edml\\datagen\\')
    #train_validation_test_split(path='C:\\Users\\group\\Desktop\\test',
    #                            output_dir='C:\\Users\\group\\Desktop',
     #                           basisset='b3lyp_6-31g(d)')
    
    #b_denisties, b_homo_lumo_gaps, b_smiles = input_fn('C:\\Users\\group\\Desktop\\train.tfrecords')
    
#    #train_validation_test_split('/mnt/orkney1/pm6', '/home/jarek/pm6nn', 'b3lyp_6-31g(d)')
    #from orbkit import grid, output
    #from cube import set_grid
    #set_grid(16, 0.5)
    #grid.init_grid()
    #sess = tf.Session()
    #res, bs = sess.run([b_denisties, b_smiles])
    #index = 1
    #print(decode_smiles(bs[index], num2token))
    #patches = tf.extract_volume_patches(d, [1,4,4,4,1], [1, 4,4,4,1], padding='SAME')
    #output.view_with_mayavi(grid.x, grid.y, grid.z, patches[0, :, :, :, 0])
    
    
    

#    (bx, by), train_init_op, valid_init_op = input_pipeline('train.tfrecords', 'valid.tfrecords')
#    sess = tf.Session()
#    sess.run(train_init_op)
#    for i in tqdm.tqdm(range(1)):
#        res = sess.run(bx)
#    output.view_with_mayavi(grid.x, grid.y, grid.z, res[1, :, :, :, 0])
#    tf.reset_default_graph()
#    a = tf.get_variable('aaa', [100, 100], tf.float32,
#                        initializer=tf.random_normal_initializer(),
#                        regularizer=tf.contrib.layers.l2_regularizer(1.0))
#    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
#