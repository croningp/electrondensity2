# -*- coding: utf-8 -*-
"""
Created on Tue May 21 11:25:04 2019

@author: group
"""
import os
import sys
import pickle
import tqdm
from collections import namedtuple
import numpy as np
import tensorflow as tf




with open('token2num.pkl', 'rb') as file:
    token2num = pickle.load(file)
    
with open('num2token.pkl', 'rb') as file:
    num2token = pickle.load(file)
    
def wrap_float(value):
    """Wraps a single float into tf FloatList"""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def wrap_float_list(value):
    """Wraps a  float list into tf FloatList"""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def wrap_int_list(value):
    """Wraps a int list into IntList"""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def prepare_TFRecord(data):
    """Builds a tfrecod from data"""
    density = wrap_float_list(data.density.reshape(-1))
    #homo_lumo_gap = wrap_float(data.HOMO_LUMO_gap)
    #smiles_string = data.smiles
    #smiles = encode_smiles(smiles_string, token2num)
    #smiles = wrap_int_list(smiles)
    
    record_dict = {'density': density,}
                   #'homo_lumo_gap':homo_lumo_gap,
                   #'smiles':smiles}
    
    record = tf.train.Features(feature=record_dict)
    tfrecord = tf.train.Example(features=record)
    return tfrecord


def parse(serialized):
    """Parse the serialized object."""
    features =\
                {
                'density': tf.FixedLenFeature([64, 64, 64], tf.float32),
                #'homo_lumo_gap':tf.FixedLenFeature([1], tf.float32),
                #'smiles':tf.FixedLenFeature([35], tf.int64)
                    }
    parsed_example = tf.parse_single_example(serialized, features=features)
    density = parsed_example['density']
    #smiles = parsed_example['smiles']
    density = tf.expand_dims(density, axis=-1)
    #homo_lumo_gap = parsed_example['homo_lumo_gap'][0]
    return density,# homo_lumo_gap, smiles


def train_preprocess(electron_density, homo_lumo_gap, smiles):
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
        uniform_random =  tf.random_uniform([], 0, 1.0)
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

    return electron_density, homo_lumo_gap, smiles

def convert_to_tfrecords(paths, out_path):
    """ Serializes all the data into one file with TFRecords.

        Args:
            path: string the main folder with cube files
            outpath: path where to save the tfrecods
        Returns:
            None
    """

    writer = tf.python_io.TFRecordWriter(out_path)

    for cube_path in tqdm.tqdm(paths):
        with open(cube_path, 'rb') as f:
            data = pickle.load(f)
        try:
            example = prepare_TFRecord(data)
            serialized = example.SerializeToString()
            writer.write(serialized)
        except ValueError:
            pass
            
    writer.close()



def train_validation_test_split(path, output_dir, basisset,
                                train_size=0.9, valid_size=0.1):
    """
    Creates tfrecords for train, validation and test set.

    """
    basissets = ['b3lyp_6-31g(d)', 'td-b3lyp_6-31g+(d)']
    if basisset not in basissets:
        raise ValueError('The available basis sets are {}'.format(basissets))

    dirs = os.listdir(path)
    compounds_path = []
    print('Reading files')
    for compound in tqdm.tqdm(dirs):
        cube_file = '{:09d}.{}.cube'.format(int(compound), basisset)
        cube_path = os.path.join(path, compound, cube_file)
        if not (os.path.exists(cube_path)) or os.path.getsize(cube_path) == 0:
            continue
        compounds_path.append(cube_path)

    # random shuffle
    compounds_path = np.array(compounds_path)
    len_data = len(compounds_path)
    idxs = np.arange(len_data)
    np.random.shuffle(idxs)
    compounds_path = compounds_path[idxs]

    train_idx = int(len_data * train_size)
    valid_idx = int(len_data * (train_size+valid_size))

    train_set = compounds_path[:train_idx]
    valid_set = compounds_path[train_idx:valid_idx]
    test_set = compounds_path[valid_idx:]

    # generate tfrecords
    print('Preparing train set {} cubes'.format(len(train_set)))
    train_path = os.path.join(output_dir, 'train.tfrecords')
    convert_to_tfrecords(train_set, train_path)
    print('Preparing valid set {}'.format(len(valid_set)))
    valid_path = os.path.join(output_dir, 'valid.tfrecords')
    convert_to_tfrecords(valid_set, valid_path)
    print('Preparing test set {}'.format(len(test_set)))
    test_path = os.path.join(output_dir, 'test.tfrecords')
    convert_to_tfrecords(test_set, test_path)

def input_fn(filenames, train=True, num_epochs=1, batch_size=16, buffer_size=100):
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

    dataset = tf.data.TFRecordDataset(filenames=filenames)
    dataset = dataset.map(map_func=parse)

    if train:
        dataset = dataset.shuffle(buffer_size=buffer_size)
        dataset = dataset.map(map_func=train_preprocess)

    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    batch_density, batch_homo_lumo, batch_smiles = iterator.get_next()

    return batch_density, batch_homo_lumo, batch_smiles


def input_pipeline(train_files,
                   valid_files,
                   num_epochs=1,
                   batch_size=16,
                   buffer_size=100):
    """ Makes a common pipeline for using training and validation set
        Args:
            train_files: a string or array of strings with paths to file/s
            containing training sets.
            valid_files: a string or array of strings with paths to file/s
            containing validation set/s.
            num_epochs: int for many epochs iterate data
            batch_size: int size of the batch
            buffer_size: int buffer size for shuffling the data for training
        Returns:
            (batch_density, batch_homo_lumo): a tuple with electron densities and
            homo-lumo gaps.
            training_init_op: a tensorflow operation to switch to training set
            validation_init_op: a tensorflow operation to switch to validation
            set

    """

    train_dataset = tf.data.TFRecordDataset(filenames=train_files)
    train_dataset = train_dataset.map(parse)
    train_dataset = train_dataset.shuffle(buffer_size=buffer_size)
    train_dataset = train_dataset.repeat(num_epochs)
    train_dataset = train_dataset.batch(batch_size)

    valid_dataset = tf.data.TFRecordDataset(filenames=valid_files)
    valid_dataset = valid_dataset.map(parse)
    valid_dataset = valid_dataset.repeat(num_epochs)
    valid_dataset = valid_dataset.batch(batch_size)

    iterator = iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                          train_dataset.output_shapes)

    batch_density, batch_homo_lumo = iterator.get_next()

    training_init_op = iterator.make_initializer(train_dataset)
    validation_init_op = iterator.make_initializer(valid_dataset)

    return (batch_density, batch_homo_lumo), training_init_op, validation_init_op



if __name__ == '__main__':
    sys.path.append('C:\\Users\\group\\Desktop\\ElectronDensityML\\edml\\datagen\\')
    #train_validation_test_split(path='C:\\Users\\group\\Desktop\\test',
    #                            output_dir='C:\\Users\\group\\Desktop',
     #                           basisset='b3lyp_6-31g(d)')
    
    b_denisties, b_homo_lumo_gaps, b_smiles = input_fn('C:\\Users\\group\\Desktop\\train.tfrecords')
    
#    #train_validation_test_split('/mnt/orkney1/pm6', '/home/jarek/pm6nn', 'b3lyp_6-31g(d)')
    from orbkit import grid, output
    from cube import set_grid
    set_grid(64, 0.625)
    grid.init_grid()
    sess = tf.Session()
    res, bs = sess.run([b_denisties, b_smiles])
    index = 1
    print(decode_smiles(bs[index], num2token))
    output.view_with_mayavi(grid.x, grid.y, grid.z, res[index, :, :, :, 0])
    
    
    

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
