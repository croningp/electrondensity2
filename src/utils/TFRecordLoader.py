##########################################################################################
#
# Loading the tfrecords files that Jarek generated. If you don't a the tfrecord file
# generate it using the generate_dataset.py script that Jarek created.
#
# This code is a mixture of https://keras.io/examples/keras_recipes/tfrecord/
# and how Jarek did it on the branch "developer" (develop/input/tfrecords.py)
#
# Author: Juan Manuel Parrilla Gutierrez (juanma@chem.gla.ac.uk)
#
##########################################################################################

import tensorflow as tf
from functools import partial


class TFRecordLoader():

    def __init__(self, filename, batch_size=64, ed_shape=[64, 64, 64, 1],
                 train=False, properties=[]):
        """Create class and set basic parameters.

        Args:
            filename: Path to the tfrecord
            batch_size (int, optional): Defaults to 64.
            ed_shape (list, optional): Electron density shape. Defaults to [64,64,64,1].
            train (bool, optional): Performs data augmentation.
            properties (list, optional): Check the properties to fetch as defined in 
                parse_fn
        """
        self.filename = filename
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE
        self.BATCH_SIZE = batch_size
        self.ED_SHAPE = ed_shape
        self.get_dataset(train, properties)  # this will set self.dataset
        self.dataset_iter = iter(self.dataset)

    def parse_fn(self, serialized, properties=[], expand_dims=True):
        """Parse the serialized object. This is used in load_dataset, in the map function.

        Args:
            serialized: A tfrecord as generated from generate_dataset.py
            properties (list, optional): See "prop in properties" below. Defaults to [].
            expand_dims (bool, optional): Expands to 4D. Defaults to True.

        Returns:
            (density, *properties): parsed tfrecord and the properties
        """

        features = {'density': tf.io.FixedLenFeature([64, 64, 64], tf.float32), }

        for prop in properties:
            if prop == 'num_atoms':
                features[prop] = tf.io.FixedLenFeature([1], tf.int64)
            elif prop == 'electrostatic_potential':
                features[prop] = tf.io.FixedLenFeature([64, 64, 64], tf.float32)
            elif prop == 'smiles':
                features[prop] = tf.io.FixedLenFeature([24], tf.int64)
            elif prop == 'fp':
                features[prop] = tf.io.FixedLenFeature([1024], tf.float32)
            elif prop == 'smiles_string':
                features[prop] = tf.io.VarLenFeature(tf.string)
            else:
                features[prop] = tf.io.FixedLenFeature([1], tf.float32)

        parsed_example = tf.io.parse_single_example(
            serialized, features=features)

        density = parsed_example['density']
        if expand_dims:
            density = tf.expand_dims(density, axis=-1)

        if 'electrostatic_potential' in properties:
            esp = parsed_example['electrostatic_potential']
            esp = tf.expand_dims(esp, axis=-1)

        parsed_properties = [parsed_example[p] for p in properties if p != 'electrostatic_potential' ]

        if 'electrostatic_potential' in properties:
            return (density, esp, *parsed_properties)
        else:
            return (density, *parsed_properties)


    def train_preprocess(self, electron_density, *args):
        """Augments the dataset by flipping the electron density
        along the x, y, z axes.

        Args:
            electron_density: An array of shape [cube_x, cube_y, cube_z]
            *args: other properties being loaded into the tfrecord
                if using ESP it must be set in args[0] if using this preprocess

        Return:
            electron_density: a new electron denisty flipped either along
            x, y, or z axis
            *args: other unchanged properties
        """

        # if the first element in args has shape of len 4, then it should be ESP
        if len(args[0].shape)==4 :
            electrostatic = args[0]
        
        input_shape = electron_density.get_shape().as_list()

        # flip each along each axes
        for flip_index in range(3):
            uniform_random = tf.random.uniform([], 0, 1.0)
            mirror_cond = tf.less(uniform_random, 0.5)

            electron_density = tf.cond(
                mirror_cond, lambda: tf.reverse(electron_density, [flip_index]),
                lambda: electron_density)

            # meaning we have esps. we flip same as with eds
            if len(args[0].shape)==4 :
                electrostatic = tf.cond(
                    mirror_cond, lambda: tf.reverse(electrostatic, [flip_index]),
                    lambda: electrostatic)

        # random rotation
        permutations = tf.range(3, dtype=tf.int32)
        permutations = tf.random.shuffle(permutations)
        last_dim = tf.constant([3], tf.int32)
        permutations = tf.concat([permutations, last_dim], axis=0)
        electron_density = tf.transpose(electron_density, perm=permutations)
        electron_density.set_shape(input_shape)

        # if using ESP, also do the rotations and return
        if len(args[0].shape)==4 :
            electrostatic = tf.transpose(electrostatic, perm=permutations)
            electrostatic.set_shape(input_shape)
            return (electron_density, electrostatic, *args[1:])

        return (electron_density, *args)


    def load_dataset(self, properties=[]):
        """ Loads a TFRecord and uses map to parse it, and stores it into self.dataset
        Check https://keras.io/examples/keras_recipes/tfrecord/ "define load methods"
        because this is basically a copy paste of that code with small modifications

        Args:
            properties (list, optional): Check parse_fn above

        Returns:
            dataset: Loadad TFRecord
        """
        ignore_order = tf.data.Options()
        ignore_order.experimental_deterministic = False  # disable order, increase speed
        dataset = tf.data.TFRecordDataset(
            self.filename
        )  # automatically interleaves reads from multiple files
        dataset = dataset.with_options(
            ignore_order
        )  # uses data as soon as it streams in, rather than in its original order
        dataset = dataset.map(
            partial(self.parse_fn, properties=properties),
            num_parallel_calls=self.AUTOTUNE
        )
        # returns the dataset as loaded
        return dataset

    def get_dataset(self, train=False, properties=[]):
        """Loads the TFRecord from the paths (filenames), and then shuffles the data and
        divides it into batches.

        Args:
            train (bool, optional): Train or test. Train will do data augmentation.
                                    Defaults to True.
            properties (list, optional): Check parse_fn above
        """
        dataset = self.load_dataset(properties)
        dataset = dataset.shuffle(2048)

        if train:
            dataset = dataset.map(
                map_func=self.train_preprocess, num_parallel_calls=self.AUTOTUNE)

        dataset = dataset.prefetch(buffer_size=self.AUTOTUNE)
        dataset = dataset.batch(self.BATCH_SIZE, drop_remainder=True)
        self.dataset = dataset  # .repeat()

    def next(self):
        """ Return the next batch

        Returns:
            tuple: next batch, you will need to get [0] for the actual tensor
        """

        return next(self.dataset_iter)
