##########################################################################################
#
# This model will aim to build a Transformer to convert from electron density to smiles.
# Heavily based on the following tutorial:
# https://keras.io/examples/audio/transformer_asr/
# Smiles and electron densities will be inputted as pre-processed by Jarek.
# I am not adding comments to most of the stuff that is directly copy-pasted. Check the
# tutorials above and then you will easily understand it.
#
# Author: Juan Manuel Parrilla Gutierrez (juanma@chem.gla.ac.uk)
#
##########################################################################################


import os
import random
from glob import glob
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.models.layers import conv_block, identity_block, drop_dimension


class TokenEmbedding(layers.Layer):
    def __init__(self, num_vocab=1000, maxlen=100, num_hid=64):
        super().__init__()
        self.emb = tf.keras.layers.Embedding(num_vocab, num_hid)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=num_hid)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        x = self.emb(x)
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        return x + positions


class ElectronDensityEmbedding(layers.Layer):
    def __init__(self, num_hid=64):
        super().__init__()
        # size embedding
        self.se = num_hid

    def call(self, x):

        # from 3D to 2D to 1D
        for i in range(2):
            x = conv_block(x, kernel_size=3, filters=se, stage=i, block='a', strides=1)
            x = identity_block(x, kernel_size=3, filters=se, stage=i, block='b')
            x = drop_dimension(x)


class SpeechFeatureEmbedding(layers.Layer):
    def __init__(self, num_hid=64, maxlen=100):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv1D(
            num_hid, 11, strides=2, padding="same", activation="relu"
        )
        self.conv2 = tf.keras.layers.Conv1D(
            num_hid, 11, strides=2, padding="same", activation="relu"
        )
        self.conv3 = tf.keras.layers.Conv1D(
            num_hid, 11, strides=2, padding="same", activation="relu"
        )
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=num_hid)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.conv3(x)
