##########################################################################################
#
# This code tries to minimise the overlapping between host and guest, AND maximise the
# polarity of the guest. I suggest before checking this you check the files 
# "host_guest_overlapping.py" and "maximise_polar.py". This script is like those two
# together.
#
# Author: juanma@chem.gla.ac.uk
#
##########################################################################################

import os
import tqdm
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

from src.models.CNN3D_featureprediction import CNN3D_singleprediction
from src.utils.optimiser_utils import load_host, load_vae_model
from src.utils import transform_back_ed


def load_PredictionModel(modelpath):
    """Model to make feature prediction"""

    with open(os.path.join(modelpath, 'params.pkl'), 'rb') as handle:
        config = pickle.load(handle)

    cnn3D = CNN3D_singleprediction(
            cubeside = 64, 
            filters = [16,32,32,64], 
            strides = [2,2,1,1], 
            dense_size = 256,
            usetanh = False,
    )
    cnn3D.load_weights(os.path.join(modelpath, 'weights/weights.h5'))

    return cnn3D

@tf.function
def grad(noise, vae, cnn3D, host):
    """Computes the gradient of fintess function with respect to latent z
    Args:
        noise: a tensor with hidden noise z of shape [batch_size, noise_size]
        vae: a trained variational autoencoder
        cnn3D: A trained CNN 3D input to single output
        host: the loaded host molecule to test for overlapping
    Returns:
        fitness: tensor with current fitness
        gradients: a tensor with shape[batch_size, noise_size]
        output: a tensor with generated electron densities with shape
                [batch_size, 64, 64, 64, 1]
    """
    # get the electron density from the noise
    guests = vae.decoder(noise)
    guests = transform_back_ed(guests)
    # host guest interaction fitness function
    fitness_overlapping = tf.reduce_sum(guests*host, axis=[1, 2, 3, 4, ]) * -10
    fitness_polarity = cnn3D.model(guests)
    fitness = fitness_overlapping + fitness_polarity
    gradients = tf.gradients(fitness, noise)
    return fitness, gradients, guests, fitness_overlapping, fitness_polarity


if __name__ == "__main__":

    BATCH_SIZE = 32
    DATA_FOLDER = '/home/nvme/juanma/Data/ED/'  # in Auchentoshan
    host = load_host(DATA_FOLDER+'cc6.pkl', BATCH_SIZE)
    vae, z_dim = load_vae_model('logs/vae/2021-05-25/')
    cnn3d = load_PredictionModel('logs/feature_prediction/2021-06-14')

    noise_t = K.random_uniform(shape=(BATCH_SIZE, z_dim), minval=-2.0, maxval=2.0)
    _, _, initial_output, _, _ = grad(noise_t, vae, cnn3d, host)

    with open('polar_initial.p', 'wb') as file:
        pickle.dump(initial_output, file)

    for i in tqdm.tqdm(range(10000)):
        f, grads, output, f1, f2 = grad(noise_t, vae, cnn3d, host)
        # print(np.mean(f.numpy()))
        print(np.mean(f1.numpy()) , np.mean(f2.numpy()))
        noise_t += 0.01 * grads[0].numpy()
        noise_t = np.clip(noise_t, a_min=-4.0, a_max=4.0)

        if i % 100 == 0:
            with open('polar_optimized.p', 'wb') as file:
                pickle.dump(output, file)
