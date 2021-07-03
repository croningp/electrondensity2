##########################################################################################
#
# This code is a small variation of "host_guest_overlapping.py" so check that one first.
# While that code is for CB6, this one is for a Pd cage.
# Because the cage is big, this code will alternate cycles of maximising size and 
# minimising overlapping. Hopefully this way we get a big molecule.
#
# Author: Juanma juanma@chem.gla.ac.uk
#
##########################################################################################

import os
import tqdm
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

from src.utils.TFRecordLoader import TFRecordLoader
from src.models.VAEresnet import VAEresnet
from src.utils import transform_ed, transform_back_ed
from src.optimisers.host_guest_overlapping import load_model


def load_host(filepath, batch_size, tanh=True):
    """Loads host saved as pickle file. The host pickles were prepared by Jarek.

    Args:
        filepath: Path to the pickle file that contains the host molecule
        batch_size: As the name says, batch size.
        tanh: If doing a tanh after loading the molecule.

    Returns:
        Returns the host repeated batch_size times
    """
    with open(filepath, 'rb') as file:
        host = pickle.load(file)
        if tanh:
            host = np.tanh(host)  # tan h needed?
        host = host.astype(np.float32)

    # cage is 80,80,80, cut out to get it as 64,64,64
    host = host[:, 8:-8, 8:-8, 8:-8, :]

    return tf.tile(host, [batch_size, 1, 1, 1, 1])


@tf.function
def grad_overlapping(noise, vae, host):
    """Computes the gradient of fintess function with respect to latent z
    Args:
        noise: a tensor with hidden noise z of shape [batch_size, noise_size]
        vae: the variational autoencoder. we will use the decoder to generate electron den
        host: the loaded host molecule
    Returns:
        fitness: tensor with current fitness
        gradients: a tensor with shape[batch_size, noise_size]
        output: a tensor with generated electron densities with shape
                [batch_size, 64, 64, 64, 1]
    """

    # get the electron density from the noise
    guest = vae.decoder(noise)
    guest = transform_back_ed(guest)
    # host guest interaction fitness function
    fitness = tf.reduce_sum(guest*host, axis=[1, 2, 3, 4, ])
    # calculate gradients of the fitness against the noise and return results
    gradients = tf.gradients(fitness, noise)
    return fitness, gradients, guest


@tf.function
def grad_size(noise, vae):
    """Computes the gradient of fintess function with respect to latent z
    Args:
        noise: a tensor with hidden noise z of shape [batch_size, noise_size]
    Returns:
        fitness: tensor with current fitness
        gradients: a tensor with shape[batch_size, noise_size]
        output: a tensor with generated electron densities with shape
                [batch_size, 64, 64, 64, 1]
    """
    output = vae.decoder(noise)
    output = transform_back_ed(output)
    fitness = tf.reduce_sum(output, axis=[1,2,3,4,])
    gradients = tf.gradients(fitness, noise)
    return fitness, gradients, output


if __name__ == "__main__":

    BATCH_SIZE = 50
    DATA_FOLDER = '/home/nvme/juanma/Data/Jarek/'
    host = load_host(DATA_FOLDER+'cage.pkl', BATCH_SIZE)
    vae, z_dim = load_model('logs/vae/2021-05-25/')

    noise_t = K.random_uniform(shape=(BATCH_SIZE, z_dim), minval=-4.0, maxval=4.0)
    _, _, initial_output = grad_overlapping(noise_t, vae, host)

    with open('initial_g.p', 'wb') as file:
        pickle.dump(initial_output, file)

    with open('initial_hg.p', 'wb') as file:
        pickle.dump(initial_output+host, file)

    # First we try to maximise size of molecule
    for i in tqdm.tqdm(range(5000)):
        f, grads, output = grad_size(noise_t, vae)
        print("size "+str(np.mean(f.numpy())))
        noise_t += 0.01 * grads[0].numpy()
        noise_t = np.clip(noise_t, a_min=-5.0, a_max=5.0)

        if i % 500 == 0:
            with open('optimized_size.p', 'wb') as file:
                pickle.dump(output, file)

    # Now we will minimize overlapping
    # we will do five cycles of optimising for both
    for factor in [1, 5, 10, 20, 50]:
        lr = 0.1 / factor

        # try to minimise overlapping
        for j in tqdm.tqdm(range(int(8000/factor))):
            f, grads, output = grad_overlapping(noise_t, vae, host)
            print("overlapping "+str(np.mean(f.numpy())))
            noise_t -= lr * grads[0].numpy()
            # noise_t = np.clip(noise_t, a_min=-5.0, a_max=5.0)

            if j % 1000 == 0:
                with open('optimized_g.p', 'wb') as file:
                    pickle.dump(output, file)

                with open('optimized_hg.p', 'wb') as file:
                    pickle.dump(output+host, file)
