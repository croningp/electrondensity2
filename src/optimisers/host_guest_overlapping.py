##########################################################################################
#
# This code will follow what Jarek did in the script he calls optimizer_2 (in branch
# develop). There, he creates random guests and then aims to minimise overlapping between
# hosts and guests via gradient descent. Main difference is that he was using the GAN to
# generate the guests, while I will use the trained VAE.
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

from src.utils.TFRecordLoader import TFRecordLoader
from src.models.VAEresnet import VAEresnet
from src.utils import transform_ed, transform_back_ed


def load_host(filepath, batch_size):
    """Loads host saved as pickle file. The host pickles were prepared by Jarek.

    Args:
        filepath: Path to the pickle file that contains the host molecule
        batch_size: As the name says, batch size.

    Returns:
        Returns the host repeated batch_size times
    """
    with open(filepath, 'rb') as file:
        host = pickle.load(file)
        host = np.tanh(host)  # tan h needed?
        host = host.astype(np.float32)

    return tf.tile(host, [batch_size, 1, 1, 1, 1])


def load_model(modelpath):
    """Create model from config file, and load the weights

    Args:
        modelpath: path to the log of the model. should be something like:
                   "logs/vae/2021-05-11"

    Returns:
        model: returns the model with loaded weights
        z_dim: size of the latent space.
    """

    # load the model configuration from the params.pkl file
    with open(os.path.join(modelpath, 'params.pkl'), 'rb') as handle:
        config = pickle.load(handle)

    # create the model
    vae = VAEresnet(
        input_dim=config[0],
        encoder_conv_filters=config[1],
        encoder_conv_kernel_size=config[2],
        encoder_conv_strides=config[3],
        dec_conv_t_filters=config[4],
        dec_conv_t_kernel_size=config[5],
        dec_conv_t_strides=config[6],
        z_dim=config[7],
        use_batch_norm=config[8],
        use_dropout=config[9],
        r_loss_factor=50000
    )

    # load the weights and return the model and the z_dim
    vae.load_weights(os.path.join(modelpath, 'weights/weights.h5'))
    return vae, config[7]


@tf.function
def grad(noise, vae):
    """Computes the gradient of fintess function with respect to latent z
    Args:
        noise: a tensor with hidden noise z of shape [batch_size, noise_size]
        vae: the variational autoencoder. we will use the decoder to generate electron den
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


def initial_population(batch_size, random=True, z_dim=400, datapath="", vae=None):
    """Create the initial population randomly, or use a batch of real data instead.

    Args:
        batch_size: Size of batch
        random (bool, optional): If random is false we use batch, else we use random
        z_dim (int, optional): Size of VAE z_dim. Only needed if radom is True
        datapath (str, optional): Path to tfrecord, only needed if random is False
        vae (optional): vae object, only needed if random is False

    Returns:
        latent vector of size (batch_size, z_dim)
    """

    if random:
        # return K.random_normal(shape=(batch_size, z_dim), mean=0., stddev=1.)
        return K.random_uniform(shape=(batch_size, z_dim), minval=-2.0, maxval=2.0)
    else:
        path2va = datapath + 'valid.tfrecords'
        tfr_va = TFRecordLoader(path2va, batch_size=batch_size)
        batch = next(tfr_va.dataset_iter)[0]
        # pre process data
        batch = tf.tanh(batch)
        batch = transform_ed(batch)
        # put through encoder and return
        _, _, z = vae.encoder(batch)
        return z


if __name__ == "__main__":

    BATCH_SIZE = 32
    DATA_FOLDER = '/home/nvme/juanma/Data/Jarek/'
    host = load_host(DATA_FOLDER+'cc6.pkl', BATCH_SIZE)
    vae, z_dim = load_model('logs/vae/2021-05-25/')

    noise_t = initial_population(BATCH_SIZE, random=True)
    # noise_t = initial_population(BATCH_SIZE, False, datapath=DATA_FOLDER, vae=vae)
    _, _, initial_output = grad(noise_t, vae)

    with open('initial_g.p', 'wb') as file:
        pickle.dump(initial_output, file)

    with open('initial_hg.p', 'wb') as file:
        pickle.dump(initial_output+host, file)

    for i in tqdm.tqdm(range(10000)):
        f, grads, output = grad(noise_t, vae)
        print(np.mean(f.numpy()))
        noise_t -= 0.05 * grads[0].numpy()
        noise_t = np.clip(noise_t, a_min=-4.0, a_max=4.0)

        if i % 1000 == 0:
            with open('optimized_g.p', 'wb') as file:
                pickle.dump(output, file)

            with open('optimized_hg.p', 'wb') as file:
                pickle.dump(output+host, file)
