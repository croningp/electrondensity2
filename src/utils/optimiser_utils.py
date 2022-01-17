##########################################################################################
#
# Some shared functionality for the optimiser scripts.
#
# Author: juanma@chem.gla.ac.uk
#
##########################################################################################

import os
import tensorflow as tf
import pickle
import numpy as np

from src.models.VAEresnet import VAEresnet
from src.utils import transform_ed, transform_back_ed
from src.utils.TFRecordLoader import TFRecordLoader


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
        # just to get different batches
        for i in range(5):
            batch = next(tfr_va.dataset_iter)[0]
        # pre process data
        batch = tf.tanh(batch)
        batch = transform_ed(batch)
        # put through encoder and return
        _, _, z = vae.encoder(batch)
        return z


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

    return tf.tile(host, [batch_size, 1, 1, 1, 1])


def load_cage_host_ed(filepath, batch_size, tanh=True):
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


def load_cage_host_ed_esp(filepathED, filepathESP, batch_size, tanh=True):
    """Loads host saved as pickle file. The host pickles were prepared by Jarek.

    Args:
        filepathED: Path to the pickle file that contains the host molecule ED
        filepathESP: Path to the pickle file that contains the host molecule ESP
        batch_size: As the name says, batch size.
        tanh: If doing a tanh after loading the molecule.

    Returns:
        Returns the host repeated batch_size times
    """
    with open(filepathED, 'rb') as file:
        hosted = pickle.load(file)
        if tanh:
            hosted = np.tanh(hosted)  # tan h needed?
        hosted = hosted.astype(np.float32)

    with open(filepathESP, 'rb') as file:
        hostesp = pickle.load(file)
        hostesp = np.expand_dims(hostesp, axis=(0,-1))
        hostesp = hostesp.astype(np.float32)
        # we need to dillate host to use voxels of 5,5,5
        datap = tf.nn.max_pool3d(hostesp, 5, 1, 'SAME')
        datan = tf.nn.max_pool3d(hostesp*-1, 5, 1, 'SAME')
        hostesp = datap + datan*-1

    # cage is 80,80,80, cut out to get it as 64,64,64
    hosted = hosted[:, 8:-8, 8:-8, 8:-8, :]
    hostesp = hostesp[:, 8:-8, 8:-8, 8:-8, :]
        
    return tf.tile(hosted, [batch_size, 1, 1, 1, 1]), tf.tile(hostesp, [batch_size, 1, 1, 1, 1])


def load_vae_model(modelpath):
    """Create VAE model from config file, and load the weights

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