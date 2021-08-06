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

from src.models.VAEresnet import VAEresnet
from src.utils import transform_back_ed


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
    vae, z_dim = load_model('logs/vae/2021-05-25/')

    noise_t = K.random_uniform(shape=(BATCH_SIZE, z_dim), minval=-4.0, maxval=4.0)
    _, _, initial_output = grad_size(noise_t, vae)

    with open('initial.p', 'wb') as file:
        pickle.dump(initial_output, file)

    for i in tqdm.tqdm(range(10000)):
        f, grads, output = grad_size(noise_t, vae)
        print(np.mean(f.numpy()))
        noise_t += 0.001 * grads[0].numpy()
        # noise_t = np.clip(noise_t, a_min=-5.0, a_max=5.0)

        if i % 500 == 0:
            with open('optimized.p', 'wb') as file:
                pickle.dump(output, file)
