##########################################################################################
#
# This code tries to minimise the overlapping between host and guest in terms of
# electron density (so that the bodies/volumes don't collide). It also tries to minimise
# the electrostatic potential relations between molecules. So it tries to put parts of
# the molecules with different signs nearby.
# In the comments and code, ED means electron density, ESP means electrostatic potential
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
from src.models.VAE_ed_esp import VAE_ed_esp
from src.optimisers.host_guest_overlapping import load_host


def preprocess_esp(data):
    """ Preprocesses esps by normalizing it between 0 and 1, and doing a dillation
    so that a data point uses a 5x5x5 area instead of a single cell"""

    # first we will do a dillation, which needs to be done for both + and -
    datap = tf.nn.max_pool3d(data, 5, 1, 'SAME')
    datan = tf.nn.max_pool3d(data*-1, 5, 1, 'SAME')
    data = datap + datan*-1

    # I have pre-calculated that data goes between -0.265 and 0.3213
    # with this division it will be roughly between -1 and 1
    data = data / 0.33
    # now we place it between 0 and 1
    data = (data+1) * 0.5
    return data

    
def load_host(filepathED, filepathESP, batch_size, tanh=True):
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
        
    return tf.tile(hosted, [batch_size, 1, 1, 1, 1]), tf.tile(hostesp, [batch_size, 1, 1, 1, 1])


def load_VAEmodel(modelpath):
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


def load_ED_to_ESP(modelpath):
    """Model to translate EDs to ESPs"""

    with open(os.path.join(modelpath, 'params.pkl'), 'rb') as handle:
        config = pickle.load(handle)

        vae_ed_esp = VAE_ed_esp(
            input_dim=[64,64,64,1],
            encoder_conv_filters=[16, 32, 64, 64],
            encoder_conv_kernel_size=[3, 3, 3, 3],
            encoder_conv_strides=[2, 2, 2, 2],
            dec_conv_t_filters=[64, 64, 32, 16],
            dec_conv_t_kernel_size=[3, 3, 3, 3],
            dec_conv_t_strides=[2, 2, 2, 2],
            z_dim=400,
            use_batch_norm=True,
            use_dropout=True,
            r_loss_factor=50000,
            )

    vae_ed_esp.load_weights(os.path.join(modelpath, 'weights/weights.h5'))

    return vae_ed_esp

@tf.function
def grad_ed_overlapping(noise, vae, host):
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


@ tf.function
def grad_esp_overlapping(noise, vae, ed2esp, hostesp):
    """Computes the gradient of fintess function with respect to latent z
    Args:
        noise: a tensor with hidden noise z of shape [batch_size, noise_size]
        vae: a trained variational autoencoder
        ed2esp: A trained CNN 3D ED to ESP
        hostesp: the loaded host molecule ESP to test for overlapping
    Returns:
        fitness: tensor with current fitness
        gradients: a tensor with shape[batch_size, noise_size]
        output: a tensor with generated electron densities with shape
                [batch_size, 64, 64, 64, 1]
    """
    # get the electron density from the noise
    guests = vae.decoder(noise)
    guests = transform_back_ed(guests)

    # get the ESP from the generated guests ED. It will be between 0s and 1s
    guests_esps = ed2esp.model(guests, usetanh=False)
    # now transform them from 0..1 to -1 .. 1
    cubes = (guests_esps*2)-1
    # now between -0.33 and 0.33 which is the range of the orig data
    guests_esps = cubes * 0.33

    # host guest interaction fitness function
    fitness = tf.reduce_sum(guests_esps*hostesp, axis = [1, 2, 3, 4, ])

    gradients = tf.gradients(fitness, noise)

    return fitness, gradients, guests, guests_esps


if __name__ == "__main__":

    BATCH_SIZE = 32
    DATA_FOLDER = '/home/nvme/juanma/Data/Jarek/'
    host_ed, host_esp = load_host(
        DATA_FOLDER+'cc6.pkl', DATA_FOLDER+'cc6_esp.pkl', BATCH_SIZE)
    vae, z_dim = load_VAEmodel('logs/vae/2021-05-25/')
    ed_to_esp = load_ED_to_ESP('logs/vae_ed_esp/2021-07-18')

    noise_t = K.random_uniform(shape = (BATCH_SIZE, z_dim),
                             minval = -2.0, maxval = 2.0)
    _, _, initial_output = grad_ed_overlapping(noise_t, vae, host_ed)

    with open('cc6_esp_opt_initial.p', 'wb') as file:
        pickle.dump(initial_output, file)

    # we will do five cycles of optimising
    for factor in [1, 5, 10, 20, 50]:
        lr = 0.05 / factor

        for j in tqdm.tqdm(range(int(10000/factor))):
            # try to minimise overlapping ESP
            f, grads, output, esps = grad_esp_overlapping(noise_t, vae, ed_to_esp, host_esp)
            print(np.mean(f.numpy()))
            noise_t -= lr * grads[0].numpy() * 0.05
            noise_t = np.clip(noise_t, a_min=-5.0, a_max=5.0)

            if j % 1000 == 0:
                with open('cc6_esp_optimizedED.p', 'wb') as file:
                    pickle.dump(output, file)
                with open('cc6_esp_optimizedESP.p', 'wb') as file:
                    pickle.dump(esps, file)

            # try to minimise overlapping ED
            f, grads, output = grad_ed_overlapping(noise_t, vae, host_ed)
            print(np.mean(f.numpy()))
            noise_t -= lr * grads[0].numpy()
            noise_t = np.clip(noise_t, a_min=-5.0, a_max=5.0)

            if j % 1000 == 0:
                with open('cc6_ed_optimizedED.p', 'wb') as file:
                    pickle.dump(output, file)

    with open('cc6_esp_optimized.p', 'wb') as file:
        pickle.dump(output, file)

