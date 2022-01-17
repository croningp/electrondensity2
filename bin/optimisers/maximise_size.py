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

from src.utils.optimiser_utils import load_vae_model, grad_size


if __name__ == "__main__":

    BATCH_SIZE = 50
    vae, z_dim = load_vae_model('logs/vae/2021-05-25/')

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
