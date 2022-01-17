##########################################################################################
#
# This code will follow what Jarek did in the script he calls optimizer_2 (in branch
# develop). There, he creates random guests and then aims to minimise overlapping between
# hosts and guests via gradient descent. Main difference is that he was using the GAN to
# generate the guests, while I will use the trained VAE.
# This code is used with the CB6 host. With the cage host, search the files with cage
# as aprt of the name.
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

from src.utils.optimiser_utils import load_host, load_vae_model, initial_population
from src.utils.optimiser_utils import grad_overlapping


if __name__ == "__main__":

    BATCH_SIZE = 32
    DATA_FOLDER = '/home/nvme/juanma/Data/Jarek/'
    host = load_host(DATA_FOLDER+'cc6.pkl', BATCH_SIZE)
    vae, z_dim = load_vae_model('logs/vae/2021-05-25/')

    # noise_t = initial_population(BATCH_SIZE, random=True)
    noise_t = initial_population(BATCH_SIZE, False, datapath=DATA_FOLDER, vae=vae)
    _, _, initial_output = grad_overlapping(noise_t, vae, host)

    with open('initial_g.p', 'wb') as file:
        pickle.dump(initial_output, file)

    with open('initial_hg.p', 'wb') as file:
        pickle.dump(initial_output+host, file)

    for i in tqdm.tqdm(range(10000)):
        f, grads, output = grad_overlapping(noise_t, vae)
        print(np.mean(f.numpy()))
        noise_t -= 0.05 * grads[0].numpy()
        noise_t = np.clip(noise_t, a_min=-4.0, a_max=4.0)

        if i % 1000 == 0:
            with open('optimized_g.p', 'wb') as file:
                pickle.dump(output, file)

            with open('optimized_hg.p', 'wb') as file:
                pickle.dump(output+host, file)
