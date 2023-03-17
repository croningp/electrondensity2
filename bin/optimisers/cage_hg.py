##########################################################################################
#
# This code is a small variation of "host_guest_overlapping.py" so check that one first.
# While that code is for CB6, this one is for a Pd cage.
# Because the cage is big, this code will alternate cycles of maximising size and 
# minimising overlapping. Hopefully this way we get a big molecule.
#
# Author: Juanma juanma.parrilla@gcu.ac.uk
#
##########################################################################################

import os
import tqdm
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

from src.utils.optimiser_utils import load_vae_model, load_cage_host_ed, grad_ed_overlapping
from src.utils.optimiser_utils import grad_size, grad_ed_overlapping


if __name__ == "__main__":

    BATCH_SIZE = 50
    DATA_FOLDER = '/home/nvme/juanma/Data/ED/'  # in Auchentoshan
    host = load_cage_host_ed(DATA_FOLDER+'cage.pkl', BATCH_SIZE)
    vae, z_dim = load_vae_model('logs/vae/2021-05-25/')

    noise_t = K.random_uniform(shape=(BATCH_SIZE, z_dim), minval=-4.0, maxval=4.0)
    _, _, initial_output = grad_ed_overlapping(noise_t, vae, host)

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
    # we will do five cycles of optimising
    for factor in [1, 5, 10, 20, 50]:
        lr = 0.1 / factor

        # try to minimise overlapping
        for j in tqdm.tqdm(range(int(8000/factor))):
            f, grads, output = grad_ed_overlapping(noise_t, vae, host)
            print("overlapping "+str(np.mean(f.numpy())))
            noise_t -= lr * grads[0].numpy()
            # noise_t = np.clip(noise_t, a_min=-5.0, a_max=5.0)

            if j % 1000 == 0:
                with open('optimized_g.p', 'wb') as file:
                    pickle.dump(output, file)

                with open('optimized_hg.p', 'wb') as file:
                    pickle.dump(output+host, file)
