##########################################################################################
#
# This code is a small variation of "host_guest_overlapping.py" so check that one first.
# While that code is for CB6, this one is for a Pd cage.
# Because the cage is big, this code will alternate cycles of maximising size and 
# minimising overlapping. Hopefully this way we get a big molecule.
# This code is an extension of the one called "cage_hg.py" but also calculating for
# electron density
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
from src.optimisers.maximise_hg_esp import load_VAEmodel, load_ED_to_ESP
from src.optimisers.maximise_hg_esp import grad_ed_overlapping, grad_esp_overlapping
from src.optimisers.maximise_size import grad_size


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

    # cage is 80,80,80, cut out to get it as 64,64,64
    hosted = hosted[:, 8:-8, 8:-8, 8:-8, :]
    hostesp = hostesp[:, 8:-8, 8:-8, 8:-8, :]
        
    return tf.tile(hosted, [batch_size, 1, 1, 1, 1]), tf.tile(hostesp, [batch_size, 1, 1, 1, 1])


if __name__ == "__main__":

    BATCH_SIZE = 32
    DATA_FOLDER = '/home/nvme/juanma/Data/Jarek/'

    host_ed, host_esp = load_host(
        DATA_FOLDER+'cage.pkl', DATA_FOLDER+'cage_esp.pkl', BATCH_SIZE)
    vae, z_dim = load_VAEmodel('logs/vae/2021-05-25/')
    ed_to_esp = load_ED_to_ESP('logs/vae_ed_esp/2021-07-18')

    noise_t = K.random_uniform(shape = (BATCH_SIZE, z_dim),
                             minval = -2.0, maxval = 2.0)
    _, _, initial_output = grad_ed_overlapping(noise_t, vae, host_ed)

    with open('cage_esp_opt_initial_guest.p', 'wb') as file:
        pickle.dump(initial_output, file)

    with open('cage_esp_opt_initial_hg.p', 'wb') as file:
        pickle.dump(initial_output+host_ed, file)

    # First we try to maximise size of molecule
    for i in tqdm.tqdm(range(5000)):
        f, grads, output = grad_size(noise_t, vae)
        print("size "+str(np.mean(f.numpy())))
        noise_t += 0.003 * grads[0].numpy()
        noise_t = np.clip(noise_t, a_min=-9.0, a_max=9.0)

    with open('cage_esp_opt_size.p', 'wb') as file:
        pickle.dump(output, file)

    # now we will do five cycles of optimising
    for factor in [1, 5, 10, 20, 50]:
        lr = 0.05 / factor

        for j in tqdm.tqdm(range(int(10000/factor))):
            # try to minimise overlapping ESP
            f, grads, output, esps = grad_esp_overlapping(noise_t, vae, ed_to_esp, host_esp)
            print(np.mean(f.numpy()))
            noise_t -= lr * grads[0].numpy() * 1.0
            # noise_t = np.clip(noise_t, a_min=-5.0, a_max=5.0)

            if j % 1000 == 0:
                with open('cage_esp_optimizedESPED.p', 'wb') as file:
                    pickle.dump(output, file)
                with open('cage_esp_optimizedESP.p', 'wb') as file:
                    pickle.dump(esps, file)

            # try to minimise overlapping ED
            f, grads, output = grad_ed_overlapping(noise_t, vae, host_ed)
            print(np.mean(f.numpy()))
            noise_t -= lr * grads[0].numpy() * 0.001
            # noise_t = np.clip(noise_t, a_min=-5.0, a_max=5.0)

            if j % 1000 == 0:
                with open('cage_esp_optimizedEDED.p', 'wb') as file:
                    pickle.dump(output, file)

    with open('cage_esp_optimized.p', 'wb') as file:
        pickle.dump(output, file)

