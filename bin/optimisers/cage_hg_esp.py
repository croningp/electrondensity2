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
#os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import tqdm
import pickle
import numpy as np
from datetime import datetime
import tensorflow as tf
from tensorflow.keras import backend as K

from src.utils.optimiser_utils import load_VAEmodel, load_ED_to_ESP
from src.utils.optimiser_utils import grad_ed_overlapping, grad_esp_overlapping
from src.utils.optimiser_utils import grad_size, load_cage_host_ed_esp



if __name__ == "__main__":

    # factor that we will use to multiply the ED part of gradient descent.
    # The ESP part will by multiplied by 1-ed_factor
    ed_factor = 0.95

    # folder where to save the logs of this run
    startdate = datetime.now().strftime('%Y-%m-%d')
    RUN_FOLDER = startdate +'_'+ str(ed_factor)

    # just do a while loop to make sure the folder doesnt exist
    n = 0
    while os.path.exists(RUN_FOLDER+'_'+str(n)+'/'):
        n += 1
    RUN_FOLDER += '_'+str(n)+'/'
    os.mkdir(RUN_FOLDER)

    BATCH_SIZE = 38
    DATA_FOLDER = '/home/nvme/juanma/Data/ED/' # in auchentoshan
    # DATA_FOLDER = '/media/extssd/juanma/' # in dragonsoop
    # DATA_FOLDER = '/home/juanma/Data/' # in maddog2020

    host_ed, host_esp = load_cage_host_ed_esp(
        DATA_FOLDER+'cage.pkl', DATA_FOLDER+'cage_esp.pkl', BATCH_SIZE)
    vae, z_dim = load_VAEmodel('logs/vae/2021-05-25/')
    ed_to_esp = load_ED_to_ESP('logs/vae_ed_esp/2021-07-18')

    noise_t = K.random_uniform(shape = (BATCH_SIZE, z_dim),
                             minval = -2.0, maxval = 2.0)
    _, _, init_eds, init_esps = grad_esp_overlapping(noise_t, vae, ed_to_esp, host_esp)

    with open(RUN_FOLDER+'cage_esp_opt_initial_g.p', 'wb') as file:
        pickle.dump(init_eds, file)

    with open(RUN_FOLDER+'cage_esp_opt_initial_g_esp.p', 'wb') as file:
        pickle.dump(init_esps, file)

    with open(RUN_FOLDER+'cage_esp_opt_initial_hg.p', 'wb') as file:
        pickle.dump(init_eds+host_ed, file)

    # First we try to maximise size of molecule
    for i in tqdm.tqdm(range(1000)):
        f, grads, output = grad_size(noise_t, vae)
        print("size "+str(np.mean(f.numpy())))
        noise_t += 0.001 * grads[0].numpy()
        # noise_t = np.clip(noise_t, a_min=-5.0, a_max=5.0)

    with open(RUN_FOLDER+'cage_esp_opt_size.p', 'wb') as file:
        pickle.dump(output, file)

    # now we will do five cycles of optimising
    for factor in [1, 5, 10, 20, 50]:
        lr = 0.01 / factor
        slr = str(factor)

        for j in tqdm.tqdm(range(int(10000/factor))):
            # try to minimise overlapping ESP
            f, grads, output, esps = grad_esp_overlapping(noise_t, vae, ed_to_esp, host_esp)
            print(np.mean(f.numpy()))
            noise_t -= lr * grads[0].numpy() * (1-ed_factor)
            # noise_t = np.clip(noise_t, a_min=-5.0, a_max=5.0)

            if j % 1000 == 0:
                with open(RUN_FOLDER+'cage_esp_optimizedESPED'+slr+'.p', 'wb') as file:
                    pickle.dump(output, file)
                with open(RUN_FOLDER+'cage_esp_optimizedESP'+slr+'.p', 'wb') as file:
                    pickle.dump(esps, file)

            # try to minimise overlapping ED
            f, grads, output = grad_ed_overlapping(noise_t, vae, host_ed)
            print(np.mean(f.numpy()))
            noise_t -= lr * grads[0].numpy() * ed_factor
            # noise_t = np.clip(noise_t, a_min=-5.0, a_max=5.0)

            if j % 1000 == 0:
                with open(RUN_FOLDER+'cage_esp_optimizedEDED'+slr+'.p', 'wb') as file:
                    pickle.dump(output, file)

    with open(RUN_FOLDER+'cage_esp_optimized_final.p', 'wb') as file:
        pickle.dump(output, file)

