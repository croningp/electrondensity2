##########################################################################################
#
# This code is a small variation of "cage_hg_esp.py" so check that one first.
# Basically, instead of doing two steps each iteration, one for ED and one for ESP,
# we will make a function that combines them into a single step.
#
# Author: Juanma juanma.parrilla@gcu.ac.uk
#
##########################################################################################

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import tqdm
import pickle
import numpy as np
from datetime import datetime
import tensorflow as tf
from tensorflow.keras import backend as K

from src.utils.optimiser_utils import load_vae_model, load_ED_to_ESP
from src.utils.optimiser_utils import grad_ed_overlapping, grad_esp_overlapping
from src.utils.optimiser_utils import grad_size, load_cage_host_ed_esp


def combined_ed_esp(latent_vector, vae, ed2esp, hosted, hostesp, ed_factor):
    """Calculates one gradient step for ED and for ESP, and combines them together
    based on ed_factor.

    Args:
        latent_vector: latent vector representing the molecule of size z_dim
        vae: vae model to obtain 3D molecules from latent_vector
        ed2esp: Model to get the ESP from ED (3d to 3d)
        hosted: the electron density of the host molecule
        hostesp: the electrostatic potential of the host molecule
        ed_factor: ed gradients will be multiplied by this, while esp gradients will be
            multiplied by (1-ed_factor)
    """

    # calculate ESP gradient step
    fitness, esp_grads, eds, esps = grad_esp_overlapping(latent_vector, vae, ed2esp, hostesp)
    print("esp "+str(np.mean(fitness.numpy())))

    # calculate ED gradient step
    fitness, ed_grads, eds = grad_ed_overlapping(latent_vector, vae, hosted)
    print("ed "+str(np.mean(fitness.numpy())))

    # combine gradients based on factor argument
    gradients = ed_grads[0].numpy() * ed_factor + esp_grads[0].numpy() * (1-ed_factor)

    return fitness, gradients, eds, esps


if __name__ == "__main__":

    for ef in [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]:
        # factor that we will use to multiply the ED part of gradient descent.
        # The ESP part will by multiplied by 1-ed_factor
        ed_factor = ef

        # folder where to save the logs of this run
        startdate = datetime.now().strftime('%Y-%m-%d')
        RUN_FOLDER = startdate +'_'+ str(ed_factor)

        # just do a while loop to make sure the folder doesnt exist
        n = 0
        while os.path.exists(RUN_FOLDER+'_'+str(n)+'/'):
            n += 1
        RUN_FOLDER += '_'+str(n)+'/'
        os.mkdir(RUN_FOLDER)

        BATCH_SIZE = 48
        # DATA_FOLDER = '/home/nvme/juanma/Data/ED/' # in auchentoshan
        # DATA_FOLDER = '/media/extssd/juanma/' # in dragonsoop
        DATA_FOLDER = '/home/juanma/Data/' # in maddog2020

        host_ed, host_esp = load_cage_host_ed_esp(
            DATA_FOLDER+'cage.pkl', DATA_FOLDER+'cage_esp.pkl', BATCH_SIZE)
        vae, z_dim = load_vae_model('logs/vae/2021-05-25/')
        ed_to_esp = load_ED_to_ESP('logs/vae_ed_esp/2021-07-18')

        noise_t = K.random_uniform(shape = (BATCH_SIZE, z_dim),
                                minval = -3.0, maxval = 3.0)
        _, _, init_eds, init_esps = combined_ed_esp(noise_t, vae, ed_to_esp, host_ed, host_esp, 1)

        with open(RUN_FOLDER+'cage_esp_opt_initial_g_ed.p', 'wb') as file:
            pickle.dump(init_eds, file)

        with open(RUN_FOLDER+'cage_esp_opt_initial_g_esp.p', 'wb') as file:
            pickle.dump(init_esps, file)

        # First we try to maximise size of molecule
        for i in tqdm.tqdm(range(2500)):
            f, grads, output = grad_size(noise_t, vae)
            print("size "+str(np.mean(f.numpy())))
            noise_t += 0.0011 * grads[0].numpy()

        # size doesnt use ESP, but do a gradient pass to get the ESP and save it
        _, _, eds, esps = combined_ed_esp(noise_t, vae, ed_to_esp, host_ed, host_esp, 1)
        
        with open(RUN_FOLDER+'cage_esp_opt_size_ED.p', 'wb') as file:
            pickle.dump(eds, file)
        with open(RUN_FOLDER+'cage_esp_opt_size_ESP.p', 'wb') as file:
            pickle.dump(esps, file)

        # now we will do five cycles of optimising
        for factor in [1, 5, 10, 20, 50]:
            lr = 0.5 / factor
            slr = str(factor)

            for j in tqdm.tqdm(range(int(1000/factor))):
                # try to minimise combined ED and esp
                f, grads, eds, esps = combined_ed_esp(noise_t, vae, ed_to_esp, host_ed, host_esp, ed_factor)
                noise_t -= lr * grads

                if j % 20 == 0:
                    with open(RUN_FOLDER+'cage_esp_optimizedED'+slr+'.p', 'wb') as file:
                        pickle.dump(eds, file)
                    with open(RUN_FOLDER+'cage_esp_optimizedESP'+slr+'.p', 'wb') as file:
                        pickle.dump(esps, file)

        # also save final results
        with open(RUN_FOLDER+'cage_esp_optimized_final_ED.p', 'wb') as file:
            pickle.dump(eds, file)
        with open(RUN_FOLDER+'cage_esp_optimized_final_ESP.p', 'wb') as file:
            pickle.dump(esps, file)

