##########################################################################################
#
# This code is a variation of "cage_hg_esp_lee.py" so check that one first.
# Basically, instead of directly multiplying ESPs from guest and hos to get the 
# interactions, we split the host into its + and its - charges, and individually multiply
# the guest with both of them, and then merge them together using a factor. This way
# you can give a bigger focus to positive or negative charges.
#
# Author: Juanma juanma@chem.gla.ac.uk
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

from src.optimisers.maximise_hg_esp import load_VAEmodel, load_ED_to_ESP
from src.optimisers.maximise_hg_esp import grad_ed_overlapping, grad_esp_overlapping
from src.optimisers.maximise_size import grad_size

from src.optimisers.cage_hg_esp import load_host

def combined_ed_esp(latent_vector, vae, ed2esp, hosted, hostesp_pos, hostesp_neg,
    ed_factor, esp_pos_factor):
    """Calculates one gradient step for ED and for ESP, and combines them together
    based on ed_factor.

    Args:
        latent_vector: latent vector representing the molecule of size z_dim
        vae: vae model to obtain 3D molecules from latent_vector
        ed2esp: Model to get the ESP from ED (3d to 3d)
        hosted: the electron density of the host molecule
        hostesp_pos: the electrostatic positive potential of the host molecule
        hostesp_neg: the electrostatic negative potential of the host molecule
        ed_factor: ed gradients will be multiplied by this, while esp gradients will be
            multiplied by (1-ed_factor)
        esp_pos_factor: esp will be divided into + and -. This factor will be used to multiply
            the + by, while the - will be (1-esp_pos_factor)
    """

    # calculate ESP gradient step. First positive regions, then negative, then combine
    fp, esp_grads_p, _, esps = grad_esp_overlapping(latent_vector, vae, ed2esp, hostesp_pos)
    fn, esp_grads_n, _, esps = grad_esp_overlapping(latent_vector, vae, ed2esp, hostesp_neg)
    esp_grads = esp_grads_p[0].numpy() * esp_pos_factor + esp_grads_n[0].numpy() * (1-esp_pos_factor)
    combined_fitness = np.mean(fp.numpy()) + np.mean(fn.numpy())
    print("esp "+str(combined_fitness))

    # calculate ED gradient step
    fitness, ed_grads, eds = grad_ed_overlapping(latent_vector, vae, hosted)
    print("ed "+str(np.mean(fitness.numpy())))

    # combine gradients based on factor argument
    gradients = ed_grads[0].numpy() * ed_factor + esp_grads * (1-ed_factor)

    return fitness, gradients, eds, esps


def split_host_esp(host_esp):
    """Given the ESP of a molecule that will have positive and negative parts, this
    function will split them and return the + and the -.

    Args:
        host_esp: the electrostatic potential of a molecule
    """

    pos_esp = tf.cast(host_esp > 0, host_esp.dtype) * host_esp
    neg_esp = tf.cast(host_esp < 0, host_esp.dtype) * host_esp

    return pos_esp, neg_esp


if __name__ == "__main__":

    for esp_pf in range(0, 8, 2):
        # factor that we will use to multiply the positive part of the host ESP.
        # The negative part will be multiplied by 1-esp_pos_factor
        esp_pos_factor = esp_pf / 10.
        # factor that we will use to multiply the ED part of gradient descent.
        # The ESP part will by multiplied by 1-ed_factor
        ed_factor = 0.9

        # folder where to save the logs of this run
        startdate = datetime.now().strftime('%Y-%m-%d')
        RUN_FOLDER = startdate +'_'+ str(ed_factor) +'_'+ str(esp_pos_factor)

        # just do a while loop to make sure the folder doesnt exist
        n = 0
        while os.path.exists(RUN_FOLDER+'_'+str(n)+'/'):
            n += 1
        RUN_FOLDER += '_'+str(n)+'/'
        os.mkdir(RUN_FOLDER)

        BATCH_SIZE = 40
        # DATA_FOLDER = '/home/nvme/juanma/Data/ED/' # in auchentoshan
        DATA_FOLDER = '/media/extssd/juanma/' # in dragonsoop
        # DATA_FOLDER = '/home/juanma/Data/' # in maddog2020

        # loading the host, splitting it, and loading the models
        host_ed, host_esp = load_host(
            DATA_FOLDER+'cage.pkl', DATA_FOLDER+'cage_esp.pkl', BATCH_SIZE)
        host_esp_pos, host_esp_neg = split_host_esp(host_esp)
        vae, z_dim = load_VAEmodel('logs/vae/2021-05-25/')
        ed_to_esp = load_ED_to_ESP('logs/vae_ed_esp/2021-07-18')

        # generating random latent space and also initial population
        noise_t = K.random_uniform(shape = (BATCH_SIZE, z_dim),
                                minval = -3.0, maxval = 3.0)
        _, _, init_eds, init_esps = combined_ed_esp(
            noise_t, vae, ed_to_esp, host_ed, host_esp_pos, host_esp_neg, 1, 1)

        with open(RUN_FOLDER+'cage_esp_opt_initial_g.p', 'wb') as file:
            pickle.dump(init_eds, file)

        with open(RUN_FOLDER+'cage_esp_opt_initial_g_esp.p', 'wb') as file:
            pickle.dump(init_esps, file)

        with open(RUN_FOLDER+'cage_esp_opt_initial_hg.p', 'wb') as file:
            pickle.dump(init_eds+host_ed, file)

        # NOW WE START THE OPTIMISATION

        # First we try to maximise size of molecule
        for i in tqdm.tqdm(range(5000)):
            f, grads, output = grad_size(noise_t, vae)
            print("size "+str(np.mean(f.numpy())))
            noise_t += 0.0012 * grads[0].numpy()
            # noise_t = np.clip(noise_t, a_min=-5.0, a_max=5.0)

        with open(RUN_FOLDER+'cage_esp_opt_size.p', 'wb') as file:
            pickle.dump(output, file)

        # now we will do five cycles of optimising
        for factor in [1, 5, 10, 20, 50]:
            lr = 0.02 / factor
            slr = str(factor)

            for j in tqdm.tqdm(range(int(5000/factor))):
                # try to minimise combined ED and esp
                _, grads, eds, esps = combined_ed_esp(
                    noise_t, vae, ed_to_esp, host_ed, host_esp_pos, host_esp_neg,
                    ed_factor, esp_pos_factor)
                noise_t -= lr * grads
                # noise_t = np.clip(noise_t, a_min=-5.0, a_max=5.0)

                if j % 1000 == 0:
                    with open(RUN_FOLDER+'cage_esp_optimizedESP'+slr+'.p', 'wb') as file:
                        pickle.dump(eds, file)
                    with open(RUN_FOLDER+'cage_esp_optimizedED'+slr+'.p', 'wb') as file:
                        pickle.dump(esps, file)

        with open(RUN_FOLDER+'cage_esp_optimized_final.p', 'wb') as file:
            pickle.dump(eds, file)

