##########################################################################################
#
# This code tries to minimise the overlapping between host and guest in terms of
# electron density (so that the bodies/volumes don't collide). It also tries to minimise
# the electrostatic potential relations between molecules. So it tries to put parts of
# the molecules with different signs nearby.
# In the comments and code, ED means electron density, ESP means electrostatic potential
#
# Author: juanma.parrilla@gcu.ac.uk
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

from src.utils.optimiser_utils import load_host_ed_esp, load_vae_model, load_ED_to_ESP
from src.utils.optimiser_utils import grad_esp_overlapping, grad_ed_overlapping


if __name__ == "__main__":

    # factor that we will use to multiply the ED part of gradient descent.
    for ef in [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]:
        
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
        # DATA_FOLDER = '/home/nvme/juanma/Data/ED/'
        DATA_FOLDER = '/home/juanma/Data/' # in maddog2020 and dragonsoop

        host_ed, host_esp = load_host_ed_esp(
            DATA_FOLDER+'cb6ED.pkl', DATA_FOLDER+'cb6ESP.pkl', BATCH_SIZE, 
            expand_dims=True, thicken=True)
        vae, z_dim = load_vae_model('logs/vae/2021-05-25/')
        ed_to_esp = load_ED_to_ESP('logs/vae_ed_esp/2021-07-18')

        noise_t = K.random_uniform(shape = (BATCH_SIZE, z_dim),
                                minval = -2.0, maxval = 2.0)
        _, _, init_eds, init_esps = grad_esp_overlapping(noise_t, vae, ed_to_esp, host_esp)

        with open(RUN_FOLDER+'cb6_esp_opt_initial_g_ed.p', 'wb') as file:
            pickle.dump(init_eds, file)

        with open(RUN_FOLDER+'cb6_esp_opt_initial_g_esp.p', 'wb') as file:
            pickle.dump(init_esps, file)

        with open(RUN_FOLDER+'cb6_esp_opt_initial_hg.p', 'wb') as file:
            pickle.dump(init_eds+host_ed, file)

        # we will do five cycles of optimising
        for factor in [1, 5, 10, 20, 50]:
            lr = 0.05 / factor
            slr = str(factor)

            for j in tqdm.tqdm(range(int(5000/factor))):
                # try to minimise overlapping ESP
                f, grads, output, esps = grad_esp_overlapping(noise_t, vae, ed_to_esp, host_esp)
                print(np.mean(f.numpy()))
                noise_t -= lr * grads[0].numpy() * (1-ed_factor)
                noise_t = np.clip(noise_t, a_min=-5.0, a_max=5.0)

                if j % 1000 == 0:
                    with open(RUN_FOLDER+'cb6_esp_optimisedESPED'+slr+'.p', 'wb') as file:
                        pickle.dump(output, file)
                    with open(RUN_FOLDER+'cb6_esp_optimisedESPESP'+slr+'.p', 'wb') as file:
                        pickle.dump(esps, file)

                # try to minimise overlapping ED
                f, grads, output = grad_ed_overlapping(noise_t, vae, host_ed)
                print(np.mean(f.numpy()))
                noise_t -= lr * grads[0].numpy() * ed_factor
                noise_t = np.clip(noise_t, a_min=-5.0, a_max=5.0)

                if j % 1000 == 0:
                    with open(RUN_FOLDER+'cb6_esp_optimisedEDED'+slr+'.p', 'wb') as file:
                        pickle.dump(output, file)

        # this last one is just to save them
        f, grads, output, esps = grad_esp_overlapping(noise_t, vae, ed_to_esp, host_esp)
        with open(RUN_FOLDER+'cb6_esped_optimised_final.p', 'wb') as file:
            pickle.dump(output, file)

        with open(RUN_FOLDER+'cb6_espesp_optimised_final.p', 'wb') as file:
            pickle.dump(esps, file)

