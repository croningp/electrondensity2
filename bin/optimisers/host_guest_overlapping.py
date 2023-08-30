##########################################################################################
#
# This code will follow what Jarek did in the script he calls optimizer_2 (in branch
# develop). There, he creates random guests and then aims to minimise overlapping between
# hosts and guests via gradient descent. Main difference is that he was using the GAN to
# generate the guests, while I will use the trained VAE.
# This code is used with the CB6 host. With the cage host, search the files with cage
# as aprt of the name.
#
# Author: juanma.parrilla@gcu.ac.uk
#
##########################################################################################

import os
import tqdm
import pickle
import numpy as np
from datetime import datetime
import tensorflow as tf
from tensorflow.keras import backend as K

from src.utils.optimiser_utils import load_host, load_vae_model, initial_population
from src.utils.optimiser_utils import grad_ed_overlapping

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

if __name__ == "__main__":

    # folder where to save the logs of this run
    startdate = datetime.now().strftime('%Y-%m-%d')
    RUN_FOLDER = startdate

    # just do a while loop to make sure the folder doesnt exist
    n = 0
    while os.path.exists(RUN_FOLDER+'_'+str(n)+'/'):
        n += 1
    RUN_FOLDER += '_'+str(n)+'/'
    os.mkdir(RUN_FOLDER)

    BATCH_SIZE = 48
    # DATA_FOLDER = '/home/nvme/juanma/Data/ED/'  # in Auchentoshan
    DATA_FOLDER = '/home/juanma/Data/' # in maddog2020
    host = load_host(DATA_FOLDER+'cb6ED.pkl', BATCH_SIZE, expand_dims=True)
    vae, z_dim = load_vae_model('logs/vae/2021-05-25/')

    # noise_t = initial_population(BATCH_SIZE, random=True)
    # noise_t = initial_population(BATCH_SIZE, False, datapath=DATA_FOLDER, vae=vae)
    
    # initial population
    noise_t = K.random_uniform(shape = (BATCH_SIZE, z_dim),
                            minval = -2.0, maxval = 2.0)
    _, _, initial_output = grad_ed_overlapping(noise_t, vae, host)

    with open(RUN_FOLDER+'initial_cb6ONLYED.p', 'wb') as file:
        pickle.dump(initial_output, file)

    # we will do five cycles of optimising
    for factor in [1, 5, 10, 20, 50]:
        lr = 0.05 / factor
        slr = str(factor)

        for j in tqdm.tqdm(range(int(10000/factor))):
            f, grads, output = grad_ed_overlapping(noise_t, vae, host)
            print(np.mean(f.numpy()))
            noise_t -= lr * grads[0].numpy()
            noise_t = np.clip(noise_t, a_min=-5.0, a_max=5.0)

            if j % 1000 == 0:
                with open(RUN_FOLDER+'optimized_cb6ONLYED'+slr+'.p', 'wb') as file:
                    pickle.dump(output, file)

    with open(RUN_FOLDER+'cb6_optimised_final.p', 'wb') as file:
        pickle.dump(output, file)
