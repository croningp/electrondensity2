from tensorflow.keras.callbacks import Callback, LearningRateScheduler
import numpy as np
import matplotlib.pyplot as plt
import os


class CustomCallback(Callback):
    
    def __init__(self, run_folder, print_every_n_epochs, initial_epoch, vae, tfr_va):
        self.epoch = initial_epoch
        self.run_folder = run_folder
        self.print_every_n_epochs = print_every_n_epochs
        self.vae = vae
        self.tfr_va = tfr_va

    def on_epoch_end(self, epoch, logs={}):  
        if epoch % self.print_every_n_epochs == 0:
            filepath = os.path.join(self.run_folder, 'edms', 'edm_' + str(self.epoch).zfill(3) + '.p')
            _ = self.vae.sample_model_validation(self.tfr_va, filepath)

        self.epoch += 1



def step_decay_schedule(initial_lr, decay_factor=0.5, step_size=1):
    '''
    Wrapper function to create a LearningRateScheduler with step decay schedule.
    '''
    def schedule(epoch):
        new_lr = initial_lr * (decay_factor ** np.floor(epoch/step_size))
        
        return new_lr

    return LearningRateScheduler(schedule)