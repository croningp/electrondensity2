from tensorflow.keras.callbacks import Callback, LearningRateScheduler
import numpy as np
import matplotlib.pyplot as plt
import os


class CustomCallback(Callback):
    """ Callback used on the VAE to generate electrondensities every few iterations
    """
    
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

class DisplayOutputs(Callback):
    def __init__(
        self, batch, idx_to_token, target_start_token_idx=30, target_end_token_idx=31
    ):
        """Callback used on the Smiles GPT. Displays a batch of outputs after every epoch.

        Args:
            batch: A test batch containing.
            idx_to_token: A List containing the vocabulary tokens corresponding to their indices
            target_start_token_idx: A start token index in the target vocabulary
            target_end_token_idx: An end token index in the target vocabulary
        """
        self.batch = batch
        self.target_start_token_idx = target_start_token_idx
        self.target_end_token_idx = target_end_token_idx
        self.idx_to_char = idx_to_token

    def on_epoch_end(self, epoch, logs=None):
        # if epoch % 5 != 0:
        #     return
        source = self.batch[1][:3]
        target = source.numpy()
        bs = source.shape[0]
        preds = self.model.generate(source, self.target_start_token_idx, 7)
        preds = preds.numpy()
        for i in range(bs):
            target_text = "".join([self.idx_to_char[str(t)] for t in target[i, :]])
            prediction = ""
            for idx in preds[i, :]:
                prediction += self.idx_to_char[str(idx)]
                if idx == self.target_end_token_idx:
                    break
            print(f"target:     {target_text}")
            print(f"prediction: {prediction}\n")



def step_decay_schedule(initial_lr, decay_factor=0.5, step_size=1):
    '''
    Wrapper function to create a LearningRateScheduler with step decay schedule.
    '''
    def schedule(epoch):
        new_lr = initial_lr * (decay_factor ** np.floor(epoch/step_size))
        
        return new_lr

    return LearningRateScheduler(schedule)