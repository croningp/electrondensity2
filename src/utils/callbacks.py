import tensorflow as tf
from tensorflow.keras.callbacks import Callback, LearningRateScheduler
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
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
        self.genfrom = 10

    def on_epoch_end(self, epoch, logs=None):
        # if epoch % 5 != 0:
        #     return
        source = self.batch[1][:5]
        target = source.numpy()
        bs = source.shape[0]
        preds = self.model.generate(source, self.target_start_token_idx, self.genfrom)
        preds = preds.numpy()
        for i in range(bs):
            target_text = "".join([self.idx_to_char[str(t)] for t in target[i, 1:]])
            #target_text = self.tokenizer.decode_smiles(target[i, :])
            #prediction = self.tokenizer.decode_smiles(preds[i, :])
            prediction = ""
            for idx in preds[i, 1:]:
                prediction += self.idx_to_char[str(idx)]
                if idx == self.target_end_token_idx:
                    break
            # add white space to make more obvious seed and prediction
            pos = self.genfrom - 1 
            target_text = target_text[:pos] + ' ' + target_text[pos:]
            prediction = prediction[:pos] + ' ' + prediction[pos:]
            print(f"target:     {target_text.replace('NULL','').replace('STOP','')}")
            print(f"prediction: {prediction.replace('STOP','')}\n")


def step_decay_schedule(initial_lr, decay_factor=0.5, step_size=1):
    '''
    Wrapper function to create a LearningRateScheduler with step decay schedule.
    '''
    def schedule(epoch):
        new_lr = initial_lr * (decay_factor ** np.floor(epoch/step_size))
        
        return new_lr

    return LearningRateScheduler(schedule)


class CustomSchedule(LearningRateSchedule):
    def __init__(
        self,
        init_lr=0.000001,
        lr_after_warmup=0.00005,
        final_lr=0.000001,
        warmup_epochs=15,
        decay_epochs=85,
        steps_per_epoch=203,
    ):
        super().__init__()
        self.init_lr = init_lr
        self.lr_after_warmup = lr_after_warmup
        self.final_lr = final_lr
        self.warmup_epochs = warmup_epochs
        self.decay_epochs = decay_epochs
        self.steps_per_epoch = steps_per_epoch

    def calculate_lr(self, epoch):
        """ linear warm up - linear decay """
        warmup_lr = (
            self.init_lr
            + ((self.lr_after_warmup - self.init_lr) / (self.warmup_epochs - 1)) * epoch
        )
        decay_lr = tf.math.maximum(
            self.final_lr,
            self.lr_after_warmup
            - (epoch - self.warmup_epochs)
            * (self.lr_after_warmup - self.final_lr)
            / (self.decay_epochs),
        )
        return tf.math.minimum(warmup_lr, decay_lr)

    def __call__(self, step):
        epoch = step // self.steps_per_epoch
        return self.calculate_lr(epoch)
