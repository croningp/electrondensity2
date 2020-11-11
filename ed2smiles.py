# -*- coding: utf-8 -*-
"""
Electron density to smiles translation models
and related utilities. 
Created on Tue Jun  9 12:07:18 2020

@author: Jaroslaw Granda
"""

import os
import pickle
import numpy as np
import tqdm
import time
import rdkit
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, AvgPool2D, LSTMCell, Dense, RNN
from tensorflow.keras.layers import Conv2DTranspose, MaxPool2D, UpSampling2D, Layer, RNN, Bidirectional
from tensorflow.keras.layers import LSTM, Conv3D, MaxPool3D, AvgPool3D, UpSampling3D, Conv3DTranspose, Activation, BatchNormalization
from tensorflow.keras import Model

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':20,
         'ytick.labelsize':20}
pylab.rcParams.update(params)


os.chdir('/home/jarek/electrondensity2')
from input.tfrecords import input_fn
from input.tokenizer import Tokenizer



class ResBlockDown(Model):
    """ Resnet like block working with 2D images"""
    def __init__(self, num_channels, pooling='AvgPool2D'):
        """
        Args:
            num_channels: int with number of output channels
            pooling: str with type of pooling used to reduce dimensions
        """
        
        super(ResBlockDown,self).__init__()
        self.num_channels = num_channels
        self.conv_1x1 = Conv2D(num_channels, 1, padding='same')
        self.conv_3x3a = Conv2D(num_channels, 3, padding='same')
        self.conv_3x3b = Conv2D(num_channels, 3, padding='same')
        
        if pooling == 'MaxPool2D':
            self.pooling = MaxPool2D(2)
        elif pooling == 'AvgPool2D':
            self.pooling = AvgPool2D(2)
        else:
            raise ValueError('Unknowne type of pooling {}'.format(pooling))
        self.activation = Activation('elu')
            
        
    def call(self, inputs):
        """
        Transoforms inputs through resnet like layer.
        Args: 
            inputs: a tensor with shape [batch_size, img_size, img_size, num_channels]
            output: a tensor with shape [batch_size, img_size/2, img_size/2, num_channels]
        
        """
        layer_1a = self.conv_1x1(inputs)
        layer_1a = self.pooling(layer_1a)
    
        layer_1b= self.activation(inputs)
        layer_1b = self.conv_3x3a(layer_1b)
        layer_1b = self.activation(layer_1b)
        layer_1b = self.conv_3x3b(layer_1b)
        layer_1b = self.pooling(layer_1b)
        
        
        output = layer_1a + layer_1b
        
        return output

class ConvSelfAttn(Model):
    """
    Convodultional self-attention from https://arxiv.org/abs/1805.08318
    """

    def __init__(self,  attn_dim, output_dim):
        """
        Args:
             attn_dim: int for the attention dimension
             output_dim: int with feature_size for the attention transformed inputs
             kernel_initializer: str or initializer for the Conv2D used in attention
        """
    
        super(ConvSelfAttn, self).__init__()
        self.f_conv = Conv2D(attn_dim, 1)
        self.g_conv = Conv2D(attn_dim, 1)
        self.h_conv = Conv2D(attn_dim, 1)
        self.v_conv = Conv2D(output_dim, 1)
        
        self.scale = tf.Variable(0.0)
    
    def flatten(self, inputs):
        """Inner flattens the inputs
        Args:
            inputs: tensor of shape [batch_size, img_dim, img_dim, hidden_dim]
            output: tensor of [batch_size, img_dim*img_dim hidden_dim]
        
        """
    
        inputs_shape = inputs.get_shape()
        batch_size = tf.TensorShape([inputs_shape[0]])
        hidden_dims = tf.TensorShape([inputs_shape[1] * inputs_shape[2]])
        last_dim = tf.TensorShape([inputs_shape[-1]])
        new_shape = batch_size + hidden_dims + last_dim
        new_shape = [inputs_shape[0], tf.reduce_prod(inputs_shape[1:-1]), inputs_shape[-1]]
        return tf.reshape(inputs, new_shape)
    
    
    def call(self, input):
        """Performs the attention
        Args:
            input: tensor of shape [batch_size, img_dim, img_dim, hidden_dim]
        Returns:
            output: tensor of shape [batch_size, img_dim, img_dim, output_dim]
            
        """

        fx = self.f_conv(input)
        gx = self.g_conv(input)
        hx = self.h_conv(input)
        
        fx_flat = self.flatten(fx)
        gx_flat = self.flatten(gx)
        hx_flat = self.flatten(hx)
        
        raw_attn_weights = tf.matmul(fx_flat, gx_flat, transpose_b=True)
        raw_attn_weights = tf.transpose(raw_attn_weights, perm=[0,2,1])
        attn_weights = tf.nn.softmax(raw_attn_weights, axis=-1)
        
        attn_flat = tf.matmul(attn_weights, hx_flat)
        attn = tf.reshape(attn_flat, hx.get_shape())
        output = self.v_conv(attn)
        
        output  = self.scale * output + input
        return output


class InputCNN(Model):
    """
    CNN used to flatten 2D electron densities slices
    int 1D flat representations.
    """
    def __init__(self):
        super(InputCNN, self).__init__()
        self.resblock_1 = ResBlockDown(4)
        self.resblock_2 = ResBlockDown(8)
        self.resblock_3 = ResBlockDown(16)
        self.resblock_4 = ResBlockDown(32)  
        self.flatten = Flatten()
     
    def call(self, inputs):
        """
        Flatten the input slices. 
        Args:
            inputs: a tensor with shape [batch_size, img_size, img_size, 1]
            output: a tensor with shape [batch_size, flat_dim]
        """
        x = self.resblock_1(inputs)    
        x = self.resblock_2(x)
        x = self.resblock_3(x)
        x = self.resblock_4(x)
        x = self.flatten(x)
        return x
    
    
    
    
class Encoder(Model):
    """
    Electron density encoder equivalent to encoder in seq2seq models.
    It transforms electron density from 3D representation into 2D representation 
    suitable as an input to attention mechanism. It does so by first reshaping the 
    input electron densities from shape [batch_size, cube_dim, cube_dim, cube_dim, 1]
    into [batch_size*cube_dim, cube_dim, cube_dim, 1] and then running CNN on each
    cube slice transforimng it into flat representation and transforimng it back into
    shape [batch_size, cube_dim, flat_dim]. The flat representations are then passed
    through BiLSTM so the interactions are allowed between them. Finally the outputs
    from biLSTM are transformed through linear layer with elu activation.
    """
    def __init__(self):
        """
        """
        super(Encoder, self).__init__()
        self.bi_lstm = Bidirectional(LSTM(128, return_sequences=True), )
        self.conv = InputCNN()
        self.linear = Dense(256, activation='elu')
    def call(self, input):
        """
        Does the transormation
        
        """
        input_shape = input.shape
        new_shape = [input_shape[0]*input_shape[1], input_shape[2], input_shape[3], input_shape[4]]
        input = tf.reshape(input, new_shape)
        output = self.conv(input)
        output = tf.reshape(output, [input_shape[0], input_shape[1], -1])
        output = self.bi_lstm(output)
        return self.linear(output)
    
    
    
    
    
    

class CustomCell(Layer):
    """
    Custom RNN cell based on multilayer LSTMs and attention for doing the ED to smiles translation.
    """
    def __init__(self, **kwargs):
        super(CustomCell, self).__init__()
        self.lstms = [LSTMCell(512), LSTMCell(512), LSTMCell(512)]
        self.linear = Dense(33,)
        self.attn_linear = Dense(256)
    
    def get_initial_state(self, inputs=None, batch_size=None, dtype=tf.float32):
        """
        Gets initial_state for the cell. Zero states for LSTMs
        and zero state for initial attention read.
        Args:
            inputs: a tensor with shape [batch_size, time_step, hidden_dim] 
            batch_size: int with batch size to generate initial state
        """
        initial_state = [lstm.get_initial_state(inputs, batch_size, dtype) for lstm in self.lstms]  
        initial_state += [tf.zeros([batch_size, 256])]
        return initial_state
    
    def set_keys(self, keys):
        """
            Sets the keys to attention mechanism to look up.
        """
        self.keys = keys
    
    def attn(self, output):
        """Luong Attention mechanism as described in 
            https://arxiv.org/pdf/1508.04025.pdf. 
            It looks up the output from lstm in the output generated by CNN-biLSTM encoder
            
        """
        attn_query = self.attn_linear(output)
        attn_query = tf.expand_dims(attn_query, axis=1)
        attn_query = tf.tile(attn_query, [1, self.keys.shape[1], 1])
        # [batch_size, 64, 256, 1]
        attn_query = tf.expand_dims(attn_query, axis=-1)
        
        # [batch_size, 64, 1, 256]
        keys_expand = tf.expand_dims(self.keys, axis=2)
        # computes raw attention weights
        raw_attn = tf.matmul(keys_expand, attn_query)
        raw_attn = tf.squeeze(raw_attn, axis=-1)
        raw_attn = tf.transpose(raw_attn, perm=[0,2,1])
        # normalized attention weights
        attn_weights = tf.nn.softmax(raw_attn, axis=-1)
        # computes the attention output by multiplying encoder states by attention weights 
        attn = tf.matmul(attn_weights, self.keys)
        attn = tf.squeeze(attn, axis=1)
        return attn
        
        
    def call(self, input, states):
        """
        A single pass through the cell computing cell output
        and the next cell state.
        """
        # concatenate the input with attention output from previous time step
        input = tf.concat([input, states[3]], axis=-1)
        # run lstm0 using inputs and state from previous time step
        output_1, new_state_1 = self.lstms[0](input, states[0])
        # produce the attention output using output from lstm0
        attn_1 = self.attn(output_1)
        
        # concatenate output from lstm0 with attention output
        concat_1 = tf.concat([output_1, attn_1], axis=1)
        # run lstm1 using concat_1 and state from previous time step
        output_2, new_state_2 = self.lstms[1](concat_1, states[1])
        # calculate the attention output
        attn_2 = self.attn(output_2)
        #concatenate with output of LSTM1
        concat_2 = tf.concat([output_2, attn_2], axis=1)
        
        # run lstm1 using concat_2 and state from previous time step
        output, new_state_3 = self.lstms[2](concat_2, states[2])
        # calculate the attention output
        attn_3 = self.attn(output)
        #concatenate with output of LSTM1
        concat = tf.concat([output, attn_3], axis=1)
        # final output with tokens probabilty logits
        output = self.linear(concat)
        
        return output, [new_state_1, new_state_2, new_state_3, attn_3]
    
    @property
    def output_size(self):
        return tf.TensorShape([33])
    
    @property
    def state_size(self):
        return [lstm.state_size for lstm in self.lstms] + [[64,256]]

    
    
    
class E2S(Layer):
    """
        Electron density to smiles translation model. It contain encoder which 
        generates 2D representation of 3D electron densities. This 2D representation
        is used as an input to attention mechanism. CustomCell which is responsible 
        for generating logits of tokens probabilty using attention mechanism. To feed
        the smiles embedding layer with size 128 is used.
    """
    def __init__(self, num_outputs, lstm_dim=512, **kwargs):
        """
        Args: 
            num_outputs: int with the size of output layer
            lstm_dim: int with the size of lstms
        """
        super(E2S, self).__init__()
        self.num_outputs = num_outputs
        
        self.encoder = Encoder()
        self.rnn_cell = CustomCell()
        self.rnn = RNN(self.rnn_cell, return_sequences=True)
        self.embedding = tf.keras.layers.Embedding(33, 128)
        #self.scale = tf.Variable([0.], tf.float32)
        
        
    def call(self, densities, smiles):
        """
        Generates logits of next token probabilty for input electron
        density. This is used for training using teacher forcing.
        """
        smiles = self.embedding(smiles)
        encoded_molecule = self.encoder(densities)
        self.rnn_cell.set_keys(encoded_molecule)
        outputs = self.rnn(smiles)
        return outputs
    
        
    def sample(self, batch_size, densities):
        """
        Sample the model to generate smiles strings.
        For each time step the model generates the probabilty distribution
        of tokens which is then sampled using categorical distribution yielding
        single token which is fed back at the next time step after embedding. 
        Args:
            batch_size: int with the batch size to used
            densities: a tensor with electron densities
            with shape[batch_size, 64,64,64,1]
        Returns:
            outputs: a tensor of ints with shape [batch_size, seq_len]
        """
        
        encoded_molecule = self.encoder(densities)
        self.rnn_cell.set_keys(encoded_molecule)
        
        prev_state = self.rnn_cell.get_initial_state(batch_size=batch_size)
        next_input = tf.zeros([batch_size, 1], tf.int64) + 30
        next_input = self.embedding(next_input)[:, 0, :]
        outputs = []
        for i in range(24):
            output, next_state = self.rnn_cell(next_input, prev_state)
            prev_state = next_state
            next_tokens = tf.random.categorical(logits=output, num_samples=1)
            outputs.append(next_tokens)
            next_input = self.embedding(next_tokens) [:, 0, :]
        outputs = tf.stack(outputs)
        outputs = tf.transpose(outputs, perm=[1, 0, 2])
        return outputs[:, :, 0]
     


     
    
def transorm_ed(density):
    
    density = density + 1e-4
    density = tf.math.log(density)
    density = density / tf.math.log(1e-4)
    
    return density

def transorm_back(density):
    
    density = density * tf.math.log(1e-4)
    density = tf.exp(density) - 1e-4
    
    return density



def benchmark_model():
    sampled_smiles_list = []
    original_smiles_list = []
    original_smiles_list__ = []
    for i in tqdm.tqdm(range(20)):
        densities, original_smiles, smiles_idxs = valid_batch_density.__next__()
        densities = transorm_ed(densities)
        sampled_smiles = model.sample(32, densities)
        sampled_smiles = tokenizer.batch_decode_smiles(list(sampled_smiles.numpy()))
        sampled_smiles_list += sampled_smiles
        original_smiles = list(original_smiles.values.numpy())
        original_smiles = [s.decode("utf-8") for s in original_smiles]
        original_smiles_list += original_smiles
        print('\n', original_smiles[0], sampled_smiles[0], sampled_smiles_list[0], '\n')
    return original_smiles_list, sampled_smiles_list



def draw_mol_pair(mol1, mol2):
    
    mol1_ = rdkit.Chem.MolFromSmiles(mol1)
    mol2_ = rdkit.Chem.MolFromSmiles(mol2)
    mols = [mol1_, mol2_]
    img=rdkit.Chem.Draw.MolsToGridImage(mols,molsPerRow=2,subImgSize=(200,200),
                                   legends=[mol1, mol2])
    img.save(mol1+'.png')


def draw_mol_grid(mols_smiles):
    """Draws mols on a grid using rdkit"""
    mols = [rdkit.Chem.MolFromSmiles(m) for m in mols_smiles]
    img=rdkit.Chem.Draw.MolsToGridImage(mols,molsPerRow=4,subImgSize=(200,200),
                                   legends=mols_smiles)
    img.save('grid'+'.png')


def TanimotoSimilarity(mol1, mol2):
    """Computes the tanimoto similarity between two
       molecules and draws them if the smiles are diffrent.
       Return None if smiles are not valid.
       Args:
            mol1, mol2: str with smiles
    """
    mol1_ = rdkit.Chem.MolFromSmiles(mol1)
    mol2_ = rdkit.Chem.MolFromSmiles(mol2)
    if mol1 != mol2:
        draw_mol_pair(mol1, mol2)
    if (mol1_ == None) or (mol2_==None):
        print(mol1, mol2)
        return None
    fp1 = rdkit.Chem.RDKFingerprint(mol1_)
    fp2 = rdkit.Chem.RDKFingerprint(mol2_)
    return rdkit.DataStructs.FingerprintSimilarity(fp1,fp2)

def get_unique_smiles(sm_list):
    """
    Returns unique smiles from the list using rdkit.
    Args: 
        sm_list: list of str with smiles
    Return:
        uniqe_smiles: list of str with uniqe smiles
    """
    mols = [rdkit.Chem.MolFromSmiles(m) for m in sm_list]
    mols = [m for m in mols if m is not None]
    can_smiles = [rdkit.Chem.MolToSmiles(m) for m in mols]
    uniqe_smiles = []
    for m in can_smiles:
        if m not in uniqe_smiles:
            uniqe_smiles.append(m)    
    return uniqe_smiles


def translate_from_single_ed(density, batch_size=32):
    """
    Generates a uniqe smiles for a single electron with shape 
         and draws to image.
    Args:
        density: a tensor with shape [64,64,64,1]
        batch_size: int with the batch size to place on GPU.
    Return:
        sampled_smiles: list of str with uniqe smiles
        
    """
    density = tf.expand_dims(density, axis=0)
    density = tf.tile(density, [batch_size, 1,1,1,1])
    density = transorm_ed(density)
    sampled_smiles = model.sample(32, density)
    sampled_smiles = tokenizer.batch_decode_smiles(list(sampled_smiles.numpy()))
    
    unique_smiles = get_unique_smiles(sampled_smiles)
    #draw_mol_grid(unique_smiles)
    return sampled_smiles


    

def generate_from_file(path, num_smiles=32):
    """
    Load pickled data from disk with ed cubes and 
    do translation for one of the cubes.
    """
    with open(path, 'rb') as pfile:
        cubes = pickle.load(pfile)    
    cubes = cubes.astype(np.float32)
    smiles = translate_from_single_ed(cubes[1], batch_size=num_smiles)
    return smiles


def analysis():
    import textdistance
    import random
    
    original, sampled = benchmark_model()
    sampled_mols = [rdkit.Chem.MolFromSmiles(s) for s in sampled]
    
    valid_sampled_mols= [m for m in sampled_mols if m is not None]
    original_mols = [rdkit.Chem.MolFromSmiles(s) for s in original]
    
    print('Percent valid smiles', len(valid_sampled_mols)/len(sampled_mols))
    print('Sampled ', len(sampled_mols), 'Valid ', len(valid_sampled_mols))
    
    
    #text_similarites = [textdistance.hamming.normalized_similarity(*mols) for mols in zip(sampled, original)]
    
    
    similarities = [TanimotoSimilarity(*mols) for mols in zip(sampled, original)]
    similarities = [s for s in similarities if s is not None]
    
    
    random_mols = [random.sample(original, 2) for i in range(len(sampled))]
    random_similarity = [TanimotoSimilarity(*mols) for mols in random_mols]
    plt.hist(similarities, bins=50, alpha=0.75, color='r')
    #plt.legend('Prediction')
    plt.hist(random_similarity, bins=50, alpha=0.75, color='g')
    #plt.legend('Random prediction')
    plt.xlabel('Tanimoto Similarity', fontsize=20, )
    plt.ylabel('Count', fontsize=20)
    plt.show()
    return sampled, original
    
    




def train_step(densities, smiles):
    """ A single train step. 
    """
    # construct the targets by removing the start token and adding pad token 
    pad = tf.zeros([smiles.shape[0], 1], tf.int64) + 32
    smiles_target = tf.concat([smiles[:, 1:], pad], axis=-1)
    # one hot encode the targets
    targets_onehot = tf.one_hot(smiles_target, depth=33, on_value=1.0, off_value=0.0)
    # computer the gradients
    with tf.GradientTape() as tape:
        # logits of probabilty
        predicted = model(densities, smiles)
        # cross entropy loss between targets and logits
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=predicted, 
                                                       labels=targets_onehot, axis=-1)
        loss = tf.reduce_mean(loss)
        # argmax for getting the inexes of tokens with highest probabilty
        predicted_idxs = tf.argmax(predicted, axis=-1)
        # accuracy - how many smiles strings are exactly the same
        accuracy = tf.reduce_all(tf.equal(predicted_idxs, smiles_target), axis=-1)
        accuracy = tf.cast(accuracy, tf.float32)
        accuracy = tf.reduce_mean(accuracy)
    #calculate the gradients and apply them
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, accuracy, predicted, targets_onehot



if __name__ == '__main__':
    
    # create data iterators
    batch_density = input_fn(['..\\train.tfrecords',],
                            train=True, batch_size=32,
                            num_epochs=100,
                            properties=['smiles'])
    batch_density = iter(batch_density)
    
    
    valid_batch_density = input_fn(['..\\valid.tfrecords',],
                            train=False, batch_size=32,
                            num_epochs=1,
                            properties=['smiles_string', 'smiles',])
    valid_batch_density = iter(valid_batch_density)
    
    
    
    # instantiate the model and optimizer
    model = E2S(num_outputs=33, lstm_dim=512)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    
    tokenizer = Tokenizer('/home/jarek/electrondensity2/data')
    # restore the model
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    checkpoint.restore('/media/group/d22cc883-8622-4ecd-8e46-e3b0850bb89a2/jarek/ed2smiles/ed2smiles.ckpt-72')
    
    
    
    
    
    
    
# # main training loop
# losses = []
# accs = []
# loss_running_avg = 0.0
# acc_running_avg = 0.0

# for i in tqdm.tqdm(range(1, 100001)):
#     densities, _, smiles = batch_density.__next__()
#     #densities = tf.tanh(densities)
#     densities = transorm_ed(densities)
#     #loss, acc, pred, targ = train_step(densities, smiles)
#     #loss_running_avg = 0.99* loss_running_avg + 0.01 * loss.numpy()
#     #acc_running_avg =  0.99* acc_running_avg + 0.01 * acc.numpy()
#     #losses.append(loss.numpy())
#     #accs.append(acc.numpy())
#     #if i % 20 == 0:
#     #   print('\nAverage loss {} Avg accs {} '.format(np.mean(losses), np.mean(accs)))
#       #  print('Average running loss {} Acc {} '.format(loss_running_avg, acc_running_avg))
#       # losses = []
#         #accs = []
#     if i % 10 == 0:
#         samples = tokenizer.batch_decode_smiles(list(model.sample(32, densities).numpy())[:10])
#         print('\n', samples)
#         #pred_ = np.argmax(pred, axis=-1)[:10]
#         #targ_ = np.argmax(targ, axis=-1)[:10]
#         #print(tokenizer.batch_decode_smiles(pred_))
#         print('\n', tokenizer.batch_decode_smiles(smiles.numpy())[:10])
#         #print('\n', tokenizer.batch_decode_smiles(targ_)[:10])
    
#     if i % 3000 == 0:
#         path = checkpoint.save('models\\ed2smiles.ckpt')
#         print('Model saved to', path)





