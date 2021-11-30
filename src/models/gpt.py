##########################################################################################
#
# This model will aim to build a small GPT that will generate smiles.
# Heavily based on the following tutorials:
# https://keras.io/examples/generative/text_generation_with_miniature_gpt
# https://keras.io/examples/audio/transformer_asr/
# Smiles will be inputted as pre-processed by Jarek.
# I am not adding comments to most of the stuff that is directly copy-pasted. Check the
# tutorials above and then you will easily understand it.
#
# Author: Juan Manuel Parrilla Gutierrez (juanma@chem.gla.ac.uk)
#
##########################################################################################

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint

import os
import pickle

from src.utils.callbacks import DisplayOutputs, CustomSchedule


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.3):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads, embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="gelu"),
             layers.Dense(embed_dim), ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def causal_attention_mask(self, batch_size, n_dest, n_src, dtype):
        """Masks the upper half of the dot product matrix in self attention.

        This prevents flow of information from future tokens to current token.
        1's in the lower triangle, counting from the lower right corner.
        """
        i = tf.range(n_dest)[:, None]
        j = tf.range(n_src)
        m = i >= j - n_src + n_dest
        mask = tf.cast(m, dtype)
        mask = tf.reshape(mask, [1, n_dest, n_src])
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1),
             tf.constant([1, 1], dtype=tf.int32)], 0
        )
        return tf.tile(mask, mult)

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        causal_mask = self.causal_attention_mask(
            batch_size, seq_len, seq_len, tf.bool)
        attention_output = self.att(inputs, inputs, attention_mask=causal_mask)
        attention_output = self.dropout1(attention_output)
        out1 = self.layernorm1(inputs + attention_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


class GPT(keras.Model):
    def __init__(
        self,
        tokenizer,
        embed_dim=256,
        num_heads=2,
        feed_forward_dim=512,
        num_trans_blocks=1,
        vocab_size=33,  # There are 33 different smiles tokens
        maxlen=24,  # max len of the smiles strings is 24, as set by Jarek
    ):
        super().__init__()

        self.tokenizer = tokenizer

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.feed_forward_dim = feed_forward_dim
        self.num_trans_blocks = num_trans_blocks
        self.vocab_size = vocab_size
        self.maxlen = maxlen

        self.loss_metric = keras.metrics.Mean(name="loss")

        self.embedding_layer = TokenAndPositionEmbedding(
            maxlen=maxlen, vocab_size=vocab_size, embed_dim=embed_dim
        )

        self.tblocks = keras.Sequential(
            [
                TransformerBlock(embed_dim, num_heads, feed_forward_dim)
                for _ in range(num_trans_blocks)
            ]
        )

        self.classifier = layers.Dense(vocab_size)

    def compile_model(self):

        learning_rate = CustomSchedule(
            init_lr=0.0001,
            lr_after_warmup=0.0005,
            final_lr=0.0001,
            warmup_epochs=100,
            decay_epochs=500,
            steps_per_epoch=1882,  # calculated beforehand, going through iter takes time
        )

        loss_fn = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True, label_smoothing=0.1,
        )

        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.compile(optimizer=optimizer, loss=loss_fn)

    def call(self, inputs):
        x = self.embedding_layer(inputs)
        x = self.tblocks(x)
        return self.classifier(x)

    @property
    def metrics(self):
        return [self.loss_metric]

    def save_build(self, folder):
        """Saves the config before the training starts. The model itself will be saved
        later on using keras checkpoints.

        Args:
            folder: Where to save the config parameters
        """
        if not os.path.exists(folder):
            os.makedirs(folder)
            os.makedirs(os.path.join(folder, 'weights'))

        with open(os.path.join(folder, 'params.pkl'), 'wb') as f:
            pickle.dump([
                self.embed_dim,
                self.num_heads,
                self.feed_forward_dim,
                self.num_trans_blocks,
                self.vocab_size,
                self.maxlen,
            ], f)

    def train_step(self, batch):
        """Processes one batch inside model.fit()."""
        source = batch[1]  # [0] are densities which we don't here. [1] are token smiles
        source_input = source[:, :-1]
        source_target = source[:, 1:]
        with tf.GradientTape() as tape:
            preds = self(source_input)
            one_hot = tf.one_hot(source_target, depth=self.vocab_size)
            mask = tf.math.logical_not(tf.math.equal(source_target, 32))  # 32 is NULL
            loss = self.compiled_loss(one_hot, preds, sample_weight=mask)
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.loss_metric.update_state(loss)
        return {"loss": self.loss_metric.result()}

    def test_step(self, batch):
        # [0] are densities which we don't here. [1] are token smiles
        source = batch[1]
        source_input = source[:, :-1]
        source_target = source[:, 1:]
        preds = self(source_input)
        one_hot = tf.one_hot(source_target, depth=self.vocab_size)
        mask = tf.math.logical_not(tf.math.equal(source_target, 32))  # 32 is NULL
        loss = self.compiled_loss(one_hot, preds, sample_weight=mask)
        self.loss_metric.update_state(loss)
        return {"loss": self.loss_metric.result()}

    def probabilistic_sampling(self, logits):
        """ Performs probabilistic selection of logits, instead of argmax."""
        logits, indices = tf.math.top_k(logits, k=10, sorted=True)
        indices = np.asarray(indices).astype("int32")
        preds = tf.keras.activations.softmax(tf.expand_dims(logits, 0))[0]
        preds = np.asarray(preds).astype("float32")
        tokens = []
        for i in range(len(preds)):
            token = np.random.choice(indices[i][-1], p=preds[i][-1])
            tokens.append(token)
        return np.array(tokens).reshape([len(tokens),-1])

    def generate(self, source, target_start_token_idx=30, startid=0, greedy=True):
        """Performs inference over one batch of inputs using greedy decoding."""
        bs = tf.shape(source)[0]

        if startid == 0:
            gpt_input = tf.ones((bs, 1), dtype=tf.int32) * target_start_token_idx
        else:
            gpt_input = tf.cast(source[:,:startid], dtype=tf.int32)

        gpt_logits = []
        for _ in range(self.maxlen - startid - 1):
            logits = self(gpt_input)
            if greedy:
                logits = tf.argmax(logits, axis=-1, output_type=tf.int32)
            else:
                logits = self.probabilistic_sampling(logits)
            last_logit = tf.expand_dims(logits[:, -1], axis=-1)
            gpt_logits.append(last_logit)
            gpt_input = tf.concat([gpt_input, last_logit], axis=-1)
        return gpt_input

    def generate_from_smiles(self, smiles, greedy=True):
        """Given a string seed "smiles", it will generate from there until STOP.
        For example if smiles is "CO" it might generate something like "COCC[STOP]"

        Args:
            smiles (string): a smiles string in the usual smiles format
            greedy: If true will pick the next token based on argmax, else random sampling
        """

        # start converting the smiles into tokens using the tokenizer
        encoded_smiles = self.tokenizer.encode_smiles(smiles)
        encoded_smiles = np.array([encoded_smiles])
        # use model to generate smiles from seed
        generated_smiles = self.generate(encoded_smiles, startid=len(smiles)+1, greedy=greedy)
        # transform for decoding
        generated_smiles = list(generated_smiles[0].numpy())
        generated_smiles = [str(e) for e in generated_smiles]

        return self.tokenizer.decode_smiles(generated_smiles)

    def train(
        self, train_dataset, valid_dataset, epochs, run_folder, initial_epoch=0, 
        print_every_n_epochs=1
    ):

        display_cb = DisplayOutputs(
            next(valid_dataset.dataset_iter), self.tokenizer.num2token, 
            run_folder=run_folder,
        )

        checkpoint_filepath = os.path.join(
            run_folder, "weights/weights-{epoch:03d}-{loss:.3f}-{val_loss:.3f}.h5")
        checkpoint1 = ModelCheckpoint(
            checkpoint_filepath, save_weights_only=True)
        checkpoint2 = ModelCheckpoint(
            os.path.join(run_folder, 'weights/weights.h5'),
            save_weights_only=True)

        callbacks_list = [checkpoint1, checkpoint2, display_cb]

        self.fit(
            train_dataset.dataset, validation_data=valid_dataset.dataset,
            epochs=epochs, initial_epoch=initial_epoch, callbacks=callbacks_list
        )
