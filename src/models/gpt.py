##########################################################################################
#
# This model will aim to build a small GPT that will generate smiles.
# Heavily based on the following tutorials:
# https://keras.io/examples/generative/text_generation_with_miniature_gpt
# https://keras.io/examples/audio/transformer_asr/
# Smiles will be inputted as pre-processed by Jarek
#
# Author: Juan Manuel Parrilla Gutierrez (juanma@chem.gla.ac.uk)
#
##########################################################################################

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads, embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"),
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
        embed_dim=256,
        num_heads=2,
        feed_forward_dim=128,
        num_trans_blocks=2,
        vocab_size=33,  # There are 33 different smiles tokens
        maxlen=24,  # max len of the smiles strings is 24, as set by Jarek
    ):
        super().__init__()

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

    def call(self, inputs):
        x = self.embedding_layer(inputs)
        x = self.tblocks(x)
        return self.classifier(x)

    @property
    def metrics(self):
        return [self.loss_metric]

    def train_step(self, batch):
        """Processes one batch inside model.fit()."""
        source = batch["source"]
        source_input = source[:, :-1]
        source_target = source[:, 1:]
        with tf.GradientTape() as tape:
            preds = self(source_input)
            one_hot = tf.one_hot(source_target, depth=self.num_classes)
            mask = tf.math.logical_not(tf.math.equal(source_target, 0))
            loss = self.compiled_loss(one_hot, preds, sample_weight=mask)
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.loss_metric.update_state(loss)
        return {"loss": self.loss_metric.result()}

    def test_step(self, batch):
        source = batch["source"]
        source_input = source[:, :-1]
        source_target = source[:, 1:]
        preds = self(source_input)
        one_hot = tf.one_hot(source_target, depth=self.num_classes)
        mask = tf.math.logical_not(tf.math.equal(source_target, 0))
        loss = self.compiled_loss(one_hot, preds, sample_weight=mask)
        self.loss_metric.update_state(loss)
        return {"loss": self.loss_metric.result()}


def create_train_model():