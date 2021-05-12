##########################################################################################
#
# This model will aim to build a Transformer to convert from electron density to smiles.
# Heavily based on the following tutorial:
# https://keras.io/examples/audio/transformer_asr/
# Smiles and electron densities will be inputted as pre-processed by Jarek.
# I am not adding comments to most of the stuff that is directly copy-pasted. Check the
# tutorials above and then you will easily understand it.
#
# Author: Juan Manuel Parrilla Gutierrez (juanma@chem.gla.ac.uk)
#
##########################################################################################


import os
import pickle
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint

from src.utils import transform_ed
from src.utils.callbacks import DisplayOutputs, CustomSchedule
from src.models.layers import ConvBlock, IdentityBlock, DropDimension


class TokenEmbedding(layers.Layer):
    def __init__(self, num_vocab=1000, maxlen=100, num_hid=64):
        super().__init__()
        self.emb = tf.keras.layers.Embedding(num_vocab, num_hid)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=num_hid)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        x = self.emb(x)
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        return x + positions


class ElectronDensityEmbedding(layers.Layer):
    def __init__(self, num_hid=64):
        super().__init__()

        self.conv32 = ConvBlock(kernel_size=3, filters=num_hid, stage=0, block='a', strides=2)
        self.conv21 = ConvBlock(kernel_size=3, filters=num_hid, stage=1, block='a', strides=1)
        self.conv11 = ConvBlock(kernel_size=3, filters=num_hid, stage=2, block='a', strides=1)

        self.id32 = IdentityBlock(kernel_size=3, filters=num_hid, stage=0, block='a')
        self.id21 = IdentityBlock(kernel_size=3, filters=num_hid, stage=1, block='a')
        self.id11 = IdentityBlock(kernel_size=3, filters=num_hid, stage=2, block='a')

        self.dd1 = DropDimension()
        self.dd2 = DropDimension()

    def call(self, x):
        # First we do the pre-processing Jarek was doing
        x = tf.tanh(x)
        x = transform_ed(x)
        # now from 3D to 2D
        x = self.conv32(x)
        x = self.id32(x)
        x = self.dd1(x)
        # from 2D to 1D
        x = self.conv21(x)
        x = self.id21(x)
        x = self.dd2(x)
        # keep 1D but to embedding size
        x = self.conv11(x)
        x = self.id11(x)

        return x


class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [
                layers.Dense(feed_forward_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, dropout_rate=0.1):
        super().__init__()
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)
        self.self_att = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.enc_att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.self_dropout = layers.Dropout(0.5)
        self.enc_dropout = layers.Dropout(0.1)
        self.ffn_dropout = layers.Dropout(0.1)
        self.ffn = tf.keras.Sequential(
            [
                layers.Dense(feed_forward_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )

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
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], 0
        )
        return tf.tile(mask, mult)

    def call(self, enc_out, target):
        input_shape = tf.shape(target)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        causal_mask = self.causal_attention_mask(batch_size, seq_len, seq_len, tf.bool)
        target_att = self.self_att(target, target, attention_mask=causal_mask)
        target_norm = self.layernorm1(target + self.self_dropout(target_att))
        enc_out = self.enc_att(target_norm, enc_out)
        enc_out_norm = self.layernorm2(self.enc_dropout(enc_out) + target_norm)
        ffn_out = self.ffn(enc_out_norm)
        ffn_out_norm = self.layernorm3(enc_out_norm + self.ffn_dropout(ffn_out))
        return ffn_out_norm


class E2S_Transformer(tf.keras.Model):
    def __init__(
        self,
        num_hid=64,
        num_head=2,
        num_feed_forward=128,
        target_maxlen=24,  # max len of the smiles strings is 24, as set by Jarek
        num_layers_enc=4,
        num_layers_dec=1,
        num_classes=33,  # There are 33 different smiles tokens
    ):
        super().__init__()
        self.loss_metric = tf.keras.metrics.Mean(name="loss")
        self.num_hid = num_hid
        self.num_head = num_head
        self.num_feed_forward = num_feed_forward
        self.num_layers_enc = num_layers_enc
        self.num_layers_dec = num_layers_dec
        self.target_maxlen = target_maxlen
        self.num_classes = num_classes

        self.enc_input = ElectronDensityEmbedding(num_hid=num_hid)
        self.dec_input = TokenEmbedding(
            num_vocab=num_classes, maxlen=target_maxlen, num_hid=num_hid
        )

        self.encoder = tf.keras.Sequential(
            [self.enc_input]
            + [
                TransformerEncoder(num_hid, num_head, num_feed_forward)
                for _ in range(num_layers_enc)
            ]
        )

        for i in range(num_layers_dec):
            setattr(
                self,
                f"dec_layer_{i}",
                TransformerDecoder(num_hid, num_head, num_feed_forward),
            )

        self.classifier = layers.Dense(num_classes)

    def decode(self, enc_out, target):
        y = self.dec_input(target)
        for i in range(self.num_layers_dec):
            y = getattr(self, f"dec_layer_{i}")(enc_out, y)
        return y

    def call(self, inputs):
        source = inputs[0]
        target = inputs[1]
        x = self.encoder(source)
        y = self.decode(x, target)
        return self.classifier(y)

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

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.compile(optimizer=optimizer, loss=loss_fn)

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
                self.num_hid,
                self.num_head,
                self.num_feed_forward,
                self.num_layers_enc,
                self.num_layers_dec,
                self.target_maxlen,
                self.num_classes,
            ], f)

    @property
    def metrics(self):
        return [self.loss_metric]

    def train_step(self, batch):
        """Processes one batch inside model.fit()."""
        source = batch[0]  # position 0 contains the electron density
        target = batch[1]  # position 1 contains the tokenized smiles
        dec_input = target[:, :-1]
        dec_target = target[:, 1:]
        with tf.GradientTape() as tape:
            preds = self([source, dec_input])
            one_hot = tf.one_hot(dec_target, depth=self.num_classes)
            mask = tf.math.logical_not(tf.math.equal(dec_target, 32))  # 32 is 'NULL'
            loss = self.compiled_loss(one_hot, preds, sample_weight=mask)
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.loss_metric.update_state(loss)
        return {"loss": self.loss_metric.result()}

    def test_step(self, batch):
        source = batch[0]  # position 0 contains the electron density
        target = batch[1]  # position 1 contains the tokenized smiles
        dec_input = target[:, :-1]
        dec_target = target[:, 1:]
        preds = self([source, dec_input])
        one_hot = tf.one_hot(dec_target, depth=self.num_classes)
        mask = tf.math.logical_not(tf.math.equal(dec_target, 32))  # 32 is 'NULL'
        loss = self.compiled_loss(one_hot, preds, sample_weight=mask)
        self.loss_metric.update_state(loss)
        return {"loss": self.loss_metric.result()}

    def generate(self, batch, target_start_token_idx=30, startid=0):
        """Performs inference over one batch of inputs using greedy decoding."""

        source = batch[0]  # electron densities
        smiles = batch[1]
        bs = tf.shape(source)[0]
        enc = self.encoder(source)
        
        if startid == 0:
            dec_input = tf.ones((bs, 1), dtype=tf.int32) * target_start_token_idx
        else:
            dec_input = tf.cast(smiles[:,:startid], dtype=tf.int32)

        dec_logits = []
        for i in range(self.target_maxlen - startid - 1):
            dec_out = self.decode(enc, dec_input)
            logits = self.classifier(dec_out)
            logits = tf.argmax(logits, axis=-1, output_type=tf.int32)
            last_logit = tf.expand_dims(logits[:, -1], axis=-1)
            dec_logits.append(last_logit)
            dec_input = tf.concat([dec_input, last_logit], axis=-1)
        return dec_input

    def train(
        self, train_dataset, valid_dataset, epochs, run_folder, tokenizer,
        initial_epoch=0, print_every_n_epochs=1
    ):

        display_cb = DisplayOutputs(
            next(valid_dataset.dataset_iter), tokenizer.num2token, run_folder=run_folder,
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

