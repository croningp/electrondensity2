##########################################################################################
#
# Variaton over ED_ESP2smiles.py, and but obtaining SELFIES instead of SMILES.
# Check that file first, the following script will just change it a bit.
#
# author: Juan Manuel Parrilla - juanma.parrilla@gcu.ac.uk
#
##########################################################################################


import tensorflow as tf
from tensorflow.keras import layers

from src.models.ED2smiles import E2S_Transformer
from src.models.ED2smiles import TransformerEncoder
from src.models.ED_ESP2smiles import ED_ESP_Embedding
from src.models.layers import ConvBlock, IdentityBlock



class ED_ESP2SF_Transformer(E2S_Transformer):
    def __init__(
        self,
        num_hid=64,
        num_head=2,
        num_feed_forward=128,
        target_maxlen=23,  # max len of the selfies strings in the database is 23
        num_layers_enc=4,
        num_layers_dec=1,
        num_classes=33,  # There are 33 different selfies tokens
    ):
        super().__init__(num_hid, num_head, num_feed_forward, target_maxlen,
                         num_layers_enc, num_layers_dec, num_classes)

        self.enc_input = ED_ESP_Embedding(num_hid=self.num_hid)

        self.encoder = tf.keras.Sequential(
            [
                TransformerEncoder(self.num_hid, self.num_head, self.num_feed_forward)
                for _ in range(self.num_layers_enc)
            ],
            name="transformer_encoder"
        )

    def call(self, inputs):
        source = inputs[0]
        target = inputs[1]
        x = self.enc_input(source)
        x = self.encoder(x)
        y = self.decode(x, target)
        return self.classifier(y)


    def train_step(self, batch):
        """Processes one batch inside model.fit()."""
        source = [batch[0], batch[1]]  # position 0 has ED position 1 contains ESP
        target = batch[2]  # position 2 contains the tokenized selfies
        dec_input = target[:, :-1]
        dec_target = target[:, 1:]
        with tf.GradientTape() as tape:
            preds = self([source, dec_input])
            one_hot = tf.one_hot(dec_target, depth=self.num_classes)
            mask = tf.math.logical_not(
                tf.math.equal(dec_target, 27))  # 27 is '[nop]'
            loss = self.compiled_loss(one_hot, preds, sample_weight=mask)
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.loss_metric.update_state(loss)
        return {"loss": self.loss_metric.result()}


    def test_step(self, batch):
        source = [batch[0], batch[1]]  # position 0 has ED position 1 contains ESP
        target = batch[2]  # position 2 contains the tokenized selfies
        dec_input = target[:, :-1]
        dec_target = target[:, 1:]
        preds = self([source, dec_input])
        one_hot = tf.one_hot(dec_target, depth=self.num_classes)
        mask = tf.math.logical_not(
            tf.math.equal(dec_target, 27))  # 27 is '[nop]'
        loss = self.compiled_loss(one_hot, preds, sample_weight=mask)
        self.loss_metric.update_state(loss)
        return {"loss": self.loss_metric.result()}


    def generate(self, batch, target_start_token_idx=14, startid=0, greedy=True):
        """Performs inference over one batch of inputs using greedy decoding."""

        source = [batch[0], batch[1]]  # position 0 has ED position 1 contains ESP
        selfies = batch[2]
        bs = tf.shape(source[0])[0]
        enc = self.enc_input(source)
        enc = self.encoder(enc)

        if startid == 0:
            dec_input = tf.ones((bs, 1), dtype=tf.int32) * \
                target_start_token_idx
            maxlen = self.target_maxlen - 1
        else:
            dec_input = tf.cast(selfies[:, :startid], dtype=tf.int32)
            maxlen = self.target_maxlen - startid

        dec_logits = []
        for _ in range(maxlen):
            dec_out = self.decode(enc, dec_input)
            logits = self.classifier(dec_out)
            if greedy:
                logits = tf.argmax(logits, axis=-1, output_type=tf.int32)
            else:
                logits = self.probabilistic_sampling(logits)
            last_logit = tf.expand_dims(logits[:, -1], axis=-1)
            dec_logits.append(last_logit)
            dec_input = tf.concat([dec_input, last_logit], axis=-1)
        return dec_input
