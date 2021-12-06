##########################################################################################
#
# Variaton over ED2smiles.py, but using both electrostatic potentials (ESP) and ED.
# The idea is to multiply ED by ESP to "decorate" it
# Check that file first, the following script will just extend it a bit.
#
##########################################################################################


import tensorflow as tf
from tensorflow.keras import layers

from src.models.ED2smiles import E2S_Transformer
from src.models.ED2smiles import TransformerEncoder
from src.models.layers import ConvBlock, IdentityBlock


class ED_ESP_Embedding(layers.Layer):
    """We need to transform a 4D into a 2D tensor. The drop of dimensionality in this 
    embedding will be achieved by using strides of 2 in 2 out of the 4 dimensions, until 
    they are 1,1,N,N and then we will squeeze them out.
    Same as V2 above but more convs
    """

    def __init__(self, num_hid=64):
        super().__init__()

        self.conv32 = ConvBlock(
            kernel_size=3, filters=num_hid, stage=0, block='a', strides=[2, 2, 2])
        self.conv16 = ConvBlock(kernel_size=3, filters=num_hid, stage=1, block='a',
                                strides=[2, 2, 1])
        self.conv8 = ConvBlock(kernel_size=3, filters=num_hid, stage=2, block='a',
                               strides=[2, 2, 1])
        self.conv4 = ConvBlock(kernel_size=3, filters=num_hid, stage=3, block='a',
                               strides=[2, 2, 1])
        self.conv2 = ConvBlock(kernel_size=3, filters=num_hid, stage=4, block='a',
                               strides=[2, 2, 1])
        self.conv1 = ConvBlock(kernel_size=3, filters=num_hid, stage=5, block='a',
                               strides=[2, 2, 1])

        self.id32 = IdentityBlock(
            kernel_size=3, filters=num_hid, stage=0, block='a')
        self.id16 = IdentityBlock(
            kernel_size=3, filters=num_hid, stage=1, block='a')
        self.id8 = IdentityBlock(
            kernel_size=3, filters=num_hid, stage=2, block='a')
        self.id4 = IdentityBlock(
            kernel_size=3, filters=num_hid, stage=3, block='a')
        self.id2 = IdentityBlock(
            kernel_size=3, filters=num_hid, stage=4, block='a')
        self.id1 = IdentityBlock(
            kernel_size=3, filters=num_hid, stage=5, block='a')

    def preprocess_esp(self, data):
        """ Preprocesses esps by normalizing it between 0 and 1, and doing a dillation
        so that a data point uses a 10x10x10 area instead of a single cell"""

        # first we will do a dillation, which needs to be done for both + and -
        datap = tf.nn.max_pool3d(data, 10, 1, 'SAME')
        datan = tf.nn.max_pool3d(data*-1, 10, 1, 'SAME')
        data = datap + datan*-1

        # I have pre-calculated that data goes between -0.265 and 0.3213
        # with this division it will be roughly between -1 and 1
        data = data / 0.33
        return data

    def call(self, x):
        # First we do the pre-processing
        x_ed = tf.tanh(x[0])
        x_esp = self.preprocess_esp(x[1])
        # decorate ED
        x = x_ed * x_esp
        # put it between 0 and 1 (now it is between -1 and 1)
        x = (x+1) * 0.5
        # now from 64,64,64,1 to 32,32,32,num_hid
        x = self.conv32(x)
        x = self.id32(x)
        # now from 32,32,32,num_hid to 16,16,32,num_hid
        x = self.conv16(x)
        x = self.id16(x)
        # now from 16,16,32,1 to 8,8,32,num_hid
        x = self.conv8(x)
        x = self.id8(x)
        # now from 8,8,32,1 to 4,4,32,num_hid
        x = self.conv4(x)
        x = self.id4(x)
        # now from 4,4,32,num_hid to 2,2,32,num_hid
        x = self.conv2(x)
        x = self.id2(x)
        # now from 2,2,32,num_hid to 1,1,32,num_hid
        x = self.conv1(x)
        x = self.id1(x)

        return tf.squeeze(x, [1, 2])


class ESP2S_Transformer(E2S_Transformer):
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
        super().__init__(num_hid, num_head, num_feed_forward, target_maxlen,
                         num_layers_enc, num_layers_dec, num_classes)

        self.enc_input = ED_ESP_Embedding(num_hid=num_hid)

        self.encoder = tf.keras.Sequential(
            [self.enc_input]
            + [
                TransformerEncoder(num_hid, num_head, num_feed_forward)
                for _ in range(num_layers_enc)
            ],
            name="transformer_encoder"
        )


    def train_step(self, batch):
        """Processes one batch inside model.fit()."""
        source = [batch[0], batch[1]]  # position 0 has ED position 1 contains ESP
        target = batch[2]  # position 2 contains the tokenized smiles
        dec_input = target[:, :-1]
        dec_target = target[:, 1:]
        with tf.GradientTape() as tape:
            preds = self([source, dec_input])
            one_hot = tf.one_hot(dec_target, depth=self.num_classes)
            mask = tf.math.logical_not(
                tf.math.equal(dec_target, 32))  # 32 is 'NULL'
            loss = self.compiled_loss(one_hot, preds, sample_weight=mask)
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.loss_metric.update_state(loss)
        return {"loss": self.loss_metric.result()}


    def test_step(self, batch):
        source = [batch[0], batch[1]]  # position 0 has ED position 1 contains ESP
        target = batch[2]  # position 2 contains the tokenized smiles
        dec_input = target[:, :-1]
        dec_target = target[:, 1:]
        preds = self([source, dec_input])
        one_hot = tf.one_hot(dec_target, depth=self.num_classes)
        mask = tf.math.logical_not(
            tf.math.equal(dec_target, 32))  # 32 is 'NULL'
        loss = self.compiled_loss(one_hot, preds, sample_weight=mask)
        self.loss_metric.update_state(loss)
        return {"loss": self.loss_metric.result()}


    def generate(self, batch, target_start_token_idx=30, startid=0, greedy=True):
        """Performs inference over one batch of inputs using greedy decoding."""

        source = [batch[0], batch[1]]  # position 0 has ED position 1 contains ESP
        smiles = batch[2]
        bs = tf.shape(source[0])[0]
        enc = self.encoder(source)

        if startid == 0:
            dec_input = tf.ones((bs, 1), dtype=tf.int32) * \
                target_start_token_idx
            maxlen = self.target_maxlen - 1
        else:
            dec_input = tf.cast(smiles[:, :startid], dtype=tf.int32)
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
