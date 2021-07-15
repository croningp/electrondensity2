##########################################################################################
#
# This code is implements pix2pix to transform from electron densities to electro static
# potentials. Check the paper (https://arxiv.org/abs/1611.07004).
# There are a lot of implementations. This code adapts the one from:
# https://machinelearningmastery.com/
#
# Author: Juan Manuel Parrilla Gutierrez (juanma@chem.gla.ac.uk)
#
##########################################################################################

# example of pix2pix gan for satellite to map image-to-image translation
from numpy import load
from numpy import zeros
from numpy import ones
from numpy.random import randint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, Conv3DTranspose, LeakyReLU
from tensorflow.keras.layers import Activation, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from matplotlib import pyplot

from src.models.layers import identity_block, conv_block


class Pix2Pix():

    def __init__(
        self, image_shape,
        disc_conv_filters, disc_conv_kernel_size, disc_conv_strides,
        gen_conv_t_filters, gen_conv_t_kernel_size, gen_conv_t_strides,
        use_batch_norm=False, use_dropout=False,
    ):

        self.name = 'pix2pix'

        self.image_shape = image_shape
        self.disc_conv_filters = disc_conv_filters
        self.disc_conv_kernel_size = disc_conv_kernel_size
        self.disc_conv_strides = disc_conv_strides
        self.gen_conv_t_filters = gen_conv_t_filters
        self.gen_conv_t_kernel_size = gen_conv_t_kernel_size
        self.gen_conv_t_strides = gen_conv_t_strides

        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout

        self.n_layers_disc = len(disc_conv_filters)
        self.n_layers_gen = len(gen_conv_t_filters)

        self._build()

    # define the discriminator model
    def define_discriminator(self):
        # weight initialization
        init = RandomNormal(stddev=0.02)
        # source image input
        in_src_image = Input(shape=self.image_shape)
        # target image input
        in_target_image = Input(shape=self.image_shape)
        # concatenate images channel-wise
        merged = Concatenate()([in_src_image, in_target_image])
        # x = merged

        # d = Conv3D(64, (4, 4), strides=(2, 2), padding='same',
        #            kernel_initializer=init)(merged)
        # d = LeakyReLU(alpha=0.2)(d)

        # for i in range(1, self.n_layers_disc):
        #     # just fetch the parameters in a variable so it doesn't get super long
        #     filters = self.disc_conv_filters[i]
        #     kernel_size = self.disc_conv_kernel_size[i]
        #     strides = self.disc_conv_strides[i]

        #     # First one doesnt do batchnorm, we won't do residual
        #     if i == 0:
        #         x = Conv3D(filters, kernel_size, strides=strides,
        #                    padding='same', kernel_initializer=init)(x)
        #         x = LeakyReLU(alpha=0.2)(x)
        #     else:
        #         x = conv_block(x, kernel_size, filters, stage=i,
        #                        block='a', strides=strides, activaton='l')

        # C64
        d = Conv3D(64, 4, strides=2, padding='same', kernel_initializer=init)(merged)
        d = LeakyReLU(alpha=0.2)(d)
        # C128
        d = Conv3D(128, 4, strides=2, padding='same', kernel_initializer=init)(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        # C256
        d = Conv3D(256, 4, strides=2, padding='same', kernel_initializer=init)(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        # C512
        d = Conv3D(512, 4, strides=2, padding='same', kernel_initializer=init)(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        # second last output layer
        d = Conv3D(512, 4, padding='same', kernel_initializer=init)(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)

        # patch output
        d = Conv3D(1, 4, padding='same', kernel_initializer=init)(d)
        patch_out = Activation('sigmoid')(d)
        # define model
        model = Model([in_src_image, in_target_image], patch_out)
        # compile model
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy',
                      optimizer=opt, loss_weights=[0.5])
        return model

    # define an encoder block
    def define_encoder_block(self, layer_in, n_filters, batchnorm=True):
        # weight initialization
        init = RandomNormal(stddev=0.02)
        # add downsampling layer
        g = Conv3D(n_filters, 4, strides=2, padding='same', kernel_initializer=init)(layer_in)
        # conditionally add batch normalization
        if batchnorm:
            g = BatchNormalization()(g, training=True)
        # leaky relu activation
        g = LeakyReLU(alpha=0.2)(g)
        return g

    # define a decoder block
    def decoder_block(layer_in, skip_in, n_filters, dropout=True):
        # weight initialization
        init = RandomNormal(stddev=0.02)
        # add upsampling layer
        g = Conv3DTranspose(n_filters, 4, strides=2, padding='same', kernel_initializer=init)(layer_in)
        # add batch normalization
        g = BatchNormalization()(g, training=True)
        # conditionally add dropout
        if dropout:
            g = Dropout(0.5)(g, training=True)
        # merge with skip connection
        g = Concatenate()([g, skip_in])
        # relu activation
        g = Activation('relu')(g)
        return g

    # define the standalone generator model
    def define_generator(self):
        # weight initialization
        init = RandomNormal(stddev=0.02)
        # image input
        in_image = Input(shape=self.image_shape)
        # encoder model
        e1 = self.define_encoder_block(in_image, 64, batchnorm=False)
        e2 = self.define_encoder_block(e1, 128)
        e3 = self.define_encoder_block(e2, 256)
        e4 = self.define_encoder_block(e3, 512)
        e5 = self.define_encoder_block(e4, 512)
        e6 = self.define_encoder_block(e5, 512)
        e7 = self.define_encoder_block(e6, 512)
        # bottleneck, no batch norm and relu
        b = Conv3D(512, 4, strides=2, padding='same', kernel_initializer=init)(e7)
        b = Activation('relu')(b)
        # decoder model
        d1 = self.decoder_block(b, e7, 512)
        d2 = self.decoder_block(d1, e6, 512)
        d3 = self.decoder_block(d2, e5, 512)
        d4 = self.decoder_block(d3, e4, 512, dropout=False)
        d5 = self.decoder_block(d4, e3, 256, dropout=False)
        d6 = self.decoder_block(d5, e2, 128, dropout=False)
        d7 = self.decoder_block(d6, e1, 64, dropout=False)
        # output
        g = Conv3DTranspose(3, 4, strides=2, padding='same', kernel_initializer=init)(d7)
        out_image = Activation('tanh')(g)
        # define model
        model = Model(in_image, out_image)
        return model

    
    # define the combined generator and discriminator model, for updating the generator
    def define_gan(self, g_model, d_model):
        # make weights in the discriminator not trainable
        for layer in d_model.layers:
            if not isinstance(layer, BatchNormalization):
                layer.trainable = False
        # define the source image
        in_src = Input(shape=self.image_shape)
        # connect the source image to the generator input
        gen_out = g_model(in_src)
        # connect the source input and generator output to the discriminator input
        dis_out = d_model([in_src, gen_out])
        # src image as input, generated image and classification output
        model = Model(in_src, [dis_out, gen_out])
        # compile model
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1,100])
        return model

    # load and prepare training images
    def load_real_samples(self, filename):
        # load compressed arrays
        data = load(filename)
        # unpack arrays
        X1, X2 = data['arr_0'], data['arr_1']
        # scale from [0,255] to [-1,1]
        X1 = (X1 - 127.5) / 127.5
        X2 = (X2 - 127.5) / 127.5
        return [X1, X2]

    # select a batch of random samples, returns images and target
    def generate_real_samples(dataset, n_samples, patch_shape):
        # unpack dataset
        trainA, trainB = dataset
        # choose random instances
        ix = randint(0, trainA.shape[0], n_samples)
        # retrieve selected images
        X1, X2 = trainA[ix], trainB[ix]
        # generate 'real' class labels (1)
        y = ones((n_samples, patch_shape, patch_shape, 1))
        return [X1, X2], y

    # generate a batch of images, returns images and targets
    def generate_fake_samples(g_model, samples, patch_shape):
        # generate fake instance
        X = g_model.predict(samples)
        # create 'fake' class labels (0)
        y = zeros((len(X), patch_shape, patch_shape, 1))
        return X, y

    # generate samples and save as a plot and save the model
    def summarize_performance(self, step, g_model, dataset, n_samples=3):
        # select a sample of input images
        [X_realA, X_realB], _ = self.generate_real_samples(dataset, n_samples, 1)
        # generate a batch of fake samples
        X_fakeB, _ = self.generate_fake_samples(g_model, X_realA, 1)
        # scale all pixels from [-1,1] to [0,1]
        X_realA = (X_realA + 1) / 2.0
        X_realB = (X_realB + 1) / 2.0
        X_fakeB = (X_fakeB + 1) / 2.0
        # plot real source images
        for i in range(n_samples):
            pyplot.subplot(3, n_samples, 1 + i)
            pyplot.axis('off')
            pyplot.imshow(X_realA[i])
        # plot generated target image
        for i in range(n_samples):
            pyplot.subplot(3, n_samples, 1 + n_samples + i)
            pyplot.axis('off')
            pyplot.imshow(X_fakeB[i])
        # plot real target image
        for i in range(n_samples):
            pyplot.subplot(3, n_samples, 1 + n_samples*2 + i)
            pyplot.axis('off')
            pyplot.imshow(X_realB[i])
        # save plot to file
        filename1 = 'plot_%06d.png' % (step+1)
        pyplot.savefig(filename1)
        pyplot.close()
        # save the generator model
        filename2 = 'model_%06d.h5' % (step+1)
        g_model.save(filename2)
        print('>Saved: %s and %s' % (filename1, filename2))

    # train pix2pix models
    def train(self, d_model, g_model, gan_model, dataset, n_epochs=100, n_batch=1):
        # determine the output square shape of the discriminator
        n_patch = d_model.output_shape[1]
        # unpack dataset
        trainA, trainB = dataset
        # calculate the number of batches per training epoch
        bat_per_epo = int(len(trainA) / n_batch)
        # calculate the number of training iterations
        n_steps = bat_per_epo * n_epochs
        # manually enumerate epochs
        for i in range(n_steps):
            # select a batch of real samples
            [X_realA, X_realB], y_real = self.generate_real_samples(dataset, n_batch, n_patch)
            # generate a batch of fake samples
            X_fakeB, y_fake = self.generate_fake_samples(g_model, X_realA, n_patch)
            # update discriminator for real samples
            d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
            # update discriminator for generated samples
            d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
            # update the generator
            g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
            # summarize performance
            print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, d_loss1, d_loss2, g_loss))
            # summarize model performance
            if (i+1) % (bat_per_epo * 10) == 0:
                self.summarize_performance(i, g_model, dataset)