'''
Convolutional autoencoder model
Arturo Pardo, 2019
'''

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime
from tqdm import tqdm
import h5py
import csv

import os
dir_path = os.path.dirname(os.path.abspath(__file__))

import ai.common_ops as common

'''
Convolutional Info-VAE (a.k.a. MMD-VAE) with deterministic bottleneck
'''
class ConvolutionalInfoVAE:
    def __init__(self, n_conv, fs, nstride, n_h, n_z, x_dims, lr=0.0001, optim='Adam', actype='lrelu', mbs=16,
                 cost_function='MSE', fixed_lr=False, regeps=0.1, quiet = 0, debug = 0,
                 fresh_init=1, name='model', reset_default_graph=1, write_path='../../../output/network_tests'):

        # Is this a new network or a pretrained one?
        self.fresh_init = fresh_init
        self.name = name

        # Quiet and debug flags -- note, these might not work for this older model
        self.quiet = quiet
        self.debug = debug

        # Order to reset default graph when instanced
        self.reset_default_graph = reset_default_graph

        # Quiet/debug flags
        self.quiet = quiet
        self.debug = debug

        # First iteration detector
        if self.fresh_init == 1:
            self.first_iter = 1
        else:
            self.first_iter = 0

        # Init_parameters
        self.n_conv = n_conv        # Convolutional layers and their sizes (in neurons)
        self.fs = fs                # Filter size per layer (we will assume square filters)
        self.nstride = nstride            # Stride per layer (usually just ones)
        self.n_h = n_h              # Feedforward layers and their sizes (in neurons)
        self.n_z = n_z              # Number of dimensions of variational latent space

        # Reversed params for decoder
        self.n_conv_d = np.flip(self.n_conv)
        self.fs_d = np.flip(self.fs)
        self.nstride_d = np.flip(self.nstride)
        self.n_h_d = np.flip(self.n_h)

        print(self.n_conv)
        print(self.n_conv_d)

        # Image properties
        self.width = x_dims[0]
        self.height = x_dims[1]
        self.channels = x_dims[2]

        # Optimizer choice
        self.optim = optim

        # Learning rate, cost function, learning rate
        self.lr = lr
        self.cost_function = cost_function
        self.fixed_lr = fixed_lr

        self.regeps = regeps

        # Minibatch size
        self.mbs = int(mbs)

        # Activation function
        self.actype = actype

        activations = {
            'lrelu': tf.nn.leaky_relu,
            'elu': tf.nn.elu,
            'tanh': tf.nn.tanh,
            'sigmoid': tf.nn.sigmoid,
            'softmax': tf.nn.softmax,
            'selu': tf.nn.selu,
            'softsign': tf.nn.softsign,
            'linear': tf.identity,
            'softplus': tf.nn.softplus,
        }
        self.hidden_activation = activations.get(self.actype, None)
        if self.hidden_activation is None:
            raise NotImplementedError('Hidden activation function not implemented. Try:' + str(activations.keys()))

        # Save logging folder (includes name)
        self.write_path = write_path + '/' + self.name

        # Generate logging folder, error folder and model folder
        self.logfolder = self.write_path + '/logs'
        os.makedirs(self.logfolder, exist_ok=True)
        self.errorfolder = self.write_path + '/errors'
        os.makedirs(self.errorfolder, exist_ok=True)
        self.modelfolder = self.write_path + '/model'
        os.makedirs(self.modelfolder, exist_ok=True)

        # Frame storage for video creation
        self.framesfolder = self.write_path + '/frames'
        os.makedirs(self.framesfolder, exist_ok=True)
        self.reconframesfolder = self.write_path + '/reconframes'
        os.makedirs(self.reconframesfolder, exist_ok=True)

        # Prepare graph, define loss function, and configure session parameters
        self.prepare_graph()

        if self.fresh_init == 1:
            self.define_loss()
            self.start()
        else:
            self.define_loss()
            self.start()

    def prepare_graph(self):

        if self.reset_default_graph == 1:
            # Clear default graph stack and reset global default path
            tf.reset_default_graph()

        # Generate placeholder for iterator
        self.placeholder_X = tf.placeholder(tf.float32, [None, self.width, self.height, self.channels],
                                            name='placeholder_X')

        # Produce dataset instance and transformations
        self.dataset = tf.data.Dataset.from_tensor_slices(self.placeholder_X)  # It's an AE so it's (X, X)
        self.dataset = self.dataset.shuffle(buffer_size=100000)
        #self.dataset = self.dataset.repeat(10)
        self.dataset = self.dataset.batch(self.mbs)

        # Define an iterator that goes through the dataset
        self.iterator = self.dataset.make_initializable_iterator()
        self.data_x = self.iterator.get_next()

        # Generate input X placeholder
        #self.X = tf.placeholder(tf.float32, shape=(None, self.width, self.height, self.channels), name='X')
        self.X = tf.identity(self.data_x)

        # Other parameters
        self.eps_throttle = tf.placeholder(tf.float32, shape=(), name='throttle')
        self.kl_weight = tf.placeholder(tf.float32, shape=(), name="kl_weight")
        self.learning_rate = tf.placeholder(tf.float32, shape=(), name='learning_rate')

        # Dropout regularization
        self.dropout_rate = tf.placeholder(tf.float32, shape=(), name='dropout_rate')

        # Parameter initialization - Xavier's
        #self.init = tf.initializers.glorot_normal()
        self.init = tf.random_normal_initializer(mean=0.0, stddev=1.0)

        ''' Convolutional part of encoder '''
        self.Wce = []
        self.bce = []
        self.hce = []
        for k in range(0, np.size(self.n_conv)):
            with tf.name_scope('conv_encoder_l' + str(k)):
                if k == 0:
                    nmw = np.sqrt(2.0/(self.fs[k]*self.fs[k]*self.channels))
                    self.Wce.append(
                        tf.Variable(nmw*self.init([self.fs[k], self.fs[k], self.channels, self.n_conv[k]], dtype=tf.float32),
                                    name='Wce' + str(k)))
                    self.bce.append(tf.Variable(tf.zeros([self.n_conv[k]], dtype=tf.float32), name='bce' + str(k)))
                    self.hce.append(tf.nn.dropout(self.hidden_activation(tf.nn.bias_add(
                        tf.nn.conv2d(self.X, self.Wce[k], strides=[1, self.nstride[k], self.nstride[k], 1], padding='SAME'),
                        self.bce[k]), name='hce' + str(k)), keep_prob=self.dropout_rate, noise_shape=[tf.shape(self.X)[0], 1, 1, self.n_conv[k]]))
                else:
                    nmw = np.sqrt(2.0 / (self.fs[k] * self.fs[k] * self.n_conv[k-1]))
                    self.Wce.append(
                        tf.Variable(nmw*self.init([self.fs[k], self.fs[k], self.n_conv[k - 1], self.n_conv[k]], dtype=tf.float32),
                                    name='Wce' + str(k)))
                    self.bce.append(tf.Variable(tf.zeros([self.n_conv[k]], dtype=tf.float32), name='bce' + str(k)))
                    self.hce.append(tf.nn.dropout(self.hidden_activation(tf.nn.bias_add(
                        tf.nn.conv2d(self.hce[k - 1], self.Wce[k], strides=[1, self.nstride[k], self.nstride[k], 1],
                                     padding='SAME'),
                        self.bce[k]), name='hce' + str(k)), keep_prob=self.dropout_rate, noise_shape=[tf.shape(self.X)[0], 1, 1, self.n_conv[k]]))
                print(k)
                print('self.n_conv[k]=' + str(self.n_conv[k]))
                print('self.hce[k].shape = ' + str(self.hce[k].shape))

        ''' Feedforward part of encoder '''
        with tf.name_scope('global_averaging'):
            self.fe = tf.transpose(tf.reduce_mean(tf.reduce_mean(self.hce[-1], axis=1), axis=1))

        print('self.fe is ' + str(np.shape(self.fe)))

        self.We = []
        self.be = []
        self.he = []
        for k in range(0, np.size(self.n_h)):
            with tf.name_scope('ff_encoder_l' + str(k)):
                if k == 0:
                    nmw = np.sqrt(2.0/int(np.size(self.fe, 0)))
                    self.We.append(tf.Variable(nmw*self.init([self.n_h[k], int(np.size(self.fe, 0))]), name="We" + str(k)))
                    self.be.append(tf.Variable(tf.zeros([self.n_h[k], 1]), name='be' + str(k)))
                    self.he.append(tf.nn.dropout(self.hidden_activation(
                                            tf.add(tf.matmul(self.We[k], self.fe), self.be[k]), name='he' + str(k)), keep_prob=self.dropout_rate))
                else:
                    nmw = np.sqrt(2.0 /self.n_h[k-1])
                    self.We.append(tf.Variable(nmw*self.init([self.n_h[k], self.n_h[k-1]]), name="We" + str(k)))
                    self.be.append(tf.Variable(tf.zeros([self.n_h[k], 1]), name='be' + str(k)))
                    self.he.append(tf.nn.dropout(self.hidden_activation(
                        tf.add(tf.matmul(self.We[k], self.he[k-1]), self.be[k]), name='he' + str(k)), keep_prob=self.dropout_rate))
                print('self.he[' + str(k) + '] is ' + str(np.shape(self.he[k])))

        '''  Latent layer '''
        k = np.size(self.n_h) - 1

        with tf.name_scope("z"):
            # Generate latent layer
            nmw = np.sqrt(2.0/int(self.n_h[-1]))
            self.Wz = tf.Variable(nmw*self.init([self.n_z, self.n_h[-1]]), name='Wz')
            self.bz = tf.Variable(tf.zeros([self.n_z, 1]), name='bz')

            # Mu can be positive or negative, but sigma can't be negative! use softplus layer
            self.z = tf.add(tf.matmul(self.Wz, self.he[-1]), self.bz, name='z')

            # Random noise generator
            self.eps = tf.random_normal(tf.shape(self.z), mean=0.0, stddev=1.0,
                                                            name='normal_samples')

        ''' Feedforward part of decoder '''
        # Decoder layers (with a linear output)
        self.Wd = []
        self.bd = []
        self.hd = []

        # W_k will be the connector between h_k and h_(k+1), with h_k == z for k==0
        # (k==1 : h_1 -> h_2, k==2: h_2 -> h_3), k==n
        for k in range(0, np.size(self.n_h_d)):
            if k == 0:
                with tf.name_scope('ff_decoder_l' + str(k)):
                    nmw = np.sqrt(2.0/self.n_z)
                    self.Wd.append(
                        tf.Variable(nmw*self.init([self.n_h_d[k], self.n_z], dtype=tf.float32), name="Wd" + str(k)))
                    self.bd.append(tf.Variable(tf.zeros([self.n_h_d[k], 1], dtype=tf.float32), name='bd' + str(k)))
                    self.hd.append(tf.nn.dropout(
                        self.hidden_activation(tf.add(tf.matmul(self.Wd[k], self.z), self.bd[k]), name='hd' + str(k)),
                        keep_prob=self.dropout_rate))
            else:
                with tf.name_scope('ff_decoder_l' + str(k)):
                    nmw = np.sqrt(2.0/self.n_h_d[k-1])
                    self.Wd.append(
                        tf.Variable(nmw*self.init([self.n_h_d[k], self.n_h_d[k - 1]], dtype=tf.float32), name="Wd" + str(k)))
                    self.bd.append(tf.Variable(tf.zeros([self.n_h_d[k], 1], dtype=tf.float32), name='bd' + str(k)))
                    self.hd.append(tf.nn.dropout(
                        self.hidden_activation(tf.add(tf.matmul(self.Wd[k], self.hd[k - 1]), self.bd[k]),
                                               name='hd' + str(k)), keep_prob=self.dropout_rate))
            print('self.hd[' + str(k) + '] is ' + str(np.shape(self.hd[k])))

        ''' Get flattened layer and un-flatten it '''
        nmw = np.sqrt(2.0/self.n_h_d[-1])
        self.Wfd = tf.Variable(nmw*self.init([int(np.size(self.fe, 0))*16, self.n_h_d[-1]], dtype=tf.float32), name='Wfd')
        self.bfd = tf.Variable(tf.zeros([int(np.size(self.fe, 0))*16, 1], dtype=tf.float32), name='bfd')
        self.fd = tf.nn.dropout(tf.reshape(tf.transpose(self.hidden_activation(tf.add(tf.matmul(self.Wfd, self.hd[-1]), self.bfd), name='fd')),
                                           [-1, 4, 4, int(np.size(self.fe, 0))]),
                                noise_shape=[tf.shape(self.z)[1], 1, 1, int(np.size(self.fe, 0))], keep_prob=self.dropout_rate)



        print('self.Wfd is ' + str(np.shape(self.Wfd)))
        print('self.hd[-1] is ' + str(np.shape(self.hd[-1])))
        print('self.bfd is ' + str(np.shape(self.bfd)))
        print('self.fd is ' + str(np.shape(self.fd)))

        ''' Convolutional part of decoder '''
        # List declarations
        self.Wcd = []
        self.bcd = []
        self.hcd = []
        self.Wcd.append(0)
        self.bcd.append(0)
        self.hcd.append(self.fd)
        self.upsample_d = []

        Nc = int(np.size(self.n_conv)) - 1
        for k in range(1, Nc + 1):
            print(k)
            print('k-Nc = ' + str(Nc - k))
            print('self.nc_d[k]=' + str(self.n_conv_d[k]))
            print('self.hce[Nc-k].shape = ' + str(self.hce[Nc - k].shape))

            with tf.name_scope('conv_decoder_l' + str(k)):
                nmw = np.sqrt(2.0/(self.fs_d[k-1]*self.fs_d[k-1]*self.n_conv_d[k-1]))
                self.Wcd.append(tf.Variable(
                    nmw*self.init([self.fs_d[k - 1], self.fs_d[k - 1], self.n_conv_d[k-1],  self.n_conv_d[k]],
                                   dtype=tf.float32), name='Wcd' + str(k)))
                print('self.Wcd[' + str(k) + '].shape = ' + str(np.shape(self.Wcd[k])))

                self.bcd.append(tf.Variable(tf.zeros([self.n_conv_d[k]], dtype=tf.float32), name='bcd' + str(k)))

                self.upsample_d.append(tf.image.resize_images(self.hcd[k - 1], size=(
                    int(self.hce[Nc - k].shape[1]), int(self.hce[Nc - k].shape[2]))))
                print('self.upsample_d[' + str(k - 1) + '].shape = ' + str(np.shape(self.upsample_d[k - 1])))
                self.hcd.append(tf.nn.dropout(
                    self.hidden_activation(tf.nn.bias_add(
                        tf.nn.conv2d(self.upsample_d[k - 1], self.Wcd[k], strides=[1, 1, 1, 1], padding='SAME'),
                        self.bcd[k]), name='hcd' + str(k)),
                    keep_prob=self.dropout_rate, noise_shape=[tf.shape(self.z)[1], 1, 1, self.n_conv_d[k]]))

        print('self.hcd[' + str(k) + '] is ' + str(np.shape(self.hcd[k])))

        ''' Final output layer '''
        with tf.name_scope("Xhat"):
            self.upsample_hat = tf.image.resize_images(self.hcd[-1], size=(self.X.shape[1], self.X.shape[2]))
            print('self.upsample_hat is ' + str(np.shape(self.upsample_hat)))

            nmw = np.sqrt(2.0/(self.n_conv_d[-1]*(self.fs_d[-1]**2)))
            self.What = tf.Variable(
                nmw*self.init([self.fs_d[-1], self.fs_d[-1], self.n_conv_d[-1], self.channels], dtype=tf.float32),
                name='What')

            print('self.What.shape = ' + str(np.shape(self.What)))

            self.bhat = tf.Variable(tf.zeros([self.channels], dtype=tf.float32), name='bhat')
            self.Xhat = tf.nn.bias_add(tf.nn.conv2d(self.upsample_hat, self.What, strides=[1, 1, 1, 1], padding='SAME'),
                                       self.bhat, name='Xhat')

            print('self.Xhat is ' + str(np.shape(self.Xhat)))

    start = common.start

    compute_kernel = common.compute_kernel
    compute_mmd = common.compute_mmd
    define_loss = common.conv_infovae_loss

    train = common.train_convolutional_ae
    show_map = common.show_map_conv
    show_reconstructions = common.show_reconstructions_conv_ae
    output = common.output_conv_ae
    synthesize = common.synthesize_conv

    save_model = common.save_model
    load_model = common.load_model


if __name__ == '__main__':
    ''' Load dataset '''
    import ai.common_ops.datasets as dts
    training_data = dts.get_mnist()

    ''' Network instance and parameters '''
    nc = [101, 102, 103, 104, 105, 106, 107]
    fs = [3, 3, 3, 3, 3, 3, 3]
    nstride = [1, 2, 1, 2, 1, 2, 1]
    nh = [301, 302, 303]
    nz = 30
    x_dims = [28, 28, 1]

    network = ConvolutionalInfoVAE(nc, fs, nstride, nh, nz, x_dims, fresh_init=1, actype='lrelu', mbs=32, cost_function='MSE',
                                   name='conv_InfoVAE', reset_default_graph=1, optim='Adam')

    # Test for Cyclic Learning Rate training
    training_schedule = {
        'mode': 'ocp',
        'eval_criterion': 'iter',
        'stop_criterion': 'none',
        'lr_max': 0.0001,
        'lr_min': 0.0,
        'beta_start': 1.0,
        'beta_end': 1.0,
        'Nwarm': 1,
        'Ncool': 100000,
        'Nwarmup': 1,
        'T': 1000,
        'Niters': 400000,
        'dropout_rate': 0.95,
        'max_patience': 50,
        'eval_size': 3000
    }
    network.train(training_data, training_schedule, show=1)