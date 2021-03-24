'''
Convolutional autoencoder model with skip-connections
Arturo Pardo, 2019
'''

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime
import h5py
import csv
from tqdm import tqdm

import os
dir_path = os.path.dirname(os.path.abspath(__file__))

import ai.common_ops as common

'''
Convolutional Info-VAE (simple)
'''
class ConvolutionalSCVAE:
    def __init__(self, n_conv, fs, nstride, n_h, n_z, x_dims, lr=0.0001, optim='Adam', actype='elu', mbs=16, regeps=0.1,
                 cost_function = 'MSE', fixed_lr=False, quiet = 0, debug = 0,
                 fresh_init=1, name='conv_scvae', reset_default_graph=1, write_path='../../../output/network_tests'):

        # Is this a new network or a pretrained one?
        self.fresh_init = fresh_init
        self.name = name

        # Debug/quiet flags
        self.quiet = quiet
        self.debug = debug

        # Order to reset default graph when instanced
        self.reset_default_graph = reset_default_graph

        # First iteration detector
        if self.fresh_init == 1:
            self.first_iter = 1
        else:
            self.first_iter = 0

        # Init_parameters
        self.n_conv = n_conv        # Convolutional layers and their sizes (in neurons)
        self.fs = fs                # Filter size per layer (we will assume square filters)
        self.nstride = nstride      # Strides per layer (usually just ones)
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

        # Epsilon for regularizing relative MSE
        self.regeps = regeps

        # Minibatch size
        self.mbs = int(mbs)

        # Activation function selector
        self.actype = actype

        activations = {
            'lrelu': tf.nn.leaky_relu,
            'elu': tf.nn.elu,
            'tanh': tf.nn.tanh,
            'sigmoid': tf.nn.sigmoid,
            'softmax': tf.nn.softmax,
            'linear': tf.identity,
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
        self.X = tf.identity(self.data_x, name='X')

        # Other hyperparameters
        self.eps_throttle = tf.placeholder(tf.float32, shape=(), name='throttle')           # Control for noise generation
        self.kl_weight = tf.placeholder(tf.float32, shape=(), name="kl_weight")             # Cost of gaussianity constraint
        self.learning_rate = tf.placeholder(tf.float32, shape=(), name='learning_rate')     # For variable learning rate

        # Dropout regularization control
        self.dropout_rate = tf.placeholder(tf.float32, shape=(), name='dropout_rate')

        # Parameter initialization - He's
        self.init = tf.random_normal_initializer(mean=0.0, stddev=1.0)

        ''' Convolutional part of encoder '''
        self.Wce = []               # Filters
        self.bce = []               # Biases
        self.gce = []               # Reference images
        self.gce_connected = []     # Connected reference images
        self.hce = []               # Current layer

        self.gce.append([self.X])

        for i in range(1, np.size(self.n_conv)+1):
            with tf.name_scope('conv_encoder_l' + str(i)):
                # Current list of inputs
                g_curr = []
                for j in range(0, i+1):
                    if j == 0:
                        # Generate weights for concatenated input
                        print('contatenated self.gce[' + str(i-1) + '], which is ' + str(np.shape(self.gce[i-1])))
                        print(str(self.gce[i-1]))
                        self.gce_connected.append(tf.concat(self.gce[i-1], axis=3))
                        print('gce_connected[i=' + str(i-1) + '] is ' + str(np.shape(self.gce_connected[i-1])))

                        # Norm weight for He init
                        nmw = np.sqrt(2.0) / np.sqrt(int(self.gce_connected[i-1].shape[3])*self.fs[i-1]*self.fs[i-1])

                        self.Wce.append(tf.Variable(nmw*self.init([self.fs[i-1], self.fs[i-1], int(self.gce_connected[i-1].shape[3]), self.n_conv[i-1]], dtype=tf.float32)*nmw, name='Wce' + str(i)))
                        self.bce.append(tf.Variable(tf.zeros([self.n_conv[i-1]], dtype=tf.float32)*nmw, name='bce' + str(i)))
                        g_curr.append(tf.nn.dropout(self.hidden_activation(tf.nn.bias_add(
                                             tf.nn.conv2d(self.gce_connected[i-1], self.Wce[i-1], strides=[1, self.nstride[i-1], self.nstride[i-1], 1], padding='SAME'),
                                                self.bce[i-1]), name='hce' + str(i)), keep_prob=self.dropout_rate, noise_shape=[tf.shape(self.X)[0], 1, 1, self.n_conv[i-1]]))
                        print('   g_curr[j=' + str(j)+ '] is ' + str(np.shape(g_curr[0])))
                    else:
                        #print(g_curr[0].shape[0])
                        g_curr.append(tf.image.resize_images(self.gce[i-1][j-1], size=(int(g_curr[0].shape[1]), int(g_curr[0].shape[2])))) # Resize previous layers
                        print('   g_curr[j=' + str(j) + '] is ' + str(np.shape(g_curr[-1])))

                print('      g_curr is finally ' + str(np.shape(g_curr)))
                self.gce.append(g_curr)
                print('      gce is ' + str(np.shape(self.gce)))

        ''' Finally, prepare a global averaging vector'''
        self.gce_connected.append(tf.concat(self.gce[-1], axis=3))
        with tf.name_scope('glob_avg_fe'):
            self.fe = tf.transpose(tf.reduce_mean(tf.reduce_mean(self.gce_connected[-1], axis=1), axis=1))

        print('self.fe is ' + str(np.shape(self.fe)))

        # Encoder layers (with a linear output)
        self.he = []
        self.ge = []
        self.We = []
        self.be = []

        print('Encoder feedforward:')
        # i-th hidden layer (starts at 1)
        for i in range(0, len(self.n_h)):
            print('Layer i = ' + str(i))
            with tf.name_scope('encoder_l' + str(i)):
                w_curr = []
                g_curr = []
                for j in range(0, i + 1):
                    print('Connection j = ' + str(j - 1) + ' --> ' + str(i))

                    print('n_e[0:' + str(i) + '] is ' + str(self.n_h[0:i]))
                    if j == 0:
                        nmw = np.sqrt(2.0 / int(self.fe.shape[0]))
                        # Create weights
                        w_curr.append(tf.Variable(nmw * self.init([self.n_h[i], int(self.fe.shape[0])]),
                                                  name='We_' + str(j) + '_' + str(i)))
                        g_curr.append(tf.matmul(w_curr[j], self.fe))

                    else:
                        nmw = np.sqrt(2.0 / self.n_h[j - 1])
                        # Create weights
                        w_curr.append(tf.Variable(nmw * self.init([self.n_h[i], self.n_h[j - 1]]),
                                                  name='We_' + str(j) + '_' + str(i)))
                        g_curr.append(tf.matmul(w_curr[j], self.he[j - 1]))

                self.We.append(w_curr)
                self.ge.append(g_curr)
                self.be.append(tf.Variable(tf.zeros([self.n_h[i], 1]), name='be_' + str(i - 1)))

                self.he.append(tf.nn.dropout(
                    self.hidden_activation(tf.add(tf.add_n(self.ge[i]), self.be[i]), name='he_' + str(i)),
                    keep_prob=self.dropout_rate))

        # Generate now the mu and sigma layer
        i = len(self.n_h)
        print('Latent space:')
        print('Mu: i = ' + str(i))

        with tf.name_scope("mu"):
            self.w_mu = []
            self.g_mu = []

            for j in range(0, i + 1):
                print('Mu: j=' + str(j - 1) + '--> ' + str(i))
                print('n_e[0:' + str(i) + '] is ' + str(self.n_h[0:i]))
                if j == 0:
                    nmw = np.sqrt(2.0 / int(self.fe.shape[0]))
                    # Create weights
                    self.w_mu.append(tf.Variable(nmw*self.init([self.n_z, int(self.fe.shape[0])]), name='We_' + str(j) + '_' + str(i)))
                    self.g_mu.append(tf.matmul(self.w_mu[j], self.fe))

                else:
                    nmw = np.sqrt(2.0 / self.n_h[j-1])
                    self.w_mu.append(tf.Variable(nmw*self.init([self.n_z, self.n_h[j - 1]]), name='We_' + str(j) + '_' + str(i)))
                    self.g_mu.append(tf.matmul(self.w_mu[j], self.he[j - 1]))

            self.b_mu = tf.Variable(tf.zeros([self.n_z, 1]), name='b_mu')
            self.mu = tf.add(tf.add_n(self.g_mu), self.b_mu, name='mu')

        print('Sigma: ')
        with tf.name_scope("sigma"):
            self.w_sigma = []
            self.g_sigma = []

            for j in range(0, i + 1):
                print('Sigma: ' + str(j - 1) + ' --> ' + str(i))
                print('n_e[0:' + str(i) + '] is ' + str(self.n_h[0:i]))
                if j == 0:
                    nmw = np.sqrt(2.0 / int(self.fe.shape[0]))
                    # Create weights
                    self.w_sigma.append(tf.Variable(nmw*self.init([self.n_z, int(self.fe.shape[0])]), name='We_' + str(j) + '_' + str(i)))
                    self.g_sigma.append(tf.matmul(self.w_sigma[j], self.fe))

                else:
                    nmw = np.sqrt(2.0 / self.n_h[j - 1])
                    self.w_sigma.append(tf.Variable(nmw*self.init([self.n_z, self.n_h[j - 1]]), name='We_' + str(j) + '_' + str(i)))
                    self.g_sigma.append(tf.matmul(self.w_sigma[j], self.he[j - 1]))

            self.b_sigma = tf.Variable(tf.zeros([self.n_z, 1]), name='b_sigma')
            self.log_sigma = tf.add(tf.add_n(self.g_sigma), self.b_sigma, name='log_sigma')

        with tf.name_scope('z'):
            # Random sample generator
            self.eps = self.eps_throttle * tf.random_normal(tf.shape(self.mu), mean=0.0, stddev=1.0,
                                                            name='normal_samples')

            # Reparameterization trick
            self.z = tf.add(tf.multiply(self.eps, tf.exp(self.log_sigma / 2.0)), self.mu, name='z')
            print('self.z is ' + str(np.shape(self.z)))

        # Decoder layers (with a linear output)
        self.hd = []
        self.gd = []
        self.Wd = []
        self.bd = []

        print('Decoder:')
        # i-th hidden layer (starts at 1)
        for i in range(0, len(self.n_h_d)):
            print('Layer i = ' + str(i))
            with tf.name_scope('decoder_l' + str(i)):
                w_curr = []
                g_curr = []
                for j in range(0, i + 1):
                    print('Connection j = ' + str(j - 1) + ' --> ' + str(i))
                    print('n_d[0:' + str(i) + '] is ' + str(self.n_h_d[0:i]))
                    if j == 0:
                        # For the i-th layer you have to find the fan-in
                        nmw = np.sqrt(2.0 / self.n_z)

                        # Create weights
                        w_curr.append(tf.Variable(nmw * self.init([self.n_h_d[i], self.n_z]),
                                                  name='Wd_' + str(j) + '_' + str(i)))
                        g_curr.append(tf.matmul(w_curr[j], self.z))

                    else:
                        nmw = np.sqrt(2.0 / self.n_h_d[j - 1])
                        w_curr.append(tf.Variable(nmw * self.init([self.n_h_d[i], self.n_h_d[j - 1]]),
                                                  name='Wd_' + str(j) + '_' + str(i)))
                        g_curr.append(tf.matmul(w_curr[j], self.hd[j - 1]))

                self.Wd.append(w_curr)
                self.gd.append(g_curr)
                self.bd.append(tf.Variable(tf.zeros([self.n_h_d[i], 1]), name='bd_' + str(i - 1)))

                self.hd.append(tf.nn.dropout(
                    self.hidden_activation(tf.add(tf.add_n(self.gd[i]), self.bd[i]), name='hd_' + str(i)),
                    keep_prob=self.dropout_rate))

        # Generate feedforward decoder first layer
        i = len(self.n_h_d)
        print('self.fd:')
        with tf.name_scope("glob_avg_fd"):
            self.w_fd = []
            self.g_fd = []

            for j in range(0, i + 1):
                print('Connection j = ' + str(j - 1) + ' --> ' + str(i))
                print('n_d[0:' + str(i) + '] is ' + str(self.n_h_d[0:i]))
                if j == 0:
                    nmw = np.sqrt(2.0 / self.n_z)
                    # Create weights
                    self.w_fd.append(tf.Variable(nmw * self.init([int(self.fe.shape[0]), self.n_z]),
                                                 name='W_' + str(j) + '_Xhat'))
                    # self.g_fd.append(tf.matmul(self.w_fd[j], self.z))

                else:
                    nmw = np.sqrt(2.0 / self.n_h_d[j - 1])
                    self.w_fd.append(tf.Variable(nmw * self.init([int(self.fe.shape[0]), self.n_h_d[j - 1]]),
                                                 name='W_' + str(j) + '_Xhat'))
                    self.g_fd.append(tf.matmul(self.w_fd[j], self.hd[j - 1]))

            self.b_fd = tf.Variable(tf.zeros([int(self.fe.shape[0]), 1]), name='b_fd')

            self.fd = tf.nn.dropout(self.hidden_activation(tf.add(tf.add_n(self.g_fd), self.b_fd), name='fd'),
                                    keep_prob=self.dropout_rate)

            # Re-scale to original desired shape
            self.fd_rescaled = tf.image.resize_images(tf.expand_dims(tf.expand_dims(tf.transpose(self.fd), 1), 1),
                                                      size=(int(self.gce_connected[-1].shape[1]),
                                                            int(self.gce_connected[-1].shape[2])))

            print('fd_rescaled is ' + str(np.shape(self.fd_rescaled)))

        ''' Convolutional part of decoder '''
        self.Wcd = []
        self.bcd = []
        self.gcd = []  # Reference images
        self.gcd_connected = []  # Connected reference images
        self.hcd = []  # Current layer

        self.gcd.append([self.fd_rescaled])

        for i in range(1, np.size(self.n_conv_d) + 1):

            with tf.name_scope('conv_decoder_l' + str(i)):
                # Current list of inputs
                g_curr = []
                for j in range(0, i + 1):
                    # print('j = ' + str(j))
                    if j == 0:
                        # Generate weights for concatenated input
                        print('contatenated self.gcd[' + str(i - 1) + '], which is ' + str(np.shape(self.gcd[i - 1])))
                        print(str(self.gcd[i - 1]))

                        self.gcd_connected.append(tf.concat(self.gcd[i - 1], axis=3))

                        print('gcd_connected[i=' + str(i - 1) + '] is ' + str(np.shape(self.gcd_connected[i - 1])))
                        nmw = np.sqrt(2.0 / (int(self.gcd_connected[i - 1].shape[3]) * self.fs_d[i - 1] * self.fs_d[i - 1]))
                        self.Wcd.append(tf.Variable(nmw * self.init(
                            [self.fs_d[i - 1], self.fs_d[i - 1], int(self.gcd_connected[i - 1].shape[3]),
                             self.n_conv_d[i - 1]],
                            dtype=tf.float32), name='Wcd' + str(i)))
                        self.bcd.append(
                            tf.Variable(tf.zeros([self.n_conv_d[i - 1]], dtype=tf.float32), name='bcd' + str(i)))

                        g_curr.append(tf.nn.dropout(self.hidden_activation(tf.nn.bias_add(
                            tf.nn.conv2d(tf.image.resize_images(self.gcd_connected[i - 1], size=(
                            int(self.gcd_connected[i - 1].shape[1]) * self.nstride[i - 1],
                            int(self.gcd_connected[i - 1].shape[2]) * self.nstride[i - 1])),
                                         self.Wcd[i - 1], strides=[1, 1, 1, 1], padding='SAME'),
                            self.bcd[i - 1]), name='hcd' + str(i)), keep_prob=self.dropout_rate,
                            noise_shape=[tf.shape(self.z)[1], 1, 1, self.n_conv_d[i - 1]]))

                        print('   g_curr[j=' + str(j) + '] is ' + str(np.shape(g_curr[0])))
                    else:
                        g_curr.append(tf.image.resize_images(self.gcd[i - 1][j - 1], size=(
                            int(g_curr[0].shape[1]), int(g_curr[0].shape[2]))))  # Resize previous layers
                        print('   g_curr[j=' + str(j) + '] is ' + str(np.shape(g_curr[-1])))

                print('      g_curr is finally ' + str(np.shape(g_curr)))
                self.gcd.append(g_curr)
                print('      gcd is ' + str(np.shape(self.gcd)))

        with tf.name_scope('Xhat'):
            self.gcd_connected.append(tf.concat(self.gcd[- 1], axis=3))
            nmw = np.sqrt(2.0 / (int(self.gcd_connected[-1].shape[3]) * self.fs_d[-1] * self.fs_d[-1]))
            self.W_Xhat = tf.Variable(
                nmw * self.init([self.fs_d[-1], self.fs_d[-1], int(self.gcd_connected[-1].shape[3]), self.channels],
                                dtype=tf.float32), name='W_Xhat' + str(i))
            self.b_Xhat = tf.Variable(tf.zeros([self.channels], dtype=tf.float32), name='bcd' + str(i))
            self.Xhat = tf.nn.bias_add(
                tf.nn.conv2d(tf.image.resize_images(self.gcd_connected[-1], size=(self.width, self.height)),
                             self.W_Xhat, strides=[1, 1, 1, 1], padding='SAME'),
                self.b_Xhat, name='Xhat')

    start = common.start
    define_loss = common.conv_beta_vae_loss

    train = common.train_convolutional_ae
    output = common.output_conv_ae
    show_reconstructions = common.show_reconstructions_conv_ae
    show_map = common.show_map_conv
    synthesize = common.synthesize_conv

    save_model = common.save_model
    load_model = common.load_model


if __name__ == '__main__':
    import ai.common_ops.datasets as dts

    ''' Load dataset '''
    training_data = dts.get_mnist(mlp=False)

    ''' Network instance and parameters '''
    nconv = [30, 31, 32, 33, 34, 35]
    fs = [3, 3, 3, 3, 3, 3]
    nstride = [1, 2, 1, 2, 1, 2]
    nh = [301, 302, 303]
    nz = 30
    x_dims = [28, 28, 1]

    network = ConvolutionalSCVAE(nconv, fs, nstride, nh, nz, x_dims, fresh_init=0,
                                 actype='lrelu', name='conv_SCVAE', reset_default_graph=1)

    # Train the network
    training_schedule = {
        'mode': 'ocp',
        'eval_criterion': 'epoch',
        'stop_criterion': 'none',
        'lr_max': 0.0001,
        'lr_min': 0.0,
        'beta_start': 1.0,
        'beta_end': 1.0,
        'Nwarm': 1,
        'Ncool': 200000,
        'Nwarmup': 1,
        'T': 1,
        'Niters': 500000,
        'dropout_rate': 0.95,
        'max_patience': 50,
        'eval_size': 3000
    }
    network.train(training_data, training_schedule, show=1)