'''
Convolutional InfoVAE with feature matching and a gated/clamped bottleneck
Arturo Pardo, 2019/2021
'''

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
from datetime import datetime
import h5py
import csv

import os
dir_path = os.path.dirname(os.path.abspath(__file__))

import ai.common_ops as common


class ConvolutionalInfoSCVAE_GCN_gated_encoded:
    def __init__(self, n_conv, fs, npad, n_h, n_z, x_dims, lr=0.0001, optim='Adam', actype='elu', mbs=16, gan_weight=0.05,
                 cost_function='MSE', fixed_lr=False, Nstart = 1, n_connect = 50, n_convhelp = 10, regeps=0.1, quiet = 0, debug = 0,
                 fresh_init=1, name='model', reset_default_graph=1, write_path='../../../output/network_tests'):

        # Is this a new network or a pretrained one?
        self.fresh_init = fresh_init
        self.name = name

        # Debug and quiet params
        self.debug = debug
        self.quiet = quiet

        # Start of fully connected layer connections and layer size
        self.Nstart = Nstart
        self.n_connect = n_connect
        self.n_convhelp = n_convhelp

        # Value of GAN-like discriminator -- usually 0.05 (Mathieu et al., 2017)
        # but sometimes we need to switch
        self.gan_weight = gan_weight

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
        self.npad = npad            # Padding per layer (usually just ones)
        self.n_h = n_h              # Feedforward layers and their sizes (in neurons)
        self.n_z = n_z              # Number of dimensions of variational latent space

        # Reversed params for decoder
        self.n_conv_d = np.flip(self.n_conv)
        self.fs_d = np.flip(self.fs)
        self.npad_d = np.flip(self.npad)
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
            'selu': tf.nn.selu,
            'softsign': tf.nn.softsign,
            'linear': tf.identity,
            'softplus': tf.nn.softplus,
        }
        self.hidden_activation = activations.get(self.actype, None)
        if self.hidden_activation is None:
            raise NotImplementedError('Hidden activation function not implemented. Try:' + str(activations.keys()))

        # Discriminator activation function -- we'll use leaky_relu
        self.gan_activation = tf.nn.leaky_relu

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
        self.dataset = self.dataset.shuffle(buffer_size=1000000)
        #self.dataset = self.dataset.repeat(10)
        self.dataset = self.dataset.batch(self.mbs)

        # Define an iterator that goes through the dataset
        self.iterator = self.dataset.make_initializable_iterator()
        self.data_x = self.iterator.get_next()

        # Generate input X placeholder
        #self.X = tf.placeholder(tf.float32, shape=(None, self.width, self.height, self.channels), name='X')
        self.X = tf.identity(self.data_x, name='X')

        # Other parameters
        self.eps_throttle = tf.placeholder(tf.float32, shape=(), name='throttle')
        self.kl_weight = tf.placeholder(tf.float32, shape=(), name="kl_weight")
        self.learning_rate = tf.placeholder(tf.float32, shape=(), name='learning_rate')

        # Dropout regularization
        self.dropout_rate = tf.placeholder(tf.float32, shape=(), name='dropout_rate')
        self.dropout_rate_AE = tf.identity(self.dropout_rate, name="dropout_rate_AE")
        self.dropout_rate_Disc = tf.identity(self.dropout_rate, name="dropout_rate_Disc")

        '''
        This is an interesting trick given by Burgess in 2017, modified because we don't use beta-VAE.
        If we force each unit in z to increasingly explain variance in the data, then the first few 
        units in z will explain most of the variance. 
        We do it via clamping instead of by annealing the beta parameter.
        '''
        self.z_throt = tf.placeholder(tf.float32, [self.n_z, None], name='z_throt')             # Throttle flag
        self.z_stopgrad = tf.placeholder(tf.float32, [self.n_z, None], name='z_stopgrad')       # Stop gradients flag

        # Parameter initialization - He's
        self.init = tf.random_normal_initializer(mean=0.0, stddev=1.0)
        self.init_conv = tf.initializers.glorot_normal()
        #self.init_conv = tf.random_normal_initializer(mean=0.0, stddev=1.0)

        with tf.name_scope('Autoencoder'):
            ''' Convolutional part of encoder '''
            self.Wce = []               # Filter list per layer
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

                            nmw = np.sqrt(1.0/(int(self.gce_connected[i-1].shape[3])*self.fs[i-1]**2))
                            self.Wce.append(tf.Variable(nmw*self.init([self.fs[i-1],
                                                                       self.fs[i-1],
                                                                       int(self.gce_connected[i-1].shape[3]),
                                                                       self.n_conv[i-1]],
                                                                       dtype=tf.float32), name='Wce' + str(i)))

                            self.bce.append(tf.Variable(tf.zeros([self.n_conv[i-1]], dtype=tf.float32), name='bce' + str(i)))
                            g_curr.append(
                                tf.nn.dropout(
                                    self.hidden_activation(
                                        tf.nn.bias_add(
                                            tf.nn.conv2d(self.gce_connected[i-1],
                                                         self.Wce[i-1],
                                                         strides=[1, self.npad[i-1], self.npad[i-1], 1],
                                                         padding='SAME'),
                                            self.bce[i-1]),
                                            name='hce' + str(i)),
                                    keep_prob=self.dropout_rate_AE,
                                    noise_shape=[tf.shape(self.X)[0], 1, 1, self.n_conv[i-1]]))

                            print('   g_curr[j=' + str(j)+ '] is ' + str(np.shape(g_curr[0])))
                        else:
                            # If we are on the last layer, create a resized version of all previous layers for the next layer.
                            g_curr.append(tf.image.resize_images(self.gce[i-1][j-1],
                                                                 size=(int(g_curr[0].shape[1]),
                                                                       int(g_curr[0].shape[2])),
                                                                 align_corners=True,
                                                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR))

                            print('   g_curr[j=' + str(j) + '] is ' + str(np.shape(g_curr[-1])))

                    print('      g_curr is finally ' + str(np.shape(g_curr)))
                    self.gce.append(g_curr)
                    print('      gce is ' + str(np.shape(self.gce)))

            with tf.name_scope('fe'):
                # Instead of using global averaging, we add a layer that takes each convolutional layer's receptive field
                # and only keeps the information of interest.

                self.W_layers = []
                self.b_layers = []
                self.flat_layers = []
                self.h_layers = []

                for k in range(self.Nstart, len(self.gce[-1])):
                    self.flat_layers.append(tf.transpose(tf.reshape(self.gce[k][0], [-1,
                                                                        int(self.gce[k][0].shape[1])* \
                                                                        int(self.gce[k][0].shape[2])* \
                                                                        int(self.gce[k][0].shape[3])])))

                    nmw = np.sqrt(1.0/int(self.flat_layers[k-self.Nstart].shape[0]))
                    self.W_layers.append(tf.Variable(nmw*self.init([self.n_connect,
                                                                    int(self.flat_layers[k-self.Nstart].shape[0])],
                                                                    dtype=tf.float32)))
                    self.b_layers.append(tf.Variable(tf.zeros([self.n_connect, 1], dtype=tf.float32)))
                    self.h_layers.append(
                        tf.nn.dropout(
                            self.hidden_activation(
                                tf.matmul(self.W_layers[k-self.Nstart], self.flat_layers[k-self.Nstart]) + self.b_layers[k-self.Nstart]),
                           keep_prob = self.dropout_rate_AE))

                self.fe = tf.concat(self.h_layers, axis=0)

            print('self.fe is ' + str(np.shape(self.fe)))

            self.dec_imgsize = np.concatenate((np.round(np.linspace(3, self.width, len(self.n_conv_d)+1))[np.newaxis, :],
                                               np.round(np.linspace(3, self.height, len(self.n_conv_d)+1))[np.newaxis, :]), axis=0)

            print('self.dec_imgsize is ' + str(np.shape(self.dec_imgsize)))
            print('self.dec_imgsize =' + str(self.dec_imgsize[0, :]))

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

                            nmw = np.sqrt(1.0/int(self.fe.shape[0]))
                            # Create weights for this projection
                            w_curr.append(tf.Variable(nmw*self.init([self.n_h[i], int(self.fe.shape[0])]),
                                                      name='We_' + str(j) + '_' + str(i)))

                            g_curr.append(tf.matmul(w_curr[j], self.fe))

                        else:
                            nmw = np.sqrt(1.0/self.n_h[j-1])

                            # Add weights to this projection
                            w_curr.append(tf.Variable(nmw*self.init([self.n_h[i], self.n_h[j - 1]]),
                                                      name='We_' + str(j) + '_' + str(i)))

                            g_curr.append(tf.matmul(w_curr[j], self.he[j - 1]))

                    self.We.append(w_curr)
                    self.ge.append(g_curr)
                    self.be.append(tf.Variable(tf.zeros([self.n_h[i], 1]), name='be_' + str(i - 1)))

                    nmw = np.sqrt(1.0/float(len(self.ge[i])))
                    self.he.append(tf.nn.dropout(
                        self.hidden_activation(tf.add(nmw*tf.add_n(self.ge[i]), self.be[i]),
                                               name='he_' + str(i)), keep_prob=self.dropout_rate_AE))

            # Bottleneck layer -- z
            i = len(self.n_h)
            with tf.name_scope("z"):
                self.w_z = []
                self.g_z = []
                for j in range(0, i + 1):
                    if j == 0:
                        self.w_z.append(tf.Variable(self.init([self.n_z, int(self.fe.shape[0])]),
                                                    name='We_' + str(j) + '_' + str(i)))
                        #self.g_z.append(0.0*tf.matmul(self.w_z[j], self.fe)) # We don't need this connection.

                    else:
                        nmw = np.sqrt(1.0/self.n_h[j-1])
                        self.w_z.append(tf.Variable(nmw*self.init([self.n_z, self.n_h[j - 1]]),
                                                     name='We_' + str(j) + '_' + str(i)))

                        self.g_z.append(tf.matmul(self.w_z[j], self.he[j - 1]))

                self.b_z = tf.Variable(tf.zeros([self.n_z, 1]), name='b_z')

                # We calculate z first,
                self.z_true =  tf.add(np.sqrt(1 / float(len(self.g_z))) * tf.add_n(self.g_z), self.b_z, name='z')
                # then define a copy of z that cannot propagate gradients backwards;
                self.z_stop = tf.stop_gradient(self.z_throt[:, :tf.shape(self.z_true)[1]]*tf.identity(self.z_true))
                # finally, we connect them with the clamping flags.
                self.z =self.z_throt[:, :tf.shape(self.z_true)[1]]* self.z_true * (1 - self.z_stopgrad[:, :tf.shape(self.z_true)[1]]) + \
                        self.z_stop * self.z_stopgrad[:, :tf.shape(self.z_true)[1]]

                # Reference noise (for MMD calculations).
                self.eps = tf.random_normal(tf.shape(self.z_true), mean=0.0, stddev=1.0, name='normal_samples')

            # Decoder layers (with a linear output)
            self.hd = []            # Hidden layers
            self.gd = []            # Concatenated previous layers
            self.Wd = []            # Weights
            self.bd = []            # biases

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
                            nmw = np.sqrt(1.0/self.n_z)         # Manual normalization with fan-in
                            w_curr.append(tf.Variable(nmw*self.init([self.n_h_d[i], self.n_z]),
                                                      name='Wd_' + str(j) + '_' + str(i)))

                            g_curr.append(tf.matmul(w_curr[j], self.z))
                        else:
                            nmw = np.sqrt(1.0/self.n_h_d[j-1]) # Manual normalization with fan-in
                            w_curr.append(tf.Variable(nmw*self.init([self.n_h_d[i], self.n_h_d[j - 1]]),
                                                      name='Wd_' + str(j) + '_' + str(i)))

                            g_curr.append(tf.matmul(w_curr[j], self.hd[j - 1]))

                    self.Wd.append(w_curr)
                    self.gd.append(g_curr)
                    self.bd.append(tf.Variable(tf.zeros([self.n_h_d[i], 1]), name='bd_' + str(i - 1)))

                    nmw = np.sqrt(1.0/float(len(self.gd[i])))
                    self.hd.append(tf.nn.dropout(
                        self.hidden_activation(tf.add(nmw*tf.add_n(self.gd[i]), self.bd[i]), name='hd_' + str(i)),
                        keep_prob=self.dropout_rate_AE))

            # Generate feedforward decoder's first layer
            i = len(self.n_h_d)
            print('self.fd:')
            with tf.name_scope("glob_avg_fd"):
                self.w_fd = []
                self.g_fd = []

                for j in range(0, i + 1):
                    print('Connection j = ' + str(j - 1) + ' --> ' + str(i))
                    print('n_d[0:' + str(i) + '] is ' + str(self.n_h_d[0:i]))
                    if j == 0:
                        nmw = np.sqrt(1.0/self.n_z)
                        self.w_fd.append(tf.Variable(nmw*self.init([int(self.fe.shape[0]), self.n_z]), name='W_' + str(j) + '_fd'))
                        # Ignore direct connection with z, this forces at least a single hidden layer.
                        #self.g_fd.append(0.0*tf.matmul(self.w_fd[j], self.z))

                    else:
                        nmw = np.sqrt(1.0/self.n_h_d[j-1])
                        self.w_fd.append(tf.Variable(nmw*self.init([int(self.fe.shape[0]), self.n_h_d[j - 1]]),
                                                       name='W_' + str(j) + '_fd'))
                        self.g_fd.append(tf.matmul(self.w_fd[j], self.hd[j - 1]))

                self.b_fd = tf.Variable(tf.zeros([int(self.fe.shape[0]), 1]), name='b_fd')

                nmw = np.sqrt(1.0/float(len(self.g_fd)))
                self.fd_brute = tf.nn.dropout(self.hidden_activation(tf.add(nmw*tf.add_n(self.g_fd), self.b_fd), name='fd'), keep_prob=self.dropout_rate_AE)

                # Re-generate fully connected layer to a proper convolutional layer.
                nmw = np.sqrt(1.0/int(self.fd_brute.shape[0]))
                self.W_start = tf.Variable(nmw*self.init([self.n_conv_d[0]*16, int(self.fd_brute.shape[0])], dtype=tf.float32))
                self.b_start = tf.Variable(tf.zeros([int(self.n_conv_d[0]*16), 1], dtype=tf.float32))
                self.h_start = tf.nn.dropout(
                                    self.hidden_activation(tf.matmul(self.W_start, self.fd_brute) + self.b_start),
                                    keep_prob=self.dropout_rate_AE)

                self.conv_start = tf.reshape(tf.transpose(self.h_start),
                                             [-1, 4, 4, int(int(self.h_start.shape[0])/16)])


            ''' Convolutional part of decoder '''
            self.Wcd = []               # Filters / kernels
            self.Wcd_confd = []         # Connection from fd layer into connect layer
            self.bcd_confd = []         # Bias from fd layer into connect layer
            self.hcd_confd = []         # Connected units
            self.Wcd_fd = []            # Additional bias from fd layer
            self.bcd_fd = []            # Input to activation function that is added to each convolution via bias_add.
            self.hcd_fd = []            # Densely connected layers that are included in the convolution
            self.bcd = []               # Convolutional bias
            self.gcd = []               # Reference images
            self.gcd_connected = []     # Connected reference images
            self.hcd = []               # Current layer

            self.gcd.append([self.conv_start])

            for i in range(1, np.size(self.n_conv_d)+1):
                with tf.name_scope('conv_decoder_l' + str(i)):
                    # Current list of inputs
                    g_curr = []
                    for j in range(0, i + 1):
                        # print('j = ' + str(j))
                        if j == 0:
                            # Generate weights for concatenated input
                            print('contatenated self.gcd[' + str(i - 1) + '], which is ' + str(np.shape(self.gcd[i - 1])))
                            print(str(self.gcd[i - 1]))

                            # Concatenate all previous convolutional layers
                            self.gcd_connected.append(tf.concat(self.gcd[i - 1], axis=3))
                            print('gcd_connected[i=' + str(i - 1) + '] is ' + str(np.shape(self.gcd_connected[i-1])))

                            # Normal weights and biases for convolutional op
                            nmw = np.sqrt(1.0 / (int(self.gcd_connected[i - 1].shape[3])*self.fs_d[i - 1] ** 2))
                            self.Wcd.append(tf.Variable(nmw*self.init([self.fs_d[i - 1], self.fs_d[i - 1],
                                                                        int(self.gcd_connected[i-1].shape[3])+self.n_convhelp,
                                                                        self.n_conv_d[i - 1]],
                                                                        dtype=tf.float32), name='Wcd' + str(i)))
                            self.bcd.append(tf.Variable(tf.zeros([self.n_conv_d[i - 1]], dtype=tf.float32), name='bcd' + str(i)))

                            # Additional weight matrix for direct connection to fd (helper layer).
                            # These create a layer of limited capacity
                            nmw = np.sqrt(1.0 / int(self.fd_brute.shape[0]))
                            self.Wcd_confd.append(tf.Variable(nmw * self.init([self.n_connect,
                                                                            int(self.fd_brute.shape[0])], dtype=tf.float32)))
                            self.bcd_confd.append(tf.Variable(tf.zeros([self.n_connect, 1])))
                            self.hcd_confd.append(self.hidden_activation(tf.matmul(self.Wcd_confd[i-1], self.fd_brute) + self.bcd_confd[i-1]))

                            # These other layers connect the limited-capacity layer to a small feature map.
                            # This imitates CoordConv layers, but instead of just using positions (x,y) we allow the net
                            # to decide what it needs to send forwards.
                            nmw = np.sqrt(1.0/int(self.n_connect))
                            self.Wcd_fd.append(tf.Variable(nmw * self.init([int(self.dec_imgsize[0, i])*int(self.dec_imgsize[1, i])*self.n_convhelp,
                                                                            self.n_connect], dtype=tf.float32)))
                            self.bcd_fd.append(tf.Variable(tf.zeros([int(self.dec_imgsize[0, i])*int(self.dec_imgsize[1, i])*self.n_convhelp, 1])))

                            self.hcd_fd.append(self.hidden_activation(
                                                        tf.reshape(
                                                            tf.transpose(tf.matmul(self.Wcd_fd[i-1],
                                                                                   self.hcd_confd[i-1]) + self.bcd_fd[i-1]),
                                                            [-1,
                                                            int(self.dec_imgsize[0, i]),
                                                            int(self.dec_imgsize[1, i]),
                                                            self.n_convhelp]
                                                        )))


                            # Concatenate both the previous layers and the helper layer, and then perform a convolution.
                            g_curr.append(
                                tf.nn.dropout(
                                    self.hidden_activation(
                                        tf.nn.bias_add(
                                            tf.nn.conv2d(
                                                tf.concat([tf.image.resize_images(self.gcd_connected[i-1],
                                                                                    size = (int(self.dec_imgsize[0, i]),
                                                                                            int(self.dec_imgsize[1, i])),
                                                                                    align_corners=True,
                                                                                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR),
                                                            self.hcd_fd[i-1]],
                                                            axis=3),
                                                self.Wcd[i - 1], strides = [1, 1, 1, 1], padding='SAME'),
                                            self.bcd[i - 1]),
                                        name='hcd' + str(i)),
                                    keep_prob=self.dropout_rate_AE,
                                    noise_shape=[tf.shape(self.z)[1], 1, 1, self.n_conv_d[i - 1]]))

                            print('   g_curr[j=' + str(j) + '] is ' + str(np.shape(g_curr[0])))
                        else:
                            # Resize each of the previous layers for the next layer to concatenate and filter.
                            g_curr.append(tf.image.resize_images(self.gcd[i - 1][j - 1],
                                                                 size=(int(g_curr[0].shape[1]),
                                                                       int(g_curr[0].shape[2])),
                                                                 align_corners=True,
                                                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR))
                            print('   g_curr[j=' + str(j) + '] is ' + str(np.shape(g_curr[-1])))

                    print('      g_curr is finally ' + str(np.shape(g_curr)))
                    self.gcd.append(g_curr)
                    print('      gcd is ' + str(np.shape(self.gcd)))

            # Output layer.
            with tf.name_scope('Xhat'):
                # The output layers take a massive set of concatenated images and features.
                self.gcd_connected.append(tf.concat(self.gcd[- 1], axis=3))
                nmw = np.sqrt(1.0/(int(self.gcd_connected[-1].shape[3])*self.fs_d[-1]**2))
                self.W_Xhat = tf.Variable(nmw*self.init([self.fs_d[-1],
                                                         self.fs_d[-1],
                                                         int(self.gcd_connected[-1].shape[3]),
                                                         self.channels],
                                                         dtype=tf.float32), name='W_Xhat' + str(i))
                self.b_Xhat = tf.Variable(tf.zeros([self.channels], dtype=tf.float32), name='bcd' + str(i))
                self.Xhat = tf.nn.bias_add(tf.nn.conv2d(
                                            tf.image.resize_images(self.gcd_connected[-1],
                                                                   size=(self.width, self.height),
                                                                   method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                                                                   align_corners=True),
                                            self.W_Xhat,
                                            strides=[1, 1, 1, 1], padding='SAME'),
                                            self.b_Xhat, name='Xhat')

            # These are the inputs being provided to the discriminator.
            # It's a copy of the input (X) and the autoencoder reconstruction (Xhat).
            self.X_vals = tf.concat([self.X, self.Xhat], axis=0)  # tensors are now (2*mbs, nx, ny, nch)
            self.X_d = self.X_vals + self.eps_throttle * tf.random_normal(tf.shape(self.X_vals), mean=0, stddev=1.0)
            self.Xsize = tf.shape(self.X)[0]

            # The output of the discriminator must be two flags:
            self.d = tf.concat([tf.concat([tf.ones([1, self.Xsize]), tf.zeros([1, self.Xsize])], axis=0),
                                tf.concat([tf.zeros([1, self.Xsize]), tf.ones([1, self.Xsize])], axis=0)], axis=1)

        with tf.name_scope('Discriminator'):
            ''' Convolutional part of encoder '''
            self.disc_Wce = []                  # Discriminator convolutional filters
            self.disc_bce = []                  # Convolutional layer biases
            self.disc_gce = []                  # List of individual previous convolutional layers
            self.disc_gce_connected = []        # Concatenated previous layers
            self.disc_hce = []                  # Current layer

            self.disc_gce.append([self.X_d])    # The first of concatenated layers is the input tensor.

            for i in range(1, np.size(self.n_conv) + 1):
                with tf.name_scope('conv_encoder_l' + str(i)):
                    # Current list of inputs
                    g_curr = []
                    for j in range(0, i + 1):
                        if j == 0:
                            # Generate weights for concatenated input
                            print('contatenated self.gce[' + str(i - 1) + '], which is ' + str(np.shape(self.disc_gce[i - 1])))
                            print(str(self.disc_gce[i - 1]))
                            self.disc_gce_connected.append(tf.concat(self.disc_gce[i - 1], axis=3))
                            print('gce_connected[i=' + str(i - 1) + '] is ' + str(np.shape(self.disc_gce_connected[i - 1])))

                            # Weights and biases with
                            nmw =  np.sqrt(1.0/(int(self.disc_gce_connected[i-1].shape[3])*self.fs[i-1]**2))
                            self.disc_Wce.append(tf.Variable(nmw*self.init([self.fs[i - 1],
                                                                            self.fs[i - 1],
                                                                            int(self.disc_gce_connected[i - 1].shape[3]),
                                                                            self.n_conv[i - 1]],
                                                                           dtype=tf.float32), name='disc_Wce' + str(i)))

                            self.disc_bce.append(tf.Variable(self.init([self.n_conv[i - 1]],
                                                                       dtype=tf.float32), name='disc_bce' + str(i)))

                            g_curr.append(
                                tf.nn.dropout(self.gan_activation(
                                    tf.nn.bias_add(
                                        tf.nn.conv2d(self.disc_gce_connected[i - 1],
                                                     self.disc_Wce[i - 1],
                                                     strides=[1, self.npad[i - 1], self.npad[i - 1], 1],
                                                     padding='SAME'),
                                        self.bce[i - 1]), name='disc_hce' + str(i)),
                                    keep_prob=self.dropout_rate_Disc,
                                    noise_shape=[tf.shape(self.X_d)[0], 1, 1, self.n_conv[i - 1]]))
                            print('   g_curr[j=' + str(j) + '] is ' + str(np.shape(g_curr[0])))
                        else:
                            # Resize previous layers
                            g_curr.append(tf.image.resize_images(self.disc_gce[i - 1][j - 1],
                                                                 size=(int(g_curr[0].shape[1]),
                                                                       int(g_curr[0].shape[2])),
                                                                 align_corners=True,
                                                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR))
                            print('   g_curr[j=' + str(j) + '] is ' + str(np.shape(g_curr[-1])))

                    print('      g_curr is finally ' + str(np.shape(g_curr)))
                    self.disc_gce.append(g_curr)
                    print('      gce is ' + str(np.shape(self.disc_gce)))

            # Get the last layer included to disc_gce
            self.disc_gce_connected.append(tf.concat(self.disc_gce[-1], axis=3))

            with tf.name_scope('disc_fe'):
                # Weighted global averaging vector.
                self.disc_W_layers = []             # Weights
                self.disc_b_layers = []             # Biases
                self.disc_flat_layers = []          # Flattened versions of certain layers
                self.disc_h_layers = []             #

                for k in range(self.Nstart, len(self.gce[-1])):
                    self.disc_flat_layers.append(tf.transpose(tf.reshape(self.disc_gce[k][0],
                                                                         [-1,
                                                                         int(self.disc_gce[k][0].shape[1]) * \
                                                                         int(self.disc_gce[k][0].shape[2]) * \
                                                                         int(self.disc_gce[k][0].shape[3])]
                                                                         )))

                    nmw = np.sqrt(1.0 / int(self.disc_flat_layers[k - self.Nstart].shape[0]))
                    self.disc_W_layers.append(
                        tf.Variable(nmw * self.init([self.n_connect,
                                                     int(self.disc_flat_layers[k - self.Nstart].shape[0])],
                                                    dtype=tf.float32)))
                    self.disc_b_layers.append(tf.Variable(tf.zeros([self.n_connect, 1], dtype=tf.float32)))
                    self.disc_h_layers.append(tf.nn.dropout(self.gan_activation(
                        tf.matmul(self.disc_W_layers[k - self.Nstart], self.disc_flat_layers[k - self.Nstart]) + self.disc_b_layers[k - self.Nstart]),
                                                       keep_prob=self.dropout_rate_Disc))

                self.disc_fe = tf.concat(self.disc_h_layers, axis=0)


            print('self.fe is ' + str(np.shape(self.fe)))

            # Encoder layers (with a linear output)
            self.disc_he = []
            self.disc_ge = []
            self.disc_We = []
            self.disc_be = []

            print('Encoder feedforward:')
            # i-th hidden layer (starts at 1)
            for i in range(0, len(self.n_h)):
                print('Layer i = ' + str(i))
                with tf.name_scope('disc_encoder_l' + str(i)):
                    w_curr = []
                    g_curr = []
                    for j in range(0, i + 1):
                        print('Connection j = ' + str(j - 1) + ' --> ' + str(i))
                        print('n_e[0:' + str(i) + '] is ' + str(self.n_h[0:i]))
                        if j == 0:
                            nmw = np.sqrt(1.0 / int(self.disc_fe.shape[0]))
                            # Create weights
                            w_curr.append(tf.Variable(nmw * self.init([self.n_h[i], int(self.disc_fe.shape[0])]),
                                                      name='disc_We_' + str(j) + '_' + str(i)))
                            g_curr.append(tf.matmul(w_curr[j], self.disc_fe))

                        else:
                            nmw = np.sqrt(1.0 / self.n_h[j - 1])

                            # Create weights
                            w_curr.append(tf.Variable(nmw * self.init([self.n_h[i], self.n_h[j - 1]]),
                                                      name='disc_We_' + str(j) + '_' + str(i)))
                            g_curr.append(tf.matmul(w_curr[j], self.disc_he[j - 1]))

                    self.disc_We.append(w_curr)
                    self.disc_ge.append(g_curr)
                    self.disc_be.append(tf.Variable(tf.zeros([self.n_h[i], 1]), name='disc_be_' + str(i - 1)))

                    nmw = np.sqrt(1.0 / float(len(self.disc_ge[i])))
                    self.disc_he.append(tf.nn.dropout(
                        self.gan_activation(tf.add(nmw * tf.add_n(self.disc_ge[i]), self.disc_be[i]),
                                               name='he_' + str(i)), keep_prob=self.dropout_rate_Disc))

            # Generate now the mu and sigma layer
            i = len(self.n_h)
            print('Discriminator output:')
            print('z: i = ' + str(i))

            with tf.name_scope("dhat"):
                self.w_dhat = []
                self.g_dhat = []

                for j in range(0, i + 1):
                    print('dhat: j=' + str(j - 1) + '--> ' + str(i))
                    print('n_e[0:' + str(i) + '] is ' + str(self.n_h[0:i]))
                    if j == 0:
                        # Create weights
                        self.w_dhat.append(tf.Variable(self.init([2, int(self.disc_fe.shape[0])]),
                                        name='We_' + str(j) + '_' + str(i)))
                        #self.g_dhat.append(0.0 * tf.matmul(self.w_dhat[j], self.disc_fe))

                    else:
                        nmw = np.sqrt(1.0/self.n_h[j-1])
                        self.w_dhat.append(tf.Variable(nmw*self.init([2, self.n_h[j - 1]]), name='dhat_We_' + str(j) + '_' + str(i)))
                        self.g_dhat.append(tf.matmul(self.w_dhat[j], self.disc_he[j - 1]))

                nmw = np.sqrt(1.0 / float(len(self.g_dhat)))
                self.b_dhat = tf.Variable(tf.zeros([2, 1]), name='b_dhat')
                self.dhat_logits = tf.add(nmw * tf.add_n(self.g_dhat), self.b_dhat, name='dhat')
                self.dhat = tf.nn.softmax(self.dhat_logits, axis=0)  # Independent probabilities

            # Get fake values
            self.fake_X = self.X_d[self.Xsize:, :, :, :]
            self.fake_dhat_logits = self.dhat_logits[:, self.Xsize:]
            self.fake_dhat = self.dhat[:, self.Xsize:]
            self.fake_d = tf.concat([tf.ones([1, self.Xsize]), tf.zeros([1, self.Xsize])], axis=0)  # should be 'real'

    # Other methods required to circunvent tensorflow
    start = common.start                                            # Initialize parameters and session

    compute_kernel = common.compute_kernel                          # Functions for MMD loss
    compute_mmd = common.compute_mmd
    define_loss = common.conv_infovaegan_loss

    lr_finder = common.mlp_aegan_gated_lrf                              # Learning rate finder (testing)
    train = common.train_convolutional_aegan_with_gating_parallel       # Training funcstions
    output = common.output_conv_ae_with_gating                          # similar to Torch's forward(x). Generates xhat, z
    show_reconstructions = common.show_reconstructions_conv_ae          # Small demonstrator, shows reconstructions.
    show_map = common.show_map_conv                                     # Shows a 2D map of the first 2 z-coordinates
    synthesize = common.synthesize_conv                                 # Produces Xhat from a given z.

    save_model = common.save_model                                      # Functions for saving and loading the model
    load_model = common.load_model                                      # from a given path.


if __name__ == '__main__':
    ''' Load dataset '''
    import ai.common_ops.datasets as dts
    training_data = dts.get_cifar10(mlp=False)

    ''' Network instance and parameters '''
    nconv = [32, 32, 32, 32,
             32, 32, 32, 32,
             32, 32, 32, 32,
             32, 32, 32]
    fs = [3, 3, 3, 3,
          3, 3, 3, 3,
          3, 3, 3, 3,
          3, 3, 3]

    npad = [1, 1, 1, 2,
            1, 1, 1, 2,
            1, 1, 1, 2,
            1, 1, 1, 2,
            1, 1, 1]
    nh = [300, 300, 300]

    x_dims = [32, 32, 3]
    n_z = 64

    network = ConvolutionalInfoSCVAE_GCN_gated_encoded(nconv, fs, npad, nh, n_z, x_dims, fresh_init=1,
                                                       mbs=16, cost_function='MSE', Nstart=5, n_connect=150,
                                                       n_convhelp=10,
                                                       gan_weight=0.0,
                                                       actype='lrelu', name='conv_infoscvae_gcn_cifar10',
                                                       reset_default_graph=1)

    # Test for Cyclic Learning Rate training
    training_schedule = {
        'mode': 'ocp',
        'eval_criterion': 'iter',
        'stop_criterion': 'none',
        'lr_max': 0.0001,
        'lr_min': 0.0,
        'beta_start': 1.0,
        'beta_end': 1.0,
        'eps_start': 0.0,
        'eps_end': 0.0,
        'Nwarm': 1,
        'Ncool': 300000,
        'Nwarmup': 1,
        'T': 1024,
        'Niters': 600000,
        'dropout_rate': 0.95,
        'max_patience': 50,
        'eval_size': 3000
    }
    # lrfinder_params = network.lr_finder(training_data, training_schedule)
    # training_data['lr_max'] = lrfinder_params['lr_first']
    network.train(training_data, training_schedule, show=1, saveframes=0)

    X_train = training_data['X_train']
    X_test = training_data['X_test']
    y_train = training_data['y_train']
    y_test = training_data['y_test']

    # Looking at bottleneck data
    fig2 = plt.figure(2)
    fig2.clf()
    Xhat, zhat = network.output(X_test[::100, :, :, :])
    ax_list = []
    N = 3 ** 2
    nrows = int(np.ceil(np.sqrt(N)))
    for k in range(100, 110):
        ax_list.append(fig2.add_subplot(nrows, nrows, k + 1 - 100))
        ax_list[k - 100].scatter(zhat[k, :], zhat[k + 1, :], marker='.', edgecolor='none', alpha=0.5, c=y_test[::100])
    fig2.tight_layout()

    width = 32
    # Try to vary generative results
    xvals = np.linspace(-4, 4, 30)
    BigPic = np.zeros([width * 16, width * np.size(xvals)])
    _, initial_random = network.output(X_train[13:20, :, :, :])
    initial_random = initial_random[:, 3]
    for i in range(0, 16):
        for j in range(0, np.size(xvals)):
            input_rand = initial_random.copy()
            input_rand[i] = xvals[j]
            BigPic[i * width:(i + 1) * width, j * width:(j + 1) * width] = np.squeeze(
                network.synthesize(input_rand[:, np.newaxis]))

    fig3 = plt.figure(3)
    ax31 = fig3.add_subplot(1, 1, 1)
    ax31.clear()
    ax31.imshow(BigPic, cmap='gray')
    fig3.tight_layout()

    fig4 = plt.figure(4)
    ax41 = fig4.add_subplot(1, 1, 1)
    ax41.clear()
    N = n_z
    BigPic2 = np.zeros([width * N, width * n_z])
    _, initial_values = network.output(X_train[0:N, :, :, :])
    for i in tqdm(range(0, 16)):
        BigPic2[i * width:(i + 1) * width, 0:width] = np.squeeze(X_train[i, :, :, :])
        for j in range(1, n_z):
            zvals = np.copy(initial_values[:, i][:, np.newaxis])
            zvals[j:, :] = 0
            BigPic2[i * width:(i + 1) * width, j * width:(j + 1) * width] = np.squeeze(network.synthesize(zvals))

    ax41.imshow(BigPic2, cmap='gray')
    fig4.canvas.draw()
    fig4.canvas.flush_events()

    N = 16
    Nt = 20
    z = np.random.randn(N, Nt * Nt)
    z[3:, :] = 0

    const = 2.0
    z[4, 200:] -= const
    z[4, :200] += const
    z[7, 200:] += const
    z[7, 200:] -= const
    xsynth = network.synthesize(z)
    fig6 = plt.figure(6)
    ax61 = fig6.add_subplot(1, 1, 1)
    BigPic = np.zeros([width * Nt, width * Nt])
    m = 0
    for i in range(0, Nt):
        for j in range(0, Nt):
            BigPic[i * width:(i + 1) * width, j * width:(j + 1) * width] = np.squeeze(xsynth[m, :, :, :])
            m += 1
    ax61.clear()
    ax61.imshow(BigPic, cmap='gray', vmin=0, vmax=1)
    ax61.axis('off')
    fig6.tight_layout()
    fig6.canvas.draw()
    fig6.canvas.flush_events()
    fig6.canvas.flush_events()

    '''
    # Quick checks of hidden layer distributions
    fig1 = plt.figure(1);
    ax11 = fig1.add_subplot(1, 1, 1);
    Xhat, zhat = network.output(np.random.randn(1000, width, width, 1))

    ax11.clear()

    # Normal distribution
    ax11.hist(np.ndarray.flatten(np.random.randn(100000)), color='k', bins=200, density=True, label='Std Gaussian')

    # Last encoder convolutional layer distribution
    gcevals = network.gce_connected[2].eval(session=network.sess,
                                             feed_dict={network.X: np.random.randn(100, width, width, 1),
                                                        network.dropout_rate: 1.0})
    ax11.hist(np.ndarray.flatten(gcevals), bins=100, density=True, label='gcd')

    # Flatten distribution
    fevals = network.fe.eval(session=network.sess,
                             feed_dict={network.X: np.random.randn(1000, width, width, 1), network.dropout_rate: 1.0})
    ax11.hist(np.ndarray.flatten(fevals), color=[0.5, 0.5, 0.0], bins=200, density=True, label='fe')

    fdvals = network.conv_start.eval(session=network.sess,
                             feed_dict={network.X: np.random.randn(1000, width, width, 1), network.dropout_rate: 1.0,
                                        network.z_throt: np.ones([nz, 1]), network.z_stopgrad: np.zeros([nz, 1])})
    ax11.hist(np.ndarray.flatten(fdvals), color=[0.5, 0.0, 0.5], bins=200, density=True, label='fd')

    # Convolutional layer distribution
    gcdvals = network.gcd_connected[-1].eval(session=network.sess,
                                             feed_dict={network.X: np.random.randn(100, width, width, 1),
                                                        network.dropout_rate: 1.0,
                                                        network.z_throt: np.ones([nz, 1]), network.z_stopgrad: np.zeros([nz, 1])})
    ax11.hist(np.ndarray.flatten(gcdvals), color=[0.5, 0.5, 0.5], bins=100, density=True, label='gcd')

    # Bottleneck distribution
    ax11.hist(np.ndarray.flatten(zhat), color='g', bins=200, density=True, label='Bottleneck')

    # Output distribution
    ax11.hist(np.ndarray.flatten(Xhat), color='b', bins=200, density=True, label='Output')

    ax11.legend(fancybox=True)

    '''
