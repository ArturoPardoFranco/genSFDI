'''
Feedforward skip-connected Variational Autoencoder (SC-InfoVAE).
Arturo Pardo, 2019
'''

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from datetime import datetime
import csv, os
dir_path = os.path.dirname(os.path.abspath(__file__))
import ai.common_ops as common

''' Standard autoencoder class '''
class MLPInfoSCVAE:
    def __init__(self, n_encoder, n_decoder, n_X, n_z, lr=0.0001, actype='elu', optim='Adam', mbs=16, regeps=0.1,
                 cost_function='MSE', fixed_lr=False, quiet = 0, debug = 0,
                 fresh_init=1, name='mlp_infoscvae', reset_default_graph=1, write_path='../../../output/network_tests'):

        # Debug and quiet params
        self.debug = debug
        self.quiet = quiet

        # Is this a new network or a pretrained one?
        self.fresh_init = fresh_init
        self.name = name

        # Epsilon for regularizing relative MSE
        self.regeps = regeps

        # Order to reset default graph when instanced
        self.reset_default_graph = reset_default_graph

        # First iteration detector
        if self.fresh_init == 1:
            self.first_iter = 1
        else:
            self.first_iter = 0

        # Optimizer choice
        self.optim = optim

        # Layers and dimensions of encoder and decoder:
        self.n_e = n_encoder
        self.n_d = n_decoder

        # Dimension of original space:
        self.n_X = n_X

        # Dimension of normal latent space:
        self.n_z = n_z

        # Learning rate, cost function choice and fixed learning rate
        self.cost_function = cost_function
        self.lr = lr
        self.fixed_lr = fixed_lr

        # Minibatch size (for learning)
        self.mbs = int(mbs)

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

        # Activation type for hidden layers
        self.actype = actype

        activations = {
            'lrelu': tf.nn.leaky_relu,
            'elu': tf.nn.elu,
            'tanh': tf.nn.tanh,
            'softplus': tf.nn.softplus,
            'sigmoid': tf.nn.sigmoid,
            'softsign': tf.nn.softsign,
            'siren': tf.math.sin,
            'softmax': tf.nn.softmax,
            'linear': tf.identity,
        }
        self.hidden_activation = activations.get(self.actype, None)
        if self.hidden_activation is None:
            raise NotImplementedError('Hidden activation function not implemented. Try:' + str(activations.keys()))

        # Prepare graph, define loss function, and configure session parameters
        self.prepare_graph()

        if self.fresh_init==1:
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
        self.placeholder_X = tf.placeholder(tf.float32, [self.n_X, None], name='placeholder_X')

        # Generate dataset for placeholder (incl. randomization) and batch learning schedule
        self.dataset = tf.data.Dataset.from_tensor_slices(tf.transpose(self.placeholder_X))
        self.dataset = self.dataset.shuffle(buffer_size=100000)
        #self.dataset = self.dataset.repeat(10)
        self.dataset = self.dataset.batch(self.mbs)

        # This is the iterator that goes through the introduced dataset
        self.iterator = self.dataset.make_initializable_iterator()
        self.data_x = self.iterator.get_next()

        # Input layer
        #self.X = tf.placeholder(tf.float32, shape=(self.n_X, None), name='X')
        self.X = tf.transpose(tf.identity(self.data_x))

        # Variables: noise throttle, KL-divergence weight, dropout rate (outdated, to fix soon)
        self.eps_throttle = tf.placeholder(tf.float32, shape=(), name='throttle')
        self.kl_weight = tf.placeholder(tf.float32, name="kl_weight")
        self.dropout_rate = tf.placeholder(tf.float32, shape = (), name="dropout_rate")
        self.learning_rate = tf.placeholder(tf.float32, shape= (), name='learning_rate')

        # Parameter initialization - Kaiming with changes
        self.init = tf.random_normal_initializer(mean=0.0, stddev=1.0)

        # Encoder layers (with a linear output)
        self.he = []
        self.ge = []
        self.We = []
        self.be = []

        if self.debug == 1: print('Encoder:')
        # i-th hidden layer (starts at 1)
        for i in range(0, len(self.n_e)):
            if self.debug == 1: print('Layer i = ' + str(i))
            with tf.name_scope('encoder_l' + str(i)):
                w_curr = []
                g_curr = []
                for j in range(0, i+1):
                    if self.debug == 1: print('Connection j = ' + str(j-1) + ' --> ' + str(i))
                    if self.debug == 1: print('n_e[0:' + str(i) + '] is ' + str(self.n_e[0:i]))
                    if j == 0:
                        # For the i-th layer you have to find the fan-in
                        nmw = np.sqrt(2.0) / np.sqrt(self.n_X)

                        # Create weights
                        w_curr.append(tf.Variable(self.init([self.n_e[i], self.n_X])*nmw, name='We_' + str(j) + '_' + str(i)))
                        g_curr.append(tf.matmul(w_curr[j], self.X))

                    else:
                        nmw = np.sqrt(2.0) / np.sqrt(self.n_e[j-1])

                        # Create weights
                        w_curr.append(tf.Variable(self.init([self.n_e[i], self.n_e[j-1]])*nmw, name='We_' + str(j) + '_' + str(i)))
                        g_curr.append(tf.matmul(w_curr[j], self.he[j - 1]))

                self.We.append(w_curr)
                self.ge.append(g_curr)
                self.be.append(tf.Variable(tf.zeros([self.n_e[i], 1]), name='be_' + str(i - 1)))

                self.he.append(tf.nn.dropout(
                    self.hidden_activation(tf.add(
                        np.sqrt(1/float(len(self.ge[i])))*tf.add_n(self.ge[i]),
                        self.be[i]), name='he_' + str(i)), keep_prob=self.dropout_rate))

        # Generate now the mu and sigma layer
        i = len(self.n_e)
        if self.debug == 1: print('Latent space:')
        if self.debug == 1: print('z: i = ' + str(i))

        with tf.name_scope("z"):
            self.w_z = []
            self.g_z = []

            for j in range(0, i+1):
                if self.debug == 1: print('z: j=' + str(j-1) + '--> ' + str(i))
                if self.debug == 1: print('n_e[0:' + str(i) + '] is ' + str(self.n_e[0:i]))
                if j == 0:
                    nmw = np.sqrt(2.0/self.n_X)
                    # Create weights
                    self.w_z.append(tf.Variable(nmw*self.init([self.n_z, self.n_X]), name='We_' + str(j) + '_' + str(i)))
                    #self.g_z.append(tf.matmul(self.w_z[j], self.X))

                else:
                    nmw = np.sqrt(2.0 / self.n_e[j-1])
                    self.w_z.append(tf.Variable(nmw*self.init([self.n_z, self.n_e[j-1]]), name='We_' + str(j) + '_' + str(i)))
                    self.g_z.append(tf.matmul(self.w_z[j], self.he[j - 1]))

            self.b_z = tf.Variable(tf.zeros([self.n_z, 1]), name='b_z')
            self.z = tf.add(np.sqrt(1/float(len(self.g_z)))*tf.add_n(self.g_z), self.b_z, name='mu')

        # Random sample generator
        self.eps = tf.random_normal(tf.shape(self.z), mean=0.0, stddev=1.0, name='normal_samples')

        # Decoder layers (with a linear output)
        self.hd = []
        self.gd = []
        self.Wd = []
        self.bd = []

        if self.debug == 1: print('Decoder:')
        # i-th hidden layer (starts at 1)
        for i in range(0, len(self.n_d)):
            if self.debug == 1: print('Layer i = ' + str(i))
            with tf.name_scope('decoder_l' + str(i)):
                w_curr = []
                g_curr = []
                for j in range(0, i + 1):
                    if self.debug == 1: print('Connection j = ' + str(j - 1) + ' --> ' + str(i))
                    if self.debug == 1: print('n_d[0:' + str(i) + '] is ' + str(self.n_d[0:i]))
                    if j == 0:
                        # For the i-th layer you have to find the fan-in
                        nmw = np.sqrt(2.0 /self.n_z)
                        # Create weights
                        w_curr.append(tf.Variable(nmw*self.init([self.n_d[i], self.n_z]), name='Wd_' + str(j) + '_' + str(i)))
                        g_curr.append(tf.matmul(w_curr[j], self.z))

                    else:
                        nmw = np.sqrt(2.0 / self.n_d[j-1])
                        w_curr.append(tf.Variable(nmw*self.init([self.n_d[i], self.n_d[j - 1]]), name='Wd_' + str(j) + '_' + str(i)))
                        g_curr.append(tf.matmul(w_curr[j], self.hd[j - 1]))

                self.Wd.append(w_curr)
                self.gd.append(g_curr)
                self.bd.append(tf.Variable(tf.zeros([self.n_d[i], 1]), name='bd_' + str(i - 1)))

                self.hd.append(tf.nn.dropout(
                    self.hidden_activation(tf.add(tf.add_n(self.gd[i]), self.bd[i]), name='hd_' + str(i)),
                    keep_prob=self.dropout_rate))

        i = len(self.n_d)
        if self.debug == 1: print('Xhat:')
        with tf.name_scope("Xhat"):
            self.w_Xhat = []
            self.g_Xhat = []

            for j in range(0, i+1):
                if self.debug == 1: print('Connection j = ' + str(j - 1) + ' --> ' + str(i))
                if self.debug == 1: print('n_d[0:' + str(i) + '] is ' + str(self.n_d[0:i]))
                if j == 0:
                    # Create weights
                    self.w_Xhat.append(tf.Variable(self.init([self.n_X, self.n_z]), name='W_' + str(j) + '_Xhat'))
                    #self.g_Xhat.append(tf.matmul(self.w_Xhat[j], self.z))

                else:
                    nmw = np.sqrt(2.0 /self.n_d[j-1])
                    self.w_Xhat.append(tf.Variable(nmw*self.init([self.n_X, self.n_d[j - 1]]), name='W_' + str(j) + '_Xhat'))
                    self.g_Xhat.append(tf.matmul(self.w_Xhat[j], self.hd[j - 1]))

            self.b_Xhat = tf.Variable(tf.zeros([self.n_X, 1]), name='b_Xhat')

            self.Xhat = tf.add(np.sqrt(1/float(len(self.g_Xhat)))*tf.add_n(self.g_Xhat), self.b_Xhat, name='Xhat')

    # General methods
    save_model = common.save_model
    load_model = common.load_model
    start = common.start

    # Universal functions:
    lr_finder = common.standard_ae_lrf
    compute_mmd = common.compute_mmd
    compute_kernel = common.compute_kernel
    define_loss = common.mlp_infovae_loss                           # Loss function(s) for InfoVAE autoencoders

    train = common.train_mlp_ae                                     # Train function for multilayer autoencoders
    output = common.output_mlp_ae                                   # Output function for multilayer autoencoders
    show_reconstructions = common.show_reconstructions_mlp_ae       # Show reconstructions for multilayer autoencoders
    show_map = common.show_map_mlp                                  # Show map for multilayer autoencoders
    synthesize = common.synthesize_mlp                              # Synthesis for multilayer autoencoders


if __name__ == '__main__':
    ''' Prepare datasets'''
    import ai.common_ops.datasets as dts
    training_data = dts.get_mnist(mlp=True)

    ''' Network instance and parameters '''
    n_encoder = [300, 300, 300, 300]
    n_decoder = [300, 300, 300, 300]
    n_X = 784
    n_z = 2

    network = MLPInfoSCVAE(n_encoder, n_decoder, n_X, n_z, lr=0.001, cost_function='MSE',
                                optim='Adam', fresh_init=1, mbs=32, actype='lrelu', name='mlp_infovae_test')


    # Test for Cyclic Learning Rate training
    training_schedule = {
        'mode': 'normal',
        'eval_criterion': 'iter',
        'stop_criterion': 'none',
        'lr_start': 0.0001,
        'lr_end': 0.0,
        'beta_start': 1.0,
        'beta_end': 1.0,
        'Nwarm': 1,
        'Ncool': 1,
        'Nwarmup': 1,
        'mult': 1.0,
        'T': 2000,
        'Niters': 200000,
        'dropout_rate': 0.95,
        'max_patience': 50,
        'eval_size': 5000
    }
    network.train(training_data, training_schedule, show=1)

    Xhat, zhat = network.output(X_train, quiet_progress=0)

    network.show_map(Ntest=40, zspan=1.5)

    fig1 = plt.figure(1)
    ax11 = fig1.add_subplot(1, 1, 1)
    ax11.clear()
    ax11.scatter(zhat[0, :], zhat[1, :], 30, c=y_train, cmap='jet', edgecolor='none', marker='.', alpha=1)
    fig1.canvas.draw()
    fig1.canvas.flush_events()

