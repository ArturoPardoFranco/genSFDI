'''
Least-Squares GAN
Arturo Pardo, 2020
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

'''
LSGANs with many Ds don't suffer from mode collapse and are pretty much identical to the original MLP GANs. 
'''
class mlpLSGAN_multiD:
    def __init__(self, n_gen, n_disc, n_z, n_x,lr=0.0001, optim='Adam', actype='elu', mbs=16,
                 cost_function='MSE', fixed_lr=False, regeps=0.1, quiet = 0, debug = 0, N_disc = 10,
                 fresh_init=1, name='model', reset_default_graph=1, write_path='../../../output/network_tests'):

        # Is this a new network or a pretrained one?
        self.fresh_init = fresh_init
        self.name = name

        # Debug and quiet params
        self.debug = debug
        self.quiet = quiet

        # Order to reset default graph when instanced
        self.reset_default_graph = reset_default_graph

        # First iteration detector
        if self.fresh_init == 1:
            self.first_iter = 1
        else:
            self.first_iter = 0

        # Init_parameters
        self.n_gen = n_gen          # Number of layers and units per layer in generator MLP
        self.n_disc = n_disc        # Number of layers and units per layer in discriminator MLP
        self.n_x = n_x              # Input size
        self.n_z = n_z              # Latent / noisy dimension

        # Optimizer choice
        self.optim = optim

        # Number of discriminators
        self.N_disc = N_disc

        # Learning rate, cost function, learning rate
        self.lr = lr
        self.cost_function = cost_function
        self.fixed_lr = fixed_lr

        # Epsilon for regularizing relative MSE
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
        self.placeholder_X = tf.placeholder(tf.float32, [self.n_x, None], name='placeholder_X')

        # Generate dataset for placeholder (incl. randomization) and batch learning schedule
        self.dataset = tf.data.Dataset.from_tensor_slices(tf.transpose(self.placeholder_X))
        self.dataset = self.dataset.shuffle(buffer_size=100000)         # (!) make this bigger if necessary!
        self.dataset = self.dataset.batch(self.mbs)                     # minibatch size specified here

        # This is the iterator that goes through the introduced dataset
        self.iterator = self.dataset.make_initializable_iterator()
        self.data_x = self.iterator.get_next()

        # Input and output layers -- Y specifies condition
        self.X = tf.transpose(tf.identity(self.data_x))
        self.Xsize = tf.shape(self.X)[1]
        self.z = tf.random_uniform(shape=(self.n_z, self.Xsize), minval=-1.0, maxval=1.0)

        # Other params
        self.eps_throttle = tf.placeholder(tf.float32, shape=(), name='eps_throttle')
        self.learning_rate = tf.placeholder(tf.float32, shape=(), name='learning_rate')

        # Dropout regularization is different depending on which network you train, i.e. p_drop = 0.0 when not training
        self.dropout_rate = tf.placeholder(tf.float32, shape=(), name='dropout_rate')
        self.dropout_rate_gen = tf.identity(self.dropout_rate, name="dropout_rate_gen")
        self.dropout_rate_disc = tf.identity(self.dropout_rate, name="dropout_rate_disc")

        # We will do He's initialization but with a trick because we have skip-connections!
        self.init = tf.random_normal_initializer(mean=0.0, stddev=1.0)

        # Generator structure: g: z --> xhat
        with tf.name_scope('Generator'):

            self.W_gen = []  # List of lists of weights
            self.b_gen = []  # Final bias for layer
            self.g_gen = []  # Concatenated input list
            self.h_gen = []  # Layer values past activation function
            self.rho_gen = []   # Layer weighting (like WeightNorm but without the Norm part)

            # Bulk DenseNet MLP generation
            for i in range(0, len(self.n_gen)):
                with tf.name_scope('gen_l' + str(i)):
                    w_curr = []     # List of weights to the i-th layer
                    g_curr = []     # List of activations to the i-th layer
                    rho_curr = []   # List of modulating weights to the i-th layer
                    for j in range(0, i + 1):
                        if j == 0:
                            # Weights init
                            rho_curr.append(tf.Variable(tf.ones([1], dtype=tf.float32)))
                            nmw = np.sqrt(1.0/int(self.z.shape[0]))
                            w_curr.append(tf.Variable(nmw*self.init([self.n_gen[i], int(self.z.shape[0])]),
                                                      name='We_' + str(j) + '_' + str(i)))

                            g_curr.append(tf.matmul(w_curr[j], self.z)*rho_curr[j])

                        else:

                            # Weights init
                            rho_curr.append(tf.Variable(tf.ones([1], dtype=tf.float32)))
                            nmw = np.sqrt(1.0 / self.n_gen[j - 1])
                            w_curr.append(tf.Variable(nmw*self.init([self.n_gen[i], self.n_gen[j - 1]]),
                                                      name='We_' + str(j) + '_' + str(i)))

                            g_curr.append(tf.matmul(w_curr[j], self.h_gen[j - 1])*rho_curr[j])

                    self.W_gen.append(w_curr)
                    self.g_gen.append(g_curr)
                    self.rho_gen.append(rho_curr)

                    # Bias init
                    self.b_gen.append(tf.Variable(tf.zeros([self.n_gen[i], 1]), name='b_gen_' + str(i - 1)))

                    # i-th layer is sum of previous layers
                    nmw = np.sqrt(1.0/float(len(self.g_gen[i])))
                    self.h_gen.append(tf.nn.dropout(
                        self.hidden_activation(tf.add(nmw*tf.add_n(self.g_gen[i]), self.b_gen[i]),
                                               name='he_' + str(i)), keep_prob=self.dropout_rate_gen))

            # Last layer
            i = len(self.n_gen)
            with tf.name_scope("Xhat"):
                self.w_Xhat = []
                self.g_Xhat = []
                self.rho_Xhat = []
                for j in range(0, i + 1):
                    if j == 0:
                        # Dummy weight
                        self.w_Xhat.append(tf.Variable(self.init([1]), name='W_xhat_' + str(j) + '_' + str(i)))
                        self.rho_Xhat.append(tf.Variable(tf.ones([1], dtype=tf.float32)))
                    else:
                        # Weight initialization
                        nmw = np.sqrt(1.0 / self.n_gen[j - 1])
                        self.rho_Xhat.append(tf.Variable(tf.ones([1], dtype=tf.float32)))
                        self.w_Xhat.append(tf.Variable(nmw * self.init([self.n_x, self.n_gen[j - 1]]),
                                                    name='We_' + str(j) + '_' + str(i)))

                        self.g_Xhat.append(tf.matmul(self.w_Xhat[j], self.h_gen[j - 1])*self.rho_Xhat[j])

                # Bias initialization
                self.b_Xhat = tf.Variable(tf.zeros([self.n_x, 1]), name='b_Xhat')

                # Final shape of linear layer
                self.Xhat = tf.add(np.sqrt(1.0 / float(len(self.g_Xhat))) * tf.add_n(self.g_Xhat), self.b_Xhat, name='Xhat')

        # We combine real and fake data for the discriminator to see
        self.X_d = tf.concat([self.X, self.Xhat], axis=1)
        self.X_vals = self.X_d + self.eps_throttle*tf.random_normal(tf.shape(self.X_d), mean=0.0, stddev=1.0)

        # Discriminator structure: d: xhat, x --> (fake, real)
        with tf.name_scope('Discriminator'):

            self.W_disc_list = []
            self.b_disc_list = []
            self.g_disc_list = []
            self.h_disc_list = []

            self.w_dhat_list = []
            self.g_dhat_list = []
            self.b_dhat_list = []
            self.dhat_list = []
            self.d_list = []

            for p in range(0, self.N_disc):

                self.W_disc_list.append([])  # List of lists of weights
                self.b_disc_list.append([])  # Final bias for layer
                self.g_disc_list.append([])  # Concatenated input list
                self.h_disc_list.append([])  # Layer values past activation function

                # Bulk DenseNet MLP generation
                for i in range(0, len(self.n_disc)):
                    with tf.name_scope('disc_l' + str(i)):
                        w_curr = []  # List of weights to the i-th layer
                        g_curr = []  # List of activations to the i-th layer
                        for j in range(0, i + 1):
                            if j == 0:
                                # Weights init
                                nmw = np.sqrt(1.0 / int(self.X_vals.shape[0]))
                                w_curr.append(tf.Variable(nmw * self.init([self.n_disc[i], int(self.X_vals.shape[0])]),
                                                          name='W_disc_' + str(j) + '_' + str(i)))

                                g_curr.append(tf.matmul(w_curr[j], self.X_vals))

                            else:

                                # Weights init
                                nmw = np.sqrt(1.0 / self.n_disc[j - 1])
                                w_curr.append(tf.Variable(nmw * self.init([self.n_disc[i], self.n_disc[j - 1]]),
                                                          name='W_disc_' + str(j) + '_' + str(i)))

                                g_curr.append(tf.matmul(w_curr[j], self.h_disc_list[p][j - 1]))

                        self.W_disc_list[p].append(w_curr)
                        self.g_disc_list[p].append(g_curr)

                        # Bias init
                        self.b_disc_list[p].append(tf.Variable(tf.zeros([self.n_disc[i], 1]), name='b_disc_' + str(i - 1)))

                        # i-th layer is sum of previous layers
                        nmw = np.sqrt(1.0 / float(len(self.g_disc_list[p][i])))
                        self.h_disc_list[p].append(tf.nn.dropout(
                            self.hidden_activation(tf.add(nmw * tf.add_n(self.g_disc_list[p][i]), self.b_disc_list[p][i]),
                                                   name='h_disc_' + str(i)), keep_prob=self.dropout_rate_disc))

                # Last layer
                i = len(self.n_disc)
                with tf.name_scope("dhat"):
                    self.w_dhat_list.append([])
                    self.g_dhat_list.append([])
                    for j in range(0, i + 1):
                        if j == 0:
                            # Dummy weight -- helps with pasta code
                            self.w_dhat_list[p].append(tf.Variable(self.init([1]), name='W_dhat_' + str(j) + '_' + str(i)))
                        else:
                            # Weight initialization
                            nmw = np.sqrt(1.0 / self.n_disc[j - 1])
                            self.w_dhat_list[p].append(tf.Variable(nmw * self.init([1, self.n_disc[j - 1]]),
                                                           name='We_' + str(j) + '_' + str(i)))

                            self.g_dhat_list[p].append(tf.matmul(self.w_dhat_list[p][j], self.h_disc_list[p][j - 1]))

                    # Bias initialization
                    self.b_dhat_list.append(tf.Variable(tf.zeros([1]), name='b_dhat'))

                    # Final shape of linear layer
                    self.dhat_list.append(tf.add(np.sqrt(1.0 / float(len(self.g_dhat_list[p]))) * tf.add_n(self.g_dhat_list[p]), self.b_dhat_list[p], name='dhat'))
                    self.d_list.append(tf.concat([tf.ones([1, self.Xsize]), tf.zeros([1, self.Xsize])], axis=1))

    start = common.start
    define_loss = common.lsgan_loss_multipled
    train = common.gan_train
    synthesize = common.synthesize_mlpGAN
    show_map = common.show_map_mlpGAN

    save_model = common.save_model
    load_model = common.load_model

if __name__ == '__main__':
    # Import MNIST dataset
    import ai.common_ops.datasets as dts
    training_data = dts.get_mnist(mlp=True)

    n_gen = [300, 300, 300, 300]
    n_disc = [300, 300, 300, 300]
    n_z = 100
    n_x = 784

    net = mlpLSGAN_multiD(n_gen, n_disc, n_z, n_x, mbs=16, fresh_init=1, actype='lrelu', name='test_2')

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
        'Ncool': 100000,
        'Nwarmup': 1,
        'T': 1000,
        'Niters': 200000,
        'dropout_rate': 1.0,
        'max_patience': 50,
        'eval_size': 3000
    }
    net.train(training_data, training_schedule, show=1, saveframes=1)

    # Test dhat values
    dhat = net.dhat.eval(session=net.sess, feed_dict={net.dropout_rate: 1.0, net.X: X_test, net.eps_throttle: 0.0})
    d = net.d.eval(session=net.sess, feed_dict={net.dropout_rate: 1.0, net.X: X_test, net.eps_throttle: 0.5})

    Xhat = net.synthesize(np.random.rand(100, 30000)*2-1)

    fig1 = plt.figure(1)
    ax11 = fig1.add_subplot(1, 1,1)
    ax11.clear()
    ax11.hist(np.ndarray.flatten(Xhat), bins=200, density=True)
    ax11.hist(np.ndarray.flatten(X_test), bins=200, density=True)