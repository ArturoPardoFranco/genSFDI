'''
Feedforward network model with n skip connections
Arturo Pardo, 2019
'''

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
import os, csv
import time
from datetime import datetime

dir_path = os.path.dirname(os.path.abspath(__file__))

import ai.common_ops as common

'''
Here, we will attempt to produce a feedforward neural network with a finite number of skip connections.
For a network with n layers, we can produce m = (n - 1) levels of skip connections. 
'''

'''
Feedforward network with n layers
'''
class MLP_SC:
    def __init__(self, n, lr, fresh_init=1, actype='elu', out_actype='softmax', cost_function='MSE', fixed_lr=False, regeps=0.1,
                 quiet = 0, debug = 0,
                 optim='Adam', mbs= 16, name='sc_mlp_cls', reset_default_graph=1, write_path='../../../output/network_tests'):

        # Quiet and debug (two layers of feedback)
        self.quiet = quiet
        self.debug = debug

        # Name for the network
        self.name = name

        # Optimizer
        self.optim = optim

        # Order to reset default graph when instanced
        self.reset_default_graph = reset_default_graph

        # Specification: is this a new network?
        self.fresh_init = fresh_init

        # First iteration detector
        if self.fresh_init == 1:
            self.first_iter = 1
        else:
            self.first_iter = 0

        # Layers and number of layer list
        self.n = n

        # Learning rate, cost function, learning rate
        self.lr = lr
        self.cost_function = cost_function
        self.fixed_lr = fixed_lr

        # Regularization epsilon
        self.regeps = regeps

        # Minibatch size
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

        # Activation type for hidden layers
        self.actype = actype
        self.out_actype = out_actype

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

        self.output_activation = activations.get(self.out_actype, None)
        if self.output_activation is None:
            raise NotImplementedError('Output activation function not implemented. Try:' + str(activations.keys()))

        # Three steps: (1) prepare the graph, (2) define the loss function, (3) configure session
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
        self.placeholder_X = tf.placeholder(tf.float32, [self.n[0], None], name='placeholder_X')
        self.placeholder_Y = tf.placeholder(tf.float32, [self.n[-1], None], name='placeholder_Y')

        # Generate dataset for placeholder (incl. randomization) and batch learning schedule
        self.dataset = tf.data.Dataset.from_tensor_slices((tf.transpose(self.placeholder_X),
                                                           tf.transpose(self.placeholder_Y)))
        self.dataset = self.dataset.shuffle(buffer_size=1000000)
        #self.dataset = self.dataset.repeat(10)
        self.dataset= self.dataset.batch(self.mbs)

        # This is the iterator that goes through the introduced dataset
        self.iterator = self.dataset.make_initializable_iterator()
        self.data_x, self.data_y = self.iterator.get_next()

        # Generate inputs and outputs
        #self.X = tf.placeholder(tf.float32, shape=(self.n[0], None), name = 'x')
        #self.Y = tf.placeholder(tf.float32, shape=(self.n[-1], None), name = 'y')
        self.X = tf.transpose(tf.identity(self.data_x))
        self.Y = tf.transpose(tf.identity(self.data_y))
        self.dropout_rate = tf.placeholder(tf.float32, shape=(), name='dropout_rate')
        self.learning_rate = tf.placeholder(tf.float32, shape = (), name='learning_rate')

        # Parameter initialization
        self.init = tf.random_normal_initializer(mean=0.0, stddev=1.0)

        self.h = []
        self.g = []
        self.W = []
        self.b = []

        # i-th hidden layer (starts at 1)
        for i in range(1, len(self.n)-1):
            with tf.name_scope('h' + str(i)):
                w_curr = []
                g_curr = []
                # He normalization depending on fan-in
                normweight = np.sqrt(2.0)/np.sqrt(np.sum(self.n[0:i]))
                for j in range(0, i):
                    # Create weights
                    w_curr.append(tf.Variable(self.init([self.n[i], self.n[j]])*normweight, name='W_' + str(j) + '_' + str(i)))

                    if j == 0:
                        g_curr.append(tf.matmul(w_curr[j], self.X))
                    else:
                        g_curr.append(tf.matmul(w_curr[j], self.h[j-1]))

                self.W.append(w_curr)
                self.g.append(g_curr)
                self.b.append(tf.Variable(self.init([self.n[i], 1])*normweight, name='b_' + str(i - 1)))

                self.h.append(tf.nn.dropout(self.hidden_activation(tf.add(tf.add_n(self.g[i-1]), self.b[i-1]), name='h_' + str(i)),
                                            keep_prob=self.dropout_rate))

        i = len(self.n)-1
        print('i = ' + str(i))

        with tf.name_scope("yhat"):
            self.w_yhat = []
            self.g_yhat = []
            # He normalization depending on fan-in
            normweight = np.sqrt(2.0) / np.sqrt(np.sum(self.n[0:i]))
            for j in range(1, i): # Don't look at the inputs directly! unless it's a good activation fcn
                print('j = ' + str(j))
                # Create weights
                self.w_yhat.append(tf.Variable(self.init([self.n[-1], self.n[j]])*normweight, name='W_' + str(j) + '_y'))

                if j == 0:
                    self.g_yhat.append(tf.matmul(self.w_yhat[j], self.X))
                else:
                    self.g_yhat.append(tf.matmul(self.w_yhat[j-1], self.h[j-1]))


            self.b_yhat = tf.Variable(self.init([self.n[-1], 1])*normweight, name='b_' + str(i - 1))

            # Linear output
            if self.out_actype == 'softmax':
                self.logits = tf.add(tf.add_n(self.g_yhat), self.b_yhat, name='logits')
                self.Yhat = self.output_activation(self.logits, axis=0, name='yhat')
            else:
                if self.out_actype != 'linear':
                    self.Yhat = self.output_activation(tf.add(tf.add_n(self.g_yhat), self.b_yhat, name='yhat'))
                else:
                    self.Yhat = tf.add(tf.add_n(self.g_yhat), self.b_yhat, name='yhat')

    # Core function definitions
    start = common.start                        # Graph initializer
    define_loss = common.classifier_loss        # Loss function for classifier

    train = common.train_mlp                    # Training schedule for mlp classifier
    output = common.output_mlp                  # Output function
    interrogate = common.interrogate_mlp        # Interrogate (for introspection, possibly fails)

    save_model = common.save_model
    load_model = common.load_model

    # Classifier-specific functions
    get_conf_matrix = common.get_conf_matrix
    one_vs_others = common.one_vs_others
    show_normalized_confusion = common.show_normalized_confusion


if __name__ == '__main__':
    ''' Import dataset '''
    import ai.common_ops.datasets as dts
    training_data = dts.get_mnist(mlp=True)

    ''' Network instance and parameters '''
    n = [784, 200, 200, 200, 200, 200, 200, 10 ]

    network = MLP_SC(n, lr=0.001, out_actype='softmax', actype='lrelu', fresh_init=1, mbs=16, name='sc_mlp_cls')

    # Test for one-cycle policy
    training_schedule = {
        'mode': 'ocp',  # Either 'clr' (Cyclic Learning Rates) 'ocp' (One-cycle Policy) or 'normal' (linear decay)
        'stop_criterion': 'acc',
        'eval_criterion': 'iter',
        'Niters': 100000,
        'lr_max': 0.0001,
        'lr_min': 0.0,
        'Nwarm': 1,
        'Ncool': 50000,
        'T': 1000,
        'dropout_rate': 0.9,
        'mult': 1,
        'max_patience': 50,
        'eval_size': 1000
    }
    network.train(training_data, training_schedule, show=1)


