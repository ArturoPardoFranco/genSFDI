'''
Loss functions for all possible networks
Arturo Pardo, 2019
'''

import numpy as np
import tensorflow as tf

# Loss for a classifier (convolutional or mlp)
def classifier_loss(self):

    # Possible mlp losses // including experimental.
    try:
        losses = {
            'MSE': tf.reduce_sum(tf.squared_difference(self.Y, self.Yhat), 0),
            'MAE': tf.reduce_sum(tf.abs(self.Y - self.Yhat), 0),
            'RMSE': tf.sqrt(tf.reduce_sum(tf.squared_difference(self.Y, self.Yhat), 0)),
            'logMSE': tf.log(tf.reduce_mean(tf.squared_difference(self.Y, self.Yhat), 0)),
            'rMSE': tf.reduce_sum(tf.squared_difference(self.Y, self.Yhat) / (self.Y ** 2 + self.regeps), 0),
            'rMAE': tf.reduce_sum(tf.abs(self.Y - self.Yhat) / (tf.abs(self.Y) + self.regeps), 0),
            'cross_entropy': tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.Y, logits=self.logits, axis=0),
        }
    except:
        # If it's not a classifier, it may not have logits or regeps. These will be available
        losses = {
            'MSE': tf.reduce_sum(tf.squared_difference(self.Y, self.Yhat), 0),
            'MAE': tf.reduce_sum(tf.abs(self.Y - self.Yhat), 0),
            'rMSE': tf.reduce_sum(tf.squared_difference(self.Y, self.Yhat) / (self.Y ** 2 + self.regeps), 0),
            'rMAE': tf.reduce_sum(tf.abs(self.Y - self.Yhat) / (tf.abs(self.Y) + self.regeps), 0),
        }

    self.sumloss = losses.get(self.cost_function, None)
    if self.sumloss == None:
        raise NotImplementedError('The specified cost function ('+ self.cost_function +') is not available. Try: ' + str(losses.keys()))

    self.loss = tf.reduce_mean(self.sumloss)

    # The optimizer is either fixed or variable:
    if self.fixed_lr == True:
        optim_options = {
            'RMSprop': tf.train.RMSPropOptimizer(learning_rate=self.lr),
            'Adam': tf.train.AdamOptimizer(learning_rate=self.lr),
            'Momentum': tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=0.9),
            'SGD': tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        }
        self.optimizer = optim_options.get(self.optim, None)
        if self.optimizer == None:
            raise NotImplementedError('The specified optimizer is not available. Try: ' + str(optim_options.keys()))
        self.optimizer = self.optimizer.minimize(self.loss)

    else:
        optim_options = {
            'RMSprop': tf.train.RMSPropOptimizer(learning_rate=self.learning_rate),
            'Adam': tf.train.AdamOptimizer(learning_rate=self.learning_rate),
            'Momentum': tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.9),
            'SGD': tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        }
        self.optimizer = optim_options.get(self.optim, None)
        if self.optimizer == None:
            raise NotImplementedError('The specified optimizer is not available. Try: ' + str(optim_options.keys()))
        self.optimizer = self.optimizer.minimize(self.loss)

# Convolutional beta-VAE losses
def conv_beta_vae_loss(self):
    # Reconstruction error at the output.
    cost_functions = {
        'MSE': tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(tf.squared_difference(self.X, self.Xhat), 1), 1), 1),
        'MAE': tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(tf.abs(self.X - self.Xhat), 1), 1), 1),
        'rMSE': tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(tf.squared_difference(self.X, self.Xhat) / (self.X ** 2 + self.regeps), 1), 1), 1),
        'rMAE': tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(tf.abs(self.X - self.Xhat) / (tf.abs(self.X) + self.regeps), 1), 1), 1)
    }
    self.recon_error = cost_functions.get(self.cost_function, None)
    if self.recon_error == None:
        raise NotImplementedError('I/O Loss function not implemented. Try: ' + str(cost_functions))

    # KL divergence term is simple
    self.KLdivergence = tf.reduce_sum(tf.exp(self.log_sigma) + self.mu ** 2.0 - 1.0 - self.log_sigma, 0)

    # The total loss function is the sum of both:
    self.loss = tf.reduce_mean(self.recon_error + self.kl_weight * self.KLdivergence)

    # The optimizer is either fixed or variable:
    if self.fixed_lr == True:
        optim_options = {
            'RMSprop': tf.train.RMSPropOptimizer(learning_rate=self.lr),
            'Adam': tf.train.AdamOptimizer(learning_rate=self.lr),
            'Momentum': tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=0.9),
            'SGD': tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        }
        self.optimizer = optim_options.get(self.optim, None)
        if self.optimizer == None:
            raise NotImplementedError('The specified optimizer is not available. Try: ' + str(optim_options.keys()))
        self.optimizer = self.optimizer.minimize(self.loss)

    else:
        optim_options = {
            'RMSprop': tf.train.RMSPropOptimizer(learning_rate=self.learning_rate),
            'Adam': tf.train.AdamOptimizer(learning_rate=self.learning_rate),
            'Momentum': tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.9),
            'SGD': tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        }
        self.optimizer = optim_options.get(self.optim, None)
        if self.optimizer == None:
            raise NotImplementedError('The specified optimizer is not available. Try: ' + str(optim_options.keys()))
        self.optimizer = self.optimizer.minimize(self.loss)

# Secondary functions for InfoVAE bottlenecks
def compute_kernel(self, x, y):
    x_size = tf.shape(x)[0]
    y_size = tf.shape(y)[0]
    dim = tf.shape(x)[1]
    tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
    tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))
    return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2)/ tf.cast(dim, tf.float32))

# MMD for two sets of vectors.
def compute_mmd(self, x, y, sigma_sqr=1.0):
    x_kernel = self.compute_kernel(x, x)
    y_kernel = self.compute_kernel(y, y)
    xy_kernel = self.compute_kernel(x, y)
    return tf.reduce_mean(x_kernel) + tf.reduce_mean(y_kernel) - 2 * tf.reduce_mean(xy_kernel)

# Loss for convolutional InfoVAE
def conv_infovae_loss(self):
    # Reconstruction error at the output.
    cost_functions = {
        'MSE': tf.reduce_mean(tf.squared_difference(self.X, self.Xhat)),
        'MAE': tf.reduce_mean(tf.abs(self.X - self.Xhat)),
        'rMSE': tf.reduce_mean(tf.squared_difference(self.X, self.Xhat) / (self.X ** 2 + self.regeps)),
        'rMAE': tf.reduce_mean(tf.abs(self.X - self.Xhat) / (tf.abs(self.X) + self.regeps))
    }
    self.recon_error = cost_functions.get(self.cost_function, None)
    if self.recon_error == None:
        raise NotImplementedError('I/O Loss function not implemented. Try: ' + str(cost_functions))

    self.KLdivergence = self.compute_mmd(tf.transpose(self.eps), tf.transpose(self.z))

    # The total loss function is the sum of both:
    self.loss = self.recon_error + self.kl_weight * self.KLdivergence
    print('(conv-infovae-loss) Loss is ' + str(np.shape(self.loss)))

    # The optimizer is either fixed or variable:
    if self.fixed_lr == True:
        optim_options = {
            'RMSprop': tf.train.RMSPropOptimizer(learning_rate=self.lr),
            'Adam': tf.train.AdamOptimizer(learning_rate=self.lr),
            'Momentum': tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=0.9),
            'SGD': tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        }
        self.optimizer = optim_options.get(self.optim, None)
        if self.optimizer == None:
            raise NotImplementedError('The specified optimizer is not available. Try: ' + str(optim_options.keys()))
        self.optimizer = self.optimizer.minimize(self.loss)


    else:
        optim_options = {
            'RMSprop': tf.train.RMSPropOptimizer(learning_rate=self.learning_rate),
            'Adam': tf.train.AdamOptimizer(learning_rate=self.learning_rate),
            'Momentum': tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.9),
            'SGD': tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        }
        self.optimizer = optim_options.get(self.optim, None)
        if self.optimizer == None:
            raise NotImplementedError('The specified optimizer is not available. Try: ' + str(optim_options.keys()))
        self.optimizer = self.optimizer.minimize(self.loss)

# Loss for LSGAN with multiple discriminators
def lsgan_loss_multipled(self):
    # Gen loss is MSE!
    self.gen_loss_list = []
    self.disc_loss_list = []
    for p in range(0, self.N_disc):
        self.gen_loss_list.append(tf.reduce_mean(tf.square(self.dhat_list[p][:, self.Xsize:]-1.0)))
        # Discriminator loss is MSE!
        self.disc_loss_list.append(tf.reduce_mean(tf.squared_difference(self.d_list[p], self.dhat_list[p])))

    self.gen_loss = (1.0/int(self.N_disc))*tf.add_n(self.gen_loss_list)
    self.disc_loss = (1.0/int(self.N_disc))*tf.add_n(self.disc_loss_list)

    # The optimizer is either fixed or variable:
    if self.fixed_lr == True:
        optim_options = {
            'RMSprop': tf.train.RMSPropOptimizer(learning_rate=self.lr),
            'Adam': tf.train.AdamOptimizer(learning_rate=self.lr),
            'Momentum': tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=0.9),
            'SGD': tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        }

        self.optimizer_gen = optim_options.get(self.optim, None)
        self.optimizer_gen = self.optimizer_gen.minimize(self.gen_loss,
                                                         var_list=tf.get_collection(
                                                             tf.GraphKeys.GLOBAL_VARIABLES,
                                                             scope='Generator'))
        if self.optimizer == None:
            raise NotImplementedError('The specified optimizer is not available. Try: ' + str(optim_options.keys()))

        self.optimizer_disc = optim_options.get(self.optim, None)
        self.optimizer_disc = self.optimizer_gen.minimize(self.disc_loss,
                                                          var_list=tf.get_collection(
                                                              tf.GraphKeys.GLOBAL_VARIABLES,
                                                              scope='Discriminator'))

    else:
        optim_options_gen = {
            'RMSprop': tf.train.RMSPropOptimizer(learning_rate=self.learning_rate),
            'Adam': tf.train.AdamOptimizer(learning_rate=self.learning_rate),
            'Momentum': tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.9),
            'SGD': tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        }

        self.optimizer_gen = optim_options_gen.get(self.optim, None)
        self.optimizer_gen = self.optimizer_gen.minimize(self.gen_loss,
                                                         var_list=tf.get_collection(
                                                             tf.GraphKeys.GLOBAL_VARIABLES,
                                                             scope='Generator'))

        optim_options_disc = {
            'RMSprop': tf.train.RMSPropOptimizer(learning_rate=self.learning_rate),
            'Adam': tf.train.AdamOptimizer(learning_rate=self.learning_rate),
            'Momentum': tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.9),
            'SGD': tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        }

        self.optimizer_disc = optim_options_disc.get(self.optim, None)
        self.optimizer_disc = self.optimizer_disc.minimize(self.disc_loss,
                                                           var_list=tf.get_collection(
                                                               tf.GraphKeys.GLOBAL_VARIABLES,
                                                               scope='Discriminator'))
