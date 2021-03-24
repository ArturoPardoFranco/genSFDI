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

# Loss for convolutional InfoVAE with GAN/GCF function
def conv_infovaegan_loss(self):
    # Reconstruction error:
    if self.cost_function == 'MSEMAE':
        self.recon_error = tf.reduce_mean(tf.squared_difference(self.X, self.Xhat)) + \
                           tf.reduce_mean(tf.abs(self.X - self.Xhat))
    else:
        if self.cost_function == 'rMSEMAE':
            self.recon_error = tf.reduce_mean(tf.squared_difference(self.X, self.Xhat) / (self.X ** 2 + self.regeps)) + \
                               tf.reduce_mean(tf.abs(self.X - self.Xhat) / (tf.abs(self.X) + self.regeps))
        else:
            if self.cost_function == 'MSE':
                self.recon_error = tf.reduce_mean(tf.squared_difference(self.X, self.Xhat))
            else:
                if self.cost_function == 'rMSE':
                    self.recon_error = tf.reduce_mean(
                        tf.squared_difference(self.X, self.Xhat) / (self.X ** 2 + self.regeps))
                else:
                    if self.cost_function == 'MAE':
                        self.recon_error = tf.reduce_mean(tf.abs(self.X - self.Xhat))
                    else:
                        if self.cost_function == 'rMAE':
                            self.recon_error = tf.reduce_mean(
                                tf.abs(self.X - self.Xhat) / (tf.abs(self.X) + self.regeps))
                        else:
                            raise NotImplementedError('The specified function is not implemented')

    self.KLdivergence = self.compute_mmd(tf.transpose(self.eps), tf.transpose(self.z_true))

    self.foolD_list = []
    for k in range(1, len(self.disc_gce)):
        self.foolD_list.append(tf.reduce_mean(tf.abs(self.disc_gce[k][0][self.Xsize:, :, :, :] - \
                                                                    self.disc_gce[k][0][:self.Xsize, :, :, :])))
    self.foolD_A = (1 / float(len(self.disc_gce))) * tf.add_n(self.foolD_list)
    # tf.reduce_mean(tf.squared_difference(self.disc_fe[:, self.Xsize:], self.disc_fe[:, :self.Xsize]))

    '''
    # Relative MSEMAE
    self.foolD_A =  tf.reduce_mean(tf.squared_difference(self.disc_fe[:, self.Xsize:], self.disc_fe[:, :self.Xsize])) #+ \
                        #tf.reduce_mean(tf.abs(self.disc_fe[:, self.Xsize:] - self.disc_fe[:, :self.Xsize]))


    # Normal MSEMAE
    self.foolD_A = tf.reduce_mean(tf.abs(self.disc_fe[:, self.Xsize:] - self.disc_fe[:, :self.Xsize])) + 
                   tf.reduce_mean(tf.squared_difference(self.disc_fe[:, self.Xsize:], self.disc_fe[:, :self.Xsize]))
    '''

    # self.foolD_B = tf.reduce_mean(tf.squared_difference(self.disc_afe_prev[self.Xsize:, :, :, :], self.disc_afe_prev[:self.Xsize, :, :, :]))
    # self.foolD_C = tf.reduce_mean(tf.squared_difference(self.disc_he[0][:, self.Xsize:], self.disc_he[0][:, :self.Xsize]))

    # self.foolD = (self.foolD_A + self.foolD_B + self.foolD_C)

    self.foolD = self.foolD_A

    # Total loss for autoencoder: l2 reconstruction + MMD + implicit fooling of D
    self.AEloss = self.recon_error + self.kl_weight * self.KLdivergence + self.gan_weight * self.foolD

    # Binary cross-entropy loss
    self.bceD = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.d, logits=self.dhat_logits, axis=0))

    # Total loss for Discriminator: BCE of D
    self.Discloss = self.bceD

    # The optimizer is either fixed or variable:
    if self.fixed_lr == True:
        if self.optim == 'RMSprop':
            self.base_optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, epsilon=1E-5)
            # self.optimizer = tf.contrib.estimator.clip_gradients_by_norm(self.base_optimizer, clip_norm=1.0).minimize(self.loss)
            self.optimizer = self.base_optimizer.minimize(self.loss)
        if self.optim == 'Adam':
            self.base_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=1E-5)
            # self.optimizer = tf.contrib.estimator.clip_gradients_by_norm(self.base_optimizer, clip_norm=1.0).minimize(self.loss)
            self.optimizer = self.base_optimizer.minimize(self.loss)
        if self.optim == 'Momentum':
            self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=0.9,
                                                        use_nesterov=True).minimize(self.loss)
        if self.optim == 'GD':
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(self.loss)
    else:
        if self.optim == 'RMSprop':
            self.base_optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, epsilon=1E-5)
            # self.optimizer = tf.contrib.estimator.clip_gradients_by_norm(self.base_optimizer, clip_norm=1.0).minimize(self.loss)
            self.optimizer = self.base_optimizer.minimize(self.loss)
        if self.optim == 'Adam':
            '''
            self.optim_base1 = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=1E-5)
            self.optim_base2 = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=1E-5)
            self.optimizer_AE = tf.contrib.estimator.clip_gradients_by_norm(self.optim_base1, clip_norm=1.0).minimize(self.AEloss,
                                                                                                  var_list=tf.get_collection(
                                                                                                      tf.GraphKeys.GLOBAL_VARIABLES,
                                                                                                      scope='Autoencoder'))
            self.optimizer_Disc = tf.contrib.estimator.clip_gradients_by_norm(self.optim_base2, clip_norm=1.0).minimize(self.Discloss,
                                                                                                    var_list=tf.get_collection(
                                                                                                        tf.GraphKeys.GLOBAL_VARIABLES,
                                                                                                        scope='Discriminator'))
            '''
            self.optimizer_AE = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.AEloss,
                                                                                                  var_list=tf.get_collection(
                                                                                                      tf.GraphKeys.GLOBAL_VARIABLES,
                                                                                                      scope='Autoencoder'))
            self.optimizer_Disc = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.Discloss,
                                                                                                    var_list=tf.get_collection(
                                                                                                        tf.GraphKeys.GLOBAL_VARIABLES,
                                                                                                        scope='Discriminator'))
        if self.optim == 'Momentum':
            self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=0.9,
                                                        use_nesterov=True).minimize(self.loss)
        if self.optim == 'GD':
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        if self.optim == 'AdamHD':
            self.global_step1 = tf.Variable(0, dtype=tf.float32)
            self.global_step2 = tf.Variable(0, dtype=tf.float32)
            self.optimizer_AE = optims.AdamHDOptimizer().minimize(self.AEloss, self.global_step1, give_var_list=1,
                                                                  var_list=tf.get_collection(
                                                                      tf.GraphKeys.GLOBAL_VARIABLES,
                                                                      scope='Autoencoder'), name='optAE')
            self.optimizer_Disc = optims.AdamHDOptimizer().minimize(self.Discloss, self.global_step2, give_var_list=1,
                                                                    var_list=tf.get_collection(
                                                                        tf.GraphKeys.GLOBAL_VARIABLES,
                                                                        scope='Discriminator'), name='optDisc')

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

# Loss for a multilayer perceptron InfoVAE
def mlp_infovae_loss(self):
    # Reconstruction error at the output.
    cost_functions = {
        'MSE': tf.reduce_mean(tf.squared_difference(self.X, self.Xhat)),
        'MAE': tf.reduce_mean(tf.abs(self.X-self.Xhat)),
        'MAPE': tf.reduce_mean(tf.abs(self.X-self.Xhat)/tf.abs(self.X+1)),
        'MAAPE': tf.reduce_mean(tf.math.atan(tf.abs((self.X-self.Xhat))/(self.X+1E-8))),
        'rMSE': tf.reduce_mean(tf.squared_difference(self.X, self.Xhat) / (self.X ** 2 + self.regeps)),
        'rMSEnMSE': tf.reduce_mean(tf.squared_difference(self.X, self.Xhat) / (self.X ** 2 + self.regeps)) + tf.reduce_mean(tf.squared_difference(self.X, self.Xhat)),
        'rMAE': tf.reduce_mean(tf.abs(self.X - self.Xhat) / (tf.abs(self.X) + self.regeps)),
        'logMSE': tf.log(tf.reduce_mean(tf.squared_difference(self.X, self.Xhat))+1E-8),
        'logMSE2': tf.log(tf.reduce_mean(tf.squared_difference(self.X, self.Xhat))) + tf.reduce_mean(tf.squared_difference(self.X, self.Xhat)),
        'MSEMAE': 0.5*(tf.reduce_mean(tf.abs(self.X - self.Xhat)) + tf.reduce_mean(tf.squared_difference(self.X, self.Xhat))),
    }
    self.recon_error = cost_functions.get(self.cost_function, None)
    if self.recon_error == None:
        raise NotImplementedError('I/O Loss function not implemented. Try: ' + str(cost_functions))

    # Kullback-Leibler divergence at the bottleneck.
    if hasattr(self, 'z_true'):
        self.KLdivergence = self.compute_mmd(tf.transpose(self.eps), tf.transpose(self.z_true))
    else:
        self.KLdivergence = self.compute_mmd(tf.transpose(self.eps), tf.transpose(self.z))


    # Add L2 activity regularization.
    self.actloss = 0.0
    '''
    try:
        self.he_means = []
        for k in range(0, np.size(self.n_e)):
            self.he_means.append(tf.reduce_mean(tf.square(self.he[k])))
        self.actloss += tf.add_n(self.he_means) / float(len(self.he_means))
        print('L2 regularization for mlp layers of encoder active.')
        self.h2_means = []
        for k in range(0, np.size(self.n_d)):
            self.h2_means.append(tf.reduce_mean(tf.square(self.hd[k])))
        self.actloss += tf.add_n(self.hd_means) / float(len(self.hd_means))
        print('L2 regularization for mlp layers of decoder active.')
    except:
        self.actloss += 0.0
    '''

    # The total loss function is the sum of both:
    self.loss = self.recon_error + self.kl_weight * self.KLdivergence + self.actloss
    print('(mlp-infovae-loss) Loss is ' + str(np.shape(self.loss)))


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
