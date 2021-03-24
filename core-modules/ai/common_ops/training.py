'''
Garden variety of training methods for specific classes/modules
Arturo Pardo, 2019
'''

import os, csv
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
plt.rc('text', usetex=False)

from datetime import datetime
from tensorflow.python.client import timeline

''' Module for training a convolutional autoencoder '''
def train_convolutional_ae(self, training_data, training_sched, show=1, saveframes=0):
    '''
    Performs a sequence of SGD training examples with a given training_schedule
    :param Xdata: Input data. Must be (batch, width, height, channels)
    :param training_sched: Training schedule. Dict with fields below.
    :return:
    '''

    # First, store the current timestamp
    now = datetime.now()
    ts = now.strftime("%Y_%m_%d_%H_%M_%S")

    ''' Training data dict '''
    X_train = training_data['X_train']      # Inputs (train)
    Y_train = training_data['Y_train']      # One-hot encoded categories (ideal softmax output)
    y_train = training_data['y_train']      # Integer class numbers (0--nclasses)

    X_test = training_data['X_test']        # Inputs (test)
    Y_test = training_data['Y_test']        # One-hot encoded categories (ideal softmax output)
    y_test = training_data['y_test']        # Integer class numbers (0--nclasses)

    # 'clr' for Cyclic Learning Rates, 'normal' for Typical training, 'ocp' for One Cycle Policy
    mode = training_sched['mode']  # Training modality

    # Learning criterion: 'acc' for Accuracy, 'mse' for Mean Squared Error, 'none' means never stop.
    stop_criterion = training_sched['stop_criterion']

    # Evaluation time criterion: 'epoch' for evaluating at the end of the epoch, 'iter' for check every T iterations
    eval_criterion = training_sched['eval_criterion']

    max_patience = training_sched['max_patience']   # Patience (for validation purposes only).
    dropout = training_sched['dropout_rate']        # Dropout rate (keep_prob = 0.8 is equivalent to 20% dropout)
    eval_size = training_sched['eval_size']         # Size for evaluating train/test set

    # Prepare learning rate plan and timestamp list
    lr_plan = np.array([])
    timestamps = []

    if mode == 'clr':
        lr_max = training_sched['lr_max']           # Maximum training step. Peak cycle value.
        lr_min = training_sched['lr_min']           # Minimum training step. Happens at the beginning and end of each cycle.
        T = training_sched['T']                     # Cycle period
        beta_start = training_sched['beta_start']   # Beta-VAE warmup start value
        beta_end = training_sched['beta_end']       # Beta-VAE warmup end value
        Nwarmup = training_sched['Nwarmup']         # Warmup period
        Ncycles = training_sched['Ncycles']         # Number of cycles in total.
        mult = training_sched['mult']               # Multiplier for cycle length at t+1
        mult0 = 1.0

        for k in range(1, Ncycles + 1):
            # Generate learning cycle
            lr_T = np.linspace(lr_min, lr_max, int(mult0 * T / 2))
            lr_T_ = np.linspace(lr_max, lr_min, int(mult0 * T / 2))
            lr_plan = np.concatenate((lr_plan, lr_T, lr_T_), axis=0)

            # End of cycle should be included
            timestamps.append(int(T * mult0))
            mult0 *= mult

        timestamps = np.cumsum(timestamps)

        # Beta-values are also calculated here
        beta_plan = np.linspace(beta_start, beta_end, Nwarmup)
        beta_plan = np.concatenate((beta_plan, np.repeat(beta_end, T * Ncycles - Nwarmup)))

    else:
        if mode == 'normal':
            lr_start = training_sched['lr_start']               # Initial learning rate
            lr_end = training_sched['lr_end']                   # Final learning rate
            beta_start = training_sched['beta_start']           # Beta-VAE warmup start value
            beta_end = training_sched['beta_end']               # Beta-VAE warmup end value
            Nwarmup = training_sched['Nwarmup']                 # Warmup period
            Niters = training_sched['Niters']                   # Number of iterations
            T = training_sched['T']                             # Verification period

            # Timestamping occurs every T iterations
            n_periods = int(np.ceil(Niters / T))
            for k in range(1, n_periods):
                timestamps.append(k * T)

            # The learning rate plan in the standard case is a lot simpler.
            lr_plan = np.linspace(lr_start, lr_end, Niters)

            # The KL divergence influence value (beta-VAE) is also simple here!
            beta_plan = np.linspace(beta_start, beta_end, Nwarmup)
            beta_plan = np.concatenate((beta_plan, np.repeat(beta_end, Niters - Nwarmup)))

        else:
            if mode == 'ocp':
                lr_min = training_sched['lr_min']   # Minimum learning rate
                lr_max = training_sched['lr_max']   # Maximum learning rate
                Niters = training_sched['Niters']   # Number of iterations
                Nwarm = training_sched['Nwarm']     # Warmup iterations
                Ncool = training_sched['Ncool']     # Cooldown iterations
                T = training_sched['T']             # Check every T steps

                # Timestamping occurs every T iterations
                n_periods = int(np.ceil(Niters / T))
                for k in range(1, n_periods):
                    timestamps.append(k * T)

                # The learning rate plan is as follows:
                lr_w = np.linspace(lr_min, lr_max, Nwarm)
                lr_0 = np.repeat(lr_max, Niters - Nwarm - Ncool, axis=0)
                lr_c = np.linspace(lr_max, lr_min, Ncool)

                lr_plan = np.concatenate((lr_w, lr_0, lr_c), axis=0)
                print('lr_plan is ' + str(np.shape(lr_plan)))
                print('lr_plan = ' + str(lr_plan))

                # The KL divergence influence value (beta-VAE) is also simple here!
                beta_start = training_sched['beta_start']   # Beta-VAE warmup start value
                beta_end = training_sched['beta_end']       # Beta-VAE warmup end value
                Nwarmup = training_sched['Nwarmup']         # Warmup period
                beta_plan = np.linspace(beta_start, beta_end, Nwarmup)
                beta_plan = np.concatenate((beta_plan, np.repeat(beta_end, Niters - Nwarmup)))

            else:
                print('(!) No established training policy ')

    print('lr_plan is ' + str(np.shape(lr_plan)))
    print('timestamps are ' + str(timestamps))

    Ne = np.size(lr_plan)
    # lr_plan = np.random.rand(Ne)*lr_max
    # lr_plan *= np.random.rand(Ne)

    # Prepare the skip sizes for evaluation
    skip_train = int(np.max([np.round(X_train.shape[0] / eval_size), 1]))
    skip_test = int(np.max([np.round(X_test.shape[0] / eval_size), 1]))

    # Start feeding the input pipeline
    self.sess.run(self.iterator.initializer, feed_dict={self.placeholder_X:X_train})

    # Training error
    E = []
    # Validation error and patience
    vE = []


    # Plot some things if that's what the user wants.
    # We can only plot learning curves and cloud shape.
    if show == 1:
        plt.rc('text', usetex=False)
        fig10 = plt.figure(10, figsize=(15, 6))
        fig10.clf()
        ax101 = fig10.add_subplot(1, 2, 1)
        ax102 = fig10.add_subplot(1, 2, 2)

        # First curves are typical train/test errors
        ax101.clear()
        curve_E, = ax101.semilogy(E, color='g', linestyle='--', label='Training error', alpha=0.6, )
        curve_vE, = ax101.semilogy(vE, color='g', label='Validation error', alpha=0.6)
        ax101.legend()

        # Second plot is a scatter plot
        scatter_z = ax102.scatter(np.random.randn(10), np.random.randn(10), 30,
                                  marker='.', edgecolor='none', alpha=0.6)

        # Other internal functions will be called with
        self.first_map = 1
        self.first_recon = 1

        plt.ion()

    patience = 0
    for t in tqdm(range(0, Ne)):
        epoch_det = 0       # Epoch detection. Is zero unless t in timestamps if the condition is right.
        try:
            _, self.E = self.sess.run([self.optimizer, self.loss], feed_dict={self.learning_rate:lr_plan[t],
                                                                              self.dropout_rate:dropout,
                                                                              self.eps_throttle: 1.0,
                                                                              self.kl_weight: beta_plan[t]})

        except tf.errors.OutOfRangeError:
            # Re-feed the pipeline once it runs out of samples (constant learning)
            self.sess.run(self.iterator.initializer,
                          feed_dict={self.placeholder_X: X_train})
            epoch_det = 1

        if eval_criterion == 'iter':
            eval_cond = (t in timestamps)
        else:
            eval_cond = epoch_det

        if eval_cond == 1:

            Xhat_train, z_train = self.output(X_train[::skip_train, :, :, :])
            E_curr = np.mean((Xhat_train - X_train[::skip_train, :, :, :]) **2 )
            E.append(E_curr)

            Xhat_test, z_test = self.output(X_test[::skip_test, :, :, :])
            vE_curr = np.mean((Xhat_test - X_test[::skip_test, :, :, :]) ** 2)
            vE.append(vE_curr)

            '''
            Storage of learning data (at validation) 
            * If network is new, then create file with writing privileges and include data
            * If network is old, then open file with read/write privileges and append data
            '''
            if self.first_iter == 1:
                self.first_iter = 0
                with open(self.write_path + '/errors/data_' + self.name + '_' +  ts + '.csv',
                          'w') as file:
                    writer = csv.writer(file)
                    writer.writerow([t, E[-1], vE[-1]])
            else:
                with open(self.write_path + '/errors/data_' + self.name + '_' +   ts + '.csv',
                          'a') as file:
                    writer = csv.writer(file)
                    writer.writerow([t, E[-1], vE[-1]])

            '''
            Plotting results (if desired)
            For a feedforward network, we can only show the training and validation errors. 
            '''
            if show == 1:
                tvec = np.arange(0, len(E))
                curve_E.set_data(tvec, np.asarray(E))
                curve_vE.set_data(tvec, np.asarray(vE))

                ax101.relim()
                ax101.autoscale_view(True, True, True)

                scatter_z.set_offsets(z_train[0:2, :].T)
                scatter_z.set_array(np.squeeze(y_train[::skip_train]))

                ax102.ignore_existing_data_limits = True
                ax102.update_datalim(scatter_z.get_datalim(ax102.transData))
                ax102.autoscale_view()
                #ax102.set_xlim(left=-3, right=3)
                #ax102.set_ylim(bottom=-3, top=3)

                fig10.canvas.draw()
                fig10.canvas.flush_events()

                self.show_map()
                self.show_reconstructions(X_test[::skip_test, :, :, :])

                if saveframes == 1:
                    plt.figure(10)
                    plt.savefig(self.framesfolder + '/learning_' + ts + '_' + str(t).zfill(20) + '.png')
                    plt.figure(30)
                    plt.savefig(self.reconframesfolder + '/learning_' + ts + '_' + str(t).zfill(20) + '.png')


        if len(vE) >= 2 and eval_cond == 1:
            '''
            Early stopping implementation
            Uses stored validation tests and verifies if the system has already learned enough
            '''

            # The model stops depending on the selected criterion!
            if stop_criterion == 'mse':
                cond = (vE[-1] > np.min(vE))            # If error isn't better than the best
            if stop_criterion == 'none':
                cond = False                            # Nothing really matters, never stop!

            if cond == True:
                patience += 1
                if patience > max_patience:
                    # Time's up!
                    print('(!) Early stopping: vE[-1] = ' + str(vE[-1]) + ' > vE[-2] = ' + str(vE[-2]))
                    break
            else:
                # Restart the patience counter
                patience = 0
                # Save current model build
                self.save_model()

            print('Error: ' + str(np.round(vE[-1], 4)) + ', patience = ' + str(patience))

    plt.figure(10)
    plt.savefig(self.write_path + '/last_learning_' + ts + '.png')
    plt.figure(20)
    plt.savefig(self.write_path + '/last_random_' + ts + '.png')
    plt.figure(30)
    plt.savefig(self.write_path + '/last_recon_' + ts + '.png')

    # If we are here we have finished training, delete last cycle if early stopping has ocurred.
    self.load_model()

''' Module for training an LSGAN '''
def gan_train(self, training_data, training_sched, show=1, saveframes=0):
    '''
        Performs a sequence of SGD training examples with a given training_schedule
        :param Xdata: Input data. Must be (dimensions x samples) and not the other way around.
        :param training_sched: Training schedule. Dict with fields below.
        :return:
        '''

    # First, store the current timestamp
    now = datetime.now()
    ts = now.strftime("%Y_%m_%d_%H_%M_%S")

    ''' Training data dict '''
    X_train = training_data['X_train']  # Inputs (train)
    Y_train = training_data['Y_train']  # One-hot encoded categories (ideal softmax output)
    y_train = training_data['y_train']  # Integer class numbers (0--nclasses)

    X_test = training_data['X_test']  # Inputs (test)
    Y_test = training_data['Y_test']  # One-hot encoded categories (ideal softmax output)
    y_test = training_data['y_test']  # Integer class numbers (0--nclasses)

    # 'clr' for Cyclic Learning Rates, 'normal' for Typical training, 'ocp' for One Cycle Policy
    mode = training_sched['mode']  # Training modality

    # Learning criterion: 'acc' for Accuracy, 'mse' for Mean Squared Error, 'none' means never stop.
    stop_criterion = training_sched['stop_criterion']

    # Evaluation time criterion: 'epoch' for evaluating at the end of the epoch, 'iter' for check every T iterations
    eval_criterion = training_sched['eval_criterion']

    max_patience = training_sched['max_patience']  # Patience (for validation purposes only).
    dropout = training_sched['dropout_rate']  # Dropout rate (keep_prob = 0.8 is good enough)
    eval_size = training_sched['eval_size']  # Size for evaluating train/test set

    # Prepare learning rate plan and timestamp list
    lr_plan = np.array([])
    timestamps = []

    if mode == 'clr':
        lr_max = training_sched['lr_max']  # Maximum training step. Peak cycle value.
        lr_min = training_sched['lr_min']  # Minimum training step. Happens at the beginning and end of each cycle.
        T = training_sched['T']  # Cycle period
        beta_start = training_sched['beta_start']  # Beta-VAE warmup start value
        beta_end = training_sched['beta_end']  # Beta-VAE warmup end value
        Nwarmup = training_sched['Nwarmup']  # Warmup period
        Ncycles = training_sched['Ncycles']  # Number of cycles in total.
        mult = training_sched['mult']  # Multiplier for cycle length at t+1
        mult0 = 1.0

        for k in range(1, Ncycles + 1):
            # Generate learning cycle
            lr_T = np.linspace(lr_min, lr_max, int(mult0 * T / 2))
            lr_T_ = np.linspace(lr_max, lr_min, int(mult0 * T / 2))
            lr_plan = np.concatenate((lr_plan, lr_T, lr_T_), axis=0)

            # End of cycle should be included
            timestamps.append(int(T * mult0))
            mult0 *= mult

        timestamps = np.cumsum(timestamps)

        # Beta-values are also calculated here
        beta_plan = np.linspace(beta_start, beta_end, Nwarmup)
        beta_plan = np.concatenate((beta_plan, np.repeat(beta_end, T * Ncycles - Nwarmup)))

    else:
        if mode == 'normal':
            lr_start = training_sched['lr_start']  # Initial learning rate
            lr_end = training_sched['lr_end']  # Final learning rate
            beta_start = training_sched['beta_start']  # Beta-VAE warmup start value
            beta_end = training_sched['beta_end']  # Beta-VAE warmup end value
            Nwarmup = training_sched['Nwarmup']  # Warmup period
            Niters = training_sched['Niters']  # Number of iterations
            T = training_sched['T']  # Verification period

            # Timestamping occurs every T iterations
            n_periods = int(np.ceil(Niters / T))
            for k in range(1, n_periods):
                timestamps.append(k * T)

            # The learning rate plan in the standard case is a lot simpler.
            lr_plan = np.linspace(lr_start, lr_end, Niters)

            # The KL divergence influence value (beta-VAE) is also simple here!
            beta_plan = np.linspace(beta_start, beta_end, Nwarmup)
            beta_plan = np.concatenate((beta_plan, np.repeat(beta_end, Niters - Nwarmup)))

        else:
            if mode == 'ocp':
                lr_min = training_sched['lr_min']  # Minimum learning rate
                lr_max = training_sched['lr_max']  # Maximum learning rate
                Niters = training_sched['Niters']  # Number of iterations
                Nwarm = training_sched['Nwarm']  # Warmup iterations
                Ncool = training_sched['Ncool']  # Cooldown iterations
                T = training_sched['T']  # Check every T steps

                # Timestamping occurs every T iterations
                n_periods = int(np.ceil(Niters / T))
                for k in range(1, n_periods):
                    timestamps.append(k * T)

                # The learning rate plan is as follows:
                lr_w = np.linspace(lr_min, lr_max, Nwarm)
                lr_0 = np.repeat(lr_max, Niters - Nwarm - Ncool, axis=0)
                lr_c = np.linspace(lr_max, lr_min, Ncool)

                lr_plan = np.concatenate((lr_w, lr_0, lr_c), axis=0)
                print('lr_plan is ' + str(np.shape(lr_plan)))

                # The KL divergence influence value (beta-VAE) is also simple here!
                beta_start = training_sched['beta_start']  # Beta-VAE warmup start value
                beta_end = training_sched['beta_end']  # Beta-VAE warmup end value
                Nwarmup = training_sched['Nwarmup']  # Warmup period
                beta_plan = np.linspace(beta_start, beta_end, Nwarmup)
                beta_plan = np.concatenate((beta_plan, np.repeat(beta_end, Niters - Nwarmup)))


            else:
                print('(!) No established training policy ')

    print('lr_plan is ' + str(np.shape(lr_plan)))
    print('timestamps are ' + str(timestamps))

    Ne = np.size(lr_plan)

    eps_plan = np.linspace(training_sched['eps_start'],
                           training_sched['eps_end'],
                           Ne)

    # Prepare pre-existing noise
    pre_noise = np.random.rand(self.n_z, 1000)*2 -1

    # Prepare the skip sizes for evaluation
    skip_train = int(np.max([np.round(X_train.shape[0] / eval_size), 1]))
    skip_test = int(np.max([np.round(X_test.shape[0] / eval_size), 1]))

    # Start feeding the input pipeline
    self.sess.run(self.iterator.initializer, feed_dict={self.placeholder_X: X_train})

    # Discriminator error
    dE = []
    # Generator error
    gE = []
    # Diffs
    diff = []

    prev_vals = self.synthesize(pre_noise)
    new_vals = self.synthesize(pre_noise)
    #diff.append(1.0)

    # Plot some things if that's what the user wants.
    # We can only plot learning curves and cloud shape.
    if show == 1:
        plt.rc('text', usetex=False)
        fig10 = plt.figure(10, figsize=(15, 6))
        fig10.clf()
        ax101 = fig10.add_subplot(1, 2, 1)
        ax101b = ax101.twinx()
        ax102 = fig10.add_subplot(1, 2, 2)

        # First curves are typical train/test errors
        ax101.clear()
        ax101b.clear()
        curve_dE, = ax101.semilogy(dE, color='g', linestyle='--', label='Discriminator error', alpha=0.6)
        curve_gE, = ax101.semilogy(gE, color='k', label='Generator error', alpha=0.6)
        curve_diff = ax101b.semilogy(diff, color='b', label='Progress', alpha=0.6)[0]
        ax101.legend()
        ax101b.legend()

        # Other internal functions will be called with
        self.first_map = 1
        self.first_recon = 1

        plt.ion()

    patience = 0
    for t in tqdm(range(0, Ne)):
        epoch_det = 0  # Epoch detection. Is zero unless t in timestamps if the condition is right.
        try:

            # Do discriminator for another batch
            _, self.dE = self.sess.run([self.optimizer_disc, self.disc_loss],
                                       feed_dict={self.learning_rate: lr_plan[t],
                                                  self.dropout_rate_gen: 1.0,
                                                  self.dropout_rate_disc: dropout,
                                                  self.eps_throttle: eps_plan[t]})

            # Do autoencoder for one batch
            _, self.gE = self.sess.run([self.optimizer_gen, self.gen_loss], feed_dict={self.learning_rate: lr_plan[t],
                                                                                   self.dropout_rate_gen: dropout,
                                                                                   self.dropout_rate_disc: 1.0,
                                                                                   self.eps_throttle: eps_plan[t]})


        except tf.errors.OutOfRangeError:
            # Re-feed the pipeline once it runs out of samples (constant learning)
            self.sess.run(self.iterator.initializer,
                          feed_dict={self.placeholder_X: X_train})
            epoch_det = 1

        if eval_criterion == 'iter':
            eval_cond = (t in timestamps)
        else:
            eval_cond = epoch_det

        if eval_cond == 1:
            dE.append(np.copy(self.dE))
            gE.append(np.copy(self.gE))

            new_vals = self.synthesize(pre_noise)
            diff.append(np.mean((prev_vals-new_vals)**2))
            prev_vals = np.copy(new_vals)

            '''
            Storage of learning data (at validation) 
            * If network is new, then create file with writing privileges and include data
            * If network is old, then open file with read/write privileges and append data
            '''
            if self.first_iter == 1:
                self.first_iter = 0
                with open(self.write_path + '/errors/data_' + self.name + '_' + ts + '.csv',
                          'w') as file:
                    writer = csv.writer(file)
                    writer.writerow([t, dE[-1], gE[-1]])
            else:
                with open(self.write_path + '/errors/data_' + self.name + '_' + ts + '.csv',
                          'a') as file:
                    writer = csv.writer(file)
                    writer.writerow([t, dE[-1], gE[-1]])

            '''
            Plotting results (if desired)
            For a feedforward network, we can only show the training and validation errors. 
            '''
            if show == 1:
                tv = np.arange(0, len(gE))
                curve_gE.set_data(tv, np.asarray(gE))
                curve_dE.set_data(tv, np.asarray(dE))
                curve_diff.set_data(tv, np.asarray(diff))

                ax101.relim()
                ax101b.relim()
                ax101.autoscale_view(True, True, True)
                ax101b.autoscale_view(True, True, True)
                '''
                scatter_z.set_offsets(z_train[0:2, :].T)
                scatter_z.set_array(np.squeeze(y_train[::skip_train]))

                ax102.ignore_existing_data_limits = True
                ax102.update_datalim(scatter_z.get_datalim(ax102.transData))
                ax102.autoscale_view()
                '''
                # ax102.set_xlim(left=-3, right=3)
                # ax102.set_ylim(bottom=-3, top=3)

                fig10.canvas.draw()
                fig10.canvas.flush_events()

                self.show_map()
                '''
                self.show_reconstructions(X_test[::skip_test, :, :, :])
                '''

                if saveframes == 1:
                    plt.figure(10)
                    plt.savefig(self.framesfolder + '/learning_' + ts + '_' + str(t).zfill(20) + '.png')
                    plt.figure(30)
                    plt.savefig(self.reconframesfolder + '/learning_' + ts + '_' + str(t).zfill(20) + '.png')

        if len(gE) >= 2 and eval_cond == 1:
            '''
            Early stopping implementation
            Uses stored validation tests and verifies if the system has already learned enough
            '''

            # The model stops depending on the selected criterion!
            if stop_criterion == 'mse':
                cond = (gE[-1] > np.min(gE))  # If error isn't better than the best
            if stop_criterion == 'none':
                cond = False  # Nothing really matters, never stop!

            if cond == True:
                patience += 1
                if patience > max_patience:
                    # Time's up!
                    print('(!) Early stopping: vE[-1] = ' + str(gE[-1]) + ' > vE[-2] = ' + str(gE[-2]))
                    break
            else:
                # Restart the patience counter
                patience = 0
                # Save current model build
                self.save_model()

            print('Error: ' + str(np.round(gE[-1], 4)) + ', patience = ' + str(patience))
    '''
    plt.figure(10)
    plt.savefig(self.write_path + '/last_learning_' + ts + '.png', dpi=200, quality=100)
    plt.figure(20)
    plt.savefig(self.write_path + '/last_random_' + ts + '.png', dpi=200, quality=100)
    plt.figure(30)
    plt.savefig(self.write_path + '/last_recon_' + ts + '.png', dpi=200, quality=100)
    '''

    # If we are here we have finished training, delete last cycle if early stopping has ocurred.
    self.load_model()

''' Training a multilayer perceptron network '''
def train_mlp(self, training_data, training_sched, show = 1, saveframes=0):
    '''
    Performs a sequence of SGD training examples with a given training_schedule
    :param Xdata: Input data. Must be (dimensions x samples) and not the other way around.
    :param training_sched: Training schedule. Dict with fields below.
    :return:
    '''

    # First, store the current timestamp
    now = datetime.now()
    ts = now.strftime("%Y_%m_%d_%H_%M_%S")

    ''' Training data dict '''
    X_train = training_data['X_train']      # Inputs (train)
    Y_train = training_data['Y_train']  # One-hot encoded categories (ideal softmax output)
    y_train = training_data['y_train']      # Integer class numbers (0--nclasses)

    X_test = training_data['X_test']        # Inputs (test)
    Y_test = training_data['Y_test']        # One-hot encoded categories (ideal softmax output)
    y_test = training_data['y_test']        # Integer class numbers (0--nclasses)

    # 'clr' for Cyclic Learning Rates, 'normal' for Typical training, 'ocp' for One Cycle Policy
    mode = training_sched['mode']               # Training modality

    # Learning criterion: 'acc' for Accuracy, 'mse' for Mean Squared Error, 'none' for just save every T epochs.
    stop_criterion = training_sched['stop_criterion']

    # Evaluation time criterion: 'epoch' for evaluating at the end of the epoch, 'iter' for check every T iterations
    eval_criterion = training_sched['eval_criterion']

    max_patience = training_sched['max_patience']       # Patience (for validation purposes only).
    dropout = training_sched['dropout_rate']            # Dropout keep probability rate (i.e., 1-p_drop)
    eval_size = training_sched['eval_size']             # Size for evaluating train/test set

    # Prepare learning rate plan and timestamp list
    lr_plan = np.array([])
    timestamps = []

    if mode == 'clr':
        lr_max = training_sched['lr_max']           # Maximum training step. Peak cycle value.
        lr_min = training_sched['lr_min']           # Minimum training step. Happens at the beginning and end of each cycle.
        T = training_sched['T']                     # Cycle period
        Ncycles = training_sched['Ncycles']         # Number of cycles in total.
        mult = training_sched['mult']               # Multiplier for cycle length at t+1
        mult0 = 1.0

        for k in range(1, Ncycles + 1):
            # Generate learning cycle
            lr_T = np.linspace(lr_min, lr_max, int(mult0 * T / 2))
            lr_T_ = np.linspace(lr_max, lr_min, int(mult0 * T / 2))
            lr_plan = np.concatenate((lr_plan, lr_T, lr_T_), axis=0)

            # End of cycle should be included
            timestamps.append(int(T * mult0))
            mult0 *= mult

        timestamps = np.cumsum(timestamps)
    else:
        if mode == 'normal':
            lr_start = training_sched['lr_start']       # Initial learning rate
            lr_end = training_sched['lr_end']           # Final learning rate
            Niters = training_sched['Niters']           # Number of iterations
            T = training_sched['T']                     # Verification period

            # Timestamping occurs every T iterations
            n_periods = int(np.ceil(Niters/T))
            for k in range(1, n_periods):
                timestamps.append(k*T)

            # The learning rate plan in the standard case is a lot simpler.
            lr_plan = np.linspace(lr_start, lr_end, Niters)
        else:
            if mode == 'ocp':
                lr_min = training_sched['lr_min']       # Minimum learning rate
                lr_max = training_sched['lr_max']       # Maximum learning rate
                Niters = training_sched['Niters']       # Number of iterations
                Nwarm = training_sched['Nwarm']         # Warmup iterations
                Ncool = training_sched['Ncool']         # Cooldown iterations
                T = training_sched['T']                 # Check every T steps

                # Timestamping occurs every T iterations
                n_periods = int(np.ceil(Niters / T))
                for k in range(1, n_periods):
                    timestamps.append(k * T)

                # The learning rate plan is as follows:
                lr_w = np.linspace(lr_min, lr_max, Nwarm)
                lr_0 = np.repeat(lr_max, Niters-Nwarm-Ncool, axis=0)
                lr_c = np.linspace(lr_max, lr_min, Ncool)

                lr_plan = np.concatenate((lr_w, lr_0, lr_c), axis=0)
                print('lr_plan is ' + str(np.shape(lr_plan)))

            else:
                print('(!) No established training policy ')

    print('lr_plan is ' + str(np.shape(lr_plan)))
    print('timestamps are ' + str(timestamps))

    Ne = np.size(lr_plan)
    # lr_plan = np.random.rand(Ne)*lr_max
    # lr_plan *= np.random.rand(Ne)

    # Prepare the skip sizes for evaluation
    skip_train = int(np.max([np.round(X_train.shape[1] / eval_size), 1]))
    skip_test = int(np.max([np.round(X_test.shape[1] / eval_size), 1]))

    print('skip_train = ' + str(skip_train))
    print('skip_test = ' + str(skip_test))

    # Start feeding the input pipeline
    self.sess.run(self.iterator.initializer, feed_dict={self.placeholder_X:X_train, self.placeholder_Y:Y_train})

    # Training error
    E = []
    # Validation error and patience
    vE = []
    # Accuracies
    A = []
    vA = []

    # Plot initial graphs that later will be updated.
    if show == 1:
        plt.rc('text', usetex=False)
        fig10 = plt.figure(10, figsize=(15, 6))
        fig10.clf()
        ax101 = fig10.add_subplot(1, 2, 1)
        ax101b = ax101.twinx()
        ax102 = fig10.add_subplot(1, 2, 2)

        # print('E is ' + str(E))
        ax101.clear()
        ax101b.clear()
        curve_E, = ax101.semilogy(E, color='g', linestyle='--', label='Training error', alpha=0.6,)
        curve_vE, = ax101.semilogy(vE, color='g', label='Validation error', alpha=0.6)
        curve_A, = ax101b.plot(A, color='b', linestyle='--', label='Train accuracy', alpha=0.6)
        curve_vA, = ax101b.plot(vA, color='b', label='Val accuracy', alpha=0.6)
        ax101.legend()
        ax101b.legend()

        ax102.clear()
        Nclasses = Y_train.shape[0]
        C0 = np.zeros([Nclasses, Nclasses])
        imshow_acc = ax102.imshow(C0, aspect='equal', interpolation='nearest', cmap='summer')

        text_list = [[0]*Nclasses for k in range(Nclasses)]
        for (j, i), label in np.ndenumerate(C0):
            text_list[i][j] = ax102.text(i, j, np.round(C0[j, i], 3), ha='center', va='center')

        plt.ion()

    patience = 0
    for t in tqdm(range(0, Ne)):
        epoch_det = 0
        try:
            _, self.E = self.sess.run([self.optimizer, self.loss], feed_dict={self.learning_rate:lr_plan[t],
                                                                            self.dropout_rate:dropout})

        except tf.errors.OutOfRangeError:
            # Re-feed the pipeline once it runs out of samples (constant learning)
            self.sess.run(self.iterator.initializer,
                          feed_dict={self.placeholder_X: X_train, self.placeholder_Y: Y_train})
            epoch_det = 1

        if eval_criterion == 'iter':
            eval_cond = (t in timestamps)
        else:
            eval_cond = epoch_det

        if eval_cond == 1:

            Yhat_train = self.output(X_train[:, ::skip_train])
            E_curr = np.mean((Yhat_train - Y_train[:, ::skip_train]) **2 )
            E.append(E_curr)

            Yhat_test = self.output(X_test[:, ::skip_test])
            vE_curr = np.mean((Yhat_test - Y_test[:, ::skip_test]) ** 2)
            vE.append(vE_curr)

            Ctrain = self.get_conf_matrix(Yhat_train, Y_train[:, ::skip_train])
            Ctest = self.get_conf_matrix(Yhat_test, Y_test[:, ::skip_test])
            A.append(np.sum(np.diag(Ctrain)) / np.sum(Ctrain))
            vA.append(np.sum(np.diag(Ctest)) / np.sum(Ctest))

            '''
            Storage of learning data (at validation) 
            * If network is new, then create file with writing privileges and include data
            * If network is old, then open file with read/write privileges and append data
            '''
            if self.first_iter == 1:
                self.first_iter = 0
                with open(self.write_path + '/errors/data_' + self.name + '_' + ts + '.csv',
                          'w') as file:
                    writer = csv.writer(file)
                    writer.writerow([t, E[-1], vE[-1], A[-1], vA[-1]])
            else:
                with open(self.write_path + '/errors/data_' + self.name + '_' + ts + '.csv',
                          'a') as file:
                    writer = csv.writer(file)
                    writer.writerow([t, E[-1], vE[-1], A[-1], vA[-1]])

            '''
            Plotting results (if desired)
            For a feedforward network, we can only show the training and validation errors. 
            '''
            if show == 1:
                tvec = np.arange(0, len(E))
                curve_E.set_data(tvec, np.asarray(E))
                curve_vE.set_data(tvec, np.asarray(vE))
                curve_A.set_data(tvec, np.asarray(A))
                curve_vA.set_data(tvec, np.asarray(vA))

                ax101.relim()
                ax101.autoscale_view(True, True, True)
                ax101b.relim()
                ax101b.autoscale_view(True, True, True)

                self.show_normalized_confusion(Yhat_test, Y_test[:, ::skip_test], ax102, imshow_acc, text_list)

                fig10.canvas.draw()
                fig10.canvas.flush_events()

        if len(vE) >= 2 and eval_cond == 1:

            '''
            Early stopping implementation
            Uses stored validation tests and verifies if the system has already learned enough
            '''

            if stop_criterion == 'mse':
                cond = (vE[-1] > np.min(vE))
            if stop_criterion == 'acc':
                cond = (vA[-1] != np.max(vA))
            if stop_criterion == 'none':
                cond = False

            # Fix for accuracy! revert if necessary
            if cond == True:
                patience += 1
                if patience > max_patience:
                    # Time's up!
                    print('(!) Early stopping: vE[-1] = ' + str(vE[-1]) + ' > min = ' + str(np.min(vE)))
                    break
            else:
                print('Model saved.')
                # Restart the patience counter
                patience = 0
                # Save current model build
                self.save_model()

            print('Error: ' + str(np.round(vE[-1], 4)) + ', patience = ' + str(patience))
            print('Accuracy (train): ' + str(np.round(A[-1], 5)) + ', Accuracy (test): ' + str(np.round(vA[-1], 5)))

    # If we are here we have finished training, delete last cycle if early stopping has ocurred.
    self.load_model()

    # Not much is needed except the final precision
    plt.figure(10)
    plt.savefig(self.write_path + '/last_learning_' + ts + '.png', dpi=200, quality=100)
