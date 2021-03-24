'''
Output functions for all possible modules
Arturo Pardo, 2019
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import tensorflow as tf
from tqdm import tqdm


'''
Output functions provide the output values in the graph given a set of input values.
Similar to forward(x) in a Torch module.
'''

# Output function for a convolutional autoencoder: returns z, xhat
def output_conv_ae(self, x, klweight=1, batchSize=64, quiet_progress=1):

    # Number of minibatches to evaluate.
    niters = int(np.ceil(x.shape[0] / batchSize))

    Xhat_ = self.Xhat.eval(session=self.sess, feed_dict={self.X: x[:batchSize, :, :, :],
                                                         self.eps_throttle: 1.0,
                                                         self.kl_weight: klweight,
                                                         self.dropout_rate: 1.0})

    z_ = self.z.eval(session=self.sess, feed_dict={self.X: x[:batchSize, :, :, :],
                                                   self.eps_throttle: 1.0,
                                                   self.kl_weight: klweight,
                                                   self.dropout_rate: 1.0})

    # Preallocate output shapes with first batch information
    Xhat = np.zeros([x.shape[0], Xhat_.shape[1], Xhat_.shape[2], Xhat_.shape[3]])
    z = np.zeros([z_.shape[0], x.shape[0]])

    # Show with tqdm or not. This is not classy but it works.
    if quiet_progress == 1:
        for b in range(1, niters):
            Xhat_ = self.Xhat.eval(session=self.sess,
                                   feed_dict={self.X: x[b * batchSize:(b + 1) * batchSize, :, :, :],
                                              self.eps_throttle: 1.0,
                                              self.kl_weight: klweight,
                                              self.dropout_rate: 1.0})

            z_ = self.z.eval(session=self.sess,
                             feed_dict={self.X: x[b * batchSize:(b + 1) * batchSize, :, :, :],
                                        self.eps_throttle: 1.0,
                                        self.kl_weight: klweight,
                                        self.dropout_rate: 1.0})

            Xhat[b * batchSize:(b + 1) * batchSize, :, :, :] = np.copy(Xhat_)
            z[:, b * batchSize:(b + 1) * batchSize] = np.copy(z_)
    else:
        for b in tqdm(range(1, niters)):
            Xhat_ = self.Xhat.eval(session=self.sess,
                                   feed_dict={self.X: x[b * batchSize:(b + 1) * batchSize, :, :, :],
                                              self.eps_throttle: 1.0,
                                              self.kl_weight: klweight,
                                              self.dropout_rate: 1.0})

            z_ = self.z.eval(session=self.sess,
                             feed_dict={self.X: x[b * batchSize:(b + 1) * batchSize, :, :, :],
                                        self.eps_throttle: 1.0,
                                        self.kl_weight: klweight,
                                        self.dropout_rate: 1.0})

            Xhat[b * batchSize:(b + 1) * batchSize, :, :, :] = np.copy(Xhat_)
            z[:, b * batchSize:(b + 1) * batchSize] = np.copy(z_)

    # Xhat is the output of the AE, z is the bottleneck.
    return Xhat, z

# Output for a general mlp
def output_mlp(self, x, batchSize=64, debug=0):
    # Assuming X is (n_X, ?) then we can just do as with output:
    niters = int(np.ceil(x.shape[1] / batchSize))
    if debug == 1: print('niters = ' + str(niters))
    if debug == 1: print('batchSize = ' + str(batchSize))

    yhat_ = self.sess.run(self.Yhat, feed_dict={self.X: x[:, 0:batchSize],
                                                self.dropout_rate: 1.0,
                                                self.learning_rate: 0.0})

    yhat = np.zeros([yhat_.shape[0], x.shape[1]])
    yhat[:, 0:batchSize] = np.copy(yhat_)

    if debug == 1: print('yhat is ' + str(np.shape(yhat)))

    for b in range(1, niters):
        yhat[:, b * batchSize:(b + 1) * batchSize] = np.copy(self.sess.run(self.Yhat,
                                                                           feed_dict={self.X: x[:, b * batchSize:(b + 1) * batchSize],
                                                                                      self.dropout_rate: 1.0,
                                                                                      self.learning_rate: 0.0}))
    return yhat

'''
Interrogation functions -- since TF cannot let you observe hidden units but Torch can, we can do the following:
'''

# Interrogate fully connected layers of multilayer perceptron
def interrogate_mlp(self, x, batchSize=64):
    # First coordinate will be niters
    niters = int(np.ceil(x.shape[1]/batchSize))

    h_ = self.sess.run(self.h, feed_dict={self.X: x[:, 0:batchSize],
                                              self.dropout_rate: 1.0,
                                              self.learning_rate: 0.0})

    # First, initialize memory. This speeds up learning.
    h = []
    for k in range(0, len(h_)):
        h.append(np.zeros([h_[k].shape[0], x.shape[1]]))
        h[k][:, 0:batchSize] = np.copy(h_[k])

    for b in range(1, niters):
        h_ = self.sess.run(self.h, feed_dict={self.X: x[:, b*batchSize:(b+1)*batchSize],
                                              self.dropout_rate: 1.0,
                                              self.learning_rate: 0.0})

        for k in range(0, len(h)):
            h[k][:, b*batchSize:(b+1)*batchSize] = np.copy(h_[k]) # h is a list (nlayers, nh, batch)

    print('h is ' + str(np.shape(h)))

    return h


'''
Synthesis functions common to all autoencoders: produces Xhat (output) out of z (bottleneck).
'''

# Synthesize data for conv autoencoder
def synthesize_conv(self, z, klweight=1, batchSize = 64, quiet_progress=1):
    # Assuming z is (n_z, ?) then we can just do as with self.output:
    niters = int(np.ceil(z.shape[1] / batchSize))

    Xhat_ = self.Xhat.eval(session=self.sess, feed_dict={self.z: z[:, 0:batchSize],
                                                        self.eps_throttle: 1.0,
                                                        self.kl_weight: klweight,
                                                        self.dropout_rate: 1.0,
                                                        self.learning_rate: 0.0})

    # Preallocate values.
    Xhat = np.zeros([z.shape[1], Xhat_.shape[1], Xhat_.shape[2], Xhat_.shape[3]])
    Xhat[0:batchSize, :, :, :] = np.copy(Xhat_)

    if quiet_progress == 1:
        for b in range(1, niters):
            Xhat_ = self.Xhat.eval(session=self.sess, feed_dict={self.z: z[:, b * batchSize:(b + 1) * batchSize],
                                                                 self.eps_throttle: 1.0,
                                                                 self.kl_weight: klweight,
                                                                 self.dropout_rate: 1.0,
                                                                 self.learning_rate: 0.0})

            Xhat[b*batchSize:(b+1)*batchSize, :, :, :] = np.copy(Xhat_)
    else:
        for b in tqdm(range(1, niters)):
            Xhat_ = self.Xhat.eval(session=self.sess, feed_dict={self.z: z[:, b * batchSize:(b + 1) * batchSize],
                                                                 self.eps_throttle: 1.0,
                                                                 self.kl_weight: klweight,
                                                                 self.dropout_rate: 1.0,
                                                                 self.learning_rate: 0.0})

            Xhat[b * batchSize:(b + 1) * batchSize, :, :, :] = np.copy(Xhat_)

    return Xhat

'''
Synthesis functions for GANs
'''

# Synthesize data for mlp autoencoder
def synthesize_mlpGAN(self, z, klweight=1, batchSize=64, quiet_progress=1):
    # Assuming z is (n_z, ?) then we can just do as with output:
    niters = int(np.ceil(z.shape[1] / batchSize))

    Xhat_ = self.Xhat.eval(session=self.sess, feed_dict={self.z: z[:, 0:batchSize],
                                                        self.eps_throttle: 1.0,
                                                        self.dropout_rate: 1.0,
                                                        self.learning_rate: 0.0})
    Xhat = np.zeros([Xhat_.shape[0], z.shape[1]])
    Xhat[:, 0:batchSize] = np.copy(Xhat_)

    if quiet_progress == 1:
        for b in range(1, niters):
            Xhat_ = self.Xhat.eval(session=self.sess, feed_dict={self.z: z[:, b * batchSize:(b + 1) * batchSize],
                                                                 self.eps_throttle: 1.0,
                                                                 self.dropout_rate: 1.0,
                                                                 self.learning_rate: 0.0})

            Xhat[:, b * batchSize:(b + 1) * batchSize] = np.copy(Xhat_)
    else:
        for b in tqdm(range(1, niters)):
            Xhat_ = self.Xhat.eval(session=self.sess, feed_dict={self.z: z[:, b * batchSize:(b + 1) * batchSize],
                                                                 self.eps_throttle: 1.0,
                                                                 self.dropout_rate: 1.0,
                                                                 self.learning_rate: 0.0})

            Xhat[:, b * batchSize:(b + 1) * batchSize] = np.copy(Xhat_)

    return Xhat

'''
Common visualization tools/methods
'''

# Show reconstructions for convolutional autoencoder
def show_reconstructions_conv_ae(self, Xdata, nrows=4, ncols=20):

    # Prepare a row with patch size
    patchSize = int(np.round(np.asarray(self.X.shape[1], dtype=int)))

    # Prepare a large picture
    bigPic = np.zeros([patchSize * nrows * 2, patchSize * ncols])

    if self.first_recon == 1:
        # Produce a figure for later use
        self.fig30 = plt.figure(30, figsize=(15, 5))
        self.fig30.clf()
        self.ax31 = self.fig30.add_subplot(1, 1, 1)
        self.imshow_recon = self.ax31.imshow(bigPic, cmap='gray')
        self.fig30.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95)
        self.ax31.set_xlabel('z1')
        self.ax31.set_ylabel('z2')
        plt.ion()
        self.first_recon=0

    else:
        # Get output data
        Xhat, z = self.output(Xdata)

        # Always produce the same random data (for video representation purposes)
        prng = np.random.RandomState(0)
        for i in range(0, nrows):
            # Get a subset of the data
            kval = np.squeeze(prng.randint(0, np.size(Xdata, 0), ncols))
            for j in range(0, ncols):
                bigPic[2*i*patchSize:(2*i+1)*patchSize, j*patchSize:(j+1)*patchSize] = \
                                                            np.squeeze(np.mean(Xdata[int(kval[j]), :, :, :], axis=2))
                bigPic[(2 * i +1)* patchSize:(2 * i + 2) * patchSize, j * patchSize:(j + 1) * patchSize] = \
                                                            np.squeeze(np.mean(Xhat[int(kval[j]), :, :, :], axis=2))

        self.imshow_recon.set_data(bigPic)
        self.imshow_recon.autoscale()
        self.fig30.canvas.draw()
        self.fig30.canvas.flush_events()

# Show 2D map (convolutional AE)
def show_map_conv(self, Ntest=20, zspan=2.5):
    '''
    Generates a map of internal representations for the various positions in z-space.
    :return: Figure with internal representation
    '''

    ''' Random patch range '''
    patchSize = self.X.shape[1]
    zrange = np.linspace(-zspan, zspan, Ntest)
    zx, zy = np.meshgrid(zrange, zrange)
    BigPic = np.zeros([Ntest * patchSize, Ntest * patchSize])

    if self.first_map == 1:
        self.fig20 = plt.figure(20, figsize=(10, 10))
        self.ax11 = self.fig20.add_subplot(1, 1, 1)
        self.imshow_map = self.ax11.imshow(np.flip(BigPic, axis=0), extent=(-zspan, zspan, -zspan, zspan), cmap='gray')
        self.fig20.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95)
        self.ax11.set_xlabel('z1')
        self.ax11.set_ylabel('z2')
        self.first_map = 0
    else:

        # For each row in the 2D mosaic:
        for i in range(0, Ntest):

            # If the bottleneck is bigger than 2D, then we only show the first two dimensions.
            zgen = np.concatenate((zx[i, :][np.newaxis, :], zy[i, :][np.newaxis, :]), axis=0)
            if self.n_z > 2:
                zgen = np.concatenate((zgen, np.zeros([self.n_z - 2, zgen.shape[1]])), axis=0)

            # Synthesize and generate this row's results.
            Xhat = self.synthesize(zgen)

            # Assign to adequate location.
            for j in range(0, Ntest):
                BigPic[i * patchSize:(i + 1) * patchSize, j * patchSize:(j + 1) * patchSize] = np.squeeze(np.mean(Xhat[j, :, :, :], 2))

        # Plot and show
        self.imshow_map.set_data(BigPic)
        self.imshow_map.autoscale()
        self.fig20.canvas.draw()
        self.fig20.canvas.flush_events()

# Show 2D map (mlp GAN)
def show_map_mlpGAN(self, Ntest = 20, zspan=2.5):
    '''
    Generates a map of internal representations for the various positions in z-space.
    :return: Figure with internal representation
    '''

    # Patch size will be the closest square root from above
    patchSize = int(np.ceil(np.sqrt(self.n_x)))

    zrange = np.linspace(-zspan, zspan, Ntest)
    zy, zx = np.meshgrid(zrange, zrange)
    BigPic = np.zeros([Ntest * patchSize, Ntest * patchSize])

    if self.first_map == 1:
        self.fig20 = plt.figure(20, figsize=(10, 10))
        self.fig20.clf()
        self.ax11 = self.fig20.add_subplot(1, 1, 1)
        self.imshow_map = self.ax11.imshow(np.flip(BigPic, axis=0), extent=(-zspan, zspan, -zspan, zspan), cmap='gray')
        self.fig20.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95)
        self.ax11.set_xlabel('z1')
        self.ax11.set_ylabel('z2')
        self.first_map = 0
    else:

        prng = np.random.RandomState(0)
        for i in range(0, Ntest):
            zgen = np.concatenate((zx[i, :][np.newaxis, :], zy[i, :][np.newaxis, :]), axis=0)
            zgen = prng.rand(self.n_z, zgen.shape[1])*2-1

            Xhat = self.synthesize(zgen)

            # Xhat will be (X.shape[0], Ntest)
            for j in range(0, Ntest):
                Xhat_ = np.zeros([patchSize, patchSize])
                Xhat_ = np.reshape(Xhat_, (patchSize**2))
                Xhat_[:self.X.shape[0]] = Xhat[:, j]
                Xhat_ = np.reshape(Xhat_, (patchSize, patchSize))
                BigPic[j * patchSize:(j + 1) * patchSize, i * patchSize:(i + 1) * patchSize] = np.flipud(Xhat_)

        self.imshow_map.set_data(np.flipud(BigPic))
        self.imshow_map.autoscale()
        self.fig20.canvas.draw()
        self.fig20.canvas.flush_events()


''' Multiclass classification member functions '''
def get_conf_matrix(self, Yhat, Ytrue):
    # First, evaluate all the data

    # Yhat will be (nclasses, nsamples); yhat will be (1, nsamples)
    yhat = np.argmax(Yhat, axis=0)
    ytrue = np.argmax(Ytrue, axis=0)

    # Create a confusion matrix
    C = np.zeros([np.size(Yhat, 0), np.size(Yhat, 0)])

    for k in range(0, np.size(Yhat, 1)):
        # Matrix will be: rows for real values, columns for results
        C[ytrue[k], yhat[k]] += 1

    return C

# One-vs-others separation
def one_vs_others(self, Xdata, Ydata):

    # Confusion matrix has rows as real data, cols as fake data
    C = self.get_conf_matrix(Xdata, Ydata)

    # The way to separate them is simple:
    Q = []
    for k in range(0, C.shape[0]):
        TP = C[k, k]
        FP = np.sum(C[:, k]) - TP
        FN = np.sum(C[k, :]) - TP
        TN = np.sum(C) - FP - FN - TP

        q = np.zeros([2, 2])
        q[0, 0] = np.copy(TN)
        q[0, 1] = np.copy(FN)
        q[1, 0] = np.copy(FP)
        q[1, 1] = np.copy(TP)

        Q.append(q)

    return Q

# Produce classifier statistics
def generate_classifier_stats(Q):
    # Q is a list [] with as many 2x2 matrices as categories. Generate all major classifiers per cat.
    stats_per_class = []
    for k in range(0, len(Q)):
        class_dict = {}
        class_dict['TN'] = Q[k][0, 0]
        class_dict['FN'] = Q[k][0, 1]
        class_dict['FP'] = Q[k][1, 0]
        class_dict['TP'] = Q[k][1, 1]

        # Sensitivity, recall, hit rate, or True Positive Rate
        class_dict['TPR'] = class_dict['TP']/(class_dict['TP'] + class_dict['FN'])
        # Specificity, selectivity, or true negative rate (TNR)
        class_dict['TNR'] = class_dict['TN']/(class_dict['TN'] + class_dict['FP'])
        # False Negative Rate or miss rate
        class_dict['FNR'] = 1- class_dict['TPR']
        # Fall-out or False Positive Rate (FPR)
        class_dict['FPR'] = 1- class_dict['TNR']

        # Precision or Positive Predictive Value
        class_dict['PPV'] = class_dict['TP']/(class_dict['TP'] + class_dict['FP'])

        # Accuracy (ACC)
        class_dict['ACC'] = (class_dict['TP'] + class_dict['TN'])/(class_dict['TP'] + class_dict['TN'] + class_dict['FP'] + class_dict['FN'])

        stats_per_class.append(class_dict)

    return stats_per_class

# Show normalized confusion matrix (std)
def show_normalized_confusion(self, Yhat, Ydata, ax, imshow_window, text_list):

    C = self.get_conf_matrix(Yhat, Ydata)

    for k in range(0, C.shape[0]):
        C[k, :] = C[k, :]/np.sum(C[k, :])

    imshow_window.set_data(C)
    imshow_window.autoscale()
    for (j, i), label in np.ndenumerate(C):
        text_list[i][j].set_text(np.round(C[j, i],3))
