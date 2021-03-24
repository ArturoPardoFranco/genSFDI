'''
Optical properties learning with a neural network
Arturo Pardo, 2020
'''

'''
This is a copy of the original experiment files, except that we don't use patient metadata (spatial freqs and wavelengths). 
Instead we just use a proxy of the known, used frequencies and wavelengths.

A couple of notes:
    (1) The network only operates with normalized values [mu_min, mu_max] --> [0, 1]. In other words, the estimator net
        only produces a number between 0 and 1, and thus the operating ranges per parameter must be stored and/or known.
        To retrieve the proper OP value, you must re-scale [0, 1] --> [mu_min, mu_max]. 
    (2) The dataset does not store OP triplets (mu_s, mu_a, gamma) that result in a non-monotonically decaying function. 
        This is a known property of MTFs; we have observed that violating this principle confuses the network, particularly
        in noise-rich environments where this property _seems_ to not apply. You can test this yourself!
    (3) The network is a cookie-cutter version of a skip-connected MLP used for both classification and regression. 
        When you train (at least in an iPython console) the module will open a figure that serves as a small real-time monitor.
        It will provide a confusion matrix -- ignore it, since this is a regression problem, and keep an eye on the 
        train/validation MSE on the left-side subplot (green curves). Accuracy makes no sense in this context.
'''

# General imports
import numpy as np
from tqdm import tqdm
import os, sys, time
import pickle as pkl

sys.path.append('../') # Project folder

# Our imports
import ai.scNets as scnets
import utils.sfdi as sfdi

''' Control variables ---------------------------------------------------------------------------------------------- '''

# Path to the optical properties dataset
op_dataset_path = '../../../output/vaeSFDI_v2/op_datasets_final_008'
op_ds_name = 'op_dataset.pkl'
os.makedirs(op_dataset_path, exist_ok=True)

# Path to the network storage area
op_network_path = '../../../output/vaeSFDI_v2/op_networks_final_008'
os.makedirs(op_network_path, exist_ok=True)

# Ranges for all variables
fx_range = np.array([0.0, 0.05, 0.15, 0.61, 0.78, 0.92, 1.37])*1E3      # Spatial frequency range
mu_s_prime_range = np.array([10, 4000])                                 # Reduced scattering coefficient (1/m)
gamma_range = np.array([1.0, 4])                                        # Phase function parameter (a.u.)
mu_a_range = np.array([10, 4000])                                       # Absorption coefficient (1/m)
n = 1.4                                                                 # Refractive index

# Generate dataset?
generate_dataset = 1

# Resolution per axis -- Note: The final dataset wize will be this number ^3.
axis_res = 300

# Range vectors -- useful for later.
mu_s_vec = np.linspace(mu_s_prime_range[0], mu_s_prime_range[1], axis_res)      # Reduced scattering coefficient (1/m)
mu_a_vec = np.linspace(mu_a_range[0], mu_a_range[1], axis_res)                  # Absorption coefficient (1/m)
gamma_vec = np.linspace(gamma_range[0], gamma_range[1], axis_res)               # Phase parameter (Legendre moments)

# Train the model?
train_model = 1

# Model structure (multilayer perceptron with skip connections)
n_net = [np.size(fx_range), 300, 300, 300, 300, 300, 3]

# Training schedule
training_schedule = {
    'mode': 'ocp',
    'stop_criterion': 'none',
    'eval_criterion': 'iter',
    'Niters': 2000000,
    'lr_max': 0.0001,
    'lr_min': 0.0,
    'Nwarm': 200000,
    'Ncool': 500000,
    'T': 5000,
    'dropout_rate': 0.8,  # (!) We use 1-p_drop (keep_prob in TF1).
    'mult': 1,
    'max_patience': 50,
    'eval_size': 5000
}

''' Dataset generation --------------------------------------------------------------------------------------------- '''
if generate_dataset == 1:

    # Training set
    X_train = []
    Y_train = []

    # Test set
    X_test = []
    Y_test = []

    for i in tqdm(range(0, axis_res), desc='Generating training set'):
        for j in range(0, axis_res):
            for k in range(0, axis_res):
                # Get the right sequence of optical properties.
                musp_rand = mu_s_vec[i]
                mua_rand = mu_a_vec[j]
                g_rand = gamma_vec[k]

                params = np.squeeze(np.array([musp_rand, mua_rand, g_rand]))
                Rd = sfdi.Rd_complete(fx_range, params[0], params[1], params[2], n, eta_mode='const')

                if np.sum(np.diff(Rd) < 0) == np.size(np.diff(Rd)): # If the obtained reflectance has monotonic decay...
                    X_train.append(Rd.copy())

                    # Actual parameters must be normalized (otherwise the net can't ever converge on estimating ~1E5)
                    params_norm = params.copy()
                    params_norm[0] = (params_norm[0]-mu_s_prime_range[0])/(mu_s_prime_range[1]- mu_s_prime_range[0])
                    params_norm[1] = (params_norm[1] - mu_a_range[0]) / (mu_a_range[1] - mu_a_range[0])
                    params_norm[2] = (params_norm[2] - gamma_range[0]) / (gamma_range[1] - gamma_range[0])
                    Y_train.append(params_norm.copy())

    for i in tqdm(range(0, axis_res, 5), desc='Generating training set'):
        for j in range(0, axis_res, 5):
            for k in range(0, axis_res, 5):
                # Get the right sequence for optical properties.
                musp_rand = mu_s_vec[i]
                mua_rand = mu_a_vec[j]
                g_rand = gamma_vec[k]

                params = np.squeeze(np.array([musp_rand, mua_rand, g_rand]))
                Rd = sfdi.Rd_complete(fx_range, params[0], params[1], params[2], n, eta_mode='const')

                if np.sum(np.diff(Rd) < 0) == np.size(np.diff(Rd)): # If the obtained reflectance has monotonic decay...
                    X_test.append(Rd.copy())

                    params_norm = params.copy()
                    params_norm[0] = (params_norm[0] - mu_s_prime_range[0]) / (mu_s_prime_range[1] - mu_s_prime_range[0])
                    params_norm[1] = (params_norm[1] - mu_a_range[0]) / (mu_a_range[1] - mu_a_range[0])
                    params_norm[2] = (params_norm[2] - gamma_range[0]) / (gamma_range[1] - gamma_range[0])
                    Y_test.append(params_norm.copy())

    out_pack = {
        'X_train': np.asarray(X_train).T,
        'Y_train': np.asarray(Y_train).T,
        'X_test': np.asarray(X_test).T,
        'Y_test': np.asarray(Y_test).T
    }

    for key in out_pack.keys():
        print(key + ' is ' + str(np.shape(out_pack[key])))

    pkl.dump(out_pack, open(op_dataset_path + '/' + op_ds_name, 'wb'), protocol=4)


''' Model training ------------------------------------------------------------------------------------------------- '''
if train_model == 1:

    # Load dataset
    input_dataset = pkl.load(open(op_dataset_path + '/' + op_ds_name, 'rb'))

    input_dataset['y_train'] = np.zeros(np.size(input_dataset['Y_train'], 1))
    input_dataset['y_test'] = np.zeros(np.size(input_dataset['Y_test'], 1))

    # Prepare network
    net = scnets.MLP_SC(n_net, lr=0.001, out_actype='sigmoid', cost_function='MSE', fixed_lr=False, regeps=0.001,
                        actype='lrelu', fresh_init=1.0, mbs=128, name='op_estimator', write_path=op_network_path)

    net.train(input_dataset, training_schedule)