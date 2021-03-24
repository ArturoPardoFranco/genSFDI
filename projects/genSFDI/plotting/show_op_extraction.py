'''
A toy example for OP extraction with a multilayer perceptron
Arturo Pardo, 2021
'''

import numpy as np
import matplotlib.pyplot as plt
import sys, os

# Imports from core-modules
import ai.scNets as scnets
import utils.sfdi as sfdi

plt.rc('text', usetex=True)
plt.rc('font', **{'size': 15, 'family':'sans-serif'})
plt.rc('axes', titlesize=15)
plt.rc('legend', fontsize=15)


'''
We use the network trained in 
../compute/train_op_estimator.py
to extract the optical properties of a specific spectrum.

Make sure your model is trained!
'''

''' Simulation parameters ------------------------------------------------------------------------------------------ '''

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


''' Loading model and preparing data ------------------------------------------------------------------------------- '''

# Layers
n_net = [np.size(fx_range), 300, 300, 300, 300, 300, 3]

# Prepare network -- note: fresh_init == 0 implies the network loads from write_path instead of initializing.
net = scnets.MLP_SC(n_net, lr=0.001, out_actype='sigmoid', cost_function='MSE', fixed_lr=False, regeps=0.001,
                    actype='lrelu', fresh_init=0, mbs=128, name='op_estimator', write_path=op_network_path)


''' Inference, plotting -------------------------------------------------------------------------------------------- '''

# Produce a spectrum with optical properties in this range.
# We will use adipose tissue as reference at 500 nm: mu_s' = 0.3 1/mm, mu_a = 0.05 1/mm, gamma = 1.8, n = 1.4
# in SI units (1/m), that is
Rd = sfdi.Rd_complete(fx_range, 300, 50, 1.8, 1.4)

# To get it into the network, it has to be a column vector
Rd = Rd[:, np.newaxis]

# Extract OPS
mu = net.output(Rd)

# Translate to true ranges:
mu_s_hat = mu_s_prime_range[0] + mu[0, :]*(mu_s_prime_range[1] - mu_s_prime_range[0])
mu_a_hat = mu_a_range[0] + mu[1, :]*(mu_a_range[1] - mu_a_range[0])
gamma_hat = gamma_range[0] + mu[2, :]*(gamma_range[1] - gamma_range[0])

print(f'Initial valuess: mu_s = {300.0}, mu_a = {50.0}, gamma = {1.8}')
print(f'Estimated values: mu_s = {np.round(mu_s_hat, 3).squeeze()}, mu_a = {np.round(mu_a_hat, 3).squeeze()}, gamma = {np.round(gamma_hat, 3).squeeze()}')


# Show all
fig1 = plt.figure(1)
fig1.clf()
ax11 = fig1.add_subplot(1, 1, 1)
ax11.clear()
ax11.semilogy(fx_range*1E-3, Rd, color='k', marker='.', alpha=0.5, label=r'True $R_d(f_x)$')
ax11.semilogy(fx_range*1E-3,  sfdi.Rd_complete(fx_range, mu_s_hat, mu_a_hat, gamma_hat, 1.4),  marker='.',
              label=r'Estimated $R_d(f_x)$')
ax11.set_xlabel(r'$f_x$ (1/mm)')
ax11.set_xlabel(r'$R_d(f_x)$ (a.u.)')
ax11.grid(True, which='both')
ax11.legend(fancybox=True, framealpha=0.5)
ax11.set_title(r'\textbf{Neural LUT for diffuse and subdiffuse MTFs}')

fig1.tight_layout()
fig1.canvas.draw()
fig1.canvas.flush_events()
fig1.canvas.draw()
fig1.canvas.flush_events()

# And that's it!