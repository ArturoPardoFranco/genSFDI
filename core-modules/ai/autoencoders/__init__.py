'''
This paper uses two unsupervised models:
'''

from ai.autoencoders.convInfoSCVAE_GCN_gated_encoded import *   # A primary VAE with a clamped bottleneck
from ai.autoencoders.mlpInfoSCVAE import *                      # A secondary MLP VAE with a 2D bottleneck