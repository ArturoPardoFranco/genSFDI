'''
Loading classic datasets
Arturo Pardo, 2021
'''

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist, cifar10, cifar100, fashion_mnist
from tensorflow.keras.utils import to_categorical
import numpy as np
import h5py
import scipy.io as sio
import glob, sys, csv
from PIL import Image
from tqdm import tqdm
import pickle as pkl
import time
import skimage as ski

import os
curr_path = os.path.dirname(os.path.abspath(__file__))

# Uses tf.keras MNIST iface to load and prepare data
def get_mnist(mlp=False, debug=0):
    # Use standard model
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Modifications for mlp/convnet
    if mlp == True:
        X_train = (np.reshape(X_train, (60000, 28 * 28)).T) / 255.0
        X_test = (np.reshape(X_test, (10000, 28 * 28)).T) / 255.0
    else:
        X_train = X_train[:, :, :, np.newaxis]/255.0
        X_test = X_test[:, :, :, np.newaxis]/255.0

    Y_train = to_categorical(y_train).T
    Y_test = to_categorical(y_test).T

    y_train = np.squeeze(y_train)
    y_test = np.squeeze(y_test)

    nameList = '0, 1, 2, 3, 4, 5, 6, 7, 8, 9'

    dataset = {
        'X_train': X_train,
        'Y_train': Y_train,
        'y_train': y_train,
        'X_test': X_test,
        'Y_test': Y_test,
        'y_test': y_test,
        'labels': nameList.split(', ')
    }

    print('Loaded MNIST.')
    for key in dataset.keys():
        print('key: ' + key + ', shape: ' + str(np.shape(dataset[key])))

    return dataset

# Ditto, for CIFAR-10
def get_cifar10(mlp=False, debug=0):
    # Use standard tf.keras load function
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    X_train = np.asarray(X_train).astype(float)
    X_test = np.asarray(X_test).astype(float)

    # Calibrate, it's uint8
    if mlp == True:
        X_train = (np.reshape(X_train, (50000, 32 * 32 * 3)).T) / 255.0
        X_test = (np.reshape(X_test, (10000, 32 * 32 * 3)).T) / 255.0
    else:
        X_train /= 255.0
        X_test /= 255.0

    Y_train = to_categorical(y_train).T
    Y_test = to_categorical(y_test).T

    y_train = np.squeeze(y_train)
    y_test = np.squeeze(y_test)

    dataset = {
        'X_train': X_train,
        'Y_train': Y_train,
        'y_train': y_train,
        'X_test': X_test,
        'Y_test': Y_test,
        'y_test': y_test,
        'labels': ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    }

    print('Loaded CIFAR-10.')
    for key in dataset.keys():
        print('key: ' + key + ', shape: ' + str(np.shape(dataset[key])))

    return dataset
