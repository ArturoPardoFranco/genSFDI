'''
Storage functions for further reuse
Arturo Pardo, 2019
'''

import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug
#tf.config.optimizer.set_jit(True)
import csv, os

''' Universal starter '''
def start(self):
    # Define saver with defined variables
    self.saver = tf.train.Saver()

    # Configure GPU to grow memory consumption as needed
    self.config = tf.ConfigProto()
    self.config.gpu_options.allow_growth = True
    #self.config.graph_options.rewrite_options.auto_mixed_precision = 1

    # Create a graph for this module
    self.graph = tf.Graph()

    if self.fresh_init == 1:
        # Initialize global variables
        self.initializer = tf.global_variables_initializer()
        # Starts a new tensorflow session.
        self.sess = tf.Session(config=self.config)
        #self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess, ui_type='readline')
        #self.sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        self.sess.run(self.initializer)
        self.save_model()
    else:
        self.sess = tf.Session(config=self.config)
        self.load_model()
        self.save_model()

    '''
    # Tracing -- comment when not debugging!
    self.options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    self.run_metadata = tf.RunMetadata()
    '''

    # Generate a summary
    self.summ_writer = tf.summary.FileWriter(self.logfolder, self.sess.graph)

''' Stores a model from the original class '''
def save_model(self):
    save_path = self.saver.save(self.sess, self.write_path + '/model/' + self.name + '_model.ckpt')
    if self.quiet == 0: print('Model saved at ' + save_path)

''' Loads the model from specified output path '''
def load_model(self):
    self.saver.restore(self.sess, self.write_path + '/model/' + self.name + '_model.ckpt')
    if self.quiet == 0: print('Model restored.')