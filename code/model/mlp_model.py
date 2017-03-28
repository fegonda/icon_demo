#---------------------------------------------------------------------------
# Utility.py
#
# Author  : Felix Gonda
# Date    : July 10, 2015
# School  : Harvard University
#
# Project : Master Thesis
#           An Interactive Deep Learning Toolkit for
#           Automatic Segmentation of Images
#
# Summary : This file contains utility functions for reading, writing, and
#           processing images.
#---------------------------------------------------------------------------

import os
import sys
import time
import ConfigParser
import pandas as pd
import numpy as np
import theano
import theano.tensor as T
import cPickle

theano.config.floatX = 'float32'

base_path = os.path.dirname(__file__)
sys.path.insert(1,os.path.join(base_path, '../external'))
sys.path.insert(2,os.path.join(base_path, '../common'))

from logistic_sgd import LogisticRegression
from mlp import HiddenLayer
from activation_functions import rectified_linear
from utility import Utility

class MLP_Model(object):


    def __init__(self,
		 rng,                          # random number generator use to initialize weights
		 input,                        # a theano.tensor.dmatrix of shape (n_examples, n_in)
		 n_in=None,                    # dimensionality of the input
		 n_hidden=None,                # list of number of hidden units for each layer
		 n_out=None,                   # dimensionality of the output
		 filename=None,                # filename to load model from
		 activation=rectified_linear   # activation function to use
		 #activation=T.tanh
                 ):

	self.params = []
	self.hiddenLayers = []
	self.n_in = n_in
	self.n_hidden = n_hidden
	self.input = input
	self.x     = input
	self.rng   = rng

	load_from_file = (filename != None and os.path.exists( filename ))

	if load_from_file:
		w, b = self.load( filename, rng, input, activation )
	else:
		w, b = self.create(rng, input, activation)

        # The logistic regression layer gets as input the hidden units
        # of the last hidden layer
        self.logRegressionLayer = LogisticRegression(
                input=self.hiddenLayers[-1].output,
                n_in=self.n_hidden[-1],
                n_out=n_out)

	if w is not None:
		self.logRegressionLayer.W.set_value( w.get_value(), borrow=True)

	if b is not None:
		self.logRegressionLayer.b.set_value( b.get_value(), borrow=True)

	self.params += self.logRegressionLayer.params
        self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood
        self.errors = self.logRegressionLayer.errors
        self.p_y_given_x = self.logRegressionLayer.p_y_given_x
	self.y_pred = self.logRegressionLayer.y_pred


    def add_hidden_layer(self, rng, input, n_in, n_out, activation, W=None, b=None):

        layer = HiddenLayer(
                rng=rng,
                input=input,
                n_in=n_in,
                n_out=n_out,
                activation=activation)

	if W != None:
		layer.W.set_value( W.get_value(), borrow=True )

	if b != None:
		layer.b.set_value( b.get_value(), borrow=True )

	self.hiddenLayers.append( layer )
	self.params += layer.params
	return layer

    def create(self, rng, input, activation):
        next_input = input
        next_n_in  = self.n_in
        for n_h in self.n_hidden:
		layer = self.add_hidden_layer(rng, next_input, next_n_in, n_h, activation)
                next_input = layer.output
                next_n_in  = n_h
	return None, None

    def save(self, filename):
	Utility.report_status('saving best model', filename)
        with open(filename, 'wb') as file:
		cPickle.dump(
			(self.hiddenLayers,
			self.logRegressionLayer,
			self.n_in,
			self.n_hidden),
			file,
			protocol=cPickle.HIGHEST_PROTOCOL)

    def load(self, filename, rng, input, activation):
	w = None
	b = None
        with open(filename, 'r') as model_file:
        	hiddenLayers, logReg, n_in, n_hidden = cPickle.load(model_file)
		self.n_in     = n_in
		self.n_hidden = n_hidden
		next_input    = input
		next_n_in     = self.n_in
		counter = 0
		for layer in hiddenLayers:
                	layer = self.add_hidden_layer( rng, next_input, next_n_in,
				self.n_hidden[ counter ],
				activation, layer.W, layer.b )
                	next_input = layer.output
                	next_n_in  = self.n_hidden[ counter ]
			counter += 1
		w = logReg.W
		b = logReg.b
	return w, b


    def get_patch_size(self):
	return np.int(np.sqrt(self.hiddenLayers[0].W.eval().shape[0]))

    def classify_image(self, image, normMean, norm_std):
	probs = np.zeros((1024,1024))
        row_range = 1
        imSize = np.shape( image )
        patchSize = np.int(np.sqrt(self.hiddenLayers[0].W.eval().shape[0]))
                
        patch = np.zeros( (imSize[0]*row_range, patchSize**2), dtype=np.float32 )
        self.test_set_x = theano.shared( patch, borrow=True)

        classify = theano.function(
                   [],
                   outputs=self.p_y_given_x,
                   givens={self.x: self.test_set_x})

        for row in xrange(0,1024,row_range):
		
		data = Utility.get_patch( image, row, row_range, patchSize)
		'''
        	data = Utility.gen_patch(
			image, 
			rowOffset=row,
			rowRange=row_range,
			patchSize=patchSize,
			imSize=imSize)
		'''
         	self.test_set_x.set_value( data )
           	result = classify()
		
          	probs[row,:] = result[:,1]
	return np.array(probs)                  

 
