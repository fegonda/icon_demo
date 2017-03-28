import cPickle
import gzip

import os
import sys
import time

import numpy
import numpy as np

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

base_path = os.path.dirname(__file__)
sys.path.insert(1,os.path.join(base_path, '..'))
sys.path.insert(2,os.path.join(base_path, '../mlp'))
sys.path.insert(3,os.path.join(base_path, '../../external'))

from activation_functions import rectified_linear
from logistic_sgd import LogisticRegression
from convlayer import LeNetConvPoolLayer
from model import Model
from mlp_online import MLP
from utility import Utility
from db import DB

import getpass

class CNN(Model):
	# batch_size needs to be the number of rows in my test set,
	# batch_size - size of the minibatch
        def __init__(   self,
                        id,             # unique identifier of the model
                        rng,            # random number generator use to initialize weights
                        input,          # a theano.tensor.dmatrix of shape (n_examples, n_in)
                        batch_size,     # size of mini batch
                        patch_size,     # size of feature map
                        nkernels,       # number of kernels for convolutional layers
                        kernel_sizes,   # kernel sizes for convolutional layers
                        hidden_sizes,   # list of number of hidden units for each layer
                        train_time=30.0,     # batch training time before resampling
                        path=None,           # model's path
                        activation=rectified_linear
                        ):

                Model.__init__( self,
                                input=input,
                                rng=rng,
                                batch_size=batch_size,
                                patch_size=patch_size,
                                train_time=train_time,
                                path=path,
                                id=id,
                                type='CNN')

		self.nkerns      = nkernels
                self.kernelSizes = kernel_sizes
                self.hiddenSizes = hidden_sizes
                self.patchSize = patch_size
                self.batchSize = batch_size
		self.activation = activation

		self.initialize()
		

	def initialize(self):	
		self.params     = []
		self.convLayers = []

		input = self.input.reshape((self.batchSize, 1, self.patchSize, self.patchSize))

		self.layer0_input = input

		rng=self.rng
		input_next = input
		numberOfFeatureMaps = 1
		featureMapSize = self.patchSize
		nkerns = self.nkerns
		kernelSizes = self.kernelSizes
		hiddenSizes = self.hiddenSizes
	 	batchSize = self.batchSize
		patchSize = self.patchSize	

		for i in range(len(nkerns)):
		    layer = LeNetConvPoolLayer(
			rng,
			input=input_next,
			image_shape=(batchSize, numberOfFeatureMaps, featureMapSize, featureMapSize),
			filter_shape=(nkerns[i], numberOfFeatureMaps, kernelSizes[i], kernelSizes[i]),
			poolsize=(2, 2)
		    )
		    input_next = layer.output
		    numberOfFeatureMaps = nkerns[i]
		    featureMapSize = np.int16(np.floor((featureMapSize - kernelSizes[i]+1) / 2))

		    self.params += layer.params
		    self.convLayers.append(layer)

		# the 2 is there to preserve the batchSize
		mlp_input = self.convLayers[-1].output.flatten(2)

                self.mlp  = MLP(rng=self.rng,
                                input=mlp_input,
                                n_in=nkerns[-1] * (featureMapSize ** 2),
                                n_hidden=hiddenSizes,
                                n_out=2,
                                activation=self.activation,
                                id=self.id)

		self.params += self.mlp.params

		self.cost = self.mlp.negative_log_likelihood
		self.errors = self.mlp.errors
		self.p_y_given_x = self.mlp.p_y_given_x
		self.debug_x = self.p_y_given_x

		if not self.path is None and os.path.exists( self.path ):
		    with open(self.path, 'r') as file:
			print 'loading from file..', self.path
			data = cPickle.load(file)
			saved_convLayers         = data[0]
			saved_hiddenLayers       = data[1]
			saved_logRegressionLayer = data[2]
			saved_nkerns             = data[3]
			saved_kernelSizes        = data[4]
			saved_batch_size         = data[5]
			saved_patchSize          = data[6]
			saved_hiddenSizes        = data[7]

		    for s_cl, cl in zip(saved_convLayers, self.convLayers):
			cl.W.set_value(s_cl.W.get_value())
			cl.b.set_value(s_cl.b.get_value())

		    for s_hl, hl in zip(saved_hiddenLayers, self.mlp.hiddenLayers):
			hl.W.set_value(np.float32(s_hl.W.eval()))
			hl.b.set_value(s_hl.b.get_value())

		    self.mlp.logRegressionLayer.W.set_value(np.float32(saved_logRegressionLayer.W.eval()))
		    self.mlp.logRegressionLayer.b.set_value(saved_logRegressionLayer.b.get_value())

	def save(self):
		with open(self.path, 'wb') as file:
		    cPickle.dump((self.convLayers, 
			self.mlp.hiddenLayers, 
			self.mlp.logRegressionLayer, 
			self.nkerns, 
			self.kernelSizes, 
			self.batchSize, 
			self.patchSize, 
			self.hiddenSizes), 
			file)

	def train(self):
		print 'do nothing...'
