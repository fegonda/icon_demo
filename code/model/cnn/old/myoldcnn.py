#---------------------------------------------------------------------------
# cnn.py
#
# Author  : Felix Gonda
# Date    : July 10, 2015
# School  : Harvard University
#
# Project : Master Thesis
#           An Interactive Deep Learning Toolkit for
#           Automatic Segmentation of Images
#
# Summary : This file contains implementation of a convolutional 
#           neural network model using Lenet5
#---------------------------------------------------------------------------

import os
import sys
import time
import numpy as np
import cPickle

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
from mlpmodel import MLP
from utility import Utility
from db import DB
from cnn_tools import CNN_Tools

class CNN(Model):

	def __init__(	self, 
			id,		# unique identifier of the model
			rng,            # random number generator use to initialize weights
			input,          # a theano.tensor.dmatrix of shape (n_examples, n_in)
			batch_size,     # size of mini batch
			patch_size,     # size of feature map
			nkernels,       # number of kernels for convolutional layers
			kernel_sizes,   # kernel sizes for convolutional layers
			hidden_sizes,   # list of number of hidden units for each layer
			train_time,     # batch training time before resampling
			path,           # model's path
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

		self.activation   = activation
		self.nkernels      = nkernels
		self.kernel_sizes = kernel_sizes
		self.hidden_sizes = hidden_sizes

		self.initialize()

	def initialize(self):
		self.params       = []
		self.convLayers   = []

		hidden_sizes      = self.hidden_sizes
		kernel_sizes      = self.kernel_sizes
		nkernels          = self.nkernels
		batch_size        = self.batch_size
		patch_size        = self.patch_size
                nLayers           = len(nkernels)
                nFeatureMaps      = 1
                featureMapSize    = patch_size

		# Reshape matrix of rasterized images of shape (batch_size, featureMapSize * featureMapSize)
    		# to a 4D tensor, compatible with our LeNetConvPoolLayer
    		# (featureMapSize,featureMapSize) is the size of Training samples.
		input = self.input.reshape((batch_size, 1, featureMapSize, featureMapSize))
		input_next = input	

		# setup the cnn model
		for i in range( nLayers ):
			layer = LeNetConvPoolLayer(
				self.rng,
				input=input_next,
				image_shape=(batch_size, nFeatureMaps, featureMapSize, featureMapSize),
				filter_shape=(nkernels[i], nFeatureMaps, kernel_sizes[i], kernel_sizes[i]),
				poolsize=(2, 2))		
			input_next = layer.output
			nFeatureMaps = nkernels[i]
			featureMapSize = np.int16(np.floor((featureMapSize - kernel_sizes[i]+1) / 2))
			self.params += layer.params
			self.convLayers.append(layer)

		print '#layers:', len(self.convLayers)

		# the 2 is there to preserve the batchSize
		mlp_input = self.convLayers[-1].output.flatten(2)

		self.mlp  = MLP(rng=self.rng,
				input=mlp_input,
				n_in=nkernels[-1] * (featureMapSize ** 2),
				n_hidden=hidden_sizes,
				n_out=2,
				activation=rectified_linear,
				id=id)

		self.params     += self.mlp.params
		self.cost        = self.mlp.negative_log_likelihood
        	self.errors      = self.mlp.errors
        	self.p_y_given_x = self.mlp.p_y_given_x
		self.y_pred      = self.mlp.y_pred

		if os.path.isfile( self.path ):
			with open(self.path, 'r') as file:
				data = cPickle.load(file)
				saved_convLayers         = data[0]
				saved_hiddenLayers       = data[1]
				saved_logRegressionLayer = data[2]
				saved_nkernels           = data[3]
				saved_kernel_sizes       = data[4]
				saved_batch_size         = data[5]
				saved_patch_size         = data[6]
				saved_hidden_sizes       = data[7]

			for s_cl, cl in zip(saved_convLayers, self.convLayers):
				cl.W.set_value(s_cl.W.get_value())
				cl.b.set_value(s_cl.b.get_value())

			for s_hl, hl in zip(saved_hiddenLayers, self.mlp.hiddenLayers):
				hl.W.set_value(np.float32(s_hl.W.eval()))
				hl.b.set_value(s_hl.b.get_value())

			self.mlp.logRegressionLayer.W.set_value(np.float32(saved_logRegressionLayer.W.eval()))
			self.mlp.logRegressionLayer.b.set_value(saved_logRegressionLayer.b.get_value())
		

	def save(self):
		Utility.report_status('saving best model', self.path)
		DB.beginSaveModel( self.id )
		with open(self.path, 'wb') as file:
            		cPickle.dump((
				self.convLayers, 
				self.mlp.hiddenLayers, 
				self.mlp.logRegressionLayer, 
				self.nkernels, 
				self.kernel_sizes, 
				self.batch_size, 
				self.patch_size, 
				self.hidden_sizes), file)
		DB.finishSaveModel( self.id )

	def train(self):

		n_required = 3*self.batch_size 

		# a minium of 3*batch_size is required
		if self.n_data < n_required:
			print 'not enough data to train. '
			print 'A min of %d samples is required,'%(n_required)
			print 'but got %d samples instead...'%(n_data)
			return

		self.sampleTrainingData()

		def gradient_updates_momentum(cost, params, learning_rate, momentum):
        		updates = []
        		for param in params:
            			param_update = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
            			updates.append((param, param - learning_rate*param_update))
            			updates.append((param_update, momentum*param_update + (1. - momentum)*T.grad(cost, param)))
        		return updates

		batch_size = self.batch_size
		index      = self.index
		lr         = self.lr
		m          = self.m
		x          = self.x
		y          = self.y

		test_set_x = self.test_x
		test_set_y = self.test_y
		train_set_x = self.train_x
		train_set_y = self.train_y
		valid_set_x = self.valid_x
		valid_set_y = self.valid_y

		# compute number of minibatches for training, validation and testing
		n_train_batches = self.n_train / batch_size
		n_valid_batches = self.n_valid / batch_size
		n_test_batches  = self.n_test / batch_size

		cost = self.cost(y)

		# create a function to compute the mistakes that are made by the model
    		test_model = theano.function(
        		[index],
        		self.errors(y),
        		givens={
            			x: test_set_x[index * batch_size: (index + 1) * batch_size],
            			y: test_set_y[index * batch_size: (index + 1) * batch_size]
        		}
		)

		validate_model = theano.function(
        		[index],
        		self.errors(y),
        		givens={
            			x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            			y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        		}
    		)


                predict_samples = theano.function(
                        inputs=[index],
			outputs=T.neq(self.y_pred, self.y),
			#outputs=-T.log(self.p_y_given_x)[T.arange(y.shape[0]), y],
                        givens={
                                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                                y: train_set_y[index * batch_size: (index + 1) * batch_size]
                        }
                )

		gparams = []
    		for param in self.params:
        		gparam = T.grad(cost, param)
        		gparams.append(gparam)

		updates = gradient_updates_momentum(cost, self.params, lr, m)
		
		train_model = theano.function(inputs=[index], outputs=cost,
            		updates=updates,
            		givens={
                		x: train_set_x[index * batch_size:(index + 1) * batch_size],
                		y: train_set_y[index * batch_size:(index + 1) * batch_size],
                		lr: self.lr_shared,
                		m: self.m_shared})

		elapsed_time = 0.0;
		start_time = time.clock()
		minibatch_index = 0

    		patience = 100  	# look as this many examples regardless
    		patience_increase = 2   # wait this much longer when a new best is
                           		# found
    		improvement_threshold = 0.995   # a relative improvement of this much is
                                   		# considered significant
    		validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

		print 'batchsize:', self.batch_size
		print 'patchsize:', self.patch_size
		print 'n_train:', self.n_train
		print 'n_valid:', self.n_valid
		print 'validation_frequency:',validation_frequency
                print 'n_train_batches:',n_train_batches
                print 'n_valid_batches:',n_valid_batches
                print 'n_test_batches:',n_test_batches

                print 'trainx:',train_set_x.get_value(borrow=True).shape
                print 'validx:',valid_set_x.get_value(borrow=True).shape
                print 'testx:',test_set_x.get_value(borrow=True).shape

		while(elapsed_time < self.train_time):

			if (minibatch_index == (n_train_batches-1)):
				minibatch_index = 0
				break

			minibatch_avg_cost = train_model(minibatch_index)
                    	iteration = minibatch_index
                    	i = minibatch_index
                    	minibatch_index += 1

                    	# test the trained samples against the target
                    	# values to measure the training performance
                   	probs = predict_samples(minibatch_index)
			#print 'probs:', probs.shape
                    	i_batch = self.i_train[ i * batch_size:(i+1)*batch_size ]
                    	self.p_data[ i_batch ] = probs
                    	good = np.where( probs == 0)[0]
                    	bad  = np.where( probs == 1)[0]		
	
			# periodically perform a validation of the the 
                    	# model and report stats
                    	if (iteration)%validation_frequency == 0:

				validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
                           	this_val_loss = np.mean(validation_losses)
                           	self.reportTrainingStats(elapsed_time, 
					minibatch_index, 
					this_val_loss, 
					minibatch_avg_cost.item(0))

				if this_val_loss < self.best_loss:
					self.best_loss = this_val_loss
					best_iter = iteration
					self.save()
				
			# update elapsed time
			elapsed_time = time.clock() - start_time

		end_time = time.clock()
		msg = 'The code ran for'
		status = '%f seconds' % ((end_time - start_time))
		Utility.report_status( msg, status )


	def classify(self, image):
		print 'classifying...'
                probs = np.zeros(np.shape(image))

                batch_size = self.batch_size
                index      = self.index
                x          = self.x
                row_range  = 1
                imSize     = np.shape( image )
		n_batches  = 1024/self.batch_size

                patch = np.zeros( (imSize[0]*row_range, self.patch_size**2), dtype=np.float32 )
                test_set_x = theano.shared( patch, borrow=True)

                classify_data = theano.function(
                        [index],
                        outputs=self.p_y_given_x,
			givens={x: test_set_x[index * batch_size: (index + 1) * batch_size]})
                        #givens={x: test_set_x})

                start_time = time.clock()
                for row in xrange(0,1024,row_range):
                        patch = Utility.get_patch(image, row, row_range, self.patch_size )
                        test_set_x.set_value( np.float32(patch) )

                        for minibatch_index in xrange(n_batches):
				result    = classify_data(minibatch_index)
				col_start = minibatch_index*batch_size
				col_end   = (minibatch_index+1)*batch_size
				probs[row,col_start:col_end] = result[:,1]
			#probs[probs >= 0.5] = 1
			#probs[probs <0.5  ] = 0
			#p = np.int64( probs )
			
			if row%64 == 0:
				print '.'
 
		end_time = time.clock()
                duration = (end_time - start_time)
                print 'Segmentation finished. Duration:', duration
                return np.array(probs)


	def predictold(self, image):
		print 'predicting...'
		prediction = np.zeros( 1024 * 1024, dtype=int )

                row_range = 1 
		#1024/512
                imSize = np.shape( image )

                patch = np.zeros( (1024*row_range, self.patch_size**2), dtype=np.float32 )
                test_set_x = theano.shared( patch, borrow=True)

		print 'rowr:', row_range
		print 'tsetx:',test_set_x.get_value(borrow=True).shape

                batch_size = self.batch_size
                index      = self.index
                x          = self.x

                predict_data = theano.function(
                        inputs=[index],
                        outputs=self.y_pred,
                        givens={x: test_set_x[index * batch_size: (index + 1) * batch_size]}
                )


		n_batches = 1024/self.batch_size

                start_time = time.clock()
		offset = 0

		start_time = time.clock()
                for row in xrange(0,1024,row_range):
                        patch = Utility.get_patch(image, row, row_range, self.patch_size)
			test_set_x.set_value( np.float32(patch) )

			for minibatch_index in xrange(n_batches):
				result = predict_data( minibatch_index )
				#print '-->',minibatch_index, np.bincount( result )
					
				#f[r*5+(mb*bs):r*5+(mb+1)*bs]
				#start = (row * 1024)+(minibatch_index*batch_size)
				#end   = (row * 1024)+((minibatch_index+1)*batch_size)
				prediction[offset:offset+batch_size] = result
				offset += batch_size
				#print offset, np.bincount( prediction )

			#print 'predicted row %d'%(row), np.bincount( prediction )
			#if row%32 == 0:
			#	print 'predicted row %d'%(row), np.bincount( prediction )

			if row%64 == 0:
                                print np.bincount( result )

		print np.bincount( prediction )
		print 'done'
		end_time = time.clock()
                duration = (end_time - start_time)
		print 'duration:', duration
		return prediction


	def predict(self, image):

		prob = CNN_Tools.classify( image, self )
		print prob
