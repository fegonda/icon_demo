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
sys.path.insert(1,os.path.join(base_path, '../../external'))
sys.path.insert(2,os.path.join(base_path, '../../common'))
sys.path.insert(3,os.path.join(base_path, '..'))

from logistic_sgd import LogisticRegression
from hiddenlayer import HiddenLayer
#from mlp import HiddenLayer
from model import Model
from activation_functions import rectified_linear
from utility import Utility
from db import DB
from stats import TrainingStats

class MLP(Model):

	#---------------------------------------------------------------------------------------------------
	# construct an MLP model
	#---------------------------------------------------------------------------------------------------
        def __init__(	self, 
			id,                           # unique identifier of the model
			rng,                          # random number generator use to initialize weights
			input,                        # a theano.tensor.dmatrix of shape (n_examples, n_in)
			n_in=None,                    # dimensionality of the input
			n_hidden=None,                # list of number of hidden units for each layer
			n_out=None,                   # dimensionality of the output
			train_time=30,                # batch training time before resampling
			batch_size=10,                # size of mini batch
			patch_size=39,                # size of feature map
			path=None,                    # where to load and save model
			activation=rectified_linear,  # activation function to use
			):
			
                Model.__init__( self, 
				rng=rng, 
				input=input,
				batch_size=batch_size, 
				patch_size=patch_size, 
				train_time=train_time, 
				path=path,
				id=id,
				type='MLP')

		self.n_in         = n_in
		self.n_hidden     = n_hidden
		self.activation   = activation
		self.n_out        = n_out
		
		self.initialize()


	#---------------------------------------------------------------------------------------------------
	# initialize the MLP model
	#---------------------------------------------------------------------------------------------------
	def initialize(self):
	
                self.params = []
                self.hiddenLayers = []
	
		from_file = (self.path != None and os.path.exists( self.path ))

		if from_file:
			# load data from file
			with open(self.path, 'r') as model_file:
				data               = cPickle.load(model_file)
				savedHiddenLayers  = data[0]
				savedLogisticReg   = data[1]
				self.n_in          = data[2]
				self.n_hidden      = data[3]

		next_input = self.input
		next_n_in  = self.n_in
 
		for n_h in self.n_hidden:
		    hl = HiddenLayer(
				rng=self.rng, 
				input=next_input,
				n_in=next_n_in, 
				n_out=n_h,
				activation=self.activation)
		    next_input = hl.output
		    next_n_in = n_h
		    self.hiddenLayers.append(hl)
		    self.params += hl.params
		    
		self.logRegressionLayer = LogisticRegression(
		    input=self.hiddenLayers[-1].output,
		    n_in=self.n_hidden[-1],
		    n_out=self.n_out)

		self.params += self.logRegressionLayer.params
		self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood
		self.errors = self.logRegressionLayer.errors
		self.p_y_given_x = self.logRegressionLayer.p_y_given_x
                self.y_pred = self.logRegressionLayer.y_pred

		if from_file:
		    for hl, shl in zip(self.hiddenLayers, savedHiddenLayers):
			hl.W.set_value(shl.W.get_value())
			hl.b.set_value(shl.b.get_value())

		    self.logRegressionLayer.W.set_value(savedLogisticReg.W.get_value())
		    self.logRegressionLayer.b.set_value(savedLogisticReg.b.get_value())


	#---------------------------------------------------------------------------------------------------
	# serializes the model to the a file
	#---------------------------------------------------------------------------------------------------
	def save(self):
		Utility.report_status('saving best model', self.path)
		DB.beginSaveModel( self.id )
        	with open(self.path, 'wb') as file:
			cPickle.dump(
				(self.hiddenLayers,
				self.logRegressionLayer,
				self.n_in,
				self.n_hidden),
				file,
				protocol=cPickle.HIGHEST_PROTOCOL)
		DB.finishSaveModel( self.id )
		

    	def get_patch_size(self):
		return np.int(np.sqrt(self.hiddenLayers[0].W.eval().shape[0]))

	#---------------------------------------------------------------------------------------------------
	# generates a probability map of an image
	#---------------------------------------------------------------------------------------------------
	'''
    	def classify(self, image):
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
         		self.test_set_x.set_value( data )
           		result = classify()
          		probs[row,:] = result[:,1]
		return np.array(probs)                  
	'''

	#---------------------------------------------------------------------------------------------------
	# trains the MLP model
	#---------------------------------------------------------------------------------------------------
        def train(self ):

		n_required = 3*self.batchSize
		n_data     = len(self.y_data)

                # a minium of 3*batch_size is required
                if n_data < n_required:
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

                batch_size = self.batchSize
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

		cost = self.negative_log_likelihood(self.y)

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


                train_model = theano.function(
			inputs=[index], 
			outputs=cost,
                        updates=updates,
                        givens={
                                x: train_set_x[index * batch_size:(index + 1) * batch_size],
                                y: train_set_y[index * batch_size:(index + 1) * batch_size],
                                lr: self.lr_shared,
                                m: self.m_shared})

                elapsed_time = 0.0;
                start_time = time.clock()
                minibatch_index = 0

                patience = 100          # look as this many examples regardless
                patience_increase = 2   # wait this much longer when a new best is
                                        # found
                improvement_threshold = 0.995   # a relative improvement of this much is
                                                # considered significant
                validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

                print 'batchsize:', self.batchSize
                print 'patchsize:', self.patchSize
                print 'n_train:', self.n_train
                print 'n_valid:', self.n_valid
                print 'validation_frequency:',validation_frequency
                print 'n_train_batches:',n_train_batches
                print 'n_valid_batches:',n_valid_batches
                print 'n_test_batches:',n_test_batches

                print 'trainx:',train_set_x.get_value(borrow=True).shape
                print 'validx:',valid_set_x.get_value(borrow=True).shape
                print 'testx:',test_set_x.get_value(borrow=True).shape


		stats = []
	
		n_rotate = 0	
                while(elapsed_time < self.train_time):

			# if all samples visited before time expired, break out
                        if (minibatch_index == (n_train_batches-1)):
                                minibatch_index = 0
				print '------rotating samples...'
				self.rotateSamples()
				n_rotate += 1
				if n_rotate > 1:
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

                        # update elapsed time
                        elapsed_time = time.clock() - start_time

                        # periodically perform a validation of the the 
                        # model and report stats
                        if (iteration)%validation_frequency == 0:

                                validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
                                this_val_loss = np.mean(validation_losses)

				# save the model if we achieved a better validation loss
                                if this_val_loss < self.best_loss:
                                        self.best_loss = this_val_loss
                                        best_iter = iteration
                                        self.save()

				# update elapsed time
				elapsed_time = time.clock() - start_time

				# report stats
                                self.reportTrainingStats(elapsed_time,
                                        minibatch_index,
                                        this_val_loss,
                                        minibatch_avg_cost.item(0))

                end_time = time.clock()
                msg = 'The code ran for'
                status = '%f seconds' % ((end_time - start_time))
                Utility.report_status( msg, status )


	#---------------------------------------------------------------------------------------------------
	# generates a probability map for the specified image
	#---------------------------------------------------------------------------------------------------
        def classify(self, image):
                probs = np.zeros((1024,1024))

		patch_size = self.patchSize
                index      = self.index
                x          = self.x
                row_range  = 1
                imSize     = np.shape( image )
                n_batches  = 1024/self.batchSize

                patch = np.zeros( (imSize[0]*row_range, patch_size**2), dtype=np.float32 )
                test_set_x = theano.shared( patch, borrow=True)

                classify_data = theano.function(
                        [],
                        outputs=self.p_y_given_x,
                        givens={x: test_set_x})

                start_time = time.clock()
                for row in xrange(0,1024,row_range):
                        patch = Utility.get_patch(image, row, row_range, patch_size )
                        test_set_x.set_value( np.float32(patch) )
			result = classify_data()
			probs[row,:] = result[:,1]

                        if row%16 == 0:
                                sys.stdout.write('.')
                                sys.stdout.flush()

                end_time = time.clock()
                duration = (end_time - start_time)
                print 'Segmentation finished. Duration:', duration
                return np.array(probs)

	#---------------------------------------------------------------------------------------------------
	# generates a prediction for the specified image
	#---------------------------------------------------------------------------------------------------
        def predict(self, path):
		image = Utility.get_image_padded(path, self.patchSize )

                prediction = np.zeros( 1024 * 1024, dtype=int )

                row_range = 1
                patch_size = self.patchSize
                batch_size = self.batchSize
                index      = self.index
                x          = self.x
                patch      = np.zeros( (1024*row_range, patch_size**2), dtype=np.float32 )
                test_set_x = theano.shared( patch, borrow=True)

		# prediction function
                predict_data = theano.function(
                        [],
                        outputs=self.y_pred,
                        givens={x: test_set_x}
		)

                start_time = time.clock()
                for row in xrange(0,1024,row_range):
                        patch = Utility.get_patch(image, row, row_range, patch_size)
                        test_set_x.set_value( np.float32(patch) )
			result = predict_data()
			if row%16 == 0:
				#sys.stdout.write('.')
				#sys.stdout.flush()
				print np.bincount( result )
			prediction[row*1024:row*1024+row_range*1024] = result

                end_time = time.clock()
                duration = (end_time - start_time)
		print ''
		print 'duration:', duration
		print 'results :', np.bincount( prediction )
                return prediction

