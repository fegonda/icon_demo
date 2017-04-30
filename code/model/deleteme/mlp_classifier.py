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
# Summary : This file contains the implementation of an MLP classifier.
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
import zlib
import StringIO
import base64
import math

theano.config.floatX = 'float32'
base_path = os.path.dirname(__file__)
sys.path.insert(1,os.path.join(base_path, '../external'))
sys.path.insert(2,os.path.join(base_path, '../common'))

from logistic_sgd import LogisticRegression
from mlp import HiddenLayer
from mlp_model import MLP_Model
from utility import Utility
from database import Database


#---------------------------------------------------------------------------
# Multi-Layer Perceptron classifier.  This class wraps an MLP model and
# perform the work of sampling training data from a randomize set and 
# serializing the best model from disk.
#---------------------------------------------------------------------------
class MLP_Classifier (object):
	
	#-------------------------------------------------------------------
	# Constructor - creates a new classifier and initializes its state
	#-------------------------------------------------------------------
	def __init__(self):
		self.model_path = None
		self.project    = 'default'
		self.best_val_interval = 4
		self.reset();


        #-------------------------------------------------------------------
        # reset - clears out the model transient state
        #-------------------------------------------------------------------
	def reset(self):
		self.new_model = False
	    	self.model = None
	    	self.best_val_loss = np.inf
                self.n_train    = 5000
                self.i_train    = []
                self.i_valid    = []
                self.i_test     = []
                self.i          = []
                self.iteration  = 0

        #-------------------------------------------------------------------
        # getPath - returns the model's serializatin path
        #-------------------------------------------------------------------
	def getPath(self):
            path = '%s/best_mlp_model.%s.pkl'%(self.model_path, self.project)
	    return path


        #-------------------------------------------------------------------
        # Main constructor of MembraneModel
        # Initialize vari
        #-------------------------------------------------------------------
	def save(self):
	    path = self.getPath()
	    Database.beginSaveModel( self.project )	
	    Utility.report_status('saving best model', path)
	    self.model.save( path )
	    Database.finishSaveModel( self.project )


        #-------------------------------------------------------------------
        # This function samples a large batch from the training data set.
        #-------------------------------------------------------------------
	def sampleTrainingData(self, 
            x,
	    y,
            p,
	    new_data,
	    train_split=0.8,
	    valid_split=0.1,
	    test_split=0.1,
	    seed=123):

            n = len(y)
            train_len = int(math.floor(n * train_split))
            valid_len = int(math.floor(n * valid_split))
            test_len  = int(math.floor(n * test_split))


	    self.iteration += 1

            if new_data:
                self.iteration = 0

	    if new_data:
		self.iteration = 0
	    	self.n_train = min(train_len, self.n_train)
		indices      = np.arange( n )
		self.i       = indices
		self.i_valid = indices[train_len:train_len + valid_len]
		self.i_train = []
	    	#self.train_indices = indices[:train_len]
	    	#self.valid_indices = indices[train_len:train_len + valid_len]
	    	#test_indices  = indices[train_len:train_len + valid_len]

	    elif (self.iteration == self.best_val_interval):
		self.best_val_loss = np.inf
		self.iteration = 0

		#indices      = np.random.choice( n, n)
                #self.i       = indices #[:train_len]
                #self.i_valid = indices[train_len:train_len + valid_len]	
		

	    if len(self.i_train) > 0:
		# create two 2D array to sort indices by train results
		p_train  = p[ self.i_train ]
		i_sorted = np.argsort( p_train, axis = 0)
		n_train  = len(self.i_train)
		n_good   = len( np.where( p_train[ i_sorted ] == 0 )[0] )
		n_good   = max( n_good, n_train/2 )
	        i_good   = i_sorted[ : n_good ]
		i_bad    = i_sorted[ n_good: ]

		n_min_shuffle = int(n_train * 0.10)
		if len(i_bad) < n_min_shuffle:
                	indices      = np.random.choice( n, n)
                	self.i       = indices #[:train_len]
                	self.i_valid = indices[train_len:train_len + valid_len] 
 
		if n_good > 0:
			print self.i_train
			print p[ self.i_train ]
			print '-----------------------'
			print 'good:',n_good
			print ' bad:',len(i_bad)

			# place the good sample back to the global indices
			self.i = np.hstack( (self.i, i_good) )
		
			# select replacement for the good samples	
			i_new = self.i[ : n_good ]
			self.i = self.i[ n_good : ]
			self.i_train = np.hstack( (i_new, i_bad) )
	    else:
		self.i_train = self.i[ 0 : self.n_train ]
		self.i       = self.i[ self.n_train: ]

	    train_indices = self.i_train
	   
	    train_x = x[train_indices]
	    train_y = y[train_indices]
	    valid_x = x[self.i_valid]
	    valid_y = y[self.i_valid]
	    test_x  = x[self.i_valid]
	    test_y  = y[self.i_valid]

            Utility.report_shape( 'training set x (%0.1f%%)'%(train_split*100), train_x.shape)
            Utility.report_shape( 'training set y (%0.1f%%)'%(train_split*100), train_y.shape)
            Utility.report_shape( 'validation set x (%0.1f%%)'%(valid_split*100), valid_x.shape)
            Utility.report_shape( 'validation set y (%0.1f%%)'%(valid_split*100), valid_y.shape)
            Utility.report_shape( 'test set x (%0.2f%%)'%(test_split*100), test_x.shape)
            Utility.report_shape( 'test set y (%0.2f%%)'%(test_split*100), test_y.shape)

	    if self.model == None:
                self.train_set_x = theano.shared( train_x, borrow=True)
                self.valid_set_x = theano.shared( valid_x, borrow=True)
                self.test_set_x  = theano.shared( test_x, borrow=True)

                self.train_set_y = theano.shared( train_y, borrow=True)
                self.valid_set_y = theano.shared( valid_y, borrow=True)
                self.test_set_y  = theano.shared( test_y, borrow=True)

                self.train_set_y = T.cast( self.train_set_y, 'int32')
                self.valid_set_y = T.cast( self.valid_set_y, 'int32')
                self.test_set_y  = T.cast( self.test_set_y, 'int32')
	    else:
            	self.train_set_x.set_value( np.float32( train_x ))
            	self.valid_set_x.set_value( np.float32( valid_x ))
            	self.test_set_x.set_value( np.float32( test_x ))

            	self.train_set_y.owner.inputs[0].set_value( np.int32( train_y ))
            	self.valid_set_y.owner.inputs[0].set_value( np.int32( valid_y ))
            	self.test_set_y.owner.inputs[0].set_value( np.int32( test_y ))


        #-------------------------------------------------------------------
        # Initialize the DNN model
        #-------------------------------------------------------------------
	def setup(self, xdata, ydata, pdata, new_data):

		if self.model != None:
			return

	    	#if new_data:
		#		self.iteration = 0

	    	#self.sampleTrainingData(xdata, ydata, pdata, new_data)

	    	#if self.model == None:
		print 'creating classifier...'
		self.t_learningrate = T.scalar('learning_rate')
		self.t_momentum =  T.scalar('momentum')

    		self.shared_learningrate = theano.shared(np.float32(self.learning_rate))
    		self.shared_momentum = theano.shared(np.float32(self.momentum))

		self.x = T.fmatrix()
		self.y = T.ivector()
		#self.index = T.iscalar()

                self.rng = np.random.RandomState(1234)

                path = '%s/best_mlp_model.%s.pkl'%(self.model_path, self.project)
                msg = 'loading best model'
                if not os.path.exists(path):
			path = None

		self.new_model = (path == None)

                self.model = MLP_Model(
                        rng=self.rng,
                        input=self.x,
                        n_in=self.n_features,
                        n_hidden=self.n_hidden,
                        n_out=self.n_classes,
                        filename=path
                    )

        #-------------------------------------------------------------------
        # Trains the model.
        #-------------------------------------------------------------------
        def train(self,
                  xdata,
                  ydata,
		  pdata,
		  new_data,
		  n_features,
		  n_classes,
		  n_hidden,
		  learning_rate,
		  momentum,
		  batch_size,
		  epochs):

		self.learning_rate = learning_rate
		self.n_classes = n_classes
		self.n_features = n_features
		self.momentum = momentum
		self.batch_size = batch_size
		self.epochs = epochs
		self.n_hidden = n_hidden

		Utility.report_status('starting training', '')

		self.sampleTrainingData(xdata, ydata, pdata, new_data)
		self.setup(xdata, ydata, pdata, new_data)

		index = T.iscalar()

		cost = self.model.negative_log_likelihood(self.y)

		valid_batch_size = min(self.batch_size, self.valid_set_x.shape[0].eval() );
		n_validation_batches = self.valid_set_x.shape[0].eval() / valid_batch_size

		train_batch_size = min(self.batch_size, self.train_set_x.shape[0].eval())
		n_train_batches = self.train_set_x.shape[0].eval() / train_batch_size

		test_batch_size = min(self.batch_size, self.test_set_x.shape[0].eval())
		n_test_batches = self.test_set_x.shape[0].eval() / test_batch_size

		validate_model = theano.function(
		    inputs=[index],
		    outputs=self.model.errors(self.y),
		    givens={
		            self.x: self.valid_set_x[index * valid_batch_size: \
                                    (index + 1) * valid_batch_size],
		            self.y: self.valid_set_y[index * valid_batch_size: \
                                    (index + 1) * valid_batch_size]
		        }
		    )

		test_model = theano.function(
		    inputs=[index],
		    outputs=-T.log(self.model.logRegressionLayer.p_y_given_x)[T.arange(self.y.shape[0]), self.y],
		    givens={
		            self.x: self.test_set_x[index * test_batch_size: \
                                    (index + 1) * test_batch_size],
		            self.y: self.test_set_y[index * test_batch_size: \
                                    (index + 1) * test_batch_size]}, on_unused_input='ignore')


                test_samples = theano.function(
                    inputs=[index],
		    outputs=T.neq(self.model.logRegressionLayer.y_pred, self.y),
                    givens={
                            self.x: self.train_set_x[index * train_batch_size: \
                               (index + 1) * train_batch_size],
                            self.y: self.train_set_y[index * train_batch_size: \
                               (index + 1) * train_batch_size]})


		gparams = [T.grad(cost, param) for param in self.model.params]

		updates = [(param, param - self.t_learningrate * gparam)
		        for param, gparam in zip(self.model.params, gparams)]

                train_model = theano.function(
                        inputs=[index],
                        outputs=cost,
                        updates=updates,
                        givens={
                            self.x: self.train_set_x[index * train_batch_size: \
                               (index + 1) * train_batch_size],
                            self.y: self.train_set_y[index * train_batch_size: \
                               (index + 1) * train_batch_size],
                            self.t_learningrate: self.shared_learningrate})


		n_epochs = self.epochs

		# early-stopping parameters
		patience = 1000  # look as this many examples regardless
		patience_increase = 2  # wait this much longer when a new best is found
		validation_frequency = int( min(n_train_batches*0.10, patience) )
		if validation_frequency == 0:
			validation_frequency = 5;

		iteration = 0

		max_train_time = 15
	        elapsed_time = 0.0;
		start_time = time.clock()

		Utility.report_status('', '')
		Utility.report_status('batch size', '%d'%(self.batch_size))
		Utility.report_status('# of epochs', '%d'%(self.epochs))
		Utility.report_status('# of training batches', '%d'%n_train_batches);
		Utility.report_status('# of validation batches', '%d'%(n_validation_batches))
		Utility.report_status('# of test batches', '%d'%(n_test_batches))
		Utility.report_status('training interval (secs)', '%d'%(max_train_time))
		Utility.report_status('validation frequency', '%d'%(validation_frequency))
		Utility.report_status('', '')

                msg = 'TIME BATCH VALIDATION ERROR'
                status = 'COST'
                Utility.report_status(msg, status)

		train_size = min( self.n_train, self.batch_size*5 )
		best_iter = 0
		epoch = 0

	        minibatch_index = 0

		#---------------------------------------------------
		# mini-batch training: train until the time expires
		# before sampling the next batch.
		#---------------------------------------------------
	        while(elapsed_time < max_train_time): 

		    if (minibatch_index == n_train_batches):
		        minibatch_index = 0
		        break
	
		    minibatch_avg_cost = train_model(minibatch_index)
		    iteration = minibatch_index
		    i = minibatch_index
		    minibatch_index += 1

		    # test the trained samples against the target
		    # values to measure the training performance
		    probs = test_samples(minibatch_index)
		    i_batch = self.i_train[ i * train_batch_size:(i+1)*train_batch_size ]
		    pdata[ i_batch ] = probs
		    good = np.where( probs == 0)[0]
		    bad  = np.where( probs == 1)[0]


		    # periodically perform a validation of the the 
		    # model and report stats
		    if (iteration)%validation_frequency == 0:
		    	   validation_losses = [validate_model(i) for i in xrange(n_validation_batches)]
		    	   this_val_loss = np.mean(validation_losses)
			   self.reportTrainingStats(elapsed_time, minibatch_index, this_val_loss, minibatch_avg_cost.item(0))
			   '''

			   status = '[%f]'%(minibatch_avg_cost)
		    	   msg = '(%0.1f)     %i/%i     %f%%' % \
		           (
		               elapsed_time,
                               minibatch_index,
		               n_train_batches,
		               this_val_loss * 100.
		           )
		    	   Utility.report_status( msg, status )
			   '''

			   if this_val_loss < self.best_val_loss:
				self.best_val_loss = this_val_loss
				best_iter = iteration
				self.save()

		    # update elapsed time
		    elapsed_time = time.clock() - start_time

		end_time = time.clock()
		msg = 'The code ran for'
		status = '%f seconds' % ((end_time - start_time))
		Utility.report_status( msg, status )
 		return True

        #-------------------------------------------------------------------
        # Performs segmentation of the specified image.
        # Load the image data into memory and then perform segmentation
        # line by line
        #-------------------------------------------------------------------
	def reportTrainingStats(self, elapsedTime, batchIndex, valLoss, avgCost):
		Database.storeBatch( self.project, batchIndex, valLoss, avgCost)
		msg = '(%0.1f)     %i     %f%%'%\
                (   
                   elapsedTime,
                   batchIndex,
                   valLoss * 100.
                )   
		status = '[%f]'%(avgCost)
                Utility.report_status( msg, status )


	def measure_performance(self, sample_size, n_classes, n_hidden, image_id, project, img_dir):
                n_features = sample_size**2
                pred_start_time = time.clock()

                Utility.report_status('segmentation started', image_id)
                loaded, image = Utility.get_image_padded( image_id, img_dir, sample_size )


	def classify_image(self, img, normMean, norm_std):
                if self.model is None:

                        model_load_start_time = time.clock()
                        path = self.getPath()
                        start_time = time.clock()
                        self.rng = np.random.RandomState(1234)
                        self.x = T.fmatrix()
                        self.model = MLP_Model( self.rng, self.x, n_features, n_hidden, n_classes, path )
                        elapsed_time = time.clock() - start_time
                        Utility.report_status( 'model load', '(%.2f secs)'%(elapsed_time) )
		
		probs = np.zeros(np.shape(image))

                row_range = 1
                imSize = np.shape( image )
		patchSize = np.int(np.sqrt(self.model.hiddenLayers[0].W.eval().shape[0]))
		
                patch = np.zeros( (imSize[0]*row_range, patchSize**2), dtype=np.float32 )
                self.test_set_x = theano.shared( patch, borrow=True)

                classify = theano.function(
			[],
                	outputs=self.model.logRegressionLayer.p_y_given_x,
                	givens={self.x: self.test_set_x})

                start_time = time.clock()
                for row in xrange(0,1024,row_range):
                        patch = Utility.get_patch(image, row, row_range, sample_size )
                        self.test_set_x.set_value( patch )
                        result = classify()
			probs[row,:] = result[:,1]

	 	return np.array(probs)	


        #-------------------------------------------------------------------
        # Performs segmentation of the specified image.
        # Load the image data into memory and then perform segmentation
        # line by line
        #-------------------------------------------------------------------
	def predict(self, sample_size, n_classes, n_hidden, image_id, project, img_dir):

		n_features = sample_size**2
		pred_start_time = time.clock()

		Utility.report_status('segmentation started', image_id)
		path = '%s/%s.tif'%(img_dir, image_id)
		loaded, image = Utility.get_image_padded( path, sample_size )
		if not loaded:
			# report error to database that this image could not be loaded
			return False

		if self.model is None:
		
			model_load_start_time = time.clock()
	
                        path = self.getPath()
                        start_time = time.clock()
                        if not os.path.exists(path):
                                Utility.report_status( msg, 'fail' )
                                return False

                        self.rng = np.random.RandomState(1234)
                        self.x = T.fmatrix()

                        self.model = MLP_Model( self.rng, self.x, n_features, n_hidden, n_classes, path )
                        elapsed_time = time.clock() - start_time
                        Utility.report_status( 'model load', '(%.2f secs)'%(elapsed_time) )

		predicted_values = np.zeros( 1024 * 1024, dtype=int )

		row_range = 1

		imSize = np.shape( image )
		patch = np.zeros( (imSize[0]*row_range, n_features), dtype=np.float32 )

		self.test_set_x = theano.shared( patch, borrow=True)

		classify = theano.function([], 
		outputs=self.model.logRegressionLayer.y_pred,
		givens={self.x: self.test_set_x})

		normMean=None #0.5
		norm_std=None #1.0
		patchSize = np.int(np.sqrt(self.model.hiddenLayers[0].W.eval().shape[0]))

		print 'sample_size:',sample_size
		print 'patchSize:', patchSize
		start_time = time.clock()
		for row in xrange(0,1024,row_range):
			patch = Utility.get_patch(image, row, row_range, sample_size )
			self.test_set_x.set_value( patch )
			pred = classify()
			predicted_values[row*1024:row*1024+row_range*1024] = pred

		#print np.unique( predicted_values )
		print np.bincount( predicted_values )
		elapsed_time = time.clock() - start_time
		Utility.report_status('segmentation', '(%.2f secs)'%(elapsed_time))

                start_time = time.clock()
                self.save_probabilitymap(predicted_values, image_id, project)
                elapsed_time = time.clock() - start_time
                Utility.report_status('serialization', '(%.2f secs)'%(elapsed_time))

		elapsed_time = time.clock() - pred_start_time
		Utility.report_status('segmentation finshed...', '(%.2f secs)'%(elapsed_time))


        #-------------------------------------------------------------------
        # Serializes segmentation data to file.
        #-------------------------------------------------------------------
        def save_probabilitymap(self, data, image_id, project):
                path = '%s/%s.%s.seg'%(self.segmentation_path, image_id, project)
                output = StringIO.StringIO()
                output.write(data.tolist())
                content = output.getvalue()
                encoded = base64.b64encode(content)
                compressed = zlib.compress(encoded)
                with open(path, 'w') as outfile:
                        outfile.write(compressed)
