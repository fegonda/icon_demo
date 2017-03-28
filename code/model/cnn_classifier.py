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
import numpy
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
sys.path

from logistic_sgd import LogisticRegression
from mlp import HiddenLayer
from mlp_model import MLP_Model
from cnn_model import CNN_Model
from utility import Utility
from database import Database

class CNN_Classifier (object):

	#-------------------------------------------------------------------
	# Main constructor of MembraneModel
	# Initialize vari
	#-------------------------------------------------------------------
	def __init__(self):
		self.model_path = None
		self.project = 'default'
		self.reset()
		self.n_train = 2000
		self.i_train = []
		self.i_valid = []
		self.i_test  = []
		self.i       = []
		self.iteration = 0
		self.reset_interval = 4


	def reset(self):
		self.new_model = False
	    	self.classifier = None
	    	self.best_validation_loss = np.inf

	def getModelPath(self):
            path = '%s/best_mlp_model.%s.pkl'%(self.model_path, self.project)
	    return path


        #-------------------------------------------------------------------
        # Main constructor of MembraneModel
        # Initialize vari
        #-------------------------------------------------------------------
	def save(self):
	    path = self.getModelPath()

	    Database.beginSaveModel( self.project )	
	    Utility.report_status('saving best model', path)
	    self.classifier.save( path )
	    Database.finishSaveModel( self.project )


	def shuffle(self, 
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
	    	self.n_train = min(train_len, self.n_train)
	    	'''
	    	train_indices =  np.random.choice( n, train_batch_size ) 
	    	valid_indices = np.random.choice( n, valid_len )
	    	test_indices = np.random.choice( n, test_len )
 	    	'''
	    	#indices      = np.random.choice( n, n)
		indices      = np.arange( n )
		self.i       = indices #[:train_len]
		self.i_valid = indices[train_len:train_len + valid_len]
		self.i_train = []
	    	#self.train_indices = indices[:train_len]
	    	#self.valid_indices = indices[train_len:train_len + valid_len]
	    	#test_indices  = indices[train_len:train_len + valid_len]

	    elif (self.iteration%self.reset_interval) == 0:
		self.best_validation_loss = np.inf
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

	    if self.classifier == None:
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

	def create_model(self,
		learning_rate=0.0001,
                nkerns=[48,48,48],
                kernelSizes=[5,5,5],
                hiddenSizes=[200],
                batch_size=10,
                patchSize=65,
                momentum=0.98):

		if self.classifier is not None:
			return

		print 'creating classifier...'
		self.t_learningrate = T.scalar('learning_rate')
		self.t_momentum =  T.scalar('momentum')

    		self.shared_learningrate = theano.shared(np.float32(learning_rate))
    		self.shared_momentum = theano.shared(np.float32(momentum))

		self.x = T.fmatrix('x')
		self.y = T.ivector('y')
		#self.index = T.iscalar()

                self.rng = np.random.RandomState(1234)

                path = '%s/best_cnn_model.%s.pkl'%(self.model_path, self.project)
                msg = 'loading best model'
                if not os.path.exists(path):
			path = None

		self.new_model = (path == None)

		print 'creatingn model...'
		self.classifier = CNN_Model(input=self.x,
                                  batch_size=batch_size,
                                  patchSize=patchSize,
                                  rng=self.rng,
                                  nkerns=nkerns,
                                  kernelSizes=kernelSizes,
                                  hiddenSizes=hiddenSizes,
                                  fileName=path)

		print 'done creating classifier...'


	def train(self,
		xdata,
                ydata,
                pdata,
                new_data,
		n_classes,
		learning_rate=0.0001,
		nkerns=[48,48,48],
		kernelSizes=[5,5,5],
		hiddenSizes=[200],
		batch_size=1,
		patchSize=65,
		momentum=0.98):

		self.batch_size = batch_size

		if new_data:
			self.best_validation_loss = np.inf

		self.shuffle(xdata, ydata, pdata, new_data)	
	
		self.create_model(
			learning_rate,
			nkerns,
			kernelSizes,
			hiddenSizes,
			batch_size,
			patchSize,
			momentum)

	    	def gradient_updates_momentum(cost, params, learning_rate, momentum):
        		updates = []
        		for param in params:
            			param_update = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
            			updates.append((param, param - learning_rate*param_update))
           		 	updates.append((param_update, momentum*param_update + (1. - momentum)*T.grad(cost, param)))
        		return updates

		cost = self.classifier.cost(self.y)

		index = T.lscalar()

		#valid_batch_size = min(self.batch_size, self.valid_set_x.shape[0].eval() );
		n_validation_batches = self.valid_set_x.shape[0].eval() / batch_size

		#train_batch_size = min(self.batch_size, self.train_set_x.shape[0].eval())
		n_train_batches = self.train_set_x.shape[0].eval() / batch_size

		#test_batch_size = min(self.batch_size, self.test_set_x.shape[0].eval())
		n_test_batches = self.test_set_x.shape[0].eval() / batch_size
		

		test_samples = theano.function(
                    inputs=[index],
                    outputs=T.neq(self.classifier.y_pred, self.y),
                    givens={
                            self.x: self.train_set_x[index * batch_size: (index + 1) * batch_size],
                            self.y: self.train_set_y[index * batch_size: (index + 1) * batch_size]})


		validate_model = theano.function(
                    inputs=[index],
                    outputs=self.classifier.errors(self.y),
                    givens={
                            self.x: self.valid_set_x[index * batch_size: (index + 1) * batch_size],
                            self.y: self.valid_set_y[index * batch_size: (index + 1) * batch_size]})


		gparams = []
    		for param in self.classifier.params:
        		gparam = T.grad(cost, param)
        		gparams.append(gparam)


		updates = gradient_updates_momentum(
			cost, 
			self.classifier.params, 
			self.t_learningrate,
			self.t_momentum)

		train_model = theano.function(
			inputs=[index], 
			outputs=cost,
            		updates=updates,
            		givens={
			self.x: self.train_set_x,
			self.y: self.train_set_y,
			self.t_learningrate: self.shared_learningrate,
			self.t_momentum: self.shared_momentum},
			on_unused_input='ignore')


		Utility.report_status('starting training', '')

		# early-stopping parameters
		patience = 1000  # look as this many examples regardless
		patience_increase = 2  # wait this much longer when a new best is found
		validation_frequency = int( min(n_train_batches*0.10, patience) )
		iteration = 0

		max_train_time = 15
	        elapsed_time = 0.0;
		start_time = time.clock()

		Utility.report_status('', '')
		Utility.report_status('batch size', '%d'%(self.batch_size))
		Utility.report_status('training interval (secs)', '%d'%(max_train_time))
		Utility.report_status('validation frequency', '%d'%(validation_frequency))
		Utility.report_status('', '')

                msg = 'EPOCH (TIME) BATCH VALIDATION ERROR'
                status = 'COST'
                Utility.report_status(msg, status)

		train_size = min( self.n_train, self.batch_size*5 )
		best_iter = 0
		epoch = 0

	        minibatch_index = 0

	        while(elapsed_time < max_train_time): 

		    if (minibatch_index == n_train_batches):
			#x, y = self.shuffle(xdata, ydata, False)
		        minibatch_index = 0
		        break
	
		    minibatch_avg_cost = train_model(minibatch_index)
		    iteration = minibatch_index
		    i = minibatch_index
		    minibatch_index += 1

		    probs = test_samples(minibatch_index)

		    #print 'i:', i
		    #print '---'
		    i_batch = self.i_train[ i * train_batch_size:(i+1)*train_batch_size ]
		    pdata[ i_batch ] = probs
		    #print i_batch
		    #print '---'
		    #print self.i_train[ i_batch ]
		    #print '---'
		    #print pdata[ i_batch ]
		    #print '---'
		    #print probs
		    #print '---'
		    good = np.where( probs == 0)[0]
		    bad  = np.where( probs == 1)[0]
		    #print 'good:',np.where( probs == 0)[0]
		    #print ' bad:',np.where( probs == 1)[0]
		    #print '----...----'

		    if (iteration)%validation_frequency == 0:
		    	   validation_losses = [validate_model(i) for i in xrange(n_validation_batches)]
		    	   this_validation_loss = np.mean(validation_losses)

			   status = '[%f]'%(minibatch_avg_cost)
		    	   msg = '%i (%0.1f)     %i/%i     %f%%' % \
		           (
		               epoch, (elapsed_time),
                               minibatch_index,
		               n_train_batches,
		               this_validation_loss * 100.
		           )
		    	   Utility.report_status( msg, status )

			   if this_validation_loss < self.best_validation_loss:
				self.best_validation_loss = this_validation_loss
				best_iter = iteration
				self.save()

		    # update elapsed time
		    elapsed_time = time.clock() - start_time

		end_time = time.clock()
		msg = 'The code ran for'
		status = '%f seconds' % ((end_time - start_time))
		Utility.report_status( msg, status )
		#msg = 'Test error is'
		#status = '%f %%' % (test_model() * 100)
		#Utility.report_status( msg, status )
		#Utility.report_status('training', 'done')

		exit(1)
 		return True


        #-------------------------------------------------------------------
        # Main constructor of MembraneModel
        # Initialize vari
        #-------------------------------------------------------------------
	def predict(self, sample_size, n_classes, n_hidden, image_id, project, img_dir):

		n_features = sample_size**2
		pred_start_time = time.clock()

		Utility.report_status('segmentation started', image_id)
		loaded, image = Utility.get_image_padded( image_id, img_dir, sample_size )
		if not loaded:
			# report error to database that this image could not be loaded
			return False

		if self.classifier is None:
		
			model_load_start_time = time.clock()
	
                        path = self.getModelPath()
                        start_time = time.clock()
                        if not os.path.exists(path):
                                Utility.report_status( msg, 'fail' )
                                return False

                        self.rng = np.random.RandomState(1234)
                        self.x = T.fmatrix()

                        self.classifier = MLP_Model( self.rng, self.x, n_features, n_hidden, n_classes, path )
                        elapsed_time = time.clock() - start_time
                        Utility.report_status( 'model load', '(%.2f secs)'%(elapsed_time) )

		predicted_values = np.zeros( 1024 * 1024, dtype=int )

		row_range = 1

		imSize = np.shape( image )
		patch = np.zeros( (imSize[0]*row_range, n_features), dtype=np.float32 )

		self.test_set_x = theano.shared( patch, borrow=True)

		classify = theano.function([], 
		outputs=self.classifier.logRegressionLayer.y_pred,
		givens={self.x: self.test_set_x})

		normMean=None #0.5
		norm_std=None #1.0
		patchSize = np.int(np.sqrt(self.classifier.hiddenLayers[0].W.eval().shape[0]))

		start_time = time.clock()
		for row in xrange(0,1024,row_range):
	            	#data = Utility.get_patch(image, rowOffset=row, rowRange=row_range, 
			#patchSize=patchSize, imSize=imSize, data_mean=normMean, data_std=norm_std)
			patch = Utility.get_patch(image, row, row_range, sample_size )
			self.test_set_x.set_value( patch )
			
			pred = classify()
			#pred = predict_model( data ).flatten()
			#print pred
			#print np.unique( pred )
			#print pred.shape
			predicted_values[row*1024:row*1024+row_range*1024] = pred
			#print predicted_values
			#break

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

        def predict1(self,
		    test_x,
	  	    test_y,
		    n_features,
		    n_classes,
		    n_hidden,
		    image_id,
		    project):

		
                Utility.report_status('segmenting', image_id)
		
		if self.classifier is None:

	                path = self.getModelPath()
        	        msg = 'loading best model'
			Utility.report_status( msg + '(please wait)', '')
			start_time = time.clock()
                	if not os.path.exists(path):
                        	Utility.report_status( msg, 'fail' )
                        	return False

                	self.test_set_x = theano.shared( test_x, borrow=True)

                	self.rng = np.random.RandomState(1234)
                	self.x = T.fmatrix()

                	self.classifier = MLP_Model( self.rng, self.x, n_features, n_hidden, n_classes, path )
			elapsed_time = time.clock() - start_time
			Utility.report_status( 'loading best model', '(%.2f secs)'%(elapsed_time) )

		else:
                	self.test_set_x.set_value( np.float32( test_x ))

		x = self.test_set_x.get_value()

                # compile a predictor function
                predict_model = theano.function(
                inputs=[self.classifier.input],
                outputs=self.classifier.logRegressionLayer.y_pred)

		Utility.report_status('segmenting image', '')
		start_time = time.clock()
                predicted_values = predict_model( x )
		elapsed_time = time.clock() - start_time
		Utility.report_status('segmenting image', '(%.2f secs)'%(elapsed_time))

		Utility.report_status('serializing segmentation results', '')
		start_time = time.clock()
		self.save_probabilitymap(predicted_values, image_id, project)
		elapsed_time = time.clock() - start_time
		Utility.report_status('serializing segmentation results', '(%.2f secs)'%(elapsed_time))

		#Utility.report_status('PREDICTION RESULTS', '%s(%s)'%(image_id, project))
                #print predicted_values
                #print np.unique( predicted_values )
                #print np.bincount( predicted_values )

                return True

        def save_probabilitymap(self, prob, image_id, project):
                path = '%s/%s.%s.seg'%(self.segmentation_path, image_id, project)
                output = StringIO.StringIO()
                output.write(prob.tolist())
                content = output.getvalue()
                encoded = base64.b64encode(content)
                compressed = zlib.compress(encoded)
                with open(path, 'w') as outfile:
                        outfile.write(compressed)
