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
import mahotas

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from scipy.ndimage.interpolation import shift

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

		self.activation   = activation
		self.nkernels     = nkernels
		self.kernel_sizes = kernel_sizes
		self.hidden_sizes = hidden_sizes


		print '-----'
                print 'patch_size:', self.patch_size
                print 'nkernels:', self.nkernels
                print 'kernelsizes:', self.kernel_sizes
                print 'hiddensizes:', self.hidden_sizes

		print '-----'
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
			print 'loading cnn from file...'
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

		test_set_x  = self.test_x
		test_set_y  = self.test_y
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

    		patience = 64    	# look as this many examples regardless
    		patience_increase = 2   # wait this much longer when a new best is
                           		# found
    		improvement_threshold = 0.995   # a relative improvement of this much is
                                   		# considered significant
    		validation_frequency = 1 #min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

		''''
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
		'''
		print 'learning rate:', self.learning_rate
		print 'momentum:', self.momentum
		print 'batchsize:', self.batch_size
		print 'n_train_batches:', n_train_batches
		print 'train_time:', self.train_time

		#self.train_time = 30
		n_rotate = 0
		while(elapsed_time < self.train_time) and (minibatch_index < n_train_batches):

			if (minibatch_index == (n_train_batches-1)):
				minibatch_index = 0
				'''
				self.rotateSamples()
                                n_rotate += 1
                                if n_rotate > 1:
                                        break
				'''
				break

			minibatch_avg_cost = train_model(minibatch_index)
                    	iteration = minibatch_index
                    	i = minibatch_index
                    	minibatch_index += 1

                    	# test the trained samples against the target
                    	# values to measure the training performance
                   	probs = predict_samples(minibatch_index)
			print '===>probs:', np.bincount( probs )
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
					print 'best model....'
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


	def predictold(self, path):
		print 'predicting...'
		image = Utility.get_image_padded(path, self.patch_size)

		prediction = np.zeros( 1024 * 1024, dtype=int )

                row_range = 1
		nsamples  = row_range * 1024
		nfeatures = self.patch_size**2
		batch_size = self.batch_size

		#1024/512
                imSize = np.shape( image )
		print 'imsize:', imSize

                #patch = np.zeros( (1024*row_range, self.patch_size**2), dtype=np.float32 )
                #test_set_x = theano.shared( patch, borrow=True)
		patch = np.zeros( (nsamples, nfeatures), dtype=np.float32 )
		test_set_x = theano.shared( patch, borrow=True)

		print 'rowr:', row_range
		print 'nsamples:', nsamples
		print 'nfeatures:', nfeatures
		print 'tsetx:',test_set_x.get_value(borrow=True).shape

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

		print 'nbatches:', n_batches
		print 'batchsize:', batch_size
		

		start_time = time.clock()
                for row in xrange(0,1024,row_range):
                        patch = Utility.get_patch(image, row, row_range, self.patch_size)

			#print 'patch size:', np.shape( patch )
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

	def predict(self, path, threshold=0.5):
		print 'CNN.predict'
		image = mahotas.imread( path )
		prob = CNN.fast_classify( image, self )
		prob[ prob >= threshold ] = 9
		prob[ prob <  threshold ] = 1
	        prob[ prob == 9 ] = 0
		#prob[ prob < 0.5] = 0
		#prob[ prob >= 0.5] = 1
		prob = prob.astype(dtype=int)
		prob = prob.flatten()
		print 'results :', np.bincount( prob )
		return prob

	@staticmethod
	def fast_classify(image, classifier):

		print 'CNN.fast_classify'

		imageSize = 1024
		image = image[0:imageSize,0:imageSize]

		start_time = time.clock()
		#image = Utility.normalizeImage(image)

		#print 'imsize after norm:', np.shape( image )

		#GPU
		image_shared = theano.shared(np.float32(image), borrow=True)
		image_shared = image_shared.reshape((1,1,imageSize,imageSize))

		fragments = [image_shared]

		#print 'imgshared size:', image_shared.eval().shape

		print "Convolutions"

		for clayer in classifier.convLayers:
			newFragments = []
			for img_sh in fragments:
			    convolved_image = CNN.get_convolution_output(image_shared=img_sh, clayer=clayer)
			    output = CNN.get_max_pool_fragments(convolved_image, clayer=clayer)
			    newFragments.extend(output)

			fragments = newFragments


		#### now the hidden layer

		print "hidden layer"

		hidden_fragments = []

		for fragment in fragments:
			hidden_out = CNN.get_hidden_output(
					image_shared=fragment, 
					hiddenLayer=classifier.mlp.hiddenLayers[0], 
					nHidden=200, 
					nfilt=classifier.nkernels[-1])
			hidden_fragments.append(hidden_out)

		### VERIFIED CORRECT UNTIL HERE

		#### and the missing log reg layer

		print "logistic regression layer"

		final_fragments = []
		for fragment in hidden_fragments:
			#print 'logreg - fragment:', fragment.shape
			logreg_out = CNN.get_logistic_regression_output(
					image_shared=fragment, 
					logregLayer=classifier.mlp.logRegressionLayer)
			logreg_out = logreg_out[0,:,:]
			logreg_out = logreg_out.eval()
			final_fragments.append(logreg_out)


		total_time = time.clock() - start_time
		print "This took %f seconds." % (total_time)

		print "assembling final image"

		prob_img = np.zeros(image.shape)

		offsets_tmp = np.array([[0,0],[0,1],[1,0],[1,1]])

		if len(classifier.convLayers)>=1:
			offsets = offsets_tmp

		if len(classifier.convLayers)>=2:
			offset_init_1 = np.array([[0,0],[0,1],[1,0],[1,1]])
			offset_init_2 = offset_init_1 * 2

			offsets = np.zeros((4,4,2))
			for o_1 in range(4):
			    for o_2 in range(4):
				offsets[o_1,o_2] = offset_init_1[o_1] + offset_init_2[o_2]
				
			offsets = offsets.reshape((16,2))

		if len(classifier.convLayers)>=3:
			offset_init_1 = offsets.copy()
			offset_init_2 =  np.array([[0,0],[0,1],[1,0],[1,1]]) * 4

			offsets = np.zeros((16,4,2))
			for o_1 in range(16):
			    for o_2 in range(4):
				offsets[o_1,o_2] = offset_init_1[o_1] + offset_init_2[o_2]

			offsets = offsets.reshape((64,2))

		# offsets = [(0,0),(0,2),(2,0),(2,2),
		#            (0,1),(0,3),(2,1),(2,3),
		#            (1,0),(1,2),(3,0),(3,2),
		#            (1,1),(1,3),(3,1),(3,3)]

		# offsets_1 = [(0,0),(0,4),(4,0),(4,4),
		#              (0,2),(0,6),(4,2),(4,6)]

		offset_jumps = np.int16(np.sqrt(len(offsets)))
		for f, o in zip(final_fragments, offsets):
			prob_size = prob_img[o[0]::offset_jumps,o[1]::offset_jumps].shape
			f_s = np.zeros(prob_size)
			f_s[:f.shape[0], :f.shape[1]] = f.copy()
			prob_img[o[0]::offset_jumps,o[1]::offset_jumps] = f_s

		total_time = time.clock() - start_time
		print "This took %f seconds." % (total_time)

		shift_amount = np.floor( classifier.patch_size/2 )
		prob_img = shift(prob_img,(shift_amount, shift_amount))

		return prob_img


	@staticmethod
	def get_max_pool_frag(convolved_image, offset1, offset2):
	    image_width = convolved_image.shape[2]
	    image_height = convolved_image.shape[3]
	    
	    '''
	    print "This is max pool:"
	    print "Input size"
	    print image_width
	    print image_height
	    '''
	    convolved_image_shared = convolved_image[:,:,offset1:,offset2:]
	    convolved_image_shared = convolved_image_shared.reshape(convolved_image_shared.shape.eval())

	    pooled_out = downsample.max_pool_2d(
		input=convolved_image_shared,
		ds=(2,2),
		ignore_border=True
	    )
	    #print "Output size"
	    #print pooled_out.shape.eval()
	    
	    return pooled_out


	@staticmethod
	def get_max_pool_fragments(convolved_image, clayer):
	    start_time = time.clock()
	    b = clayer.b.dimshuffle('x', 0, 'x', 'x')
	    max_pooled_0_0 = CNN.get_max_pool_frag(convolved_image=convolved_image, offset1=0, offset2=0)
	    out_0_0 = rectified_linear(max_pooled_0_0 + b)

	    max_pooled_0_1 = CNN.get_max_pool_frag(convolved_image=convolved_image, offset1=0, offset2=1)
	    out_0_1 = rectified_linear(max_pooled_0_1 + b)

	    max_pooled_1_0 = CNN.get_max_pool_frag(convolved_image=convolved_image, offset1=1, offset2=0)
	    out_1_0 = rectified_linear(max_pooled_1_0 + b)

	    max_pooled_1_1 = CNN.get_max_pool_frag(convolved_image=convolved_image, offset1=1, offset2=1)
	    out_1_1 = rectified_linear(max_pooled_1_1 + b)

	    return (out_0_0, out_0_1, out_1_0, out_1_1)

	@staticmethod
	def basic_convolution(image_shared, filterMap):
	    conv_out = conv.conv2d(
		input=image_shared,
		filters=filterMap,
	    )
	    return conv_out

	@staticmethod
	def get_convolution_output(image_shared, clayer):
	    #print 'get_convolution_output.image_shared:', image_shared.eval().shape
	    output = CNN.basic_convolution(image_shared, clayer.W)
	    #print 'get_convolution_output.output:', output.eval().shape
	    output = theano.shared(np.float32(output.eval()), borrow=True)
	    return output

	@staticmethod
	def get_hidden_output(image_shared, hiddenLayer, nHidden, nfilt):
	    W = hiddenLayer.W
	    patchSize = np.int16(np.sqrt(W.shape.eval()[0] / np.double(nfilt)))
	    W = np.rollaxis(W,1)
	    W = W.reshape((nHidden,nfilt,patchSize,patchSize))

	    b = hiddenLayer.b

	    #flip kernel for convolution
	    output = rectified_linear(
			CNN.basic_convolution(
				image_shared, 
				W[:,:,::-1,::-1]) + b.dimshuffle('x', 0, 'x', 'x'))

	    return output

	@staticmethod
	def get_logistic_regression_output(image_shared, logregLayer):
	    output_shape = image_shared.shape

	    W_lreg = logregLayer.W
	    W_shape = W_lreg.shape
	    W_lreg = np.rollaxis(W_lreg, 1)
	    W_lreg = W_lreg.reshape((W_shape[1],W_shape[0],1,1))

	    b_lreg = logregLayer.b

	    # flip kernel for convolution
	    output = CNN.basic_convolution(
			image_shared, 
			W_lreg[:,:,::-1,::-1]) + b_lreg.dimshuffle('x', 0, 'x', 'x')

	    output =  T.addbroadcast(output, 0)
	    output = output.squeeze()
	    output = output.flatten(2)
	    output = T.nnet.softmax(output.T).T
	    #print 'get_logis:', output.shape

	    return output.reshape((2,output_shape[2], output_shape[3]))

