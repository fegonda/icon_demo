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
import multiprocessing

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from scipy.ndimage.interpolation import shift

base_path = os.path.dirname(__file__)
sys.path.insert(1,os.path.join(base_path, '..'))
sys.path.insert(2,os.path.join(base_path, '../../external'))
sys.path.insert(3,os.path.join(base_path, '../../common'))


from activation_functions import rectified_linear
from cnn import CNN
from utility import Utility
from db import DB
from stats import TrainingStats


from generateTrainValTestData import gen_data_supervised, shared_dataset, normalizeImage, stupid_map_wrapper
from vsk_utils import shared_single_dataset

class CNN_Offline(CNN):

	def __init__(	self, 
			id,		# unique identifier of the model
			rng,            # random number generator use to initialize weights
			input,          # a theano.tensor.dmatrix of shape (n_examples, n_in)
			batch_size,     # size of mini batch
			patch_size,     # size of feature map
			learning_rate,
			momentum,
			nkernels,       # number of kernels for convolutional layers
			kernel_sizes,   # kernel sizes for convolutional layers
			hidden_sizes,   # list of number of hidden units for each layer
			train_time,     # batch training time before resampling
			path,           # model's path
			activation=rectified_linear
			):

                CNN.__init__( 	self,
				id=id,
				rng=rng,
				input=input,
                                batch_size=batch_size,
                                patch_size=patch_size,
				nkernels=nkernels,
				kernel_sizes=kernel_sizes,
				hidden_sizes=hidden_sizes,
                                path=path,
				activation=activation)

		self.learning_rate = learning_rate
		self.momentum      = momentum

		print '-----'
                print 'patch_size:', self.patchSize
                print 'nkernels:', self.nkerns
                print 'kernelsizes:', self.kernelSizes
                print 'hiddensizes:', self.hiddenSizes
		print 'OFFLINE.........'
		print 'path:', self.path
		print '-----'
		self.initialize()


	def train(self):
		print 'offline training...'

                def gradient_updates_momentum(cost, params, learning_rate, momentum):
                        updates = []
                        for param in params:
                            param_update = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
                            updates.append((param, param - learning_rate*param_update))
                            updates.append((param_update, momentum*param_update + (1. - momentum)*T.grad(cost, param)))
                        return updates

                train_samples=10000
                val_samples=10000
                test_samples=1000

                learning_rate = self.learning_rate
                momentum   = self.momentum
                patch_size = self.patchSize
                batch_size = self.batchSize

		print 'batch_size:', batch_size
		print 'patch_size:', patch_size
		print 'hidden:', self.hiddenSizes
		print 'kernelsizes:', self.kernelSizes
		print 'nkernels:', self.nkerns
		

		# LOAD TRAINING DATA
                d = gen_data_supervised(
                        purpose='train',
                        nsamples=train_samples,
                        patchSize=patch_size,
                        balanceRate=0.5,
                        data_mean=0.5,
                        data_std=1.0)

                data = d[0]
                norm_mean = d[1]
                norm_std = d[2]
                grayImages = d[3]
                labelImages = d[4]
                maskImages = d[5]
                train_set_x, train_set_y = shared_dataset(data, doCastLabels=True)


		#LOAD VALIDATION DATA
                d = gen_data_supervised(
                        purpose='validate',
                        nsamples=val_samples,
                        patchSize=patch_size,
                        balanceRate=0.5,
                        data_mean=norm_mean,
                        data_std=norm_std)

                data = d[0]
                valid_set_x, valid_set_y = shared_dataset(data, doCastLabels=True)

		# LOAD TEST DATA
                d = gen_data_supervised(
                        purpose='test',
                        nsamples=test_samples,
                        patchSize=patch_size,
                        balanceRate=0.5,
                        data_mean=norm_mean,
                        data_std=norm_std)

                data = d[0]
                test_set_x, test_set_y = shared_dataset(data, doCastLabels=True)

		# compute number of minibatches for training, validation and testing
		n_train_batches = train_samples / batch_size
		n_valid_batches = val_samples / batch_size
		n_test_batches = test_samples / batch_size

                learning_rate_shared = theano.shared(np.float32(learning_rate))
                momentum_shared = theano.shared(np.float32(momentum))


		print '----------------------------'
		print 'training data....'
		print 'min: ', np.min( train_set_x.eval() )
		print 'max: ', np.max( train_set_x.eval() )

                # allocate symbolic variables for the data
                index = T.lscalar()  # index to a [mini]batch
                #x = T.matrix('x')  # the data is presented as rasterized images
                x = self.x
                y = T.ivector('y')  # the labels are presented as 1D vector of
                                # [int] labels
                lr = T.scalar('learning_rate')
                m = T.scalar('momentum')

                #cost = self.negative_log_likelihood(y)
		cost = self.cost( y )

                test_model = theano.function(
				[index],
                                self.errors(y),
                                givens={
                                        x: test_set_x[index * batch_size:(index + 1) * batch_size],
                                        y: test_set_y[index * batch_size:(index + 1) * batch_size]
				}
		)

                validate_model = theano.function(
				[index],
                                self.errors(y),
                                givens={
                                        x: valid_set_x[index * batch_size:(index + 1) * batch_size],
                                        y: valid_set_y[index * batch_size:(index + 1) * batch_size]
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
                                        lr: learning_rate_shared,
                                        m: momentum_shared})


                print '... training'
                best_validation_loss = np.inf
                best_iter = 0
                decrease_epoch = 1
                decrease_patience = 1
                test_score = 0.

                elapsed_time = 0.0;
                start_time = time.clock()

                epoch = 0
                done_looping = False

                # start pool for data
                print "Starting worker."
                pool = multiprocessing.Pool(processes=1)
                futureData = pool.apply_async(
                                stupid_map_wrapper,
                                [[gen_data_supervised,True, 'train', train_samples, patch_size, 0.5, 0.5, 1.0]])

                n_epochs = 100
                doResample = False

                validation_frequency = 100
                validation_diff = np.inf

                while (epoch < n_epochs) and not self.done:
                        minibatch_avg_costs = []
                        epoch = epoch + 1

                        if doResample and epoch>1:
                                print "Waiting for data."
                                data = futureData.get()
                                print "GOT NEW DATA"
                                train_set_x.set_value(np.float32(data[0]))
                                train_set_y.set_value(np.int32(data[1]))
                                futureData = pool.apply_async(
                                                stupid_map_wrapper,
                                                [[gen_data_supervised,True, 'train', train_samples, patch_size, 0.5, 0.5, 1.0]])

                        for minibatch_index in xrange(n_train_batches):
                                minibatch_avg_costs.append(train_model(minibatch_index))
                                iter = (epoch - 1) * n_train_batches + minibatch_index

                                if (iter + 1) % validation_frequency == 0:
                                        validation_losses = np.array([validate_model(i) for i
                                                             in xrange(n_valid_batches)])
                                        this_validation_loss = np.mean(validation_losses) #*100.0)

                                        # update elapsed time
                                        elapsed_time = time.clock() - start_time

					'''
                                        # report stats
                                        self.reportTrainingStats(elapsed_time,
                                                minibatch_index,
                                                this_validation_loss,
                                                minibatch_avg_costs[-1].item(0))

					'''
                                        msg = 'epoch %i, minibatch %i/%i, training error %.3f, validation error %.2f %%' %\
                                         (epoch, minibatch_index + 1, n_train_batches, minibatch_avg_costs[-1], this_validation_loss)
                                        print msg

                                        # if we got the best validation score until now
                                        if this_validation_loss < best_validation_loss:
                                                best_validation_loss = this_validation_loss
                                                best_iter = iter
                                                print "New best score!"
                                                self.save()

                pool.close()
                pool.join()
                print 'done:', self.done
                print "Pool closed."
                end_time = time.clock()
                print(('Optimization complete. Best validation score of %f %% '
                'obtained at iteration %i, with test performance %f %%') %
                (best_validation_loss * 100., best_iter + 1, test_score * 100.))
                print >> sys.stderr, ('The code for file ' +
                                          os.path.split(__file__)[1] +
                                          ' ran for %.2fm' % ((end_time - start_time) / 60.))



