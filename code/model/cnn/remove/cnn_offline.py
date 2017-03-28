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
sys.path.insert(1,os.path.join(base_path, '../../external'))
sys.path.insert(2,os.path.join(base_path, '../mlp'))
from logistic_sgd import LogisticRegression, load_data

from hiddenlayer import HiddenLayer
from mlpv import MLP
from activation_functions import rectified_linear

from generateTrainValTestData import gen_data_supervised, shared_dataset, normalizeImage, stupid_map_wrapper
import multiprocessing
from classifyImage import generate_patch_data_rows
from vsk_utils import shared_single_dataset

#import matplotlib
#import matplotlib.pyplot as plt
import getpass
from cnn import CNN

class CNN_Offline(CNN):
    # batchSize needs to be the number of rows in my test set,
    # batchSize - size of the minibatch
    def __init__(   self,
            id,     # unique identifier of the model
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

        CNN.__init__(   
            self,
            id=id,
            input=input,
            batch_size=batch_size,
            patch_size=patch_size,
            rng=rng,
            nkerns=nkernels,
            kernel_sizes=kernel_sizes,
            hidden_sizes=hidden_sizes,
            path=path,
            activation=activation)

        self.activation = activation
        self.learning_rate = learning_rate
        self.momentum      = momentum
    
 
    def train(self):
        print 'training....'
        train_samples=6144 #10000
        val_samples=1024 #1000
        test_samples=1024 #1000
        n_epochs=100
        patchSize = self.patchSize
        batchSize = self.batchSize
        learning_rate  = self.learning_rate
        momentum = self.momentum


        def gradient_updates_momentum(cost, params, learning_rate, momentum):
            updates = []
            for param in params:
                param_update = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
                updates.append((param, param - learning_rate*param_update))
                updates.append((param_update, momentum*param_update + (1. - momentum)*T.grad(cost, param)))
            return updates

        rng = numpy.random.RandomState(23455)

        # training data
        d = gen_data_supervised(
            purpose='train', 
            nsamples=train_samples, 
            patchSize=patchSize, 
            balanceRate=0.5, 
            data_mean=0.5, 
            data_std=1.0)
        data = d[0]
        train_set_x, train_set_y = shared_dataset(data, doCastLabels=True)

        norm_mean = d[1]
        norm_std  = d[2]
        grayImages = d[3]
        labelImages = d[4]
        maskImages = d[5]

        # validation data
        d = gen_data_supervised(
            purpose='validate', 
            nsamples=val_samples, 
            patchSize=patchSize, 
            balanceRate=0.5, 
            data_mean=norm_mean, 
            data_std=norm_std)[0]
        valid_set_x, valid_set_y = shared_dataset(d, doCastLabels=True)

        # test data
        d = gen_data_supervised(
            purpose='test', 
            nsamples=test_samples, 
            patchSize=patchSize, 
            balanceRate=0.5, 
            data_mean=norm_mean, 
            data_std=norm_std)[0]
        test_set_x, test_set_y = shared_dataset(d, doCastLabels=True)

        # compute number of minibatches for training, validation and testing
        n_train_batches = train_samples / batchSize
        n_valid_batches = val_samples / batchSize
        n_test_batches = test_samples / batchSize

        learning_rate_shared = theano.shared(np.float32(learning_rate))
        momentum_shared = theano.shared(np.float32(momentum))

        # allocate symbolic variables for the data
        index = T.lscalar()  # index to a [mini]batch

        # start-snippet-1
        x = self.x #T.matrix('x')   # the data is presented as rasterized images
        y = T.ivector('y')  # the labels are presented as 1D vector of
                            # [int] labels
        cost = self.cost(y)

        lr = T.scalar('learning_rate')
        m = T.scalar('momentum')

        print 'training data....'
        print 'min: ', np.min( train_set_x.eval() )
        print 'max: ', np.max( train_set_x.eval() )
        print 'n_train_batches:',n_train_batches
        print 'n_valid_batches:',n_valid_batches
        print 'n_test_batches:',n_test_batches

        # create a function to compute the mistakes that are made by the model
        test_model = theano.function(
            [index],
            self.errors(y),
            givens={
                x: test_set_x[index * batchSize: (index + 1) * batchSize],
                y: test_set_y[index * batchSize: (index + 1) * batchSize]
            }
        )

        validate_model = theano.function(
            [index],
            self.errors(y),
            givens={
                x: valid_set_x[index * batchSize: (index + 1) * batchSize],
                y: valid_set_y[index * batchSize: (index + 1) * batchSize]
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
                    x: train_set_x[index * batchSize:(index + 1) * batchSize],
                    y: train_set_y[index * batchSize:(index + 1) * batchSize],
                    lr: learning_rate_shared,
                    m: momentum_shared})


        ###############
        # TRAIN MODEL #
        ###############
        print '... training'
        best_validation_loss = numpy.inf
        best_iter = 0
        decrease_epoch = 1
        decrease_patience = 1
        test_score = 0.
        doResample = False

        validation_frequency = 1

        start_time = time.clock()

        epoch = 0
        done_looping = False

        # start pool for data
        print "Starting worker."
        pool = multiprocessing.Pool(processes=1)
        futureData = pool.apply_async(
                        stupid_map_wrapper, 
                        [[gen_data_supervised,True, 'train', train_samples, patchSize, 0.5, 0.5, 1.0]])


        
        while (epoch < n_epochs) and (not self.done):
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
                                [[gen_data_supervised,True, 'train', train_samples, patchSize, 0.5, 0.5, 1.0]])


            for minibatch_index in xrange(n_train_batches):
                if self.done:
                    break
                minibatch_avg_costs.append(train_model(minibatch_index))
                # iteration number
                iter = (epoch - 1) * n_train_batches + minibatch_index

                if (iter + 1) % validation_frequency == 0:
                    #self.save()
                    # compute zero-one loss on validation set
                    validation_losses = np.array([validate_model(i) for i
                                         in xrange(n_valid_batches)])
                    this_validation_loss = numpy.sum(validation_losses) * 100.0 / val_samples

                    msg = 'epoch %i, minibatch %i/%i, training error %.3f, validation error %.2f %%' % (epoch, minibatch_index + 1, n_train_batches, minibatch_avg_costs[-1], this_validation_loss)

                    print(msg)

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:
                        best_validation_loss = this_validation_loss
                        best_iter = iter

                        self.save()
                        print "New best score!"

        pool.close()
        pool.join()
        print "Pool closed."

        end_time = time.clock()
        print(('Optimization complete. Best validation score of %f %% '
               'obtained at iteration %i, with test performance %f %%') %
              (best_validation_loss * 100., best_iter + 1, test_score * 100.))
        print >> sys.stderr, ('The code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.2fm' % ((end_time - start_time) / 60.))



    
