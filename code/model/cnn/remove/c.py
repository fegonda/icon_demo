import cPickle
import gzip

import os
import sys
import time

import numpy
import numpy as np

import multiprocessing

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

base_path = os.path.dirname(__file__)
sys.path.insert(1,os.path.join(base_path, '..'))
sys.path.insert(2,os.path.join(base_path, '../../mlp'))
sys.path.insert(3,os.path.join(base_path, '../../common'))

from logistic_sgd import LogisticRegression
from mlp import MLP
from activation_functions import rectified_linear

from generateTrainValTestData import gen_data_supervised, shared_dataset, normalizeImage, stupid_map_wrapper
from classifyImage import generate_patch_data_rows
from vsk_utils import shared_single_dataset
from fast_segment import *

#import matplotlib
#import matplotlib.pyplot as plt
import getpass
from convlayer import LeNetConvPoolLayer

class CNN(object):
    def __init__(
        self,
        id,
        input, 
        batch_size, 
        patch_size, 
        rng, 
        nkerns, 
        kernel_sizes, 
        hidden_sizes, 
        path=None,
        train_time=30.0,
        learning_rate=0.1,
        momentum=0.9,
        activation=rectified_linear):

        self.type = 'CNN'
        self.done = False
        self.path = path    
        self.rng = rng
        self.activation = activation
        self.input = input
        self.nkerns = nkerns
        self.kernelSizes = kernel_sizes
        self.hiddenSizes = hidden_sizes
        self.batchSize = batch_size
        self.patchSize = patch_size 
        self.convLayers = []
        self.trainingCost = []
        self.validationError = []
        self.nkerns = nkerns
        self.kernelSizes = kernel_sizes
        self.hiddenSizes = hidden_sizes
        self.learning_rate = learning_rate
        self.momentum      = momentum
        self.x           =  input

        self.resample = False       
        self.initialize() 
    
    def initialize(self):
        input = self.input
        input = self.input.reshape((self.batchSize, 1, self.patchSize, self.patchSize))

        self.layer0_input = input
        self.params = []

        input_next = input
        numberOfFeatureMaps = 1
        featureMapSize = self.patchSize

        for i in range(len(self.nkerns)):
            layer = LeNetConvPoolLayer(
                self.rng,
                input=input_next,
                image_shape=(self.batchSize, numberOfFeatureMaps, featureMapSize, featureMapSize),
                filter_shape=(self.nkerns[i], numberOfFeatureMaps, self.kernelSizes[i], self.kernelSizes[i]),
                poolsize=(2, 2)
            )
            input_next = layer.output
            numberOfFeatureMaps = self.nkerns[i]
            featureMapSize = np.int16(np.floor((featureMapSize - self.kernelSizes[i]+1) / 2))

            self.params += layer.params
            self.convLayers.append(layer)

        # the 2 is there to preserve the batchSize
        mlp_input = self.convLayers[-1].output.flatten(2)

        self.mlp = MLP(
                    rng=self.rng, 
                    input=mlp_input, 
                    n_in=self.nkerns[-1] * (featureMapSize ** 2), 
                    n_hidden=self.hiddenSizes,
                    n_out=2, 
                    patch_size=self.patchSize,
                    batch_size=self.batchSize,
                    activation=self.activation)
        self.params += self.mlp.params

        self.cost = self.mlp.negative_log_likelihood
        self.errors = self.mlp.errors
        self.p_y_given_x = self.mlp.p_y_given_x
        self.debug_x = self.p_y_given_x

        if not self.path is None and os.path.exists(self.path):
            with open(self.path, 'r') as file:
                print 'loading cnn model from file...', self.path
                data = cPickle.load(file)
                saved_convLayers         = data[0]
                saved_hiddenLayers       = data[1]
                saved_logRegressionLayer = data[2]
                saved_nkerns             = data[3]
                saved_kernelSizes        = data[4]
                saved_batchSize         = data[5]
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


    def classify(self, image):
        return classify_image( image, self )

    def predict(self, path, threshold=0.5):
     
        image = mahotas.imread( path )
        #imageSize = 1024
        #image = image[0:imageSize,0:imageSize]
        image = Utility.normalizeImage( image ) - 0.5

        print 'max:', np.max( image.flatten() )
        print 'min:', np.min( image.flatten() )
        prob = classify_image( image, self )
        prob[ prob >= threshold ] = 9
        prob[ prob <  threshold ] = 1
        prob[ prob == 9 ] = 0
        prob = prob.astype(dtype=int)
        prob = prob.flatten()
        print 'results :', np.bincount( prob )
        return prob

    def train(self, offline=False, data=None):
        if offline:
            self.train_offline()
        else:
            self.train_online(data)

    def train_online(self, data):
        print 'train online...'

        d = data.sample()
        train_x = d[0]
        train_y = d[1]
        valid_x = d[2]
        valid_y = d[3]

        if self.resample:
            self.lr_shared.set_value( np.float32(self.learning_rate) )
            self.m_shared.set_value( np.float32(self.momentum) )

            self.train_x.set_value( np.float32( train_x ) )
            self.valid_x.set_value( np.float32( valid_x ) )

            self.train_y.owner.inputs[0].set_value( np.int32( train_y ))
            self.valid_y.owner.inputs[0].set_value( np.int32( valid_y ))
        else:
            self.resample  = True
            self.y         = T.ivector('y')  # the labels are presented as 1D vector of [int] labels
            self.lr        = T.scalar('learning_rate')
            self.m         = T.scalar('momentum')

            self.lr_shared = theano.shared(np.float32(self.learning_rate))
            self.m_shared  = theano.shared(np.float32(self.momentum))

            self.train_x   = theano.shared( train_x, borrow=True)
            self.valid_x   = theano.shared( valid_x, borrow=True)

            self.train_y = theano.shared( train_y, borrow=True)
            self.valid_y = theano.shared( valid_y, borrow=True)

            self.train_y = T.cast( self.train_y, 'int32')
            self.valid_y = T.cast( self.valid_y, 'int32')

        print 'done...'

    def train_offline(self):

        print 'training....'
        train_samples=10000
        val_samples=1000
        test_samples=1000
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



