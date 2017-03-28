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

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer, MLP, rectified_linear, send_email

from utils import tile_raster_images

from generateTrainValTestData import generate_experiment_data_supervised, shared_dataset, normalizeImage, stupid_map_wrapper
import multiprocessing
from classifyImage import generate_patch_data_rows
from vsk_utils import shared_single_dataset

import matplotlib
import matplotlib.pyplot as plt
import getpass

class LeNetConvPoolLayer(object):
    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2), activation=rectified_linear):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input
        self.poolsize = poolsize
        self.image_shape = image_shape
        self.filter_shape = filter_shape

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = activation(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

        self.conv_out = conv_out
        self.pooled_out = pooled_out

    def visualize_filters(self):
        self.W = self.W * 1.0
        W = self.W.eval()
        print W.shape
        patchSize = self.filter_shape[2]
        print patchSize
        filterSize = self.filter_shape[2]**2
        print filterSize
        numFilters = self.filter_shape[0]
        print numFilters

        W = np.reshape(W, (numFilters, filterSize))
        print W.shape

        return tile_raster_images(X=W, img_shape=(patchSize, patchSize), tile_shape=(10,10), tile_spacing=(1, 1),
                                  scale_rows_to_unit_interval=True,
                                  output_pixel_vals=True)


class CNN(object):
    def __init__(self,input, batch_size, patchSize, rng, nkerns, kernelSizes, hiddenSizes, fileName=None, activation=rectified_linear):
        
        self.convLayers = []
        self.trainingCost = []
        self.validationError = []
        self.nkerns = nkerns 
        self.kernelSizes = kernelSizes 
        self.hiddenSizes = hiddenSizes

        self.patchSize = patchSize
        self.batch_size = batch_size

        input = input.reshape((self.batch_size, 1, self.patchSize, self.patchSize))

        self.layer0_input = input 
        self.params = [] 
        
        input_next = input
        numberOfFeatureMaps = 1
        featureMapSize = patchSize
        
        for i in range(len(nkerns)):
            layer = LeNetConvPoolLayer(
                rng,
                input=input_next,
                image_shape=(batch_size, numberOfFeatureMaps, featureMapSize, featureMapSize),
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
        
        self.mlp = MLP(rng=rng, input=mlp_input, n_in=nkerns[-1] * (featureMapSize ** 2), n_hidden=hiddenSizes, 
                               n_out=2, activation=rectified_linear) 
        self.params += self.mlp.params
                
        self.cost = self.mlp.negative_log_likelihood 
        self.errors = self.mlp.errors
        self.p_y_given_x = self.mlp.p_y_given_x
        self.debug_x = self.p_y_given_x


        if not fileName is None:
            with open(fileName, 'r') as file:
                saved_convLayers, saved_hiddenLayers, saved_logRegressionLayer, self.trainingCost, self.validationError, saved_nkerns, saved_kernelSizes, saved_batch_size, saved_patchSize, saved_hiddenSizes = cPickle.load(file)

            for s_cl, cl in zip(saved_convLayers, self.convLayers):
                cl.W.set_value(s_cl.W.get_value())
                cl.b.set_value(s_cl.b.get_value())

            for s_hl, hl in zip(saved_hiddenLayers, self.mlp.hiddenLayers):
                hl.W.set_value(np.float32(s_hl.W.eval()))
                hl.b.set_value(s_hl.b.get_value())
            
            self.mlp.logRegressionLayer.W.set_value(np.float32(saved_logRegressionLayer.W.eval()))
            self.mlp.logRegressionLayer.b.set_value(saved_logRegressionLayer.b.get_value())


    def save_CNN(self, filename):
        with open(filename, 'wb') as file:
            cPickle.dump((self.convLayers, self.mlp.hiddenLayers, self.mlp.logRegressionLayer, self.trainingCost, self.validationError, self.nkerns, self.kernelSizes, self.batch_size, self.patchSize, self.hiddenSizes), file)

    def classify_image(self, img, normMean=None, norm_std=None):
        start_time = time.clock()
        
        row_range = 1
        img = normalizeImage(img)
        imSize = np.shape(img)
        membraneProbabilities = np.zeros(np.shape(img))
        patchSize = self.patchSize
        
        data_shared = shared_single_dataset(np.zeros((imSize[0]*row_range,patchSize**2)), borrow=True)

        classify = theano.function(
            [],
            self.p_y_given_x,
            givens={x: data_shared}
        )


        for row in xrange(0,1024,row_range):
            if row%100 == 0:
                print row
            data = generate_patch_data_rows(img, rowOffset=row, rowRange=row_range, patchSize=patchSize, imSize=imSize, data_mean=normMean, data_std=norm_std)
            data_shared.set_value(np.float32(data))
            result = classify()
            membraneProbabilities[row,:] = result[:,1]
            
        end_time = time.clock()
        total_time = (end_time - start_time)
        
        print "Image classification took %f seconds" % (total_time)
        
        return np.array(membraneProbabilities)


    def debug_whole_image_classification(self, img, normMean=None, norm_std=None):
        start_time = time.clock()

        row_range = 1
        img = normalizeImage(img)
        imSize = np.shape(img)
        membraneProbabilities = np.zeros(np.shape(img))
        patchSize = self.patchSize
        
        data_shared = shared_single_dataset(np.zeros((imSize[0]*row_range,patchSize**2)), borrow=True)

        classify = theano.function(
            [],
            self.debug_x,
            givens={x: data_shared}
        )


        for row in xrange(0,33,row_range):
            if row%100 == 0:
                print row
            data = generate_patch_data_rows(img, rowOffset=row, rowRange=row_range, patchSize=patchSize, imSize=imSize, data_mean=normMean, data_std=norm_std)
            data_shared.set_value(np.float32(data))
            result = classify()
            #membraneProbabilities[row,:] = result[:,1]
            membraneProbabilities = result

        end_time = time.clock()
        total_time = (end_time - start_time)
        
        print "Image classification took %f seconds" % (total_time)
        
        return np.array(membraneProbabilities)


def evaluate_lenet5(learning_rate=0.0001, n_epochs=20000, nkerns=[48,48,48], kernelSizes=[5,5,5], hiddenSizes=[200], doResample=True, batch_size=1, patchSize=65, train_samples=50000, val_samples=10000, test_samples=1000, validation_frequency = 100, doEmailUpdate=False, momentum=0.98, filename='tmp_cnn.pkl'):

    def gradient_updates_momentum(cost, params, learning_rate, momentum):
        updates = []
        for param in params:
            param_update = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
            updates.append((param, param - learning_rate*param_update))
            updates.append((param_update, momentum*param_update + (1. - momentum)*T.grad(cost, param)))
        return updates


    rng = numpy.random.RandomState(23455)
    
    data, norm_mean, norm_std, grayImages, labelImages, maskImages = generate_experiment_data_supervised(purpose='train', nsamples=train_samples, patchSize=patchSize, balanceRate=0.5, data_mean=0.5, data_std=1.0)
    train_set_x, train_set_y = shared_dataset(data, doCastLabels=True)

    data = generate_experiment_data_supervised(purpose='validate', nsamples=val_samples, patchSize=patchSize, balanceRate=0.5, data_mean=norm_mean, data_std=norm_std)[0]
    valid_set_x, valid_set_y = shared_dataset(data, doCastLabels=True)

    data = generate_experiment_data_supervised(purpose='test', nsamples=test_samples, patchSize=patchSize, balanceRate=0.5, data_mean=norm_mean, data_std=norm_std)[0]
    test_set_x, test_set_y = shared_dataset(data, doCastLabels=True)

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_samples / batch_size
    n_valid_batches = val_samples / batch_size
    n_test_batches = test_samples / batch_size

    learning_rate_shared = theano.shared(np.float32(learning_rate))
    momentum_shared = theano.shared(np.float32(momentum))


    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    lr = T.scalar('learning_rate')
    m = T.scalar('momentum')



    if doEmailUpdate:
        gmail_pwd = getpass.getpass()

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'
    
    classifier = CNN(input=x, batch_size=batch_size, patchSize=patchSize, rng=rng, 
                     nkerns=nkerns, kernelSizes=kernelSizes, hiddenSizes=hiddenSizes, 
                     fileName=filename)

    cost = classifier.cost(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )


    gparams = []
    for param in classifier.params:
        gparam = T.grad(cost, param)
        gparams.append(gparam)

    #SGD
#    updates = []
#    for param, gparam in zip(classifier.params, gparams):
#        updates.append((param, param - lr * gparam))

    #updates = adadelta_updates(classifier.params, gparams, lr, 0.000001)
    updates = gradient_updates_momentum(cost, classifier.params, lr, m)
    

    train_model = theano.function(inputs=[index], outputs=cost,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size:(index + 1) * batch_size],
                y: train_set_y[index * batch_size:(index + 1) * batch_size],
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

    start_time = time.clock()

    epoch = 0
    done_looping = False

    # start pool for data
    print "Starting worker."
    pool = multiprocessing.Pool(processes=1)
    futureData = pool.apply_async(stupid_map_wrapper, [[generate_experiment_data_supervised,True, 'train', train_samples, patchSize, 0.5, 0.5, 1.0]])

    while (epoch < n_epochs) and (not done_looping):
        minibatch_avg_costs = []
        epoch = epoch + 1

        if doResample and epoch>1:
            print "Waiting for data."
            data = futureData.get()
            print "GOT NEW DATA"
            train_set_x.set_value(np.float32(data[0]))
            train_set_y.set_value(np.int32(data[1]))
            futureData = pool.apply_async(stupid_map_wrapper, [[generate_experiment_data_supervised,True, 'train', train_samples, patchSize, 0.5, 0.5, 1.0]])
#            try:
#                data = futureData.get(timeout=1)
#                print "GOT NEW DATA"
#                train_set_x.set_value(np.float32(data[0]))
#                train_set_y.set_value(np.int32(data[1]))
#                futureData = pool.apply_async(stupid_map_wrapper, [[generate_experiment_data_supervised,True, 'train', train_samples, patchSize, 0.5, norm_mean, 1.0]])
#            except multiprocessing.TimeoutError:
#                print "TIMEOUT, TRAINING ANOTHER ROUND WITH CURRENT DATA"
#                pass
#


        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_costs.append(train_model(minibatch_index))
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                classifier.save_CNN('current_cnn.pkl')
                # compute zero-one loss on validation set
                validation_losses = np.array([validate_model(i) for i
                                     in xrange(n_valid_batches)])
                this_validation_loss = numpy.sum(validation_losses) * 100.0 / val_samples 
                
                msg = 'epoch %i, minibatch %i/%i, training error %.3f, validation error %.2f %%' % (epoch, minibatch_index + 1, n_train_batches, minibatch_avg_costs[-1], this_validation_loss)

                print(msg)

                classifier.trainingCost.append(minibatch_avg_costs[-1])
                classifier.validationError.append(this_validation_loss*100)

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    classifier.save_CNN('best_cnn_so_far.pkl')
                    print "New best score!"
                    if doEmailUpdate:
                        send_email(gmail_pwd, msg)
                    # test it on the test set
                    #test_losses = [test_model(i) for i
                    #               in xrange(n_test_batches)]
                    #test_score = numpy.mean(test_losses)
                    #
                    #print(('epoch %i, minibatch %i/%i, test error of '
                    #       'best model %f %%') %
                    #      (epoch, minibatch_index + 1, n_train_batches,
                    #       test_score * 100.))

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

    return classifier



if __name__ == '__main__':
    rng = numpy.random.RandomState(929292)

    import mahotas
    import matplotlib.pyplot as plt
    image = mahotas.imread('ac3_input_0141.tif')

    x = T.matrix('x')


    doDebug = False

###############################################
### DEBUGGING START
###############################################

    if doDebug:
        image = image[0:170,0:170]
        test2 = CNN(input=x, batch_size=170, patchSize=65, rng=rng, nkerns=[48,48,48], kernelSizes=[5,5,5], hiddenSizes=[200], fileName='tmp_cnn.pkl') 
        fooBa = test2.debug_whole_image_classification(image, normMean=0.5, norm_std=1.0)
    

    

###############################################
### DEBUGGING END
###############################################

    if not doDebug:
        test2 = CNN(input=x, batch_size=1024, patchSize=65, rng=rng, nkerns=[48,48,48], kernelSizes=[5,5,5], hiddenSizes=[200], fileName='tmp_cnn.pkl') 
        
        
        prob = test2.classify_image(img=image, normMean=0.5, norm_std=1.0)
        plt.imshow(1-prob)
        plt.show()
        mahotas.imsave('tmp_output_cnn_09.png', np.uint8((1-prob)*255))
        
        cl = test2.convLayers[0]
        plt.imshow(cl.visualize_filters())
        plt.show()
        mahotas.imsave('filter_output_cnn_09.png', np.uint8(cl.visualize_filters()))
        
        plt.plot(np.array(test2.trainingCost), label='training')
        plt.plot(np.array(test2.validationError), label='validation')
        plt.legend()
        plt.show()
        
        if len(test2.validationError) > 5000:
            plt.plot(np.array(test2.trainingCost)[-5000:], label='training')
            plt.plot(np.array(test2.validationError)[-5000:], label='validation')
            plt.legend()
            plt.show()
        
        
        print "best validation score: ", test2.validationError[-1]

