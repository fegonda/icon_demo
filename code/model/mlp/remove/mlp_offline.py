import cPickle
import gzip
import os
import sys
import time

import numpy
import numpy as np

import theano
import theano.tensor as T

base_path = os.path.dirname(__file__)
sys.path.insert(1,os.path.join(base_path, '../'))
#sys.path.insert(2,os.path.join(base_path, '../../external'))
sys.path.insert(2,os.path.join(base_path, '../../common'))

from hiddenlayer import HiddenLayer
from logistic_sgd import LogisticRegression
from generateTrainValTestData import gen_data_supervised, shared_dataset, normalizeImage
import multiprocessing
from vsk_utils import shared_single_dataset
from activation_functions import rectified_linear
import smtplib
import getpass

class MLP_Offline(object):
    def __init__(
        id,                           # unique identifier of the model
        rng,                          # random number generator use to initialize weights
        input,                        # a theano.tensor.dmatrix of shape (n_examples, n_in)
        n_in=None,                    # dimensionality of the input
        n_hidden=None,                # list of number of hidden units for each layer
        n_out=None,                   # dimensionality of the output
        train_time=30,                # batch training time before resampling
        batch_size=10,                # size of mini batch
        patch_size=39,                # size of feature map
        learning_rate=0.01,
        momentum=0.9,
        path=None,                    # where to load and save model
        activation=rectified_linear):  # activation function to use

        self.hiddenLayers = []
        self.params = []
        self.trainingCost = []
        self.validationError = []
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.x = input

        self.type = 'MLP'
        self.done = False
        self.path = path    
        self.activation = activation
        self.hiddenSizes = n_hidden
        self.batchSize = batch_size
        self.patchSize = patch_size 
        self.hiddenSizes = n_hidden
       
        fromFile = (self.path is not None) and os.path.exists( self.path )

        if fromFile: 
            with open(self.path, 'r') as file:
                d = ckle.load(file)
                savedhiddenLayers        = d[0]
                saved_logRegressionLayer = d[1]
                self.n_in                = d[2]
                self.n_hidden            = d[3]

        next_input = input
        next_n_in = self.n_in
            
        for n_h in self.n_hidden:
            hl = HiddenLayer(rng=rng, input=next_input,
                             n_in=next_n_in, n_out=n_h,
                             activation=activation)
            next_input = hl.output
            next_n_in = n_h
            self.hiddenLayers.append(hl)
            self.params += hl.params
            
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayers[-1].output,
            n_in=self.n_hidden[-1],
            n_out=n_out)
        
        self.params += self.logRegressionLayer.params
        
        self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood

        self.errors = self.logRegressionLayer.errors
        self.p_y_given_x = self.logRegressionLayer.p_y_given_x

        if fromFile:
            for hl, shl in zip(self.hiddenLayers, savedhiddenLayers):
                hl.W.set_value(shl.W.get_value())
                hl.b.set_value(shl.b.get_value())

            self.logRegressionLayer.W.set_value(saved_logRegressionLayer.W.get_value())
            self.logRegressionLayer.b.set_value(saved_logRegressionLayer.b.get_value())

    def save(self):
        with open(self.path, 'wb') as file:
            cPickle.dump((
                self.hiddenLayers, 
                self.logRegressionLayer, 
                self.n_in, 
                self.n_hidden), file)


    def get_patch_size(self):
        return np.int(np.sqrt(self.hiddenLayers[0].W.eval().shape[0]))

    def predict_image(self, x, img, normMean=None, norm_std=None):
        start_time = time.clock()

        row_range = 1
        img = normalizeImage(img)
        imSize = np.shape(img)
        membraneProbabilities = np.zeros(1024*1024, dtype=int )
        patchSize = np.int(np.sqrt(self.hiddenLayers[0].W.eval().shape[0]))

        data_shared = shared_single_dataset(np.zeros((imSize[0]*row_range,patchSize**2)), borrow=True)

        classify = theano.function(
            [],
            self.logRegressionLayer.y_pred,
            givens={x: data_shared}
        )
        for row in xrange(0,1024,row_range):
            if row%100 == 0:
                print row
            data = generate_patch_data_rows(img, rowOffset=row, rowRange=row_range, patchSize=patchSize, imSize=imSize, data_mean=normMean, data_std=norm_std)
            data_shared.set_value(np.float32(data))
            membraneProbabilities[row*1024:row*1024+row_range*1024] = classify()

        end_time = time.clock()
        total_time = (end_time - start_time)
        print >> sys.stderr, ('Running time: ' +
                              '%.2fm' % (total_time / 60.))

        return np.array(membraneProbabilities)
   
 
    def classify_image(self, img, normMean=None, norm_std=None):
        start_time = time.clock()
        
        row_range = 1
        img = normalizeImage(img)
        imSize = np.shape(img)
        membraneProbabilities = np.zeros(np.shape(img))
        patchSize = np.int(np.sqrt(self.hiddenLayers[0].W.eval().shape[0]))
        
        data_shared = shared_single_dataset(np.zeros((imSize[0]*row_range,patchSize**2)), borrow=True)

        classify = theano.function(
            [],
            self.p_y_given_x,
            givens={self.x: data_shared}
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
        print >> sys.stderr, ('Running time: ' +
                              '%.2fm' % (total_time / 60.))
        
        return np.array(membraneProbabilities)
        

def send_email(gmail_pwd, msg):
    to = 'vkaynig@seas.harvard.edu'
    gmail_user = 'vkaynig@gmail.com'
    smtpserver = smtplib.SMTP("smtp.gmail.com",587)
    smtpserver.ehlo()
    smtpserver.starttls()
    smtpserver.ehlo
    smtpserver.login(gmail_user, gmail_pwd)
    header = 'To:' + to + '\n' + 'From: ' + gmail_user + '\n' + 'Subject:DNN update \n'
    msg = header + '\n' + msg + '\n\n'
    smtpserver.sendmail(gmail_user, to, msg)
    smtpserver.close()


def train_mlp(learning_rate=0.01, n_epochs=10, batch_size=500, n_hidden=[500], patchSize=19, train_samples=1000, val_samples=10000, test_samples=1000, doResample=False, validation_frequency = 1, activation=rectified_linear, doEmailUpdate=False, momentum=0.0):

    def adadelta_updates(parameters,gradients,rho,eps):
        # create variables to store intermediate updates
        gradients_sq = [ theano.shared(np.zeros(p.get_value().shape, dtype=theano.config.floatX)) for p in parameters ]
        deltas_sq = [ theano.shared(np.zeros(p.get_value().shape, dtype=theano.config.floatX)) for p in parameters ]
        # calculates the new "average" delta for the next iteration
        gradients_sq_new = [ rho*g_sq + (1-rho)*(g**2) for g_sq,g in zip(gradients_sq,gradients) ]
        
        # calculates the step in direction. The square root is an approximation to getting the RMS for the average value
        deltas = [ (T.sqrt(d_sq+eps)/T.sqrt(g_sq+eps))*grad for d_sq,g_sq,grad in zip(deltas_sq,gradients_sq_new,gradients) ]
        
        # calculates the new "average" deltas for the next step.
        deltas_sq_new = [ rho*d_sq + (1-rho)*(d**2) for d_sq,d in zip(deltas_sq,deltas) ]
        
        # Prepare it as a list f
        gradient_sq_updates = zip(gradients_sq,gradients_sq_new)
        deltas_sq_updates = zip(deltas_sq,deltas_sq_new)
        parameters_updates = [ (p,p - d) for p,d in zip(parameters,deltas) ]
        return gradient_sq_updates + deltas_sq_updates + parameters_updates

    def gradient_updates_momentum(cost, params, learning_rate, momentum):
        updates = []
        for param in params:
            param_update = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
            updates.append((param, param - learning_rate*param_update))
            updates.append((param_update, momentum*param_update + (1. - momentum)*T.grad(cost, param)))
        return updates

    if doEmailUpdate:
        gmail_pwd = getpass.getpass()

    rng = numpy.random.RandomState(1234)

    data, norm_mean, norm_std, grayImages, labelImages, maskImages = gen_data_supervised(purpose='train', nsamples=train_samples, patchSize=patchSize, balanceRate=0.5, data_mean=0.5, data_std=1.0)
    train_set_x, train_set_y = shared_dataset(data, doCastLabels=True)

    data = gen_data_supervised(purpose='validate', nsamples=val_samples, patchSize=patchSize, balanceRate=0.5, data_mean=norm_mean, data_std=norm_std)[0]
    valid_set_x, valid_set_y = shared_dataset(data, doCastLabels=True)

    data = gen_data_supervised(purpose='test', nsamples=test_samples, patchSize=patchSize, balanceRate=0.5, data_mean=norm_mean, data_std=norm_std)[0]
    test_set_x, test_set_y = shared_dataset(data, doCastLabels=True)

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_samples / batch_size
    n_valid_batches = val_samples / 1000
    n_test_batches = test_samples / 1000

    learning_rate_shared = theano.shared(np.float32(learning_rate))
    momentum_shared = theano.shared(np.float32(momentum))

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels
    lr = T.scalar('learning_rate')
    m = T.scalar('momentum')

    # construct the MLP class
    classifier = MLP(rng=rng, input=x, n_in=patchSize**2,
                             n_hidden=n_hidden, n_out=2, activation=activation)


    cost = classifier.negative_log_likelihood(y) 

    test_model = theano.function(inputs=[index],
                                 outputs=classifier.errors(y),
                                 givens={
                                     x: test_set_x[index * batch_size:(index + 1) * batch_size],
                                     y: test_set_y[index * batch_size:(index + 1) * batch_size]})
    validate_model = theano.function(inputs=[index],
                                     outputs=classifier.errors(y),
                                     givens={
                                         x: valid_set_x[index * batch_size:(index + 1) * batch_size],
                                         y: valid_set_y[index * batch_size:(index + 1) * batch_size]}) 
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
    futureData = pool.apply_async(stupid_map_wrapper, [[gen_data_supervised,True, 'train', train_samples, patchSize, 0.5, 0.5, 1.0]])

    while (epoch < n_epochs) and (not done_looping):
        minibatch_avg_costs = []
        epoch = epoch + 1

        if doResample and epoch>1:
            print "Waiting for data."
            data = futureData.get()
            print "GOT NEW DATA"
            train_set_x.set_value(np.float32(data[0]))
            train_set_y.set_value(np.int32(data[1]))
            futureData = pool.apply_async(stupid_map_wrapper, [[gen_data_supervised,True, 'train', train_samples, patchSize, 0.5, 0.5, 1.0]])
#            try:
#                data = futureData.get(timeout=1)
#                print "GOT NEW DATA"
#                train_set_x.set_value(np.float32(data[0]))
#                train_set_y.set_value(np.int32(data[1]))
#                futureData = pool.apply_async(stupid_map_wrapper, [[gen_data_supervised,True, 'train', train_samples, patchSize, 0.5, norm_mean, 1.0]])
#            except multiprocessing.TimeoutError:
#                print "TIMEOUT, TRAINING ANOTHER ROUND WITH CURRENT DATA"
#                pass
#


        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_costs.append(train_model(minibatch_index))
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                classifier.save('current.pkl')
                # compute zero-one loss on validation set
                validation_losses = np.array([validate_model(i) for i
                                     in xrange(n_valid_batches)])
                this_validation_loss = numpy.mean(validation_losses*100.0)
                
                msg = 'epoch %i, minibatch %i/%i, training error %.3f, validation error %.2f %%' % (epoch, minibatch_index + 1, n_train_batches, minibatch_avg_costs[-1], this_validation_loss)

                print(msg)

                classifier.trainingCost.append(minibatch_avg_costs[-1])
                classifier.validationError.append(this_validation_loss*100)

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    classifier.save('best_so_far.pkl')
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
    import mahotas
    import matplotlib.pyplot as plt
#    image = mahotas.imread('train-input0099.tif')
    image = mahotas.imread('ac3_input_0141.tif')

    x = T.matrix('x')

    test2 = MLP(rng=numpy.random.RandomState(1), input=x, n_out=2, fileName = 'tmp.pkl')

    prob = test2.classify_image(img=image, normMean=0.5, norm_std=1.0)
    plt.imshow(1-prob)
    plt.show()
    mahotas.imsave('tmp_output_05.png', np.uint8((1-prob)*255))

    hl = test2.hiddenLayers[0]
    plt.imshow(hl.visualize_filters())
    plt.show()
    mahotas.imsave('filter_output_05.png', np.uint8(hl.visualize_filters()))

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

