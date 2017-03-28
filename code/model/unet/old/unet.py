
import os
import sys

base_path = os.path.dirname(__file__)
sys.path.insert(1,os.path.join(base_path, '../mlp'))
sys.path.insert(2,os.path.join(base_path, '../../common'))
sys.path.insert(3,os.path.join(base_path, '../../database'))

from db import DB
from paths import Paths


# yet another version of the IDSIA network
# based on code from keras tutorial 
# http://keras.io/getting-started/sequential-model-guide/
from keras.models import Model, Sequential, model_from_json
from keras.layers import Dense, Activation, Flatten, Input
from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D, merge, ZeroPadding2D, Dropout, Lambda
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras.optimizers import SGD
from keras.regularizers import l2
from generate_data import *
import multiprocessing
import sys
import matplotlib
import matplotlib.pyplot as plt
# loosing independence of backend for 
# custom loss function
import theano
import theano.tensor as T
from evaluation import Rand_membrane_prob
from theano.tensor.shared_randomstreams import RandomStreams

class UNET(object):

    def __init(self, 
        id,
        input,
        patch_size,
        offline=False,
        path=None,
        train_time=5.0,
        learning_rate=0.1,
        momentum=0.9,
        patchSize_out = 388):

        self.id = id
        self.type = 'UNET'
        self.offline = offline
        self.done = False
        self.rng = rng
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.patchSize = patch_size
           
        self.initialize(self)


    def initialize(self):

        # run as python unet.py 1 (for training) 0 (prediction)

        rng = np.random.RandomState(7)

        train_samples = 30 
        val_samples = 20
        learning_rate = 0.01
        momentum = 0.95

        patchSize = 572 #140
        patchSize_out = 388 #132

        weight_decay = 0.
        weight_class_1 = 1.

        patience = 100
        patience_reset = 100

        doBatchNormAll = False
        doFineTune = False

        purpose = 'train'
        initialization = 'glorot_uniform'
        filename = 'unet_Cerebellum_clahe'
        print "filename: ", filename

        srng = RandomStreams(1234)

    def train(self):
        print 'train...'

    def predict(self, image, mean=None, std=None, threshold=0.5):
        print 'predict...'

        return None

    def save(self):
        print 'save...'

        path = self.path
        revision = 0
        if not self.offline:
            revision = DB.getRevision( self.id )
            revision = (revision+1)%10
            path = '%s/best_%s.%s.%d.pkl'%(Paths.Models, self.id, self.type, revision)
            path = path.lower()

        print 'saving...', path
        # do actual saving here...

        if not self.offline:
            DB.finishSaveModel( self.id, revision )

    def get_path(self):
        if self.offline:
            return self.path

        rev  = DB.getRevision( self.id )
        path = '%s/best_%s.%s.%d.pkl'%(Paths.Models, self.id, self.type, rev )
        return path.lower()


    # need to define a custom loss, because all pre-implementations
    # seem to assume that scores over patch add up to one which
    # they clearly don't and shouldn't
    @staticmethod
    def unet_crossentropy_loss(y_true, y_pred):
        epsilon = 1.0e-4
        y_pred_clipped = T.clip(y_pred, epsilon, 1.0-epsilon)
        loss_vector = -T.mean(weight_class_1*y_true * T.log(y_pred_clipped) + (1-y_true) * T.log(1-y_pred_clipped), axis=1)
        average_loss = T.mean(loss_vector)
        return average_loss

    @staticmethod
    def unet_crossentropy_loss_sampled(y_true, y_pred):
        epsilon = 1.0e-4
        y_pred_clipped = T.flatten(T.clip(y_pred, epsilon, 1.0-epsilon))
        y_true = T.flatten(y_true)
        # this seems to work
        # it is super ugly though and I am sure there is a better way to do it
        # but I am struggling with theano to cooperate
        # filter the right indices
        indPos = T.nonzero(y_true)[0] # no idea why this is a tuple
        indNeg = T.nonzero(1-y_true)[0]
        # shuffle
        n = indPos.shape[0]
        indPos = indPos[srng.permutation(n=n)]
        n = indNeg.shape[0]
        indNeg = indNeg[srng.permutation(n=n)]
        # take equal number of samples depending on which class has less
        n_samples = T.cast(T.min([T.sum(y_true), T.sum(1-y_true)]), dtype='int64')

        indPos = indPos[:n_samples]
        indNeg = indNeg[:n_samples]
        loss_vector = -T.mean(T.log(y_pred_clipped[indPos])) - T.mean(T.log(1-y_pred_clipped[indNeg]))
        average_loss = T.mean(loss_vector)
        return average_loss

    @staticmethod
    def unet_block_down(input, nb_filter, doPooling=True, doDropout=False, doBatchNorm=False):
        # first convolutional block consisting of 2 conv layers plus activation, then maxpool.
        # All are valid area, not same
        act1 = Convolution2D(nb_filter=nb_filter, nb_row=3, nb_col=3, subsample=(1,1),
                                 init=initialization, activation='relu',  border_mode="valid", W_regularizer=l2(weight_decay))(input)
        if doBatchNorm:
            act1 = BatchNormalization(mode=0, axis=1)(act1)

        act2 = Convolution2D(nb_filter=nb_filter, nb_row=3, nb_col=3, subsample=(1,1),
                                 init=initialization, activation='relu', border_mode="valid", W_regularizer=l2(weight_decay))(act1)
        if doBatchNorm:
            act2 = BatchNormalization(mode=0, axis=1)(act2)

        if doDropout:
            act2 = Dropout(0.5)(act2)
        
        if doPooling:
            # now downsamplig with maxpool
            pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode="valid")(act2)
        else:
            pool1 = act2

        return (act2, pool1)

    # need to define lambda layer to implement cropping
    # input is a tensor of size (batchsize, channels, width, height)
    @staticmethod
    def crop_layer(x, cs):
        cropSize = cs
        return x[:,:,cropSize:-cropSize, cropSize:-cropSize]

    @staticmethod
    def unet_block_up(input, nb_filter, down_block_out, doBatchNorm=False):
        print "This is unet_block_up"
        print "input ", input._keras_shape
        # upsampling
        up_sampled = UpSampling2D(size=(2,2))(input)
        print "upsampled ", up_sampled._keras_shape
        # up-convolution
        conv_up = Convolution2D(nb_filter=nb_filter, nb_row=2, nb_col=2, subsample=(1,1),
                                 init=initialization, activation='relu', border_mode="same", W_regularizer=l2(weight_decay))(up_sampled)
        print "up-convolution ", conv_up._keras_shape
        # concatenation with cropped high res output
        # this is too large and needs to be cropped
        print "to be merged with ", down_block_out._keras_shape

        #padding_1 = int((down_block_out._keras_shape[2] - conv_up._keras_shape[2])/2)
        #padding_2 = int((down_block_out._keras_shape[3] - conv_up._keras_shape[3])/2)
        #print "padding: ", (padding_1, padding_2)
        #conv_up_padded = ZeroPadding2D(padding=(padding_1, padding_2))(conv_up)
        #merged = merge([conv_up_padded, down_block_out], mode='concat', concat_axis=1)
        
        cropSize = int((down_block_out._keras_shape[2] - conv_up._keras_shape[2])/2)
        down_block_out_cropped = Lambda(crop_layer, output_shape=conv_up._keras_shape[1:], arguments={"cs":cropSize})(down_block_out)
        print "cropped layer size: ", down_block_out_cropped._keras_shape
        merged = merge([conv_up, down_block_out_cropped], mode='concat', concat_axis=1)

        print "merged ", merged._keras_shape
        # two 3x3 convolutions with ReLU
        # first one halves the feature channels
        act1 = Convolution2D(nb_filter=nb_filter, nb_row=3, nb_col=3, subsample=(1,1),
                                 init=initialization, activation='relu', border_mode="valid", W_regularizer=l2(weight_decay))(merged)

        if doBatchNorm:
            act1 = BatchNormalization(mode=0, axis=1)(act1)

        print "conv1 ", act1._keras_shape
        act2 = Convolution2D(nb_filter=nb_filter, nb_row=3, nb_col=3, subsample=(1,1),
                                 init=initialization, activation='relu', border_mode="valid", W_regularizer=l2(weight_decay))(act1)
        if doBatchNorm:
            act2 = BatchNormalization(mode=0, axis=1)(act2)


        print "conv2 ", act2._keras_shape
        
        return act2
        
