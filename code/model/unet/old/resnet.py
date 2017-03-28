# https://raw.githubusercontent.com/raghakot/keras-resnet/master/resnet.py

import os
from keras.optimizers import SGD
from keras.models import Model, model_from_json
from keras.layers import (
    Input,
    Activation,
    merge,
    Dense,
    Flatten
)
from keras.layers.convolutional import (
    Convolution2D,
    MaxPooling2D,
    AveragePooling2D
)
from keras.layers.normalization import BatchNormalization
from generate_data import *
import sys
import mahotas
import multiprocessing
import matplotlib
import matplotlib.pyplot as plt


# Helper to build a conv -> BN -> relu block
def _conv_bn_relu(nb_filter, nb_row, nb_col, subsample=(1, 1)):
    def f(input):
        conv = Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=subsample,
                             init="uniform", border_mode="same")(input)
        norm = BatchNormalization(mode=0, axis=1)(conv)
        return Activation("relu")(norm)

    return f


# Helper to build a BN -> relu -> conv block
# This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
def _bn_relu_conv(nb_filter, nb_row, nb_col, subsample=(1, 1)):
    def f(input):
        norm = BatchNormalization(mode=0, axis=1)(input)
        activation = Activation("relu")(norm)
        return Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=subsample,
                             init="uniform", border_mode="same")(activation)

    return f


# Bottleneck architecture for > 34 layer resnet.
# Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
# Returns a final conv layer of nb_filters * 4
def _bottleneck(nb_filters, init_subsample=(1, 1)):
    #print
    #print "this is _botteleneck"
    def f(input):
        #print "input ", input._keras_shape
        conv_1_1 = _bn_relu_conv(nb_filters, 1, 1, subsample=init_subsample)(input)
        #print "conv_1_1 ", conv_1_1._keras_shape
        conv_3_3 = _bn_relu_conv(nb_filters, 3, 3)(conv_1_1)
        #print "conv_3_3 ", conv_3_3._keras_shape
        residual = _bn_relu_conv(nb_filters * 4, 1, 1)(conv_3_3)
        #print "residual ", residual._keras_shape
        #print
        return _shortcut(input, residual)

    return f


# Basic 3 X 3 convolution blocks.
# Use for resnet with layers <= 34
# Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
def _basic_block(nb_filters, init_subsample=(1, 1)):
    def f(input):
        conv1 = _bn_relu_conv(nb_filters, 3, 3, subsample=init_subsample)(input)
        residual = _bn_relu_conv(nb_filters, 3, 3)(conv1)
        return _shortcut(input, residual)

    return f

# Basic 3 X 3 convolution blocks.
# Use for resnet with layers <= 34
# Follows old scheme from original paper
def _basic_block_old(nb_filters, init_subsample=(1, 1)):
    def f(input):
        conv1 = _conv_bn_relu(nb_filters, 3, 3, subsample=init_subsample)(input)
        residual = _conv_bn_relu(nb_filters, 3, 3)(conv1)
        return _shortcut(input, residual)

    return f


# Adds a shortcut between input and residual block and merges them with "sum"
def _shortcut(input, residual):
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    stride_width = int(np.ceil(input._keras_shape[2] / np.double(residual._keras_shape[2])))
    stride_height = int(np.ceil(input._keras_shape[3] / np.double(residual._keras_shape[3])))
    equal_channels = residual._keras_shape[1] == input._keras_shape[1]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Convolution2D(nb_filter=residual._keras_shape[1], nb_row=1, nb_col=1,
                                 subsample=(stride_width, stride_height),
                                 init="uniform", border_mode="valid")(input)

    # FIXME this should do zero padding. Other option is to use a 1x1 convolution
    return merge([shortcut, residual], mode="sum")


# Builds a residual block with repeating bottleneck blocks.
def _residual_block(block_function, nb_filters, repetations, is_first_layer=False):
    def f(input):
        for i in range(repetations):
            init_subsample = (1, 1)
            # first layer of each block subsamples by using strided convolution
            # only very first layer is an exception
            if i == 0 and not is_first_layer:
                init_subsample = (2, 2)
            input = block_function(nb_filters=nb_filters, init_subsample=init_subsample)(input)
        return input

    return f


# http://arxiv.org/pdf/1512.03385v1.pdf
# 34 Layer resnet from figure 3
# uses old scheme
def resnet_old():
    print 
    print "======="
    print "This is resnet with old scheme"
    input = Input(shape=(1, 65, 65))
    print "input: ", input._keras_shape
    # part of all residual networks. First layer downsamples by using stride two convolution
    # output should be half of the original input. I changed this to not subsample in the first
    # layer. We have less resolution to spare than for Imagenet object categorization
    conv1 = _conv_bn_relu(nb_filter=64, nb_row=5, nb_col=5, subsample=(1, 1))(input)
    print "conv1: ", conv1._keras_shape
    # next is max pooling. again output is halved
    #pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode="same")(conv1)
    #print "pool1: ", pool1._keras_shape
    # doing a strided convolution instead of max pooling to save computation time
    conv2 = _conv_bn_relu(nb_filter=64, nb_row=5, nb_col=5, subsample=(2, 2))(input)
    print "conv2: ", conv2._keras_shape

    # Build residual blocks without bottlenecks and following architecture from paper
    block_fn = _basic_block_old
    print "=== BLOCK 1 ==="
    # build the first block
    # output shape should not change here 
    block1 = _residual_block(block_fn, nb_filters=64, repetations=2, is_first_layer=True)(conv2)
    print "block1: ", block1._keras_shape
    print
    print "=== BLOCK 2 ==="
    # build second block
    # first layer of this block subsamples by factor of 2 using strided convolution
    # this means shortucut needs to adjust the dimension
    block2 = _residual_block(block_fn, nb_filters=128, repetations=2)(block1)
    print "block2: ", block2._keras_shape
    print
    print "=== BLOCK 3 ==="
    # build third block
    block3 = _residual_block(block_fn, nb_filters=256, repetations=2)(block2)
    print "block3: ", block3._keras_shape

    # Classifier block
    # not subsampling further, its already at only 9x9
    # pool2 = AveragePooling2D(pool_size=(7, 7), strides=(1, 1), border_mode="same")(block4)

    # flatten layer to feed into logistic regression
    flatten1 = Flatten()(block3)
    # FIXME really no hidden layer here?
    dense = Dense(output_dim=2, init="uniform", activation="softmax")(flatten1)

    model = Model(input=input, output=dense)

    print "======="
    print
    return model

def resnet_old_deep():
    print 
    print "======="
    print "This is resnet with old scheme"
    input = Input(shape=(1, 65, 65))
    print "input: ", input._keras_shape
    # part of all residual networks. First layer downsamples by using stride two convolution
    # output should be half of the original input. I changed this to not subsample in the first
    # layer. We have less resolution to spare than for Imagenet object categorization
    conv1 = _conv_bn_relu(nb_filter=64, nb_row=5, nb_col=5, subsample=(1, 1))(input)
    print "conv1: ", conv1._keras_shape
    # next is max pooling. again output is halved
    #pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode="same")(conv1)
    #print "pool1: ", pool1._keras_shape
    # doing a strided convolution instead of max pooling to save computation time
    conv2 = _conv_bn_relu(nb_filter=64, nb_row=5, nb_col=5, subsample=(2, 2))(input)
    print "conv2: ", conv2._keras_shape

    # Build residual blocks without bottlenecks and following architecture from paper
    block_fn = _basic_block_old
    print "=== BLOCK 1 ==="
    # build the first block
    # output shape should not change here 
    block1 = _residual_block(block_fn, nb_filters=64, repetations=6, is_first_layer=True)(conv2)
    print "block1: ", block1._keras_shape
    print
    print "=== BLOCK 2 ==="
    # build second block
    # first layer of this block subsamples by factor of 2 using strided convolution
    # this means shortucut needs to adjust the dimension
    block2 = _residual_block(block_fn, nb_filters=128, repetations=6)(block1)
    print "block2: ", block2._keras_shape
    print
    print "=== BLOCK 3 ==="
    # build third block
    block3 = _residual_block(block_fn, nb_filters=256, repetations=6)(block2)
    print "block3: ", block3._keras_shape

    # Classifier block
    # not subsampling further, its already at only 9x9
    # pool2 = AveragePooling2D(pool_size=(7, 7), strides=(1, 1), border_mode="same")(block4)

    # flatten layer to feed into logistic regression
    flatten1 = Flatten()(block3)
    # FIXME really no hidden layer here?
    dense = Dense(output_dim=2, init="uniform", activation="softmax")(flatten1)

    model = Model(input=input, output=dense)

    print "======="
    print
    return model


if __name__ == '__main__':
    train_samples = 50000
    val_samples = 10000
    learning_rate = 0.1
    rng = np.random.RandomState(7)
    
    doTrain = int(sys.argv[1])
    
    if doTrain:
        import time
        start = time.time()
        model = resnet_old()
        duration = time.time() - start
        print "{} s to make model".format(duration)
    
        start = time.time()
        model.output
        duration = time.time() - start
        print "{} s to get output".format(duration)
        
        start = time.time()
        sgd = SGD(lr=0.0001, decay=0, momentum=0.0, nesterov=False)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd)
        duration = time.time() - start
        print "{} s to get compile".format(duration)
        
        data_val = generate_experiment_data_supervised(purpose='validate', nsamples=val_samples, patchSize=65, balanceRate=0.5, rng=rng)
        
        data_x_val = data_val[0].astype(np.float32)
        data_x_val = np.reshape(data_x_val, [-1, 1, 65, 65])
        data_y_val = data_val[1].astype(np.float32)
        
        # start pool for data
        print "Starting worker."
        pool = multiprocessing.Pool(processes=1)
        futureData = pool.apply_async(stupid_map_wrapper, [[generate_experiment_data_supervised,'train', train_samples, 65, 0.5, rng]])
        
        best_val_loss_so_far = 100
        
        for epoch in xrange(10000):
            print "Waiting for data."
            data = futureData.get()
            
            data_x = data[0].astype(np.float32)
            data_x = np.reshape(data_x, [-1, 1, 65, 65])
            data_y = data[1].astype(np.float32)
            
            print "got new data"
            futureData = pool.apply_async(stupid_map_wrapper, [[generate_experiment_data_supervised, 'train', train_samples, 65, 0.5, rng]])
            
            model.fit(data_x, data_y, batch_size=100, nb_epoch=1)
        
            validation_loss = model.evaluate(data_x_val, data_y_val, batch_size=100)
            print "validation loss ", validation_loss
            
            json_string = model.to_json()
            open('resnet_keras.json', 'w').write(json_string)
            model.save_weights('resnet_keras_weights.h5', overwrite=True) 
            
            if validation_loss < best_val_loss_so_far:
                best_val_loss_so_far = validation_loss
                print "NEW BEST MODEL"
                json_string = model.to_json()
                open('resnet_keras_best.json', 'w').write(json_string)
                model.save_weights('resnet_keras_best_weights.h5', overwrite=True) 
                
    else:
        model = model_from_json(open('resnet_keras.json').read())
        model.load_weights('resnet_keras_weights.h5')
        
        sgd = SGD(lr=learning_rate, decay=0, momentum=0.0, nesterov=False)
        # this is summed, not averaged loss => need to adjust learning rate with batch size
        model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd)
        
        image = mahotas.imread('ac3_input_0141.tif')
        image = image[:512,:512]
        prob_img = np.zeros(image.shape)
        
        start_time = time.clock()
        for rows in xrange(image.shape[0]):
            patch_data = generate_image_data(image, patchSize=65, rows=[rows]).astype(np.float32)
            patch_data = np.reshape(patch_data, [-1, 1, 65, 65])
            probs = model.predict(x=patch_data, batch_size = image.shape[0])[:,0]
            prob_img[rows,:] = probs
            
            if rows%10==0:
                print rows
                print "time so far: ", time.clock()-start_time
                
        mahotas.imsave('keras_prediction_resnet_08.png', np.uint8(prob_img*255))
        
        plt.imshow(prob_img)
        plt.show()
        







