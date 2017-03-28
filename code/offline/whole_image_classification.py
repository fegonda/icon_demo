import mahotas
import matplotlib.pyplot as plt
import numpy as np
import theano
import theano.tensor as T

from cnn import *
from mlp import HiddenLayer, MLP, rectified_linear
from generateTrainValTestData import generate_experiment_data_supervised, shared_dataset, normalizeImage, stupid_map_wrapper
from scipy.ndimage.interpolation import shift

import time

def get_max_pool_frag(convolved_image, offset1, offset2):
    image_width = convolved_image.shape[2]
    image_height = convolved_image.shape[3]
    
    convolved_image_shared = convolved_image[:,:,offset1:,offset2:]
    convolved_image_shared = convolved_image_shared.reshape(convolved_image_shared.shape.eval())

    pooled_out = downsample.max_pool_2d(
        input=convolved_image_shared,
        ds=(2,2),
        ignore_border=True
    )

    return pooled_out
    

def get_max_pool_fragments(convolved_image, clayer):
    start_time = time.clock()
    b = clayer.b.dimshuffle('x', 0, 'x', 'x')
    max_pooled_0_0 = get_max_pool_frag(convolved_image=convolved_image, offset1=0, offset2=0)
    out_0_0 = rectified_linear(max_pooled_0_0 + b)

    max_pooled_0_1 = get_max_pool_frag(convolved_image=convolved_image, offset1=0, offset2=1)
    out_0_1 = rectified_linear(max_pooled_0_1 + b)

    max_pooled_1_0 = get_max_pool_frag(convolved_image=convolved_image, offset1=1, offset2=0)
    out_1_0 = rectified_linear(max_pooled_1_0 + b)

    max_pooled_1_1 = get_max_pool_frag(convolved_image=convolved_image, offset1=1, offset2=1)
    out_1_1 = rectified_linear(max_pooled_1_1 + b)

    return (out_0_0, out_0_1, out_1_0, out_1_1)


def basic_convolution(image_shared, filterMap):
    conv_out = conv.conv2d(
        input=image_shared,
        filters=filterMap,
    )

    return conv_out


def get_convolution_output(image_shared, clayer):
    output = basic_convolution(image_shared, clayer.W)
    return output

def get_hidden_output(image_shared, hiddenLayer, nHidden, nfilt):
    W = hiddenLayer.W
    patchSize = np.int16(np.sqrt(W.shape.eval()[0] / np.double(nfilt)))
    W = np.rollaxis(W,1)
    W = W.reshape((nHidden,nfilt,patchSize,patchSize))

    b = hiddenLayer.b

    #flip kernel for convolution
    output = rectified_linear(basic_convolution(image_shared, W[:,:,::-1,::-1]) + b.dimshuffle('x', 0, 'x', 'x'))

    return output


def get_logistic_regression_output(image_shared, logregLayer):
    output_shape = image_shared.shape

    W_lreg = logregLayer.W
    W_shape = W_lreg.shape
    W_lreg = np.rollaxis(W_lreg, 1)
    W_lreg = W_lreg.reshape((W_shape[1],W_shape[0],1,1))

    b_lreg = logregLayer.b

    # flip kernel for convolution
    output = basic_convolution(image_shared, W_lreg[:,:,::-1,::-1]) + b_lreg.dimshuffle('x', 0, 'x', 'x')

    output =  T.addbroadcast(output, 0)
    output = output.squeeze()
    output = output.flatten(2)
    output = T.nnet.softmax(output.T).T

    return output.reshape((2,output_shape[2], output_shape[3]))

    
if __name__ == '__main__':
    rng = numpy.random.RandomState(929292)

    import mahotas
    import matplotlib.pyplot as plt
    #CPU
    #image = mahotas.imread('ac3_input_0141.tif')
    image = mahotas.imread('test-input-1.tif')
    imageSize = 1024
    image = image[0:imageSize,0:imageSize]

    start_time = time.clock()
    image = normalizeImage(image) - 0.5
    
    #GPU
    image_shared = theano.shared(np.float32(image))
    image_shared = image_shared.reshape((1,1,imageSize,imageSize))

    x = T.matrix('x')

    classifier = CNN(input=x, batch_size=imageSize, patchSize=65, rng=rng, nkerns=[48,48,48], kernelSizes=[5,5,5], hiddenSizes=[200], fileName='tmp_cnn.pkl')
    
    fragments = [image_shared]

    print "Convolutions"

    for clayer in classifier.convLayers:
        newFragments = []
        for img_sh in fragments:
            # CPU???
            convolved_image = get_convolution_output(image_shared=img_sh, clayer=clayer)
            #GPU
            output = get_max_pool_fragments(convolved_image, clayer=clayer)
            newFragments.extend(output)

        fragments = newFragments

    #### now the hidden layer
    
    print "hidden layer"

    hidden_fragments = []

    for fragment in fragments:
        hidden_out = get_hidden_output(image_shared=fragment, hiddenLayer=classifier.mlp.hiddenLayers[0], nHidden=200, nfilt=classifier.nkerns[-1])
        hidden_fragments.append(hidden_out)

    ### VERIFIED CORRECT UNTIL HERE

    #### and the missing log reg layer

    print "logistic regression layer"

    final_fragments = []
    for fragment in hidden_fragments:
        logreg_out = get_logistic_regression_output(image_shared=fragment, logregLayer=classifier.mlp.logRegressionLayer)
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

    prob_img = shift(prob_img,(32,32))

    plt.imshow(prob_img)
    plt.show()
    #mahotas.imsave('syn_tmp_output_cnn_12.png', np.uint8((prob_img)*255))
    mahotas.imsave('membrane_thin.png', np.uint8((prob_img)*255))
        
    cl = classifier.convLayers[0]
    plt.imshow(cl.visualize_filters())
    plt.show()
    #mahotas.imsave('syn_filter_output_cnn_12.png', np.uint8(cl.visualize_filters()))
    
    plt.plot(np.array(classifier.trainingCost), label='training')
    plt.plot(np.array(classifier.validationError), label='validation')
    plt.legend()
    plt.show()
    
    with open('before.pkl', 'r') as file:
        _, _, _, trainingCost, validationError, _, _, _, _, _ = cPickle.load(file)

    plt.plot(np.array(classifier.validationError), label='new')
    plt.plot(np.array(validationError), label='old')
    plt.legend()
    plt.show()
    

    if len(classifier.validationError) > 5000:
        plt.plot(np.array(classifier.trainingCost)[-5000:], label='training')
        plt.plot(np.array(classifier.validationError)[-5000:], label='validation')
        plt.legend()
        plt.show()
        
        
    print "best validation score: ", classifier.validationError[-1]

    
 
    

