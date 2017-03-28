# yet another version of the IDSIA network
# based on code from keras tutorial 
# http://keras.io/getting-started/sequential-model-guide/
from keras.models import Model, Sequential, model_from_json
from keras.layers import Dense, Activation, Flatten, Input
from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D, merge, ZeroPadding2D, Dropout, Lambda
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

rng = np.random.RandomState(7)

train_samples = 20 
val_samples = 10
learning_rate = 0.01
momentum = 0.95
doTrain = int(sys.argv[1])

patchSize = 572 #140
patchSize_out = 388 #132

weight_decay = 0.0
weight_class_1 = 1.

patience = 10

purpose = 'train'
nr_layers = 3
initialization = 'glorot_uniform'
filename = 'unet_3d'
print "filename: ", filename

srng = RandomStreams(1234)

# need to define a custom loss, because all pre-implementations
# seem to assume that scores over patch add up to one which
# they clearly don't and shouldn't
def unet_crossentropy_loss(y_true, y_pred):
    epsilon = 1.0e-4
    y_pred_clipped = T.clip(y_pred, epsilon, 1.0-epsilon)
    loss_vector = -T.mean(weight_class_1*y_true * T.log(y_pred_clipped) + (1-y_true) * T.log(1-y_pred_clipped), axis=1)
    average_loss = T.mean(loss_vector)
    return average_loss

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
    # subset assuming each class has at least 100 samples present
    indPos = indPos[:200]
    indNeg = indNeg[:200]
    loss_vector = -T.mean(T.log(y_pred_clipped[indPos])) - T.mean(T.log(1-y_pred_clipped[indNeg]))
    average_loss = T.mean(loss_vector)
    return average_loss

def unet_block_down(input, nb_filter, doPooling=True, doDropout=False):
    # first convolutional block consisting of 2 conv layers plus activation, then maxpool.
    # All are valid area, not same
    act1 = Convolution2D(nb_filter=nb_filter, nb_row=3, nb_col=3, subsample=(1,1),
                             init=initialization, activation='relu',  border_mode="valid", W_regularizer=l2(weight_decay))(input)
    act2 = Convolution2D(nb_filter=nb_filter, nb_row=3, nb_col=3, subsample=(1,1),
                             init=initialization, activation='relu', border_mode="valid", W_regularizer=l2(weight_decay))(act1)
    

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
def crop_layer(x, cs):
    cropSize = cs
    return x[:,:,cropSize:-cropSize, cropSize:-cropSize]


def unet_block_up(input, nb_filter, down_block_out):
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
    print "conv1 ", act1._keras_shape
    act2 = Convolution2D(nb_filter=nb_filter, nb_row=3, nb_col=3, subsample=(1,1),
                             init=initialization, activation='relu', border_mode="valid", W_regularizer=l2(weight_decay))(act1)
    print "conv2 ", act2._keras_shape
    
    return act2
    

if doTrain:
    # input data should be large patches as prediction is also over large patches
    print 
    print "=== building network ==="

    print "== BLOCK 1 =="
    input = Input(shape=(nr_layers, patchSize, patchSize))
    print "input ", input._keras_shape
    block1_act, block1_pool = unet_block_down(input=input, nb_filter=64)
    print "block1 act ", block1_act._keras_shape
    print "block1 ", block1_pool._keras_shape

    print "== BLOCK 2 =="
    block2_act, block2_pool = unet_block_down(input=block1_pool, nb_filter=128)
    print "block2 ", block2_pool._keras_shape

    print "== BLOCK 3 =="
    block3_act, block3_pool = unet_block_down(input=block2_pool, nb_filter=256)
    print "block3 ", block3_pool._keras_shape

    print "== BLOCK 4 =="
    block4_act, block4_pool = unet_block_down(input=block3_pool, nb_filter=512, doDropout=True)
    print "block4 ", block4_pool._keras_shape

    print "== BLOCK 5 =="
    print "no pooling"
    block5_act, block5_pool = unet_block_down(input=block4_pool, nb_filter=1024, doDropout=True, doPooling=False)
    print "block5 ", block5_pool._keras_shape

    print "=============="
    print

    print "== BLOCK 4 UP =="
    block4_up = unet_block_up(input=block5_act, nb_filter=512, down_block_out=block4_act)
    print "block4 up", block4_up._keras_shape
    print

    print "== BLOCK 3 UP =="
    block3_up = unet_block_up(input=block4_up, nb_filter=256, down_block_out=block3_act)
    print "block3 up", block3_up._keras_shape
    print

    print "== BLOCK 2 UP =="
    block2_up = unet_block_up(input=block3_up, nb_filter=128, down_block_out=block2_act)
    print "block2 up", block2_up._keras_shape

    print
    print "== BLOCK 1 UP =="
    block1_up = unet_block_up(input=block2_up, nb_filter=64, down_block_out=block1_act)
    print "block1 up", block1_up._keras_shape

    print "== 1x1 convolution =="
    output = Convolution2D(nb_filter=1, nb_row=1, nb_col=1, subsample=(1,1),
                             init=initialization, activation='sigmoid', border_mode="valid")(block1_up)
    print "output ", output._keras_shape
    output_flat = Flatten()(output)
    print "output flat ", output_flat._keras_shape
    model = Model(input=input, output=output_flat)
    #model = Model(input=input, output=block1_act)
    sgd = SGD(lr=learning_rate, decay=0, momentum=momentum, nesterov=False)
    #model.compile(loss='mse', optimizer=sgd)
    model.compile(loss=unet_crossentropy_loss_sampled, optimizer=sgd)
    data_val = generate_experiment_data_patch_prediction_layers(purpose='validate', nsamples=val_samples, patchSize=patchSize, outPatchSize=patchSize_out, nr_layers=nr_layers)
   
    data_x_val = data_val[0].astype(np.float32)
    data_x_val = np.reshape(data_x_val, [-1, nr_layers, patchSize, patchSize])
    data_y_val = data_val[1].astype(np.float32)
    data_label_val = data_val[2]

    # start pool for data
    print "Starting worker."
    pool = multiprocessing.Pool(processes=1)
    futureData = pool.apply_async(stupid_map_wrapper, [[generate_experiment_data_patch_prediction_layers, purpose, train_samples, patchSize, patchSize_out, nr_layers]])
    
    best_val_loss_so_far = 0
    
    patience_counter = 0
    for epoch in xrange(10000000):
        print "Waiting for data."
        data = futureData.get()
        
        data_x = data[0].astype(np.float32)
        data_x = np.reshape(data_x, [-1, nr_layers, patchSize, patchSize])
        data_y = data[1].astype(np.float32)
        
        print "got new data"
        futureData = pool.apply_async(stupid_map_wrapper, [[generate_experiment_data_patch_prediction_layers, purpose, train_samples, patchSize, patchSize_out, nr_layers]])
 
        print "current learning rate: ", model.optimizer.lr.get_value()
        model.fit(data_x, data_y, batch_size=1, nb_epoch=1)


        im_pred = 1-model.predict(x=data_x_val, batch_size = 1)

        mean_val_rand = 0
        for val_ind in xrange(val_samples):
            im_pred_single = np.reshape(im_pred[val_ind,:], (patchSize_out,patchSize_out))
            im_gt = np.reshape(data_label_val[val_ind], (patchSize_out,patchSize_out))
            validation_rand = Rand_membrane_prob(im_pred_single, im_gt)
            mean_val_rand += validation_rand
        mean_val_rand /= np.double(val_samples)
        print "validation RAND ", mean_val_rand
        
        json_string = model.to_json()
        open(filename+'.json', 'w').write(json_string)
        model.save_weights(filename+'_weights.h5', overwrite=True) 
        
        print mean_val_rand, " > ",  best_val_loss_so_far
        print mean_val_rand - best_val_loss_so_far
        if mean_val_rand > best_val_loss_so_far:
            best_val_loss_so_far = mean_val_rand
            print "NEW BEST MODEL"
            json_string = model.to_json()
            open(filename+'_best.json', 'w').write(json_string)
            model.save_weights(filename+'_best_weights.h5', overwrite=True) 
            patience_counter=0
        else:
            patience_counter +=1

        # no progress anymore, need to decrease learning rate
        if patience_counter == patience:
            print "DECREASING LEARNING RATE"
            print "before: ", learning_rate
            learning_rate *= 0.1
            print "now: ", learning_rate
            model.optimizer.lr.set_value(learning_rate)
            patience = 10
            patience_counter = 0
        
        # stop if not learning anymore
        if learning_rate < 1e-7:
            break

else:
    start_time = time.clock()

    network_file_path = 'to_evaluate/'
    file_search_string = network_file_path + '*.json'
    files = sorted( glob.glob( file_search_string ) )
    pathPrefix = '/media/vkaynig/Data1/all_data/testing/AC4_small/'
    #pathPrefix = '/media/vkaynig/Data1/all_data/testing/AC4/'

    for file_index in xrange(np.shape(files)[0]):
        print files[file_index]
        model = model_from_json(open(files[file_index]).read())
        weight_file = ('.').join(files[file_index].split('.')[:-1])
        model.load_weights(weight_file+'_weights.h5')
        model_name = os.path.splitext(os.path.basename(files[file_index]))[0]
        
        # create directory
        if not os.path.exists(pathPrefix+'boundaryProbabilities/'+model_name):
            os.makedirs(pathPrefix+'boundaryProbabilities/'+model_name)

        sgd = SGD(lr=0.01, decay=0, momentum=0.0, nesterov=False)
        model.compile(loss='categorical_crossentropy', optimizer=sgd)
        
        img_search_string = pathPrefix + 'gray_images/*.tif'
        img_files = sorted( glob.glob( img_search_string ) )
        
        for img_index in xrange(np.shape(img_files)[0]):
            print img_files[img_index]
            img_cs = int(np.floor(nr_layers/2))
            img_valid_range_indices = np.clip(range(img_index-img_cs,img_index+img_cs+1),0,np.shape(img_files)[0]-1)

            padding_ul = int(np.ceil((patchSize - patchSize_out)/2.0))
            needed_ul_padding = patchSize - padding_ul

            image = mahotas.imread(img_files[0])
            # need large padding for lower right corner
            paddedImage = np.pad(image, patchSize, mode='reflect')
            paddedImage = paddedImage[needed_ul_padding:, needed_ul_padding:]
            paddedSize = paddedImage.shape

            layer_image = np.zeros((nr_layers,paddedSize[0],paddedSize[1]))
            for ind, read_index in enumerate(img_valid_range_indices):
                image = mahotas.imread(img_files[read_index])
                image = normalizeImage(image) 
                image = image - 0.5
                paddedImage = np.pad(image, patchSize, mode='reflect')
                paddedImage = paddedImage[needed_ul_padding:, needed_ul_padding:]
                layer_image[ind] = paddedImage

            probImage = np.zeros(image.shape)
            # count compilation time to init
            row = 0
            col = 0
            patch = layer_image[:,row:row+patchSize,col:col+patchSize]
            data = np.reshape(patch, (1,nr_layers,patchSize,patchSize))
            probs = model.predict(x=data, batch_size=1)
            
            init_time = time.clock()
            #print "Initialization took: ", init_time - start_time
                            
            probImage_tmp = np.zeros(image.shape)
            for row in xrange(0,image.shape[0],patchSize_out):
                for col in xrange(0,image.shape[1],patchSize_out):
                    patch = layer_image[:,row:row+patchSize,col:col+patchSize]
                    data = np.reshape(patch, (1,nr_layers,patchSize,patchSize))
                    probs = 1-model.predict(x=data, batch_size = 1)
                    probs = np.reshape(probs, (patchSize_out,patchSize_out))
                    
                    row_end = patchSize_out
                    if row+patchSize_out > probImage.shape[0]:
                        row_end = probImage.shape[0]-row
                    col_end = patchSize_out
                    if col+patchSize_out > probImage.shape[1]:
                        col_end = probImage.shape[1]-col
                        
                    probImage_tmp[row:row+row_end,col:col+col_end] = probs[:row_end,:col_end]
            probImage = probImage_tmp
            
            print pathPrefix+'boundaryProbabilities/'+model_name+'/'+str(img_index).zfill(4)+'.tif'
            mahotas.imsave(pathPrefix+'boundaryProbabilities/'+model_name+'/'+str(img_index).zfill(4)+'.tif', np.uint8(probImage*255))
            
            end_time = time.clock()
            print "Prediction took: ", end_time - init_time
            print "Speed: ", 1./(end_time - init_time)
            print "Time total: ", end_time-start_time
            
            
            print "min max output ", np.min(probImage), np.max(probImage)

