# yet another version of the IDSIA network
# based on code from keras tutorial 
# http://keras.io/getting-started/sequential-model-guide/
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.initializations import uniform
from keras import backend as K
from generate_data import *
import multiprocessing
import sys
import matplotlib
import matplotlib.pyplot as plt

rng = np.random.RandomState(7)
train_samples = 500
val_samples = 100

learning_rate = 0.1
patience = 2

doTrain = int(sys.argv[1])
filename = 'thick_membranes'

if doTrain:
    model = Sequential()
    # input: 100x100 images with 3 channels -> (3, 100, 100) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    # layer 1
    model.add(Convolution2D(48, 5, 5, border_mode='valid', activation='relu', input_shape=(1, 65, 65), init='uniform'))
    W = model.layers[-1].get_weights()
    #model.layers[-1].set_weights([W[0]/2.0, W[1]])
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # layer 2
    model.add(Convolution2D(48, 5, 5, border_mode='valid', activation='relu', init='uniform'))
    #W = model.layers[-1].get_weights()
    #model.layers[-1].set_weights([W[0]/2.0, W[1]])
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # layer 3
    model.add(Convolution2D(48, 5, 5, border_mode='valid', activation='relu', init='uniform'))
    #W = model.layers[-1].get_weights()
    #model.layers[-1].set_weights([W[0]/2.0, W[1]])
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    
    # fc layer 1
    model.add(Dense(200, activation='relu', init='uniform'))
    # fc layer 2 to two classes
    model.add(Dense(2, init='uniform', activation='softmax'))
    
    sgd = SGD(lr=learning_rate, decay=0, momentum=0.0, nesterov=False)
    #<<<<<<< HEAD
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    #model.compile(loss=categorical_crossentropy_int, optimizer=sgd)
    #=======
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd)
    #>>>>>>> d487df725f2e135e439bfe443dc3c8f1a19188b6

    data_val = generate_experiment_data_supervised(purpose='validate', nsamples=val_samples, patchSize=65, balanceRate=0.5, rng=rng)
    
    data_x_val = data_val[0].astype(np.float32)
    data_x_val = np.reshape(data_x_val, [-1, 1, 65, 65])
    data_y_inds = data_val[1].astype(np.int64)
    data_y_val = np.zeros((data_y_inds.shape[0],2))
    data_y_val[np.arange(data_y_inds.shape[0]),data_y_inds] = 1
    
    # start pool for data
    print "Starting worker."
    pool = multiprocessing.Pool(processes=1)
    futureData = pool.apply_async(stupid_map_wrapper, [[generate_experiment_data_supervised,'train', train_samples, 65, 0.5, rng]])
    
    best_val_loss_so_far = 100
    patience_counter = 0

    for epoch in xrange(1000000000):
        print "Waiting for data."
        data = futureData.get()
        #data = generate_experiment_data_supervised(purpose='train', nsamples=1000, patchSize=65, balanceRate=0.5, rng=np.random)
        
        data_x = data[0].astype(np.float32)
        data_x = np.reshape(data_x, [-1, 1, 65, 65])
        # need one hot encoding for keras
        data_y_inds = data[1].astype(np.int64)
        data_y = np.zeros((data_y_inds.shape[0],2))
        data_y[np.arange(data_y_inds.shape[0]),data_y_inds] = 1

        print "got new data"
        futureData = pool.apply_async(stupid_map_wrapper, [[generate_experiment_data_supervised, 'train', train_samples, 65, 0.5, rng]])
        
        model.fit(data_x, data_y, batch_size=100, nb_epoch=1)
        
        print "current learning rate: ", model.optimizer.lr.get_value()
        model.fit(data_x, data_y, batch_size=100, nb_epoch=1)

        validation_loss = model.evaluate(data_x_val, data_y_val, batch_size=100)
        print "validation loss ", validation_loss
        
        json_string = model.to_json()
        open(filename+'_simple_cnn.json', 'w').write(json_string)
        model.save_weights(filename+'_simple_cnn_weights.h5', overwrite=True) 
        
        if validation_loss < best_val_loss_so_far:
            best_val_loss_so_far = validation_loss
            print "NEW BEST MODEL"
            json_string = model.to_json()
            open(filename+'_simple_cnn_best.json', 'w').write(json_string)
            model.save_weights(filename+'simple_cnn_best_weights.h5', overwrite=True) 
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
            patience = 40
            patience_counter = 0
        
        # stop if not learning anymore
        if learning_rate < 1e-7:
            break

else:
    model = model_from_json(open('simple_cnn_keras.json').read())
    model.load_weights('simple_cnn_keras_weights.h5')
    
    sgd = SGD(lr=0.01, decay=0, momentum=0.0, nesterov=False)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd)
    
    image = mahotas.imread('ac3_input_0141.tif')
    image = image[:256,:256]
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
            
    mahotas.imsave('keras_prediction_cnn_16.png', np.uint8(prob_img*255))
    
    plt.imshow(prob_img)
    plt.show()
        
