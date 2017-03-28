import numpy
import theano
import theano.tensor as T

from mlpv import *
if __name__ == '__main__':
  
    '''
    train_mlp(learning_rate=0.01, n_epochs=10, 
	batch_size=500, n_hidden=[500], patchSize=19, 
	trainsamples=1000, val_samples=10000, test_samples=1000, 
	doResample=False, validation_frequency = 1, activation=rectified_linear, 
	doEmailUpdate=False, momentum=0.0):
    '''

    
    train_mlp(
    learning_rate=0.01,
    batch_size=500,
	n_hidden=[500,500,500],
	n_epochs=10000,
	train_samples=100000,
	patchSize=39)  

    # cnn patchsIZE= 65

    '''

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
    '''
