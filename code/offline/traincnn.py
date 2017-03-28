import numpy
import theano
import theano.tensor as T

from cnn import *
if __name__ == '__main__':
 
	#def evaluate_lenet5(learning_rate=0.0001, n_epochs=5, nkerns=[10,10,10], kernelSizes=[5,5,5], hiddenSizes=[200], doResample=False, batch_size=100, patchSize=39, train_samples=10000, val_samples=1000, test_samples=1000, validation_frequency = 1, doEmailUpdate=False, momentum=0.98, filename='tmp_cnn.pkl'):
 

	evaluate_lenet5(
		learning_rate=0.1,   # 0.05  (make learning rate tunable)
		n_epochs=100,        # 
		nkerns=[48,48],   # try (2 conv layers) reducing layers or the number of kernels
		kernelSizes=[5,5], # >= 5
		hiddenSizes=[200],   #
		batch_size=16,      #
		patchSize=39, #65,        # 39, 65
		train_samples=10000,
		val_samples=1000,
		test_samples=1000,
		filename=None)
