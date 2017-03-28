import os
import sys
import time

import numpy as np
import theano
import theano.tensor as T

base_path = os.path.dirname(__file__)
sys.path.insert(1,os.path.join(base_path, '../../common'))
sys.path.insert(2,os.path.join(base_path, '../../database'))
#sys.path.insert(4,os.path.join(base_path, '../model'))

from paths import Paths
from project import Project
from db import DB
from cnn import CNN
from datasets import DataSets


if __name__ == '__main__':

	project = DB.getProject('testcnn')

	x = T.matrix('x')

	rng = np.random.RandomState(929292)

	print project.batchSize, project.patchSize, project.nKernels
	print project.kernelSizes, project.hiddenUnits, project.trainTime,
	print project.path
	print project.type
	print project.id
	cnn = CNN(
		rng=rng, 
		input=x,
		batch_size=project.batchSize,
		patch_size=project.patchSize,
		nkernels=project.nKernels,
		kernel_sizes=project.kernelSizes,
		hidden_sizes=project.hiddenUnits,
		train_time=project.trainTime,
		path=project.path,
		type=project.type,
		id=project.id)

	print 'loading data...'
	dataset = DataSets()
	dataset.load( project )

	cnn.setTrainingData(
		x=dataset.x, 
		y=dataset.y, 
		p=dataset.p,
		learning_rate=project.learningRate,
		momentum=project.momentum )

	for i in range(10):
		print 'training...', i
		cnn.train()

	#evaluate_lenet5(filename=None)
