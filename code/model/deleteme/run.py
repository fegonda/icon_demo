#---------------------------------------------------------------------------
# Utility.py
#
# Author  : Felix Gonda
# Date    : July 10, 2015
# School  : Harvard University
#
# Project : Master Thesis
#           An Interactive Deep Learning Toolkit for
#           Automatic Segmentation of Images
#
# Summary : This file contains utility functions for reading, writing, and
#           processing images.
#---------------------------------------------------------------------------

import os
import sys
import time
import ConfigParser
import pandas as pd
import numpy
import numpy as np
import theano
import theano.tensor as T
import cPickle

theano.config.floatX = 'float32'

base_path = os.path.dirname(__file__)
sys.path.insert(1,os.path.join(base_path, '../external'))
sys.path.insert(2,os.path.join(base_path, '../common'))
sys.path

from logistic_sgd import LogisticRegression
#from mlp import HiddenLayer
#from mlp import MLP
from lenet import LeNetConvPoolLayer
from cnn_model import CNN_Model
from cnn_classifier import CNN_Classifier
from settings import Settings
from utility import Utility
from paths import Paths
from datasets import DataSets
from database import Database

testimage = '../../data/images/checkerboard.tif'

if __name__ == '__main__':
    rng = numpy.random.RandomState(929292)

    import mahotas
    import matplotlib.pyplot as plt
    image = mahotas.imread(testimage)

    x = T.matrix('x')

    doDebug = False

    classifier = CNN_Classifier()

    project = 'testcnn'
    settings = Settings.get( project )
    print settings.learning_rate

    dataset = DataSets(project)
    has_new_data = Database.hasTrainingTasks( project )
    if has_new_data:
	print 'has new data'
	if not dataset.load_training(settings.sample_size):
		exit(1)

    # cache the dataset
    if not dataset.valid():
	print 'bad data'
	exit(1)

    print 'good data'
    print dataset.y
    print dataset.x
    print dataset.p

    classifier.train(
		dataset.x,
		dataset.y,
		dataset.p,
                True,
                len(settings.labels),
                settings.learning_rate)

    print 'done running...'
    exit(1)
