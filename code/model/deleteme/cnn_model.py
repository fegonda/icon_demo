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
from mlp import HiddenLayer
from mlp_model import MLP_Model
from lenet import LeNetConvPoolLayer
from activation_functions import rectified_linear

class CNN_Model(object):
    def __init__(self,
		input, 
		batch_size, 
		patchSize, 
		rng, 
		nkerns, 
		kernelSizes, 
		hiddenSizes, 
		fileName=None, 
		activation=rectified_linear):
        
        self.convLayers = []
        self.trainingCost = []
        self.validationError = []
        self.nkerns = nkerns 
        self.kernelSizes = kernelSizes 
        self.hiddenSizes = hiddenSizes

        self.patchSize = patchSize
        self.batch_size = batch_size

        input = input.reshape((self.batch_size, 1, self.patchSize, self.patchSize))

        self.layer0_input = input 
        self.params = [] 
        
        input_next = input
        numberOfFeatureMaps = 1
        featureMapSize = patchSize
        
        for i in range(len(nkerns)):
            layer = LeNetConvPoolLayer(
                rng,
                input=input_next,
                image_shape=(batch_size, numberOfFeatureMaps, featureMapSize, featureMapSize),
                filter_shape=(nkerns[i], numberOfFeatureMaps, kernelSizes[i], kernelSizes[i]),
                poolsize=(2, 2)
            )
            input_next = layer.output
            numberOfFeatureMaps = nkerns[i]
            featureMapSize = np.int16(np.floor((featureMapSize - kernelSizes[i]+1) / 2))
            
            self.params += layer.params
            self.convLayers.append(layer)
            
        # the 2 is there to preserve the batchSize
        mlp_input = self.convLayers[-1].output.flatten(2)
       
	self.mlp = MLP_Model(
                        rng=rng,
                        input=mlp_input,
                        n_in=nkerns[-1] * (featureMapSize ** 2),
                        n_hidden=hiddenSizes,
                        n_out=2,
			activation=rectified_linear
                    )
 
        self.params += self.mlp.params
                
        self.cost = self.mlp.negative_log_likelihood 
        self.errors = self.mlp.errors
        self.p_y_given_x = self.mlp.p_y_given_x
	self.y_pred = self.mlp.y_pred
        self.debug_x = self.p_y_given_x


        if not fileName is None:
            with open(fileName, 'r') as file:
                saved_convLayers, 
		saved_hiddenLayers, 
		saved_logRegressionLayer, 
		self.trainingCost, 
		self.validationError, 
		saved_nkerns, 
		saved_kernelSizes, 
		saved_batch_size, 
		saved_patchSize, 
		saved_hiddenSizes = cPickle.load(file)

            for s_cl, cl in zip(saved_convLayers, self.convLayers):
                cl.W.set_value(s_cl.W.get_value())
                cl.b.set_value(s_cl.b.get_value())

            for s_hl, hl in zip(saved_hiddenLayers, self.mlp.hiddenLayers):
                hl.W.set_value(np.float32(s_hl.W.eval()))
                hl.b.set_value(s_hl.b.get_value())
            
            self.mlp.logRegressionLayer.W.set_value(np.float32(saved_logRegressionLayer.W.eval()))
            self.mlp.logRegressionLayer.b.set_value(saved_logRegressionLayer.b.get_value())


    def save(self, filename):
        with open(filename, 'wb') as file:
            cPickle.dump((self.convLayers, 
		self.mlp.hiddenLayers, 
		self.mlp.logRegressionLayer, 
		self.trainingCost, 
		self.validationError, 
		self.nkerns, 
		self.kernelSizes, 
		self.batch_size, 
		self.patchSize, 
		self.hiddenSizes), file)

