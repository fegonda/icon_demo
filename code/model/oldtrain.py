#---------------------------------------------------------------------------
# prediction_task.py
#
# Author  : Felix Gonda
# Date    : July 10, 2015
# School  : Harvard University
#
# Project : Master Thesis
#           An Interactive Deep Learning Toolkit for
#           Automatic Segmentation of Images
#
# Summary : This file contains the implementation of a prediction task
#           thread.  The prediction task is responsible for segmenting
#           an entire image based a trained model.  It loads the trained
#           model from file.
#---------------------------------------------------------------------------

import os
import sys
import signal
import threading
import time
import numpy as np

base_path = os.path.dirname(__file__)
sys.path.insert(1,os.path.join(base_path, '../common'))

from mlp_classifier import MLP_Classifier
from cnn_classifier import CNN_Classifier
from utility import Utility
from settings import Settings
from paths import Paths
from datasets import DataSets
from database import Database


#---------------------------------------------------------------------------
class TrainingManager(threading.Thread):

        #-------------------------------------------------------------------
        # Reads the input image and normalize it.  Also creates
        # a version of the input image with corner pixels
        # mirrored based on the sample size
        # arguments: - path : directory path where image is located
        #            - id   : the unique identifier of the image
        #            - pad  : the amount to pad the image by.
        # return   : returns original image and a padded version
        #-------------------------------------------------------------------
	def __init__(self):
		threading.Thread.__init__(self)
		self.waittime    = 1.0

		# one per project
		self.projectId   = None
		self.classifiers = None
		self.dataset     = None
		self.done        = False

	#-------------------------------------------------------------------
	# Create a classifier for the specified project
	#-------------------------------------------------------------------
	def create_classifier(self, project):

		print 'creating clalssifier...'
		projectId = project['id']

		if project['model_type'] == 'MLP':
			self.classifier =  MLP_Classifier()
		elif project['model_type'] == 'CNN':
			self.classifier =  CNN_Classifier()

                self.classifier.model_path = Paths.Models
                self.classifier.segmentation_path = Paths.Segmentation
                self.classifier.project = projectId

		self.dataset = DataSets(projectId)
		
        #-------------------------------------------------------------------
	# Main method of the threat - runs utnil self.done is false
        #-------------------------------------------------------------------
	def run(self):
	    while not self.done:

		projects  = Database.getProjects()
		running   = None
		runningId = None

		print '-------------------'
		# the first active project is the running project,
		# all others are deactivated 
		for project in projects:
			projectId     = project['id']
			projectStatus = project['training_mod_status']
			print project['start_time'], project['id'], projectStatus
			projectStatusStr = Database.projectStatusToStr( projectStatus )
			if projectStatus >= 1:
				if running == None:
					running = project
					runningId = projectId
				else:
					print 'shutting down....', projectId
					print projectStatusStr
					print projectStatus
					print runningId
					Database.stopProject( projectId )
					Utility.report_status('stopping (%s)' %(projectId), '(%s) -> (Deactive)'%(projectStatusStr))

		# start the new project if changed.
		if self.projectId != runningId and running != None:
			projectStatus = running['training_mod_status']
			projectStatusStr = Database.projectStatusToStr( projectStatus )
			self.create_classifier( running )
			Utility.report_status('starting (%s)' %(projectId), '(Deactive) -> (Active)')

		self.projectId = runningId
		self.train( running )
		time.sleep(self.waittime)

        #-------------------------------------------------------------------
        # Retrieve training trasks from database and call classifier to
	# perform actual training
        #-------------------------------------------------------------------
	def train(self, project):

		if project is None:
			#print 'no project...'
			return

		#print 'training......running....'

		sample_size = project['sample_size']
		n_hidden = [ h['units'] for h in project["hidden_layers"] ]
		learning_rate = project['learning_rate']
		momentum = project['momentum']
		batch_size = project['batch_size']
		epochs = project['epochs']

		# no training until all labels are annotated
		labels = Database.getLabels( self.projectId )

		# must have training and hidden layer units to train
                if len(labels) == 0 or len(project['hidden_layers']) == 0:
			print 'no labels or hidden layers....'
                        return

		# check for new data
		new_data = self.dataset.load_training(sample_size)
			
		# cache the dataset
		if not self.dataset.valid():
			return

		# train the classifier
		self.classifier.train( self.dataset.x, 
                                       self.dataset.y,
				       self.dataset.p,
				       new_data,
				       sample_size**2,
				       len(labels),
				       n_hidden,
				       learning_rate,
				       momentum,
				       batch_size,
				       epochs )

		# save statistics
		self.dataset.save_stats()

	#-------------------------------------------------------------------
        # shuts down the application
        #-------------------------------------------------------------------
        def shutdown(self):
                Utility.report_status('shutting down training manager', '')
                self.done = True


manager = None

#---------------------------------------------------------------------------
# CTRL-C signal handler
#---------------------------------------------------------------------------
def signal_handler(signal, frame):
    manager.shutdown()

#---------------------------------------------------------------------------
# Entry point to the main function of the program.
#---------------------------------------------------------------------------
if __name__ == '__main__':
	Utility.report_status('running training manager', '')
        signal.signal(signal.SIGINT, signal_handler)
        manager = TrainingManager( )
        manager.run()
