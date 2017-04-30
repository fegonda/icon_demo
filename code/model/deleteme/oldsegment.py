#---------------------------------------------------------------------------
# predict.py
#
# Author  : Felix Gonda
# Date    : July 10, 2015
# School  : Harvard University
#
# Project : Master Thesis
#           An Interactive Deep Learning Toolkit for
#           Automatic Segmentation of Images
#
# Summary : This file contains the implementation of a module that manages
#           the module for segmenting images.  It runs the latest activated
#           project's classifier to perform segmentation on all images in
#           in the project. 
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
from performance import Performance

#---------------------------------------------------------------------------
class SegmentationManager(threading.Thread):

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
		self.high        = []
		self.low         = []
		self.priority    = 0
		self.modTime     = None

	#-------------------------------------------------------------------
	# Create a classifier for the specified project
	#-------------------------------------------------------------------
	def create_classifier(self, project):

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

		# the first active project is the running project,
		# all others are deactivated 
		for project in projects:
			projectId     = project['id']
			projectStatus = project['training_mod_status']
			projectStatusStr = Database.projectStatusToStr( projectStatus )
			if projectStatus >= 1:
				if running == None:
					running = project
					runningId = projectId
					if projectId != self.projectId:
						self.create_classifier( project )
						Utility.report_status('starting (%s)' %(projectId), '(%s)'%(projectStatusStr))
				else:
					Database.stopProject( projectId )
					Utility.report_status('stopping (%s)' %(projectId), '(%s)'%(projectStatusStr))


		self.projectId = runningId
		# save the running project
		self.segment( running )
		time.sleep(self.waittime)

        #-------------------------------------------------------------------
        # Retrieve segmentation tasks from database and call classifier
        # to perform actual work.
        #-------------------------------------------------------------------
	def segment(self, project):

		start_time = time.clock()

		if project is None:
			return

		#print 'training......running....'

                if len(self.high) == 0:
                        self.high = Database.getPredictionTasks( self.projectId, 1)

                if len(self.low) == 0:
                        self.low = Database.getPredictionTasks( self.projectId, 0 )

                task = None
                if (self.priority == 0 or len(self.low) == 0) and len(self.high) > 0:
                        self.priority = 1
                        task = self.high[0]
                        del self.high[0]
                elif len(self.low) > 0:
                        self.priority = 0
                        task = self.low[0]
                        del self.low[0]

                if task == None:
                        return

                project = Database.getProject( self.projectId )
                labels = Database.getLabels( self.projectId )

                imageId = task['image_id']
                modTime = project['model_mod_time']
                sample_size = project['sample_size']
                n_hidden = [ h['units'] for h in project["hidden_layers"] ]
                labels = Database.getLabels( self.projectId )

		has_new_model = (self.modTime != modTime)
		if has_new_model:
			self.classifier.reset()
			self.modTime = modTime

                self.classifier.predict(
                        sample_size,
                        len(labels),
                        n_hidden,
                        imageId,
                        self.projectId,
                        Paths.TrainGrayscale)

		end_time = time.clock()
		duration = (end_time - start_time)
                Database.finishPredictionTask( self.projectId, imageId, duration )

		
		# measure performance if new model
                #if has_new_model:
		#	Performance.measureOnline( self.classifier.model, self.projectId )
			

	#-------------------------------------------------------------------
        # shuts down the application
        #-------------------------------------------------------------------
        def shutdown(self):
                Utility.report_status('shutting down segmentation manager', '')
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
	Utility.report_status('running segmentation manager', '')
        signal.signal(signal.SIGINT, signal_handler)
        manager = SegmentationManager( )
        manager.run()
