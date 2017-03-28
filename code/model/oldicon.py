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
import threading
import time
import signal
import json
import glob

base_path = os.path.dirname(__file__)
sys.path.insert(1,os.path.join(base_path, '../external'))
sys.path.insert(2,os.path.join(base_path, '../common'))

from mlp_classifier import MLP_Classifier
from tasks import TrainingTask
from tasks import PredictionTask
from utility import Utility
from database import Database

sys.path
from settings import Settings
from settings import Paths

INPUT_DIR = '../../data/images'
APP_NAME = 'icon (interactive connectomics)'


class Icon:

	#-------------------------------------------------------------------
	# construct the application and initalizes task list to empty
	#-------------------------------------------------------------------
	def __init__(self, mode):
		self.mode     = mode
		self.tasks    = []
		self.done     = False
		#self.projects = Settings.getall()
		self.projects = Database.getProjects()

        #-------------------------------------------------------------------
	# initializes a project
  	#-------------------------------------------------------------------
	def start_project(self, project):
		print project['model_type']

                if project['model_type'] == 'MLP':
			if self.mode == 'train':
                        	self.tasks.append( TrainingTask( MLP_Classifier(), project['id'] ) )
				return True
			elif self.mode == 'predict':
                        	self.tasks.append( PredictionTask( MLP_Classifier(), project['id'] ))
				return True
                elif settings.model_type == 'CNN':
                        Utility.report_status('Error', 'CNN implemented')
                else:
                        Utility.report_status('Error', 'No model specified')
                return False

        #-------------------------------------------------------------------
        # runs the application
        #-------------------------------------------------------------------
	def run(self):

		Utility.report_status('running', '%s'%(APP_NAME))


		if (self.mode == 'setup'):
			self.setup()
			return

		#for project, settings in self.projects.iteritems():
		for project in self.projects:
			print 'starting project...'
			self.start_project( project )

		# start the tasks
		for task in self.tasks:
		     	task.start()

		while not self.done:
		     	time.sleep( 1.0 )

		while len(self.tasks) > 0:
			try:
		            self.tasks = [t.join(1) for t in self.tasks if t is not None and t.isAlive()]
			except KeyboardInterrupt:
			    self.shutdown()

		Utility.report_status('shutdown', 'done')


	def setup(self):
		self.setup_database()
		Utility.report_status('setup', 'done')

	def setup_database(self):

		# start in a blank slate
		Database.reset()

		# install the tables
		Database.initialize()

		# install the default model
		Database.install_testproject( 'default' )

        # 	Database.storeLabel(project, 0, 'membrane one', 255,0,0)
        # 	Database.storeLabel(project, 1, 'membrane two', 0,255,0)
        # 	Database.storeProject(project, project, '', 'MLP', 39, 0.01, 0.9, 20, 20, 15, True)
        # 	Database.storeHiddenLayerUnit(project, 0, 500)
        # 	Database.storeHiddenLayerUnit(project, 1, 120)
        # 	Database.storeHiddenLayerUnit(project, 2, 100)
		# Icon.setup_images( project )
		# Utility.report_status('creating default project', 'done')

	# @staticmethod
	# def setup_images( project ):
	# 	# setup images
    #     	images = glob.glob(Paths.Images + '/*.tif')
	# 	for image in images:
	# 		tokens = image.split('/');
	# 		image_name = tokens[ len(tokens)-1 ]
    #         		tokens = image_name.split('.')
    #         		image_name = tokens[0]
	#
	# 		#if not image_name.startswith('train-labels'):
	# 		#	continue
	#
	# 		segFile = '%s/%s.%s.seg'%(Paths.Segmentation, image_name, project)
	# 		annFile = '%s/%s.%s.json'%(Paths.Labels, image_name, project)
	# 		if not os.path.exists( segFile ):
	# 			segFile = None
	# 		if not os.path.exists( annFile ):
	# 			annFile = None
	#
	# 		Database.storeTask( project, image_name, annFile, segFile )
	# 	Utility.report_status('setting up image meta data', 'done')


        #-------------------------------------------------------------------
        # shuts down the application
        #-------------------------------------------------------------------
	def shutdown(self):
		Utility.report_status('shutting down', '%s'%(APP_NAME))
		self.done = True
		for task in self.tasks:
			task.abort()

icon = None
def signal_handler(signal, frame):
    icon.shutdown()

#---------------------------------------------------------------------------
# Entry point to the main function of the program.
#---------------------------------------------------------------------------
if __name__ == '__main__':
	signal.signal(signal.SIGINT, signal_handler)
	if len(sys.argv) <= 1 or sys.argv[1] not in ['train', 'predict', 'setup']:
		print 'Usage: python icon.py <train | predict | setup>'
	else:
		icon = Icon( sys.argv[ 1 ] )
		icon.run()
