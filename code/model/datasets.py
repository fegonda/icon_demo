#---------------------------------------------------------------------------
# datasets.py
#
# Author  : Felix Gonda
# Date    : July 12, 2015
# School  : Harvard University
#
# Project : Master Thesis
#           An Interactive Deep Learning Toolkit for
#           Automatic Segmentation of Images
#
# Summary : This file contains the implementation of a dataset class
#           that houses all data used by the learning model.  The data
#           is loaded through the load_* methods.
#---------------------------------------------------------------------------


import os
import sys
import math
import theano
import theano.tensor as T
import numpy as np

base_path = os.path.dirname(__file__)
sys.path.insert(1,os.path.join(base_path, '../common'))

from utility import Utility
from settings import Paths
#from database import Database
from project import Project
from db import DB


class DataEntry:
	def __init__(self, name, offset, length, loffsets, lsizes):
	 	self.name          = name 	
		self.offset        = offset
		self.length        = length
		self.label_offsets = loffsets
		self.label_sizes   = lsizes

class DataSets:

        #-------------------------------------------------------------------
        # Main constructor of the datasets. Initializes the associated
	# image id and settings, and invalidates the datasets until
	# later loaded via the load_* methods
        #-------------------------------------------------------------------
	def __init__(self):
		self.entries  = []
		self.x        = None
		self.y        = None
		self.p        = None
		self.l        = []

	def save_stats(self, project):
		print '-------------------------------------------------------------------'
		for entry in self.entries:
			i = np.arange( entry.offset, entry.offset+entry.length )
			y = self.y[ i ]
			p = self.p[ i ]
			n_good  = len( np.where( p == 0 )[0] )
			score = 0.0 if n_good == 0 else float(n_good)/len(p)
			DB.storeTrainingScore( project.id, entry.name, score )
			print 'image (%s)(%.2f)'%(entry.name, score)
		print '-------------------------------------------------------------------'

	def valid(self):
		return len(self.entries) > 0

        #-------------------------------------------------------------------
        # This function loads training, test, and validation data to be
        # use for learning features.
        # returns True if new data is added, False otherwise
        #-------------------------------------------------------------------
	def load(self, project):

		get_new_data = (len(self.entries) > 0)

		# Build the training sets by reading in annotations
		# for all images seen so far.

		#labels = Database.getLabels( self.project )

		num_images = 0
		#tasks = Database.getTrainingTasks( self.project, not first_time )
		#Database.finishLoadingTrainingTask( self.project )

		labels  = DB.getLabels( project.id )
		images  = DB.getTrainingImages( project.id, new=get_new_data )

		entries = []
		offset = 0
		num_entries = 0

		max_images = 15
		for task in images:

			if len(entries) > max_images:
				break

			image_id = task.id


			path = '%s/%s.tif'%(Paths.TrainGrayscale, image_id)
			image_p = Utility.get_image_padded( path, project.patchSize )

			data =	Utility.get_training_data(
				project.id,
				image_id, 
				image_p, 
				project.patchSize,
				Paths.Labels)

			success = data[0]
			xdata   = data[1]
			ydata   = data[2]
			loffsets= data[3]
			lsizes  = data[4]

			if not success:
				continue

			Utility.report_status('loading training data for', image_id)

			n_ydata = len(ydata)
			offset = num_entries
			num_entries += n_ydata
			entry = DataEntry( image_id, offset, n_ydata, loffsets, lsizes )
			entries.append( entry )

			if offset == 0:
				x = xdata
				y = ydata
				p = np.ones( n_ydata, dtype=np.int )
			else:
				x = np.vstack( (x, xdata) )
				y = np.hstack( (y, ydata) )		
				p = np.hstack( (p, np.ones( n_ydata, dtype=np.int )) ) 


			Utility.report_memused()
			print 'mem size x:', x.nbytes 
			print 'mem size y:', y.nbytes
			print 'mem size p:', p.nbytes
			print '    nydata:', n_ydata
			print '  #entries:', num_entries
			print '     sizes:', lsizes
			print '   offsets:', loffsets


		if num_entries == 0:
			print 'no new data entries found...'
			return False

		print 'begin stacking.....'
		Utility.report_memused()
		print 'mem size x:', x.nbytes
		print 'mem size y:', y.nbytes 
		print 'mem size p:', p.nbytes

		#append old entries
		if len(self.entries) > 0:
			offset = len(y)
			print entries[-1].name, entries[-1].offset, entries[-1].length
			mask = np.ones( len(self.y), dtype=bool)
			names = [ e.name for e in entries ]

			for entry in self.entries:
				if entry.name in names:
					mask[ entry.offset : entry.offset+entry.length ] = False
				else:
					entry.offset = offset
					offset += entry.length
					entries.append( entry )
					print entry.name, entry.offset, entry.length
	
			x_keep = self.x[ mask ]
			y_keep = self.y[ mask ]
			p_keep = self.p[ mask ]
			x = np.vstack( (x, x_keep) )
			y = np.hstack( (y, y_keep) )
			p = np.hstack( (p, p_keep) )	

		print 'done stacking...'

		# cumulative number of labels in annotations must
		# equal the number of labels in project
		if len(np.unique( y )) != len(labels):
			# update status to not enugh annotations
			DB.updateTrainingStatus(project.id, 2)
			return


		# generate indices based on labels
		self.l = [[] for l in labels]
		#self.l = np.array( self.l )

		total = 0
		for i in range( len(self.l) ):
			print 'i:', i
			for entry in entries:
				offset    = entry.offset + entry.label_offsets[ i ]
				size      = entry.label_sizes[ i ]
				indices   = np.arange( size, dtype=np.int ) + offset
				self.l[i] = np.concatenate( (self.l[i], indices), axis=0).astype(dtype=int)
				print '---'
				print offset, offset+size, size
				print self.l[i]
				total += entry.label_sizes[ i ]

		print '==>total:', total
		self.l = np.array( self.l )
		#if len(self.entries) > 0:
		#	exit(1)

		Utility.report_status('loading complete...', '')
		self.x = x
		self.y = y
		self.p = p
		self.entries = entries

		print '-----------------------------------'
                print '#entries:', len(self.entries)
                print '#newentries:', len(entries)
                print num_entries
		print 'x:', x.shape
		print 'y:', y.shape
		print 'p:', p.shape
		print 'l:', self.l.shape
                print y
                print p

		# TODO.
		# you need to shuffle your data if you're going to split by percentages.
		# restrict upper limit on validation set:q

		DB.finishLoadingTrainingset( project.id )
		return True

