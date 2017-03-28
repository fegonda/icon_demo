import numpy
import theano
import theano.tensor as T

import mahotas
import partition_comparison
import StringIO
import glob

from mlpv import *

base_path = os.path.dirname(__file__)
sys.path.insert(1,os.path.join(base_path, '../common'))
from database import Database
from utility import *
from performance import Performance
from paths import Paths

if __name__ == '__main__':

	# load the model to use for performance evaluation
	x = T.matrix('x')
	
	model = MLP(rng=numpy.random.RandomState(1), input=x, n_out=2, fileName = 'best_so_far.pkl')

	#----------------------------------------------------------
	# compute baseline performance measurements
	#----------------------------------------------------------
	#Performance.measure_baseline( model )
	Performance.measure_offline( model, 'default' )
	exit(1)

	# get the test set
	testImages = glob.glob('%s/*.tif'%(Paths.TestGrayscale))

	# track the performance results
	performances = []
	i = 0

	# measure performance for each test image
	for path in testImages:

		# extract the name of the image from the path
		name = Utility.get_filename_noext( path )

		# load the test image
		test_image = mahotas.imread( path )

		# compute the probabilities of the test image
        	prob = model.classify_image(x, img=test_image, normMean=0.5, norm_std=1.0)

		# compute the pixel error and variation of information
		performance = Performance( image_id=name )
		performance.measure( prob )
		performances.append( performance )

		#pe, vi = compute_performance( prob=prob, image_id=name )
		#results.append( (pe, vi ) )

		#print 'pe:',pe,'vi:', vi

		i = i+1
		if i>0:
			break

	n_results = len(performances)
	n_thresholds = len(Performance.Thresholds)

	projectId = 'default'
	perfType  = 'offline'

	# compute the average pixel error and variation of information
	# for each threshold and store the results in the SQL Lite database
	for threshold_index in range(n_thresholds):
		vi = 0.0
		pe = 0.0

		# sum the pixel errors and variation infos
		for performance in performances:
			pe += performance.get_pixel_error( threshold_index )
			vi += performance.get_variation_info( threshold_index )

		# compute the averages
		vi /= n_results
		pe /= n_results

		# store results in the database
		Database.storeModelPerformance(
			projectId,
			perfType,
			Performance.Thresholds[ threshold_index ],
			vi,
			pe)
		print 'th:', Performance.Thresholds[ threshold_index ], 'vi:', vi, 'pe:', pe
