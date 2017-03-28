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

thresholds = np.arange(0.1, 1.0, 0.1)

#import matplotlib
#import matplotlib.pyplot as plt

def save_prob(prob, path):
	output = StringIO.StringIO()
	output.write(prob.tolist())
	#path = 'prob.seg'	
	with open(path, 'w') as outfile:
		outfile.write(output.getvalue())

def save_results(tImage, vImage, mImage, pError, vInfo):
	path = 'results.txt'
	with open(path, 'a') as outfile:
		outtext = "%s:%s:%s:%f:%f\n"%(tImage, vImage, mImage, pError, vInfo)
		outfile.write(outtext)

def compute_pixel_erroe(image, prob):
	image = image.flatten()
	prob = prob.flatten()

	print 'image:', np.unique( image )
	print 'prob:', np.unique( prob )

	matched = (image == prob)
	return (1.0 - float( np.sum(matched) )/image.shape[0])

def compute_variation_info(image, prob):

	# invert probabilities inorder to compute 
	# connected components
        prob = (1 - prob)*255

 	# label the image	
	'''
	print '--*--(a)'
	print prob.shape
	print type(prob)
	print prob[:3]
	print np.unique( prob )
	'''
        prob_labeled, n_labels = mahotas.label( prob )
        prob_labeled = np.int64(prob_labeled.flatten())
	prob_labeled = prob_labeled.flatten()

        print 'computing variation...'
	print '#labels:', n_labels
	'''
        print prob_labeled.shape
        print image.shape
	print np.unique( prob_labeled )
	#print np.unique( image )
        print type(prob_labeled[0]), type(image[0])
	'''
        return partition_comparison.variation_of_information( prob_labeled, image )

def compute_results(x, model, labelPath, testPath, memPath):
        mem_image = mahotas.imread( memPath )
        test_image = mahotas.imread( testPath )
        label_image = mahotas.imread( labelPath )
        '''	
	print 'label:', labelPath
	print 'test:', testPath
	print 'mem:', memPath

        print 'labels:', np.unique(label_image )
	print 'mem:', np.unique(mem_image)
	print 'test:', np.unique(test_image)
        exit(1)
	'''


	# compute probabilities of the test image
	prob = model.classify_image(x, img=test_image, normMean=0.5, norm_std=1.0)

	# compute the pixel error based on the membrane image
	# and the computed probabilities
	#pe = compute_pixel_error( mem_image, np.uint8( prob ) )
	#print 'pe:', pe

	# compute the variation of information based on the
	# labeled image and thresholds of the probabilities.
	label_image = np.int64( label_image.flatten() )
	pes = []
	vis = []
	i = 0
	for t in thresholds:
		p = np.copy( prob )
		p[p < t] = 0.0
		p[p >=t] = 1.0
		p_int = np.int64( p )
		#print 't:', t
		#print 'p_int:', np.unique(p_int)
		save_prob(p_int, 'prob%d.seg'%(i))
		i = i+1

		pe = compute_pixel_error( mem_image, p_int )
		pes.append( pe )

		vi = compute_variation_info( label_image, p_int )
		#vi = partition_comparison.variation_of_information( label_image, p_int )
		vis.append( vi )

	# return results to caller
	return pes, vis

def compute_baseline(memPath,labelPath):
	print 'compute_baseline...'
	print 'label:', labelPath
	print 'mem:', memPath
        mem_image = mahotas.imread( memPath )
	label_image = mahotas.imread( labelPath )

        # compute the variation of information based on the
        # labeled image and thresholds of the probabilities.
	label_image = np.int64( label_image.flatten() )
        pes = []
        vis = []
        for t in thresholds:
		threshold = int(t*255)
                p = np.copy( mem_image )
                p[p <= threshold] = 0
                p[p  > threshold] = 1
                p_int = np.int64( p )

                pe = 0.0
                vi = compute_variation_info( label_image, p_int )

		print 'bins:',vi,t,threshold,np.bincount(p_int.flatten())

                Database.storeModelPerformance(
                        'baseline',
                        'baseline',
                        t,
                        vi,
                        pe)


def compute_resultsold(x, model, labelPath, testPath, memPath):

	mem_image = mahotas.imread( memPath )
        test_image = mahotas.imread( testPath )
        #plt.imshow(test_image), plt.show()
        #prob = model.predict_image(x, img=test_image, normMean=0.5, norm_std=1.0)
	prob = model.classify_image(x, img=test_image, normMean=0.5, norm_std=1.0)
	# threshold here with 0.1 - 0.9

	
	#patience counter to stop when validation error
	# is not changing after a certain number of epochs
	
	#save_prob( prob )
	print np.unique( prob )

	mem_image = mem_image.flatten()
	matched = (mem_image == prob)
	percent = float(np.sum(matched))/mem_image.shape[0]
	pe = (1.0 - percent)

        #prob_uint = np.uint8( prob*255 )
	#prob = np.invert( prob*255 )
	#prob = 1 - prob
	prob_uint = np.uint8( prob )
	prob_uint = 1 - prob_uint
	prob_uint = prob_uint.reshape(1024,1024)
	prob_labeled, n_labels = mahotas.label( prob_uint )
	prob_labeled = np.uint8(prob_labeled.flatten())
	print '# labels:', n_labels
	print prob_labeled.shape
	prob_labeled = np.int64( prob_labeled )

	# compare prediction results to membrane test label


	# threshold the probabilities
	# try ypred
	# try probability threshold
	# threshold before invert
        label_image = mahotas.imread( labelPath )
	label_image = label_image.flatten()
        #label_image = np.int32( label_image.flatten() )
	label_image = np.int64( label_image.flatten() )

        print 'computing variation...'
	print prob_labeled.shape
	print label_image.shape
	print type(prob_labeled[0]), type(label_image[0])
        vi = partition_comparison.variation_of_information( prob_labeled, label_image )
	return vi, pe

if __name__ == '__main__':

	x = T.matrix('x')
	model = MLP(rng=numpy.random.RandomState(1), input=x, n_out=2, fileName = 'best_so_far.pkl')

    	#pathPrefix = '/home/fgonda/icon/data/reference/'
	pathPrefix = '/n/home00/fgonda/icon/data/reference/'
    	srcPath = '%simages/test/train-input_0090.tif'%(pathPrefix)
    	dstPath = 'train-labels_0090.png'

	memPath = '%slabels/membranes/test/train-labels_0090.tif'%(pathPrefix)
  	valPath = '%slabels/test/train-labels_0090.tif'%(pathPrefix)
	resPath = 'train-labels_0090.png'

	# compute the baseline on the gray scale image
	compute_baseline(srcPath, valPath)
	exit(1)

	testPath = '%simages/test/'%(pathPrefix)
	valPath  = '%slabels/test/'%(pathPrefix)
	memPath  = '%slabels/membranes/test/'%(pathPrefix)


	print 'testpath:', testPath 
	testImages = glob.glob('%s/*.tif'%(testPath))
	results = []
	i = 0
	for image in testImages:
		path, filename = os.path.split(image)
		testImage = image
		postfix  = filename.replace('input', 'labels')		
		valImage = '%s%s'%(valPath,postfix)
		memImage = '%s%s'%(memPath,postfix)
		if not os.path.exists(testImage):
			print 'tst error %s not found'%(testImage)
			exit(1)
                if not os.path.exists(valImage):
                        print 'val error %s not found'%(valImage)
                        exit(1)
                if not os.path.exists(memImage):
                        print 'mem error %s not found'%(memImage)
                        exit(1)

		pe,vi = compute_results( x, model, valImage, testImage, memImage )
		results.append( (pe, vi ) )
		print 'pe:',pe,'vi:', vi
		
		i = i+1
		if i>2:
			break

	
	n_results = len(results)
	n_thresholds = 0 if (n_results == 0) else len( results[0][0] )

	projectId = 'default'
	perfType  = 'offline'

	# compute the average pixel error and variation of information
	# for each threshold and store the results in the SQL Lite database
	for threshold in range(n_thresholds):
		vi = 0.0
		pe = 0.0
		for result in results:
			pe += result[0][threshold]
			vi += result[1][threshold]
		vi /= n_results
		pe /= n_results
		Database.storeModelPerformance( 
			projectId, 
			perfType,
			thresholds[ threshold ],
			vi,
			pe)
		print 'th:', thresholds[ threshold ], 'vi:', vi, 'pe:', pe
		
		
		#save_results(testImage, valImage, memImage, pe, vi)
	#vi, pe = compute_results( x, model, valPath, srcPath, memPath )
	#print results


