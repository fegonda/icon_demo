import numpy
import theano
import theano.tensor as T

import mahotas
import partition_comparison
import StringIO
import glob

import base64
import zlib

from mlpv import *

base_path = os.path.dirname(__file__)
sys.path.insert(1,os.path.join(base_path, '../common'))
from utility import *
from performance import Performance
from paths import Paths
from cnn import CNN
from fast_segment import *


def save_probs(data, projectId, imageId):
	path = '%s/%s.%s.seg'%(Paths.Segmentation, imageId, projectId)
	output = StringIO.StringIO()
	output.write(data.tolist())
	content = output.getvalue()
	encoded = base64.b64encode(content)
	compressed = zlib.compress(encoded)
	with open(path, 'w') as outfile:
		outfile.write(compressed)

if __name__ == '__main__':

	rng = numpy.random.RandomState(929292)

	imageId='ac3_input_0019'
	path = '%s/%s.tif'%(Paths.TrainGrayscale, imageId)
	image = mahotas.imread( path )

	# load the model to use for performance evaluation
	x = T.matrix('x')

        cnn = CNN(
		input=x, 
		batch_size=100,
		patchSize=39, #65, 
		rng=rng, 
		nkerns=[32,32],
		kernelSizes=[5,5], 
		hiddenSizes=[200], 
		fileName='best_cnn_so_far.pkl')

        #prob = cnn.classify_image(img=image, x=x, normMean=0.5, norm_std=1.0)
	threshold = 0.5
	prob = classify_image( path, cnn )
	prob[ prob >= threshold ] = 9
	prob[ prob <  threshold ] = 1
	prob[ prob == 9 ] = 0
	prob = prob.astype(dtype=int)
	prob = prob.flatten()
	print 'results :', np.bincount( prob )

	save_probs( prob, 'testcnn', imageId)
