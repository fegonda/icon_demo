import os
import sys
import theano
import theano.tensor as T
import numpy
import numpy as np
import mahotas
import partition_comparison
import StringIO
import glob

base_path = os.path.dirname(__file__)
sys.path.insert(1,os.path.join(base_path, '../common'))
sys.path.insert(2,os.path.join(base_path, '../database'))
from db import DB
from project import Project
from performance import Performance
from paths import Paths

from mlpv import MLP


print 'base_path:', base_path

if __name__ == '__main__':

    # load the model to use for performance evaluation
    x = T.matrix('x')

    rng = numpy.random.RandomState(1234)

    project = DB.getProject('testmlpv2')

    model = MLP(
            rng=rng,
            input=x, 
            n_out=len(project.labels),
            n_hidden=project.hiddenUnits,
            fileName = 'best_so_far.pkl')

    nTests = 2
    projectId = 'testverenamlp'
    print 'measuring offline performance...'
    Performance.measureOffline(model, projectId , mean=project.mean, std=project.std,maxNumTests=nTests)
