import os
import sys
import time


base_path = os.path.dirname(__file__)
sys.path.insert(1,os.path.join(base_path, '../external'))

from logistic_sgd import LogisticRegression, load_data
from mlpv import HiddenLayer, MLP, rectified_linear, send_email

