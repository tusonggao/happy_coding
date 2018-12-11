# https://github.com/bartgras/XGBoost-Tensorflow-Wide-and-deep-comparison/blob/master/Tensorflow%20-%20wide%20and%20deep.ipynb

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

from six.moves import urllib

import pandas as pd
import tensorflow as tf

print(tf.__version__)