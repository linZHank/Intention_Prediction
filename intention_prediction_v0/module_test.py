"""Test for DNNClassifier
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import matplotlib.pyplot as plt
import tensorflow as tf
import utils

img_te, lbl_te = utils.loadImages("test")
