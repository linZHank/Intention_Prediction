from __future__ import absolute_import, division, print_function

import os
import glob
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.contrib.eager as tfe

tf.enable_eager_execution()

print("TensorFlow version: {}".format(tf.VERSION))
print("Eager execution: {}".format(tf.executing_eagerly()))

tfr_dir = '/media/linzhank/DATA/Works/Intention_Prediction/Dataset/Ball pitch/pit2d9blk/tfrecord_20180418'
filenames = glob.glob(os.path.join(tfr_dir, 'train*'))

