"""Test validate tfrecords, create dataset, parse it, and check what do I got for labels and images
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import os
import glob
from matplotlib import pyplot as plt


tfrecords_dir = "/media/linzhank/DATA/Works/Intention_Prediction/Dataset/Ball pitch/pit2d9blk/tfrecord_20180507"
# tfrecords_dir = "/media/linzhank/850EVO_1T/Works/Data/Ball pitch/pit2d9blk/tfrecord_20180423"  
train_filenames = glob.glob(os.path.join(tfrecords_dir, "train*"))
eval_filenames = glob.glob(os.path.join(tfrecords_dir, "validate*"))

def parse_function(example_proto):
  # example_proto, tf_serialized
  keys_to_features = {
    "colorspace": tf.FixedLenFeature(shape=(), dtype=tf.string, default_value="RGB"),
    "channels": tf.FixedLenFeature(shape=(), dtype=tf.int64, default_value=3), 
    "format": tf.FixedLenFeature(shape=(), dtype=tf.string, default_value="PNG"), 
    "filename": tf.FixedLenFeature(shape=(), dtype=tf.string, default_value=""), 
    "encoded_image": tf.FixedLenFeature(shape=(), dtype=tf.string, default_value=""), 
    "label": tf.FixedLenFeature(shape=(), dtype=tf.int64, default_value=1),
    "height": tf.FixedLenFeature(shape=(), dtype=tf.int64, default_value=360),
    "width": tf.FixedLenFeature(shape=(), dtype=tf.int64, default_value=640),
    "pitcher": tf.FixedLenFeature(shape=(), dtype=tf.string, default_value=""),
    "trial": tf.FixedLenFeature(shape=(), dtype=tf.string, default_value=""),
    "frame": tf.FixedLenFeature(shape=(), dtype=tf.string, default_value="")
  }
  # parse all features in a single example according to the dics
  parsed_features = tf.parse_single_example(example_proto, keys_to_features)
  # decode the encoded image to the (360, 640, 3) uint8 array
  decoded_image = tf.image.decode_image((parsed_features["encoded_image"]))
  # reshape image
  reshaped_image = tf.reshape(decoded_image, [360, 640, 3])
  # resize decoded image
  resized_image = tf.cast(tf.image.resize_images(reshaped_image, [224, 224]), tf.float32)
  # label
  label = tf.cast(parsed_features["label"], tf.int32)
  # label = tf.one_hot(indices=label-1, depth=9)

  parsed_example = {
    "image_bytes": parsed_features["encoded_image"],
    "image_decoded": decoded_image,
    "image_reshaped": reshaped_image,
    "image_resized": resized_image,
    "label": label,
    "pitcher": parsed_features["pitcher"]
  }
  return parsed_example

dataset = tf.data.TFRecordDataset(eval_filenames, num_parallel_reads=8)
parsed_dataset = dataset.map(parse_function)
iterator = parsed_dataset.make_one_shot_iterator()
nexelem = iterator.get_next()

sess = tf.InteractiveSession()
i = 1
labels = []
while True:
  try:
    label, pitcher = sess.run([
      # tf.cast(nexelem["image_resized"], tf.uint8),
      nexelem["label"],
      nexelem["pitcher"]
  ])
    labels.append(label)
  except tf.errors.OutOfRangeError:
    break
  else:
    print('==============example %s ==============' %i)
    print("label: {}; pitcher: {}".format(label, pitcher))
  i += 1


