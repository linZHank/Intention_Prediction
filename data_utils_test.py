from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import tensorflow as tf
import numpy as np
import os


def parser(record):
  """Use `tf.parse_single_example()` to extract data from a `tf.Example` protocol buffer, 
and perform any additional per-record preprocessing.
  """
  keys_to_features = {
    "image/encoded": tf.FixedLenFeature(shape=(), dtype=tf.string),
    "image/label": tf.FixedLenFeature(shape=(), dtype=tf.int64)
    }

  # parse all features in a single example according to the dics
  parsed_features = tf.parse_single_example(record, keys_to_features)
  # decode the encoded image to the (784,) uint8 array
  decoded_image = tf.decode_raw(parsed_features["image/encoded"], tf.float32)
  # reshape image
  reshaped_image = tf.reshape(decoded_image, [28, 28, 1])
  # label
  label = tf.cast(parsed_features["image/label"], tf.int32)

  parsed_example = {
    "image_bytes": parsed_features["image/encoded"],
    "image_decoded": decoded_image,
    "image_reshaped": reshaped_image
  }

  return parsed_example, label

  
def make_input_fn(filenames, name, buffer_size=4096, batch_size=128, num_epoch=128):
  """Make input function for Estimator API
  
  Args:
    filenames: list of TFRecord filenames
    name: specify purpose of input_fn: train or eval
    batch_size:
    buffer_size: random shuffling is done on the buffer, so it must be big enough
  Returns:
    features:
    labels:
  """
  # Create Dataset object using TFRecord
  dataset = tf.data.TFRecordDataset(filenames)
  # Parse Dataset object using parse_function
  dataset = dataset.map(parser)
  # Generate different Dataset according to its purpose of usage
  if name == "train":
    dataset = dataset.shuffle(buffer_size=buffer_size) # shuffle
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_epoch)
  elif name == "eval":
    dataset = dataset.repeat(1)
  else:
    raise("Dataset usage wrong! Please specify a valid name: train or eval")
  # Create an iterator
  iterator = dataset.make_one_shot_iterator()
  # Get the next batch
  features, labels = iterator.get_next()

  return features, labels
