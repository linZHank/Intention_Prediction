#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Convolutional Neural Network Estimator for pitch2d, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import os
import glob
import time


tfrecords_dir = "/media/linzhank/DATA/Works/Intention_Prediction/Dataset/Ball pitch/pit2d9blk/tfrecord_20180418"
# tfrecords_dir = "/media/linzhank/850EVO_1T/Works/Data/Ball pitch/pit2d9blk/tfrecord_20180423"  
train_filenames = glob.glob(os.path.join(tfrecords_dir, 'train*'))
eval_filenames = glob.glob(os.path.join(tfrecords_dir, 'validate*'))

def train_input_fn():
  dataset = tf.data.TFRecordDataset(train_filenames, num_parallel_reads=16)
  def parse_function(example_proto):
    # example_proto, tf_serialized
    keys_to_features = {
      'image/colorspace': tf.FixedLenFeature(shape=(), dtype=tf.string, default_value="RGB"),
      'image/channels': tf.FixedLenFeature(shape=(), dtype=tf.int64, default_value=3), 
      'image/format': tf.FixedLenFeature(shape=(), dtype=tf.string, default_value="PNG"), 
      'image/filename': tf.FixedLenFeature(shape=(), dtype=tf.string, default_value=""), 
      'image/encoded': tf.FixedLenFeature(shape=(), dtype=tf.string, default_value=""), 
      'image/class/label': tf.FixedLenFeature(shape=(), dtype=tf.int64, default_value=1),
      'image/height': tf.FixedLenFeature(shape=(), dtype=tf.int64, default_value=360),
      'image/width': tf.FixedLenFeature(shape=(), dtype=tf.int64, default_value=640),
      'image/pitcher': tf.FixedLenFeature(shape=(), dtype=tf.string, default_value=""),
      'image/trial': tf.FixedLenFeature(shape=(), dtype=tf.string, default_value=""),
      'image/frame': tf.FixedLenFeature(shape=(), dtype=tf.string, default_value="")
    }
    # parse all features in a single example according to the dics
    parsed_features = tf.parse_single_example(example_proto, keys_to_features)
    # decode the encoded image to the (360, 640, 3) uint8 array
    decoded_image = tf.image.decode_image((parsed_features['image/encoded']))
    # reshape image
    reshaped_image = tf.reshape(decoded_image, [360, 640, 3])
    # resize decoded image
    resized_image = tf.cast(tf.image.resize_images(reshaped_image, [224, 224]), tf.float32)
    # label
    label = tf.cast(parsed_features['image/class/label']-1, tf.int32)
    # label = tf.one_hot(indices=label-1, depth=9)
    
    return {"image_bytes": parsed_features['image/encoded'],
            "image_decoded": decoded_image,
            "image_reshaped": reshaped_image,
            "image_resized": resized_image,
            "label": label}

  # Use `Dataset.map()` to build a pair of a feature dictionary and a label
  # tensor for each example.
  dataset = dataset.map(parse_function)
  dataset = dataset.shuffle(1024)
  dataset = dataset.batch(64)
  # dataset = dataset.repeat(128)
  dataset = dataset.repeat(1) # debug 
  iterator = dataset.make_one_shot_iterator()

  # `features` is a dictionary in which each value is a batch of values for
  # that feature; `labels` is a batch of labels.
  features = iterator.get_next()
  return features

def eval_input_fn():
  dataset = tf.data.TFRecordDataset(eval_filenames)
  def parse_function(example_proto):
    # example_proto, tf_serialized
    keys_to_features = {
      'image/colorspace': tf.FixedLenFeature(shape=(), dtype=tf.string, default_value="RGB"),
      'image/channels': tf.FixedLenFeature(shape=(), dtype=tf.int64, default_value=3), 
      'image/format': tf.FixedLenFeature(shape=(), dtype=tf.string, default_value="PNG"), 
      'image/filename': tf.FixedLenFeature(shape=(), dtype=tf.string, default_value=""), 
      'image/encoded': tf.FixedLenFeature(shape=(), dtype=tf.string, default_value=""), 
      'image/class/label': tf.FixedLenFeature(shape=(), dtype=tf.int64, default_value=1),
      'image/height': tf.FixedLenFeature(shape=(), dtype=tf.int64, default_value=360),
      'image/width': tf.FixedLenFeature(shape=(), dtype=tf.int64, default_value=640),
      'image/pitcher': tf.FixedLenFeature(shape=(), dtype=tf.string, default_value=""),
      'image/trial': tf.FixedLenFeature(shape=(), dtype=tf.string, default_value=""),
      'image/frame': tf.FixedLenFeature(shape=(), dtype=tf.string, default_value="")
    }
    # parse all features in a single example according to the dics
    parsed_features = tf.parse_single_example(example_proto, keys_to_features)
    # decode the encoded image to the (360, 640, 3) uint8 array
    decoded_image = tf.image.decode_image((parsed_features['image/encoded']))
    # reshape image
    reshaped_image = tf.reshape(decoded_image, [360, 640, 3])
    # resize decoded image
    resized_image = tf.cast(tf.image.resize_images(reshaped_image, [224, 224]), tf.float32)
    # label
    label = tf.cast(parsed_features['image/class/label']-1, tf.int32)
    # label = tf.one_hot(indices=label-1, depth=9)
    
    return {"image_bytes": parsed_features['image/encoded'],
            "image_decoded": decoded_image,
            "image_reshaped": reshaped_image,
            "image_resized": resized_image,
            "label": label}

  # Use `Dataset.map()` to build a pair of a feature dictionary and a label
  # tensor for each example.
  dataset = dataset.map(parse_function)
  iterator = dataset.make_one_shot_iterator()

  # `features` is a dictionary in which each value is a batch of values for
  # that feature; `labels` is a batch of labels.
  features = iterator.get_next()
  return features


def model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # pitch2d images are 224x224 pixels, and have 3 RGB color channel
  input_layer = tf.reshape(features["image_resized"], [-1, 224, 224, 3])
  
  # Convolutional Layer #1
  # Computes 96 features using a 11x11x3 filter with step of 4 plus ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 224, 224, 3]
  # Output Tensor Shape: [batch_size, 55, 55, 96]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=96,
      kernel_size=11,
      strides=4,
      padding="same",
      activation=tf.nn.relu)

  # Local Response Normalization Layer #1
  # sqr_sum[a, b, c, d] = sum(input[a, b, c, d - depth_radius : d + depth_radius + 1] ** 2)
  # output = input / (bias + alpha * sqr_sum) ** beta
  lrn1 = tf.nn.local_response_normalization(
      input=conv1,
      depth_radius=5,
      bias=2,
      alpha=1e-4,
      beta=0.75)

  # Pooling Layer #1
  # First max pooling layer with a 3x3 filter and stride of 2
  # Input Tensor Shape: [batch_size, 55, 55, 96]
  # Output Tensor Shape: [batch_size, 27, 27, 96]
  pool1 = tf.layers.max_pooling2d(
    inputs=lrn1,
    pool_size=3,
    strides=2)

  # Convolutional Layer #2
  # Computes 256 features using a 5x5x96 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 27, 27, 96]
  # Output Tensor Shape: [batch_size, 27, 27, 256]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=256,
      kernel_size=5,
      padding="same",
      activation=tf.nn.relu)

  # Local Response Normalization Layer #2
  # sqr_sum[a, b, c, d] = sum(input[a, b, c, d - depth_radius : d + depth_radius + 1] ** 2)
  # output = input / (bias + alpha * sqr_sum) ** beta
  lrn2 = tf.nn.local_response_normalization(
      input=conv2,
      depth_radius=5,
      bias=2,
      alpha=1e-4,
      beta=0.75)

  # Pooling Layer #2
  # Second max pooling layer with a 3x3 filter and stride of 2
  # Input Tensor Shape: [batch_size, 27, 27, 256]
  # Output Tensor Shape: [batch_size, 13, 13, 256]
  pool2 = tf.layers.max_pooling2d(
    inputs=lrn2,
    pool_size=3,
    strides=2)

  # Convolutional Layer #3
  # Computes 384 features using a 3x3x256 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 13, 13, 256]
  # Output Tensor Shape: [batch_size, 13, 13, 384]
  conv3 = tf.layers.conv2d(
      inputs=pool2,
      filters=384,
      kernel_size=3,
      padding="same",
      activation=tf.nn.relu)

  # Convolutional Layer #4
  # Computes 384 features using a 3x3x384 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 13, 13, 384]
  # Output Tensor Shape: [batch_size, 13, 13, 384]
  conv4 = tf.layers.conv2d(
      inputs=conv3,
      filters=384,
      kernel_size=3,
      padding="same",
      activation=tf.nn.relu)

  # Convolutional Layer #5
  # Computes 256 features using a 3x3x384 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 13, 13, 384]
  # Output Tensor Shape: [batch_size, 13, 13, 256]
  conv5 = tf.layers.conv2d(
      inputs=conv4,
      filters=256,
      kernel_size=3,
      padding="same",
      activation=tf.nn.relu)
  
  # Pooling Layer #5
  # Second max pooling layer with a 3x3 filter and stride of 2
  # Input Tensor Shape: [batch_size, 13, 13, 256]
  # Output Tensor Shape: [batch_size, 5, 5, 256]
  pool5 = tf.layers.max_pooling2d(
    inputs=conv5,
    pool_size=3,
    strides=2)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 6, 6, 256]
  # Output Tensor Shape: [batch_size, 6 * 6 * 256]
  pool5_shape = pool5.get_shape()
  num_features = pool5_shape[1:4].num_elements()
  pool5_flat = tf.reshape(pool5, [-1, num_features])
  # pool5_flat = tf.reshape(pool5, [-1, 6 * 6 * 256])

  # Dense Layer #1
  # Densely connected layer with 4096 neurons
  # Input Tensor Shape: [batch_size, 5 * 5 * 256]
  # Output Tensor Shape: [batch_size, 4096]
  dense1 = tf.layers.dense(inputs=pool5_flat, units=4096, activation=tf.nn.relu)

  # Add dropout operation; 0.5 probability that element will be kept
  dropout1 = tf.layers.dropout(
      inputs=dense1, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Dense Layer #2
  # Densely connected layer with 4096 neurons
  # Input Tensor Shape: [batch_size, 4096]
  # Output Tensor Shape: [batch_size, 4096]
  dense2 = tf.layers.dense(inputs=dense1, units=4096, activation=tf.nn.relu)

  # Add dropout operation; 0.5 probability that element will be kept
  dropout2 = tf.layers.dropout(
      inputs=dense2, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 10]
  logits = tf.layers.dense(inputs=dropout2, units=9)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      # "classes": tf.one_hot(indices=tf.argmax(input=logits), depth=9),
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=features["label"], logits=logits)
  # loss = tf.losses.softmax_cross_entropy(onehot_labels=features["label"], logits=logits)
  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=features["label"], predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
  # Create the Estimator
  # disable checkpoint saving
  start_time = time.time()
  run_config = tf.estimator.RunConfig(save_summary_steps=None,
                                      save_checkpoints_secs=None)
  pitch2d_predictor = tf.estimator.Estimator(model_fn=model_fn,
                                             model_dir="/tmp/pitch2d_alexnet_model")
                                             # config=run_config,) # use model_dir to restore model

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

  # Train the model
  pitch2d_predictor.train(input_fn=train_input_fn,
                          hooks=[logging_hook]) # used to have a "step" argument

  # Evaluate the model and print results
  eval_results = pitch2d_predictor.evaluate(input_fn=eval_input_fn)
  print(eval_results)
  end_time = time.time()
  print("Time elapsed: {}".format(end_time-start_time))


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
