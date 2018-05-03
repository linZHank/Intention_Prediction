# data: 1-d numpy array
# model: same as self-developed 2017 model: https://drive.google.com/drive/u/1/folders/0B48QxlXgxW-GdGJ0M1ZNZDlrbW8
"""AlexNet Intention Predictor for pitch2d, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import tensorflow as tf

import data_utils



# file_dir = "/media/linzhank/850EVO_1T/Works/Data/Ball pitch/pit2d9blk/dataset_config/travaltes_20180420"
file_dir = "/media/linzhank/DATA/Works/Intention_Prediction/Dataset/Ball pitch/pit2d9blk/dataset_config/travaltes_20180415"
height = 224
width = 224
img_format = "color"
num_channels = 3

def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # Pitch2d images are 224x224 pixels, and have 3 color channel
  input_layer = tf.reshape(features["x"], [-1, height, width, num_channels])

  # Convolutional Layer #1
  # Computes 96 features using a 11x11x3 filter with step of 4 plus ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 224, 224, 3]
  # Output Tensor Shape: [batch_size, 55, 55, 96]
  conv1 = tf.layers.conv2d(inputs=input_layer,
                           filters=96,
                           kernel_size=11,
                           strides=4,
                           padding="same")

  # Pooling Layer #1
  # First max pooling layer with a 3x3 filter and stride of 2
  # Input Tensor Shape: [batch_size, 55, 55, 96]
  # Output Tensor Shape: [batch_size, 27, 27, 96]
  pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                  pool_size=3,
                                  strides=2,
                                  padding="same")

  # Local Response Normalization Layer #1
  # sqr_sum[a, b, c, d] = sum(input[a, b, c, d - depth_radius : d + depth_radius + 1] ** 2)
  # output = input / (bias + alpha * sqr_sum) ** beta
  lrn1 = tf.nn.local_response_normalization(input=pool1)

  # Convolutional Layer #2
  # Computes 256 features using a 5x5x96 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 27, 27, 96]
  # Output Tensor Shape: [batch_size, 27, 27, 256]
  conv2 = tf.layers.conv2d(inputs=lrn1,
                           filters=256,
                           kernel_size=5,
                           padding="same")

  # Pooling Layer #2
  # Second max pooling layer with a 3x3 filter and stride of 2
  # Input Tensor Shape: [batch_size, 27, 27, 256]
  # Output Tensor Shape: [batch_size, 13, 13, 256]
  pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                  pool_size=3,
                                  strides=2,
                                  padding="same")

  # Local Response Normalization Layer #2
  # sqr_sum[a, b, c, d] = sum(input[a, b, c, d - depth_radius : d + depth_radius + 1] ** 2)
  # output = input / (bias + alpha * sqr_sum) ** beta
  lrn2 = tf.nn.local_response_normalization(input=pool2)
  
  # Convolutional Layer #3
  # Computes 384 features using a 3x3x256 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 13, 13, 256]
  # Output Tensor Shape: [batch_size, 13, 13, 384]
  conv3 = tf.layers.conv2d(inputs=lrn2,
                           filters=384,
                           kernel_size=3,
                           padding="same",
                           activation=tf.nn.relu)

  # Convolutional Layer #4
  # Computes 384 features using a 3x3x384 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 13, 13, 384]
  # Output Tensor Shape: [batch_size, 13, 13, 384]
  conv4 = tf.layers.conv2d(inputs=conv3,
                           filters=384,
                           kernel_size=3,
                           padding="same")

  # Convolutional Layer #5
  # Computes 256 features using a 3x3x384 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 13, 13, 384]
  # Output Tensor Shape: [batch_size, 13, 13, 256]
  conv5 = tf.layers.conv2d(inputs=conv4,
                           filters=256,
                           kernel_size=3,
                           padding="same")
  
  # Pooling Layer #5
  # Second max pooling layer with a 3x3 filter and stride of 2
  # Input Tensor Shape: [batch_size, 13, 13, 256]
  # Output Tensor Shape: [batch_size, 5, 5, 256]
  # pool5 = tf.layers.max_pooling2d(inputs=conv5,
  #                                 pool_size=3,
  #                                 strides=2)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 6, 6, 256]
  # Output Tensor Shape: [batch_size, 6 * 6 * 256]
  conv5_shape = conv5.get_shape()
  num_features = conv5_shape[1:4].num_elements()
  conv5_flat = tf.reshape(conv5, [-1, num_features])

  # Dense Layer #1
  # Densely connected layer with 4096 neurons
  # Input Tensor Shape: [batch_size, 5 * 5 * 256]
  # Output Tensor Shape: [batch_size, 4096]
  dense1 = tf.layers.dense(inputs=conv5_flat, units=4096)

  # Add dropout operation; 0.5 probability that element will be kept
  dropout1 = tf.layers.dropout(
      inputs=dense1, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Dense Layer #2
  # Densely connected layer with 4096 neurons
  # Input Tensor Shape: [batch_size, 4096]
  # Output Tensor Shape: [batch_size, 4096]
  dense2 = tf.layers.dense(inputs=dense1, units=4096)

  # Add dropout operation; 0.5 probability that element will be kept
  dropout2 = tf.layers.dropout(
      inputs=dense2, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 4096]
  # Output Tensor Shape: [batch_size, 9]
  logits = tf.layers.dense(inputs=dropout2, units=9)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

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
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)  

def main(unused_argv):
  # Load training and eval data
  train_data, train_labels = data_utils.get_train_data(file_dir, height, width, img_format)
  eval_data, eval_labels = data_utils.get_eval_data(file_dir, height, width, img_format)

  # Create the Estimator
  run_config = tf.estimator.RunConfig(save_summary_steps=None,
                                    save_checkpoints_secs=None)
  pitch2d_predictor = tf.estimator.Estimator(model_fn=cnn_model_fn,
                                             config=run_config)

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=32,
      num_epochs=None,
      shuffle=True)
  pitch2d_predictor.train(
      input_fn=train_input_fn,
      steps=20000,
      hooks=[logging_hook])

  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)
  eval_results = pitch2d_predictor.evaluate(input_fn=eval_input_fn)
  print(eval_results)

if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
