"""Use alexnet on pitch2dv0 color image data"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import numpy as np
import os
import pandas as pd
import scipy.io as spio
import matplotlib.pyplot as plt
import time
from datetime import datetime
from sklearn.metrics import confusion_matrix

import utils
# Specify result storage location
TODAY = datetime.today().strftime("%Y%m%d")
# result_path= "/media/linzhank/850EVO_1T/Works/Action_Recognition/Data/result{}".format(TODAY)
result_path= "/media/linzhank/DATA/Works/Action_Recognition/Data/result{}".format(TODAY)

def alexnet_model_fn(features, labels, mode):
  """Model function for AlexNet"""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # pitch2d images are 224x224 pixels, and have 3 RGB color channel
  x = tf.cast(features["x"], dtype=tf.float32)
  input_layer = tf.reshape(x, [-1, 224, 224, 3], )
  
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
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
  # loss = tf.losses.softmax_cross_entropy(onehot_labels=features["label"], logits=logits)
  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-5)
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


train_data, train_labels, train_classes = utils.loadImages(
  name="train",
  imformat=1,
)
num_examples_train = train_labels.shape[0]

test_data, test_labels, test_classes = utils.loadImages(
  name="test",
  imformat=1,
)
num_examples_test = test_labels.shape[0]

# Use 5, 10, 15,...,40 frames of data to train 8 svm predictor
num_frames = 5*np.arange(1,9)
# Init best kernel storage 
best_layers = np.zeros((8,2), dtype=int)
# Init highest train score storage
high_score_train = np.zeros(num_frames.shape[0])
# Init highest test score storage
high_score_test = np.zeros(num_frames.shape[0])
# Init cnn-vote prediction storage
pred_even = np.zeros((num_frames.shape[0], test_labels.shape[0])).astype(int)
pred_disc = np.zeros((num_frames.shape[0], test_labels.shape[0])).astype(int)
pred_logr = np.zeros((num_frames.shape[0], test_labels.shape[0])).astype(int)
# Init prediction accuracy storage
acc_even = np.zeros(num_frames.shape[0])
acc_disc = np.zeros(num_frames.shape[0])
acc_logr = np.zeros(num_frames.shape[0])
# Init time consumption storage
time_elapsed = np.zeros(num_frames.shape)

# Main
for i,nf in enumerate(num_frames):
  # On your mark
  start_t = time.time()
  # Prepare data for model feed in
  Xtr, ytr = utils.prepImageData(
    train_data,
    train_classes,
    nf,
    std=False,
    shuffle=True
  )
  Xte, yte = utils.prepImageData(
    test_data,
    test_classes,
    nf,
    std=False
  )
  # Build AlexNet classifier and make prediction
  classifier = tf.estimator.Estimator(
    model_fn=alexnet_model_fn,
    model_dir="/tmp/pitch2dv0_color_alex_frame{}".format(nf)
  )
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=100)
  # Train the model, and evaluate train data
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": Xtr},
    y=ytr,
    batch_size=128,
    num_epochs=None,
    shuffle=True,
    # num_threads=8
  )
  classifier.train(
    input_fn=train_input_fn,
    steps=nf*500,
    hooks=[logging_hook]
  )
  high_score_train[i] = classifier.evaluate(
    input_fn=tf.estimator.inputs.numpy_input_fn(
      x={"x": Xtr},
      y=ytr,
      num_epochs=1,
      shuffle=False
    )
  )["accuracy"]
  # Evaluate test data
  test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": Xte},
    y=yte,
    num_epochs=1,
    shuffle=False
  )
  high_score_test[i] = classifier.evaluate(input_fn=test_input_fn)["accuracy"]
  print("{} frames, training accuracy: {}".format(nf, high_score_train[i]))
  print("{} frames, testing accuracy: {}".format(nf, high_score_test[i]))
  # Predict test data
  indte = range(Xte.shape[0]) # index of test examples
  predictions = classifier.predict(input_fn=test_input_fn)
  clste = np.zeros(yte.shape).astype(int)
  correct_sum = 0
  for pred, ind in zip(predictions, indte):
    clste[ind] = pred["classes"]
    probability = pred["probabilities"][clste[ind]]
    print("Prediction is {} {:.1f}%, expected {}".format(clste[ind], 100*probability, yte[ind]))
    if clste[ind] == yte[ind]:
      correct_sum += 1
  acc_te = float(correct_sum / len(indte))
  assert abs(acc_te-high_score_test[i])<1e-4

  # Vote prediction for trials, even weight
  pred_even[i] = utils.vote(clste, nf, vote_opt="even")
  assert pred_even[i].shape == test_labels.shape
  # Calculate even prediction accuracy
  acc_even[i] = np.sum(pred_even[i]==test_labels, dtype=np.float32)/num_examples_test
  # Vote prediction for trials, discount weight
  pred_disc[i] = utils.vote(clste, nf, vote_opt="disc")
  # Calculate discounted prediction accuracy
  acc_disc[i] = np.sum(pred_disc[i]==test_labels, dtype=np.float32)/num_examples_test
  # Vote prediction for trials, logarithmic weight  
  pred_logr[i] = utils.vote(clste, nf, vote_opt="logr")
  # Calculate logarithm prediction accuracy
  acc_logr[i] = np.sum(pred_logr[i]==test_labels)/num_examples_test
  # Times up
  end_t = time.time()
  time_elapsed[i] = end_t - start_t
  print("Training and testing with {} frames consumed {:g} seconds".format(nf, time_elapsed[i]))


# Find best prediction
pred_accs = np.array([acc_even, acc_disc, acc_logr])
ind = np.unravel_index(np.argmax(pred_accs, axis=None), pred_accs.shape)
assert len(ind) == 2
assert ind[0] <= 2 or ind[1] <= 7
if ind[0] == 0:
  high_prediction = pred_even[ind[1]]
elif ind[0] == 1:
  high_prediction = pred_disc[ind[1]]
else:
  high_prediction = pred_logr[ind[1]]
  
# Save frame-wise and trial-wise accuracies in pandas DataFrmae
df = pd.DataFrame(
  {
    "frames": num_frames,
    "score_train": high_score_train,
    "score_test": high_score_test,
    "accuracy_even": acc_even,
    "accuracy_disc": acc_disc,
    "accuracy_logr": acc_even,
    "time_consume": time_elapsed
  }
)
dffilename = os.path.join(result_path, "color_alex2g_scores.csv")
if not os.path.exists(os.path.dirname(dffilename)):
  os.makedirs(os.path.dirname(dffilename))
df.to_csv(dffilename)
    
# Plot confusion matrix
target_names = ["intent"]*9
for i in range(9):
  target_names[i] += str(i+1)
cnf_matrix = confusion_matrix(test_labels, high_prediction)
utils.plotConfusionMatrix(cnf_matrix, target_names)

# Plot trates accuracies
utils.plotAccBar(high_score_train, high_score_test, num_frames)

# Save predictions to files
# Save even weighted predictions
predevenfilename = os.path.join(result_path, "color_alex2g_pred_even.txt")
if not os.path.exists(os.path.dirname(predevenfilename)):
  os.makedirs(os.path.dirname(predevenfilename))
np.savetxt(predevenfilename, pred_even, fmt="%d")
# Save discont weighted predictions
preddiscfilename = os.path.join(result_path, "color_alex2g_pred_disc.txt")
np.savetxt(preddiscfilename, pred_disc, fmt="%d")
# Save logarithm weighted predictions
predlogrfilename = os.path.join(result_path, "color_alex2g_pred_logr.txt")
np.savetxt(predlogrfilename, pred_logr, fmt="%d")

