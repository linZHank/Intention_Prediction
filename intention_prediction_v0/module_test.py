from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import scipy.io as spio
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import tensorflow as tf
import utils

train_path = "/media/linzhank/850EVO_1T/Works/Action_Recognition/Data/train/joint/"
test_path = "/media/linzhank/850EVO_1T/Works/Action_Recognition/Data/test/joint/"
result_path= "/media/linzhank/850EVO_1T/Works/Action_Recognition/Data/result"

# Load data
train_data = spio.loadmat(train_path + "joint_train.mat")["joint_train"]
num_examples_train = train_data.shape[0]
initid_train = utils.detectInit(train_data)
train_data = train_data.reshape(num_examples_train,150,75)
train_classes = spio.loadmat(train_path + "labels_train.mat")["labels_train"]
train_labels = np.argmax(train_classes, axis=1)
                                
test_data = spio.loadmat(test_path + "joint_test.mat")["joint_test"]
num_examples_test = test_data.shape[0]
initid_test = utils.detectInit(test_data)
test_data = test_data.reshape(num_examples_test,150,75)
test_classes = spio.loadmat(test_path + "labels_test.mat")["labels_test"]
test_labels = np.argmax(test_classes, axis=1)

nf = 20
Xtr, ytr = utils.prepJointData(
  train_data,
  train_labels,
  initid_train,
  nf,
  shuffle=True)
Xte, yte = utils.prepJointData(
  test_data,
  test_labels,
  initid_test,
  nf)

l1 = 64
l2 = 16
feat_cols = [tf.feature_column.numeric_column(key="x", shape=[75])]
hid_units = [l1, l2]
classifier = tf.estimator.Estimator(
  feature_columns=feat_cols,
  hidden_units=hid_units,
  optimizer=tf.train.AdamOptimizer(1e-4),
  n_classes=9
  # model_dir="/tmp/pitch2dv0_model_{}-{}".format(l1, l2)
)
train_input_fn = tf.estimator.inputs.numpy_input_fn(
  x={"x": Xtr},
  y=ytr,
  batch_size=256,
  num_epochs=16,
  shuffle=True
)
classifier.train(input_fn=train_input_fn)
test_input_fn = tf.estimator.inputs.numpy_input_fn(
  x={"x": Xte},
  y=yte,
  num_epochs=1,
)
accuracy_score=classifier.evaluate(input_fn=test_input_fn)["accuracy"]
print("\nTest Accuracy: {0:f}%\n".format(accuracy_score*100))
