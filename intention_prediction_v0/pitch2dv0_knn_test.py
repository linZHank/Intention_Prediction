"""Test knn on pitch2dv0"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.io as spio
import matplotlib.pyplot as plt
import time
from sklearn import neighbors
from sklearn.metrics import confusion_matrix

import joint_utils

num_classes = 9
num_k = 16

def vote(classes, num_frames, vote_opt="even"):
  """ Vote pitching trial by frame-wise classes using vote options
  """
  num_examples = int(classes.shape[0]/num_frames)
  prediction = np.zeros(num_examples).astype(int)
  for i in range(num_examples):
    if vote_opt == "even":
      accumulation = np.sum(classes[i*num_frames:(i+1)*num_frames], axis=0)
      prediction[i] = np.argmax(accumulation, axis=0)
    elif vote_opt == "disc":
      gamma = 0.9 # discount
      accumulation = np.zeros(num_classes)
      for f in range(num_frames):
        accumulation += np.power(gamma, (num_frames-f))*classes[i*num_frames+f]
      prediction[i] = np.argmax(accumulation, axis=0)
    elif vote_opt == "logr":
      accumulation = np.zeros(num_classes)
      for f in range(num_frames):
        accumulation += np.log1p(f)*classes[i*num_frames+f]
      prediction[i] = np.argmax(accumulation, axis=0)

  return prediction

train_path = "/media/linzhank/850EVO_1T/Works/Action_Recognition/Data/train/joint/"
test_path = "/media/linzhank/850EVO_1T/Works/Action_Recognition/Data/test/joint/"

# Load data
train_data = spio.loadmat(train_path + "joint_train.mat")["joint_train"]
num_examples_train = train_data.shape[0]
initid_train = joint_utils.detect_init(train_data)
train_data = train_data.reshape(num_examples_train,150,75)
train_classes = spio.loadmat(train_path + "labels_train.mat")["labels_train"]
train_labels = np.argmax(train_classes, axis=1)
                                
test_data = spio.loadmat(test_path + "joint_test.mat")["joint_test"]
num_examples_test = test_data.shape[0]
initid_test = joint_utils.detect_init(test_data)
test_data = test_data.reshape(num_examples_test,150,75)
test_classes = spio.loadmat(test_path + "labels_test.mat")["labels_test"]
test_labels = np.argmax(test_classes, axis=1)


num_frames = 5*np.arange(1,9)
for nf in num_frames:
  # Create training dataset
  Xtr = []
  ytr = []
  for i in range(num_examples_train):
    x = train_data[i,initid_train[i]:initid_train[i]+nf,:]
    Xtr.append(x)
    y = np.array([train_labels[i]]*nf)
    ytr.append(y)
  Xtr = np.array(Xtr).reshape(num_examples_train*nf, -1)
  ytr = np.array(ytr).reshape(Xtr.shape[0], -1)
  # Create test dataset
  Xte = []
  yte = []
  for i in range(num_examples_test):
    x = test_data[i,initid_test[i]:initid_test[i]+nf,:]
    Xte.append(x)
    y = np.array([test_labels[i]]*nf)
    yte.append(y)
  Xte = np.array(Xte).reshape(num_examples_test*nf, -1)
  yte = np.array(yte).reshape(Xte.shape[0], -1)

  # Build KNN classifier and make prediction
  num_k = 16
  score_train = np.zeros(num_k)
  score_test = np.zeros(num_k)
  for k in range(num_k):
    knn = neighbors.KNeighborsClassifier(k+1)
    score_train[k] = knn.fit(Xtr, ytr).score(Xtr, ytr)
    score_test[k] = knn.fit(Xtr, ytr).score(Xte, yte)
    class_test = knn.predict(Xte)
    pred = vote(class_test, nf)
    
    print("{} frames, training accuracy: {} @ k={}".format(nf, score_train[k], k+1))
    print("{} frames, testing accuracy: {} @ k={}".format(nf, score_test[k], k+1))
    
