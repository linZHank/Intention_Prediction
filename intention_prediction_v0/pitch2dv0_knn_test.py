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



train_path = "/media/linzhank/850EVO_1T/Works/Action_Recognition/Data/train/joint/"
test_path = "/media/linzhank/850EVO_1T/Works/Action_Recognition/Data/test/joint/"

# Load data
train_data = spio.loadmat(train_path + "joint_train.mat")["joint_train"]
num_examples_train = train_data.shape[0]
initid_train = joint_utils.detect_init(train_data)
train_data = train_data.reshape(num_examples_train,150,75)
train_labels = spio.loadmat(train_path + "labels_train.mat")["labels_train"]
                                
test_data = spio.loadmat(test_path + "joint_test.mat")["joint_test"]
num_examples_test = test_data.shape[0]
initid_test = joint_utils.detect_init(test_data)
test_data = test_data.reshape(num_examples_test,150,75)
test_labels = spio.loadmat(test_path + "labels_test.mat")["labels_test"]


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

  
  for k in range(1,17):
    knn = neighbors.KNeighborsClassifier(k)
    print(knn.fit(Xtr, ytr).score(Xte, yte))
  
