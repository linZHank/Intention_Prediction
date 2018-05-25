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
train_mat = spio.loadmat(train_path + 'joint_train.mat')
train_labels = spio.loadmat(train_path + 'labels_train.mat')["labels_train"]
train_data = train_mat['joint_train']
initid_train = joint_utils.detect_init(train_data)
train_data = train_data.reshape(train_data.shape[0],150,75)
                                
test_mat = spio.loadmat(test_path + 'joint_test.mat')
test_labels = spio.loadmat(test_path + 'labels_test.mat')
test_data = test_mat['joint_test']
Xte = []

# Training

num_frames = 5*np.arange(1,9)

for nf in num_frames:
  Xtr = []
  ytr = []
  for i in range(train_data.shape[0]):
    x = train_data[i,initid_train[i]:initid_train[i]+nf,:]
    Xtr.append(x)
    y = np.array([train_labels[i]]*nf)
    ytr.append(y)

  Xtr = np.array(Xtr).reshape(train_data.shape[0]*nf, -1)
  ytr = np.array(ytr).reshape(Xtr.shape[0], -1)
  
