"""Test knn on pitch2dv0"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import scipy.io as spio
import matplotlib.pyplot as plt
import time
from sklearn import neighbors
from sklearn.metrics import confusion_matrix

import utils

num_classes = 9
num_k = 16

train_path = "/media/linzhank/850EVO_1T/Works/Action_Recognition/Data/train/joint/"
test_path = "/media/linzhank/850EVO_1T/Works/Action_Recognition/Data/test/joint/"

# Load data
train_data = spio.loadmat(train_path + "joint_train.mat")["joint_train"]
num_examples_train = train_data.shape[0]
initid_train = utils.detect_init(train_data)
train_data = train_data.reshape(num_examples_train,150,75)
train_classes = spio.loadmat(train_path + "labels_train.mat")["labels_train"]
train_labels = np.argmax(train_classes, axis=1)
                                
test_data = spio.loadmat(test_path + "joint_test.mat")["joint_test"]
num_examples_test = test_data.shape[0]
initid_test = utils.detect_init(test_data)
test_data = test_data.reshape(num_examples_test,150,75)
test_classes = spio.loadmat(test_path + "labels_test.mat")["labels_test"]
test_labels = np.argmax(test_classes, axis=1)

# Use 5, 10, 15,...,40 frames of data to train 8 knn predictor
num_frames = 5*np.arange(1,9)
# Init best k storage 
best_k = np.zeros(num_frames.shape).astype(int)
# Init highest train score storage
high_score_train = np.zeros(best_k.shape)
# Init highest test score storage
high_score_test = np.zeros(best_k.shape)
# Init knn-vote prediction storage
pred_even = np.zeros((num_frames.shape[0], test_labels.shape[0])).astype(int)
pred_disc = np.zeros((num_frames.shape[0], test_labels.shape[0])).astype(int)
pred_logr = np.zeros((num_frames.shape[0], test_labels.shape[0])).astype(int)
for i,nf in enumerate(num_frames):
  Xtr, ytr = utils.prepjointdata(
    train_data,
    train_labels,
    initid_train,
    nf,
    shuffle=True)
  Xte, yte = utils.prepjointdata(
    test_data,
    test_labels,
    initid_test,
    nf)
  # Build KNN classifier and make prediction
  num_k = 2
  score_train = np.zeros(num_k)
  score_test = np.zeros(num_k)
  knn = []
  for k in range(num_k):
    knn.append(neighbors.KNeighborsClassifier(k+1))
    score_train[k] = knn[k].fit(Xtr, ytr).score(Xtr, ytr)
    score_test[k] = knn[k].fit(Xtr, ytr).score(Xte, yte)
    print("{} frames, training accuracy: {} @ k={}".format(nf, score_train[k], k+1))
    print("{} frames, testing accuracy: {} @ k={}".format(nf, score_test[k], k+1))
    
  max_index = np.argmax(score_test)
  best_k[i] = max_index + 1
  high_score_train[i] = score_train[max_index]
  high_score_test[i] = score_test[max_index]
  # Predict trials with knn-vote
  classes_test = knn[max_index].predict(Xte)
  pred_even[i] = utils.vote(classes_test, nf, vote_opt="even")
  pred_disc[i] = utils.vote(classes_test, nf, vote_opt="disc")
  pred_logr[i] = utils.vote(classes_test, nf, vote_opt="logr")

df = pd.DataFrame({
  "num_frames": num_frames,
  "best_k": best_k,
  "high_score_train": high_score_train,
  "high_score_test": high_score_test
  #"pred_even":pred_even
  })
    
    
