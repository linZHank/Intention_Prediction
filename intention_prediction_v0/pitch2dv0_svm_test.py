"""Test svm on pitch2dv0"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import pandas as pd
import scipy.io as spio
import matplotlib.pyplot as plt
import time
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

import utils

num_classes = 9
num_kernels = 4

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

# Use 5, 10, 15,...,40 frames of data to train 8 svm predictor
num_frames = 5*np.arange(1,9)
# Init best kernel storage 
best_kernel = np.array([""]*num_frames.shape[0], dtype="|S8")
# Init best kernel storage 
best_predictor = []
# Init highest train score storage
high_score_train = np.zeros(best_kernel.shape)
# Init highest test score storage
high_score_test = np.zeros(best_kernel.shape)
# Init svm-vote prediction storage
pred_even = np.zeros((num_frames.shape[0], test_labels.shape[0])).astype(int)
pred_disc = np.zeros((num_frames.shape[0], test_labels.shape[0])).astype(int)
pred_logr = np.zeros((num_frames.shape[0], test_labels.shape[0])).astype(int)
# Init prediction accuracy storage
acc_even = np.zeros(best_kernel.shape)
acc_disc = np.zeros(best_kernel.shape)
acc_logr = np.zeros(best_kernel.shape)

for i,nf in enumerate(num_frames):
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
  # Build SVM classifier and make prediction
  kernel = ["linear", "poly", "rbf", "sigmoid"]
  score_train = np.zeros(len(kernel))
  score_test = np.zeros(len(kernel))
  svm = []
  for k in range(len(kernel)):
    svm.append(SVC(kernel=kernel[k]))
    score_train[k] = svm[k].fit(Xtr, ytr).score(Xtr, ytr)
    score_test[k] = svm[k].fit(Xtr, ytr).score(Xte, yte)
    print("{} frames, training accuracy: {} @ {} kernel".format(nf, score_train[k], kernel[k]))
    print("{} frames, testing accuracy: {} @ {} kernel".format(nf, score_test[k], kernel[k]))
    
  ind_max = np.argmax(score_test)
  best_kernel[i] = kernel[ind_max]
  best_predictor.append(svm[ind_max])
  high_score_train[i] = score_train[ind_max]
  high_score_test[i] = score_test[ind_max]
  # Predictions on all frames
  classes_test = svm[ind_max].predict(Xte)
  # Vote prediction for trials, even weight
  pred_even[i] = utils.vote(classes_test, nf, vote_opt="even")
  assert pred_even[i].shape == test_labels.shape
  # Calculate even prediction accuracy
  acc_even[i] = np.sum(pred_even[i]==test_labels, dtype=np.float32)/num_examples_test
  # Vote prediction for trials, discount weight
  pred_disc[i] = utils.vote(classes_test, nf, vote_opt="disc")
  # Calculate discounted prediction accuracy
  acc_disc[i] = np.sum(pred_disc[i]==test_labels, dtype=np.float32)/num_examples_test
  # Vote prediction for trials, logarithmic weight  
  pred_logr[i] = utils.vote(classes_test, nf, vote_opt="logr")
  # Calculate logarithm prediction accuracy
  acc_even[i] = np.sum(pred_logr[i]==test_labels)/num_examples_test

# Find best predictor
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
df = pd.DataFrame({
  "frames": num_frames,
  "neighbors": best_kernel,
  "score_train": high_score_train,
  "score_test": high_score_test,
  "accuracy_even": acc_even,
  "accuracy_disc": acc_disc,
  "accuracy_logr": acc_even
  })
dffilename = os.path.join(result_path, "svm_joint.csv")
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
predevenfilename = os.path.join(result_path, "svm_joint_even.txt")
if not os.path.exists(os.path.dirname(predevenfilename)):
  os.makedirs(os.path.dirname(predevenfilename))
np.savetxt(predevenfilename, pred_even, fmt="%d")
# Save discont weighted predictions
preddiscfilename = os.path.join(result_path, "svm_joint_disc.txt")
if not os.path.exists(os.path.dirname(preddiscfilename)):
  os.makedirs(os.path.dirname(preddiscfilename))
np.savetxt(preddiscfilename, pred_disc, fmt="%d")
# Save logarithm weighted predictions
predlogrfilename = os.path.join(result_path, "svm_joint_logr.txt")
if not os.path.exists(os.path.dirname(predlogrfilename)):
  os.makedirs(os.path.dirname(predlogrfilename))
np.savetxt(predlogrfilename, pred_logr, fmt="%d")


