"""Use Support Vector Machine on  pitch2dv0 color image data"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import pandas as pd
import scipy.io as spio
import matplotlib.pyplot as plt
import time
from datetime import datetime
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

import utils

num_classes = 9
num_kernels = 4

TODAY = datetime.today().strftime("%Y%m%d")
result_path= "/media/linzhank/850EVO_1T/Works/Action_Recognition/Data/result{}".format(TODAY)

# Load data
train_data, train_labels, train_classes = utils.loadImages(
  name="train",
  imformat=0,
  scale=0.25
)
num_examples_train = train_labels.shape[0]
test_data, test_labels, test_classes = utils.loadImages(
  name="test",
  imformat=0,
  scale=0.25
)
num_examples_test = test_labels.shape[0]

# Use 5, 10, 15,...,40 frames of data to train 8 svm predictor
num_frames = 5*np.arange(1,9)
# Init best kernel storage 
best_kernel = np.array([""]*num_frames.shape[0], dtype="|S8")
# Init best kernel storage 
best_classifier = []
# Init highest train score storage
high_score_train = np.zeros(best_kernel.shape)
# Init highest test score storage
high_score_test = np.zeros(best_kernel.shape)
# Init svm-vote prediction storage
pred_even = np.zeros((num_frames.shape[0], test_labels.shape[0])).astype(int)
pred_disc = np.zeros((num_frames.shape[0], test_labels.shape[0])).astype(int)
pred_logr = np.zeros((num_frames.shape[0], test_labels.shape[0])).astype(int)
p# Init prediction accuracy storage
acc_even = np.zeros(num_frames.shape)
acc_disc = np.zeros(num_frames.shape)
acc_logr = np.zeros(num_frames.shape)

for i,nf in enumerate(num_frames):
  # On your mark
  start_t = time.time()
  # Prepare data for model feed in
  Xtr, ytr = utils.prepImageData(
    train_data,
    train_classes,
    nf,
    shuffle=True)
  Xte, yte = utils.prepImageData(
    test_data,
    test_classes,
    nf)
  # Build SVM classifier and make prediction
  kernel = ["linear", "poly", "rbf"]
  score_train = np.zeros(len(kernel))
  score_test = np.zeros(len(kernel))
  svm = []
  for k in range(len(kernel)):
    svm.append(SVC(kernel=kernel[k]))
    score_train[k] = svm[k].fit(Xtr, ytr).score(Xtr, ytr)
    score_test[k] = svm[k].fit(Xtr, ytr).score(Xte, yte)
    print("{} frames, training accuracy: {} @ {} kernel".format(nf, score_train[k], kernel[k]))
    print("{} frames, testing accuracy: {} @ {} kernel".format(nf, score_test[k], kernel[k]))
    
  bki = np.argmax(score_test)
  best_kernel[i] = kernel[bki]
  best_classifier.append(svm[bki])
  high_score_train[i] = score_train[bki]
  high_score_test[i] = score_test[bki]
  # Predictions on all frames
  classes_test = best_classifier[i].predict(Xte)
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
  acc_logr[i] = np.sum(pred_logr[i]==test_labels)/num_examples_test
  # Times up
  end_t = time.time()
  time_elapsed[i] = end_t - start_t
  print("Training and testing with {} frames consumed {:g} seconds".format(nf, time_elapsed[i]))

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
df = pd.DataFrame(
  {
    "frames": num_frames,
    "neighbors": best_kernel,
    "score_train": high_score_train,
    "score_test": high_score_test,
    "accuracy_even": acc_even,
    "accuracy_disc": acc_disc,
    "accuracy_logr": acc_even,
    "time_consume": time_elapsed
  }
)
dffilename = os.path.join(result_path, "color_svm_scores.csv")
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
predevenfilename = os.path.join(result_path, "color_svm_pred_even.txt")
if not os.path.exists(os.path.dirname(predevenfilename)):
  os.makedirs(os.path.dirname(predevenfilename))
np.savetxt(predevenfilename, pred_even, fmt="%d")
# Save discont weighted predictions
preddiscfilename = os.path.join(result_path, "color_svm_pred_disc.txt")
np.savetxt(preddiscfilename, pred_disc, fmt="%d")
# Save logarithm weighted predictions
predlogrfilename = os.path.join(result_path, "color_svm_pred_logr.txt")
np.savetxt(predlogrfilename, pred_logr, fmt="%d")


