"""Test knn on pitch2dv0 color image data"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import pandas as pd
import scipy.io as spio
import matplotlib.pyplot as plt
import time
from sklearn import neighbors
from sklearn.metrics import confusion_matrix

import utils

num_classes = 9
num_k = 16

result_path= "/media/linzhank/850EVO_1T/Works/Action_Recognition/Data/result"

# Time on
start_t = time.time()
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

# Use 5, 10, 15,...,40 frames of data to train 8 knn predictor
num_frames = 5*np.arange(1,9)
# Init best k storage 
best_k = np.zeros(num_frames.shape).astype(int)
# Init best predictor storage
best_predictor = []
# Init highest train score storage
high_score_train = np.zeros(best_k.shape)
# Init highest test score storage
high_score_test = np.zeros(best_k.shape)
# Init knn-vote prediction storage
pred_even = np.zeros((num_frames.shape[0], test_labels.shape[0])).astype(int)
pred_disc = np.zeros((num_frames.shape[0], test_labels.shape[0])).astype(int)
pred_logr = np.zeros((num_frames.shape[0], test_labels.shape[0])).astype(int)
# Init prediction accuracy storage
acc_even = np.zeros(best_k.shape)
acc_disc = np.zeros(best_k.shape)
acc_logr = np.zeros(best_k.shape)

for i,nf in enumerate(num_frames):
  Xtr, ytr = utils.prepImageData(
    train_data,
    train_classes,
    nf,
    shuffle=True
  )
  Xte, yte = utils.prepImageData(
    test_data,
    test_classes,
    nf
  )
  # Build KNN classifier and make prediction
  num_k = 16
  score_train = np.zeros(num_k)
  score_test = np.zeros(num_k)
  knn = []
  for k in range(num_k):
    knn.append(neighbors.KNeighborsClassifier(k+1))
    score_train[k] = knn[k].fit(Xtr, ytr).score(Xtr, ytr)
    score_test[k] = knn[k].fit(Xtr, ytr).score(Xte, yte)
    print("{} frames, training accuracy: {} @ k={}".format(nf, score_train[k], k+1))
    print("{} frames, testing accuracy: {} @ k={}".format(nf, score_test[k], k+1))
  # Find index of best test score
  ind_max = np.argmax(score_test)
  best_k[i] = ind_max + 1
  best_predictor.append(knn[ind_max])
  high_score_train[i] = score_train[ind_max]
  high_score_test[i] = score_test[ind_max]
  # Predictions frame-wise
  classes_test = best_predictor[i].predict(Xte)
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
pred_accs = np.array([acc_even, acc_disc, acc_logr]) # group prediction accuracies together
ind = np.unravel_index(np.argmax(pred_accs, axis=None), pred_accs.shape) # index of highest acc
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
  "neighbors": best_k,
  "score_train": high_score_train,
  "score_test": high_score_test,
  "accuracy_even": acc_even,
  "accuracy_disc": acc_disc,
  "accuracy_logr": acc_even
  })
dffilename = os.path.join(result_path, "color_knn_scores.csv")
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
predevenfilename = os.path.join(result_path, "color_knn_pred_even.txt")
if not os.path.exists(os.path.dirname(predevenfilename)):
  os.makedirs(os.path.dirname(predevenfilename))
np.savetxt(predevenfilename, pred_even, fmt="%d")
# Save discont weighted predictions
preddiscfilename = os.path.join(result_path, "color_knn_pred_disc.txt")
if not os.path.exists(os.path.dirname(preddiscfilename)):
  os.makedirs(os.path.dirname(preddiscfilename))
np.savetxt(preddiscfilename, pred_disc, fmt="%d")
# Save logarithm weighted predictions
predlogrfilename = os.path.join(result_path, "color_knn_pred_logr.txt")
if not os.path.exists(os.path.dirname(predlogrfilename)):
  os.makedirs(os.path.dirname(predlogrfilename))
np.savetxt(predlogrfilename, pred_logr, fmt="%d")

# Times up
end_t = time.time()
print("Total running time: {:g} seconds".format(end_t - start_t))

