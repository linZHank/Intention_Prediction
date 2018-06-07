"""Use Multilayer Perceptron on pitch2dv0 joint data"""

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

num_classes = 9

train_path = "/media/linzhank/850EVO_1T/Works/Action_Recognition/Data/train/joint/"
test_path = "/media/linzhank/850EVO_1T/Works/Action_Recognition/Data/test/joint/"
TODAY = datetime.today().strftime("%Y%m%d")
result_path= "/media/linzhank/850EVO_1T/Works/Action_Recognition/Data/result{}".format(TODAY)

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
best_layers = np.zeros((8,2), dtype=int)
# Init highest train score storage
high_score_train = np.zeros(num_frames.shape[0])
# Init highest test score storage
high_score_test = np.zeros(num_frames.shape[0])
# Init svm-vote prediction storage
pred_even = np.zeros((num_frames.shape[0], test_labels.shape[0])).astype(int)
pred_disc = np.zeros((num_frames.shape[0], test_labels.shape[0])).astype(int)
pred_logr = np.zeros((num_frames.shape[0], test_labels.shape[0])).astype(int)
# Init prediction accuracy storage
acc_even = np.zeros(num_frames.shape[0])
acc_disc = np.zeros(num_frames.shape[0])
acc_logr = np.zeros(num_frames.shape[0])
# Init time consumption storage
time_elapsed = np.zeros(num_frames.shape)

for i,nf in enumerate(num_frames):
  # On your mark
  start_t = time.time()
  # Prepare data for model feed in
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
  # Build MLP classifier and make prediction
  layer1 = np.array([16, 64, 128, 512])
  layer2 = np.array([16, 64, 128, 512])
  score_train =  np.zeros(layer1.shape[0]*layer2.shape[0])
  score_test = np.zeros(layer1.shape[0]*layer2.shape[0])
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": Xtr},
    y=ytr,
    batch_size=128,
    num_epochs=256,
    shuffle=True
  )
  test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": Xte},
    y=yte,
    num_epochs=1,
    shuffle=False
  )
  ct = 0 # counter
  for l1 in layer1:
    for l2 in layer2:
      feat_cols = [tf.feature_column.numeric_column(key="x", shape=[75])]
      hid_units = [l1, l2]
      classifier = tf.estimator.DNNClassifier(
        feature_columns = feat_cols,
        hidden_units=hid_units,
        optimizer=tf.train.AdamOptimizer(1e-3),
        n_classes=9,
        model_dir="/tmp/pitch2dv0_mlp_frame{}_{}-{}".format(nf, l1, l2)
      )   
      classifier.train(input_fn=train_input_fn)
      score_train[ct] = classifier.evaluate(
        input_fn=tf.estimator.inputs.numpy_input_fn(
        x={"x": Xtr},
        y=ytr,
        num_epochs=1,
        shuffle=False
        )
      )["accuracy"]
      score_test[ct] = classifier.evaluate(input_fn=test_input_fn)["accuracy"]
      print("{} frames, training accuracy: {} @ layer1: {} / layer2: {}".format(nf, score_train[ct], l1, l2 ))
      print("{} frames, testing accuracy: {} @ layer1: {} / layer2: {}".format(nf, score_test[ct], l1, l2))
      ct += 1
  # Find best l1, l2 combination for current frames setting
  bli = np.argmax(score_test)
  best_layers[i] = np.array([layer1[int(bli/layer2.shape[0])], layer2[int(bli%layer2.shape[0])]])
  high_score_train[i] = score_train[bli]
  high_score_test[i] = score_test[bli]
  # Predictions on all frames
  best_classifier = tf.estimator.DNNClassifier(
    feature_columns = feat_cols,
    hidden_units=best_layers[i],
    n_classes=9,
    model_dir="/tmp/pitch2dv0_mlp_frame{}_{}-{}".format(nf, best_layers[i][0], best_layers[i][1])
  )
  indte = range(Xte.shape[0])
  predictions = best_classifier.predict(input_fn=test_input_fn)
  clste = np.zeros(yte.shape).astype(int)
  correct_sum = 0
  for pred, ind in zip(predictions, indte):
    clste[ind] = pred["class_ids"][0]
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
    "layer1": best_layers[:,0],
    "layer2": best_layers[:,1],
    "score_train": high_score_train,
    "score_test": high_score_test,
    "accuracy_even": acc_even,
    "accuracy_disc": acc_disc,
    "accuracy_logr": acc_even,
    "time_consume": time_elapsed
  }
)
dffilename = os.path.join(result_path, "joint_mlp_scores.csv")
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
predevenfilename = os.path.join(result_path, "joint_mlp_pred_even.txt")
if not os.path.exists(os.path.dirname(predevenfilename)):
  os.makedirs(os.path.dirname(predevenfilename))
np.savetxt(predevenfilename, pred_even, fmt="%d")
# Save discont weighted predictions
preddiscfilename = os.path.join(result_path, "joint_mlp_pred_disc.txt")
np.savetxt(preddiscfilename, pred_disc, fmt="%d")
# Save logarithm weighted predictions
predlogrfilename = os.path.join(result_path, "joint_mlp_pred_logr.txt")
np.savetxt(predlogrfilename, pred_logr, fmt="%d")


