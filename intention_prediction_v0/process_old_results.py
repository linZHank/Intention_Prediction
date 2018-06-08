"""Plot confusion matrix from previous prediction results
   Save accuracies into pd dataframe
"""

import numpy as np
import pandas as pd
import scipy.io as spio
import os
import glob
from sklearn.metrics import confusion_matrix
import utils

# Load old results
results_path = "/media/linzhank/850EVO_1T/Works/Action_Recognition/results/color/cnn"
filename = results_path+"/disc_color_prediction.txt" # This is where the predictions locate
# Read in best predictions
with open(filename) as f:
  preds = f.read().splitlines()
predictions = np.array(preds).astype(int)
# Init prediction accuracy storage
# acc_even = np.zeros(8)
# acc_disc = np.zeros(8)
# acc_logr = np.zeros(8)
acc = np.zeros((3,8))
# Glob three acc files
acc_files = sorted(glob.glob(os.path.join(results_path, "*accuracy.txt")))
for i, af in enumerate(acc_files):
  acc[i] = np.loadtxt(af).diagonal()[:-1]

# Load test labels
test_path = "/media/linzhank/850EVO_1T/Works/Action_Recognition/Data/test/joint/"
test_classes = spio.loadmat(test_path + "labels_test.mat")["labels_test"]
test_labels = np.argmax(test_classes, axis=1)

# Use 5, 10, 15,...,40 frames of data to train 8 knn classifier
num_frames = 5*np.arange(1,9)


# Plot confusion matrix
target_names = ["intent"]*9
for i in range(9):
  target_names[i] += str(i+1)
cnf_matrix = confusion_matrix(test_labels, predictions)
utils.plotConfusionMatrix(cnf_matrix, target_names)

# Save training and evaluation scores in pandas DataFrmae
df = pd.DataFrame(
  {
    "frames": num_frames,
    "accuracy_even": acc[1],
    "accuracy_disc": acc[0],
    "accuracy_logr": acc[2],
  }
)
dffilename = "/media/linzhank/850EVO_1T/Works/Action_Recognition/Data/result20180607/color_alex_scores.csv"
if not os.path.exists(os.path.dirname(dffilename)):
  os.makedirs(os.path.dirname(dffilename))
df.to_csv(dffilename)
