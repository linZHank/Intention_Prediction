from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import scipy.io as spio
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.metrics import confusion_matrix

import utils

result_path= "/media/linzhank/850EVO_1T/Works/Action_Recognition/Data/result"
df = pd.read_csv(result_path+"/knn_joint.csv")
# # Plot confusion matrix
# target_names = ["intent"]*9
# for i in range(9):
#   target_names[i] += str(i+1)
# cnf_matrix = confusion_matrix(test_labels, pred_disc[-1])
# utils.plotConfusionMatrix(cnf_matrix, target_names)

# # Plot trates accuracies
# utils.plotAccBar(high_score_train, high_score_test, num_frames)
 
