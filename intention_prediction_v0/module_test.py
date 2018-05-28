from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.io as spio
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.metrics import confusion_matrix

import utils

path = "/media/linzhank/850EVO_1T/Works/Action_Recognition/Data/train/joint/"
data = spio.loadmat(path + "joint_train.mat")["joint_train"].reshape(256,150,75)
classes = spio.loadmat(path + "labels_train.mat")["labels_train"]
labels = np.argmax(classes, axis=1)
initid = utils.detect_init(data)
Xtr, ytr = utils.prepjointdata(data, labels, initid, num_frames=20, shuffle="True")

path = "/media/linzhank/850EVO_1T/Works/Action_Recognition/Data/test/joint/"
data = spio.loadmat(path + "joint_test.mat")["joint_test"].reshape(36,150,75)
classes = spio.loadmat(path + "labels_test.mat")["labels_test"]
labels = np.argmax(classes, axis=1)
initid = utils.detect_init(data)
Xte, yte = utils.prepjointdata(data, labels, initid, num_frames=20)

knn = neighbors.KNeighborsClassifier(4)
knn.fit(Xtr, ytr)
cls = knn.predict(Xte)
pred_even = utils.vote(cls, 20)
