import cv2
import numpy as np
import tensorflow as tf


data_dir = "/media/linzhank/850EVO_1T/Works/Data/Ball pitch/pit2d9blk/dataset_config/travaltes_20180420"
height = 360
width = 640
channels = 3

# Load training data
pathfile_train = data_dir+"/train_paths.txt"
labelfile_train = data_dir+"/train_labels.txt"
# Read in training image file paths and load images accordingly
with open(pathfile_train) as pf:
    imgpaths = pf.readlines()
    train_data = np.zeros((len(imgpaths), height, width, channels), dtype=np.float32)
for i in range(len(imgpaths)):
    train_data[i] = cv2.imread(imgpaths[i].split('\n')[0])
    # Read in training label file paths and load labels
    with open(labelfile_train) as lf:
        train_labels = np.asarray(lf.read().splitlines(), dtype=np.int32)
            
# Load evaluation data
pathfile_eval = data_dir+"/validate_paths.txt"
labelfile_eval = data_dir+"/validate_labels.txt"
# Read in evaluation image file paths and load images accordingly
with open(pathfile_eval) as pf:
    imgpaths = pf.readlines()
    eval_data = np.zeros((len(imgpaths), height, width, channels), dtype=np.float32)
for i in range(len(imgpaths)):
    eval_data[i] = cv2.imread(imgpaths[i].split('\n')[0])
    # Read in evaluation label file paths and load labels
with open(labelfile_eval) as lf:
    eval_labels = np.asarray(lf.read().splitlines(), dtype=np.int32)
