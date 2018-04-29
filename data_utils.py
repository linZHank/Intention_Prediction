from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import tensorflow as tf
import numpy as np
import os


filedir = "/media/linzhank/850EVO_1T/Works/Data/Ball pitch/pit2d9blk/dataset_config/travaltes_20180420"
# filedir = "/media/linzhank/DATA/Works/Intention_Prediction/Dataset/Ball pitch/pit2d9blk/dataset_config/travaltes_20180415"
width = 28
height = 28
imformat = 'gray'

def load_images(datapaths, height, width, imformat='color'):
  """Helper function for loading images and pre-processing 'em

  Args: 
    datapaths: a list contains all the paths and names of the images we want to load
    height: images resized height
    width: images resized width
    imformat: images loaded as color or grayscale
  Returns:
    image_data: 2-d array with shape [len(datapaths, height*width*num_channels)]
  """
  if imformat == 'gray':
    num_chnls = 1
  elif imformat =='color':
    num_chnls = 3
  image_data = np.empty([len(datapaths), height*width*num_chnls], dtype=np.float32)
  for i in range(len(datapaths)):
    # read in image according to imformat 
    if imformat == 'gray':
      img_raw = cv2.imread(datapaths[i], 0)
    elif imformat == 'color':
      img_raw = cv2.imread(datapaths[i], 1)
    # resize image according to height and width
    img_rsz = cv2.resize(img_raw, (height, width))
    # flatten image tensor to 1-d and save into the image_data array
    image_data[i] = np.resize(img_rsz, (height*width*num_chnls))

  return image_data

def load_files(fdir, usage, contents):
  """Generate list of paths and files, which are going to be loaded later

  Args:
    fdir: directory to the files that contains all data files
    usage: purpose of the file: train, validate or test
    content:  contents type of file: paths or labels
    Returns:
    datapaths: list of data file paths
  """
  fnames = os.path.join(fdir, usage+'_'+contents+'.txt')
  with open(fnames) as f:
    datapaths = f.read().splitlines()
  return datapaths

def get_train_data():
  train_images_paths = load_files(fdir=filedir,
                                  usage='train',
                                  contents='paths')
  train_images = load_images(datapaths=train_images_paths,
                             height=height,
                             width=width,
                             imformat=imformat)

  train_labels_paths = load_files(fdir=filedir,
                                  usage='train',
                                  contents='labels')
  train_labels = np.asarray(train_labels_paths, dtype=np.int32)
    
  return train_images, train_labels

def get_eval_data():
  eval_images_paths = load_files(fdir=filedir,
                                 usage='validate',
                                 contents='paths')
  eval_images = load_images(datapaths=eval_images_paths,
                            height=height,
                            width=width,
                            imformat=imformat)
  eval_labels_paths = load_files(fdir=filedir,
                                 usage='validate',
                                 contents='labels')
  eval_labels = np.asarray(eval_labels_paths, dtype=np.int32)
    
  return eval_images, eval_labels
