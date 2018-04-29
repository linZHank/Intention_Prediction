from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import tensorflow as tf
import numpy as np
import os
import glob

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
        img_raw = cv2.imread(datapaths, 0)
      elif imformat == 'color':
        img_raw = cv2.imread(datapaths, 1)
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
  fnames = os.path.join(fdir, usage+'_'+contents)
  with open(fnames) as f:
    datapaths = f.read().splitlines()
  return

class Pitch2dData(object):
  """Helper class for dealing with pitch2d dataset
  """
  def __init__(self):
    self.filedir = "/media/linzhank/DATA/Works/Intention_Prediction/Dataset/Ball pitch/pit2d9blk/dataset_config/travaltes_20180415"
    self.width = 28
    self.height = 28
    self.imformat = 'gray'
    
  def train_data():
    train_data_paths = load_files(fdir=self.filedir,
                                  usage='train',
                                  contents='paths')
    train_data = load_images(datapaths=train_data_paths,
                             width=self.width,
                             height=self.height,
                             imformat=self.imformat)

    return train_data

  def train_labels():
    train_labels_paths = load_files(fdir=self.filedir,
                                    usage='train',
                                    contents='labels')
    train_labels = np.asarray(train_labels_paths, dtype=np.int32)
    
    return train_labels

  def eval_data():
    eval_data_paths = load_files(fdir=self.filedir,
                                 usage='validate',
                                 contents='paths')
    eval_data = load_images(datapaths=eval_data_paths,
                            width=self.width,
                            height=self.height,
                            imformat=self.imformat)

    return eval_data

  def eval_labels():
    eval_labels_paths = load_files(fdir=self.filedir,
                                   usage='validate',
                                   contents='labels')
    eval_labels = np.asarray(eval_labels_paths, dtype=np.int32)
    
    return eval_labels
