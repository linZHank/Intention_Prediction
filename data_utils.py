from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import tensorflow as tf
import numpy as np
import os


def load_images(imgpaths, h, w, imf='color'):
  """Read in images and pre-processing 'em

  Args: 
    imgpaths: a list contains all the paths and names of the images we want to load
    h: height image is going to resized to
    width: width image is going to resized to
    imf: image format when loaded as color or grayscale
  Returns:
    image_data: 2-d array with shape [len(imgpaths, h*w*num_channels)]
  """
  if imf == 'gray':
    num_chnls = 1
  elif imf =='color':
    num_chnls = 3
  image_data = np.empty([len(imgpaths), h*w*num_chnls], dtype=np.float32)
  for i in range(len(imgpaths)):
    # read in image according to imf 
    if imf == 'gray':
      img_raw = cv2.imread(imgpaths[i], 0)
    elif imf == 'color':
      img_raw = cv2.imread(imgpaths[i], 1)
    # resize image according to h and w
    img_rsz = cv2.resize(img_raw, (h, w))
    # flatten image tensor to 1-d and save into the image_data array
    image_data[i] = np.resize(img_rsz, (h*w*num_chnls))

  return image_data

def load_files(fdir, usg, ctns):
  """Generate list of paths and files, which are going to be loaded later

  Args:
    fdir: directory to all data files
    usg: purpose of the file: train, eval or test
    cnts:  contents type in file: paths or labels
    Returns:
    contents: list of contents in the file
  """
  fnames = os.path.join(fdir, usg+"_"+ctns+".txt")
  with open(fnames) as f:
    contents = f.read().splitlines()
  return contents

def get_train_data(filedir, height, width, imformat):
  """Helper function get data and lables for training
  
  Args:
    filedir: directory to all the data files
    usage: purpose of the file: train, eval and test
    contents: what's inside the file? image 'paths' or 'labels'
    height: image is going to resized to
    width: image is going to resized to
    imformat: image reading format, 'gray' for 1 channel or 'color' for 3 channels
  Returns:
    train_images: feed ready image data array in shape (num_examples, height, width, channels)
    train_labels: feed read image labels array in shape (num_examples,)
  """
  train_images_paths = load_files(filedir,
                                  "train",
                                  "paths")
  train_images = load_images(train_images_paths, height, width, imformat)

  train_labels_list = load_files(filedir,
                                 "train",
                                 "labels")
  train_labels = np.asarray(train_labels_list, dtype=np.int32)

  # shuffle
  assert len(train_images) == len(train_labels)
  for _ in range(1000):
    p = np.random.permutation(len(train_images))
    
  return train_images[p], train_labels[p]

def get_eval_data(filedir, height, width, imformat):
  """Helper function get data and lables for evaling
  
  Args:
    filedir: directory to all the data files
    usage: purpose of the file: train, eval and test
    contents: what's inside the file? image 'paths' or 'labels'
    height: image is going to resized to
    width: image is going to resized to
    imformat: image reading format, 'gray' for 1 channel or 'color' for 3 channels
  Returns:
    eval_images: feed ready image data array in shape (num_examples, height, width, channels)
    eval_labels: feed read image labels array in shape (num_examples,)
  """
  eval_images_paths = load_files(filedir,
                                  "validate",
                                  "paths")
  eval_images = load_images(eval_images_paths, height, width, imformat)

  eval_labels_list = load_files(filedir,
                                 "validate",
                                 "labels")
  eval_labels = np.asarray(eval_labels_list, dtype=np.int32)
    
  return eval_images, eval_labels

def make_input_fn(filenames, num_threads, name, batch_size=64, buffer_size=4096):
  """Make input function for Estimator API
  
  Args:
    filenames: list of TFRecord filenames
    name: specify purpose of input_fn: train or eval
    batch_size:
    buffer_size: random shuffling is done on the buffer, so it must be big enough
  Returns:
    features:
    labels:
  """
  dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=num_threads)
