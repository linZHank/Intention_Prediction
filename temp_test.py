from datetime import datetime
import os
import glob
import random
import sys
import threading

import numpy as np
import scipy.io as spio
import tensorflow as tf

# DATA_DIR = '/media/linzhank/DATA/Works/Intention_Prediction/Dataset/Ball pitch/pit2d9blk'
DATA_DIR = '/media/linzhank/850EVO_1T/Works/Data/Ball pitch/pit2d9blk'

class ImageCoder(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Create a single Session to run all image coding calls.
    self._sess = tf.Session()

    # Initializes function that decodes RGB JPEG data.
    self._decode_image_data = tf.placeholder(dtype=tf.string)
    self._decode_image = tf.image.decode_image(self._decode_image_data, channels=3)

  def decode_image(self, image_data):
    image = self._sess.run(self._decode_image,
                           feed_dict={self._decode_image_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image


def _process_image(filename, coder):
  """Process a single image file.

  Args:
    filename: string, path to an image file e.g., '/path/to/example.JPG'.
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
  Returns:
    image_buffer: string, JPEG encoding of RGB image.
    height: integer, image height in pixels.
    width: integer, image width in pixels.
  """
  # Read the image file.
  with tf.gfile.FastGFile(filename, 'rb') as f:
    image_data = f.read()

  # Decode the RGB JPEG.
  image = coder.decode_image(image_data)

  # Check that image converted to RGB
  assert len(image.shape) == 3
  height = image.shape[0]
  width = image.shape[1]
  assert image.shape[2] == 3

  return image_data, height, width

def find_pitch_init(joint_path, intent, pitcher, trial):
  """Find the moment of the initiating of each pitching trial.
  
  Args: 
    joint_path: string, path to the root dirctory of pitch data
    intent: string, pitching intent with numbers e.g. "intent02"
    pitcher: string, pitcher's name e.g. "ZL"
    trial: string, trial id with timestamp e.g. trial_201801302254

  Returns:
    dist: list of float, each float number indicate the euclidean distance between 
      joint positions of current frame and joint positions of first frame. 
    init_frame_id: integer, this number indicate the frame index of the pitching initiation.
  """
  matfile_path = os.path.join(joint_path, intent, pitcher, trial, '*.mat')
  matfile_name = glob.glob(matfile_path)[0]
  joint_position = spio.loadmat(matfile_name)['joint_positions_3d']
  window_size = 20
  dist = []
  for i in range(joint_position.shape[2]):
    d = np.linalg.norm(joint_position[:,:,i] - joint_position[:,:,0])
    dist.append(d)
  inc_inarow = 0
  di = 0 # index of distance
  while di < len(dist)-45 and inc_inarow <= window_size:
    if dist[di+1] > dist[di]:
      inc_inarow += 1
    else:
      inc_inarow = 0
    di += 1
  initframe = di - window_size
  return dist, initframe 

def _find_image_files(data_dir):
  """Build a list of all images files and labels in the data set.

  Args:
    data_dir: string, path to the root directory of images.

      Assumes that the ImageNet data set resides in PNG files located in
      the following directory structure.
        data_dir/intent##/ID/datetime_trial##/trial_datetime_frame_####.png

      We start the integer labels at 1 is to reserve label 0 as an
      unused background class.

  Returns:
    trainpaths: list of strings; each string is a path to an image file.
    validatepaths: list of strings; each string is a path to an image file.
    testpaths: list of strings; each string is a path to an image file.
    trainlabels: list of integer; each integer identifies the ground truth.
    validatelabels: list of integer; each integer identifies the ground truth.
    testlabels: list of integer; each integer identifies the ground truth.
  """
  color_path = os.path.join(data_dir, 'color')
  joint_path = os.path.join(data_dir, 'joint')
  print('Determining list of input files and labels from %s.' % data_dir)
  # Prepare training, validation and test data 
  trainpaths = []
  validatepaths = []
  testpaths = []

  trainlabels = []
  validatelabels = []
  testlabels = []

  intents = []
  labels = []
  filenames = []

  # Leave label index 0 empty as a background class.
  label_index = 1

  # Construct the list of PNG files and labels.
  intent_paths = sorted(glob.glob(color_path+'/*'))
  for ipath in intent_paths:
    intent = ipath.split('/')[-1]
    intents.append(intent)
    labels.append(label_index)

    pitcher_paths = sorted(glob.glob(ipath+'/*'))
    for ppath in pitcher_paths:
      pitcher = ppath.split('/')[-1]
      trial_paths = sorted(glob.glob(ppath+'/*'))
      np.random.shuffle(trial_paths) # shuffle all 10 trials, before travaltes arrangement
      #separate images to train, val, test (travaltes), 6/2/2
      train_trial_paths = trial_paths[:int(0.6*len(trial_paths))]
      val_trial_paths = trial_paths[int(0.6*len(trial_paths)):int(0.8*len(trial_paths))]
      test_trial_paths = trial_paths[int(0.8*len(trial_paths)):]
      for trnpath in train_trial_paths:
        trial = trnpath.split('/')[-1]
        _, init_frmid = find_pitch_init(joint_path, intent, pitcher, trial)
        train_img_paths = sorted(glob.glob(trnpath+'/*.png'))[init_frmid:init_frmid+45]
        trainpaths += train_img_paths
        trainlabels += [label_index] * len(train_img_paths)

      for valpath in val_trial_paths:
        trial = trnpath.split('/')[-1]
        _, init_frmid = find_pitch_init(joint_path, intent, pitcher, trial)
        val_img_paths = glob.glob(valpath+'/*.png')[init_frmid:init_frmid+45]
        validatepaths += val_img_paths
        validatelabels += [label_index] * len(val_img_paths)

      for tespath in test_trial_paths:
        trial = trnpath.split('/')[-1]
        _, init_frmid = find_pitch_init(joint_path, intent, pitcher, trial)
        test_img_paths = glob.glob(tespath+'/*.png')[init_frmid:init_frmid+45]
        testpaths += test_img_paths
        testlabels += [label_index] * len(test_img_paths)

    # Construct the list of PNG files and labels
    print('Finished finding files in {}.'.format(intent))
    label_index += 1 # label index increase when investigating new intent

  print('Found {num_trn} images for training; \nFound {num_val} images for validating; \nFound {num_tes} images for testing.'.format(num_trn=len(trainpaths),
                                                  num_val=len(validatepaths),
                                                  num_tes=len(testpaths)))

  return trainpaths, validatepaths, testpaths, trainlabels, validatelabels, testlabels


trainpaths, validatepaths, testpaths, trainlabels, validatelabels, testlabels = _find_image_files(DATA_DIR)

