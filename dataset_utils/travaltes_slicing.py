""" 
Separate the raw dataset to train/valiadate/test. 
Make notations for each image with filename, label, pitcher, trial, frame for each image file.
Save lists of above to root of raw data.
"""

import numpy as np
import scipy.io as spio
import os
import glob
import datetime
import errno

data_dir = '/media/linzhank/DATA/Works/Intention_Prediction/Dataset/Ball pitch/pit2d9blk'

def find_pitch_init(joint_path, intent, pitcher, trial):
  """Find the initiating moment of each pitching trial.
  
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

def notate_image_files(data_dir):
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
  # paths
  trainpaths = []
  validatepaths = []
  testpaths = []
  # labels
  trainlabels = []
  validatelabels = []
  testlabels = []
  # pitchers
  trainpitchers = []
  validatepitchers = []
  testpitchers = []
  # trials
  traintrials = []
  validatetrials = []
  testtrials = []
  # frames
  trainframes = []
  validateframes = []
  testframes = []

  intents = []
  labels = []
  
  
  filenames = []

  # Leave label index 0 empty as a background class.
  label_index = 1

  # Construct the list of PNG files and labels.
  intent_paths = sorted(glob.glob(color_path+'/*'))
  for ipath in intent_paths:
    intent = ipath.split('/')[-1] # e.g. 4
    intents.append(intent)
    labels.append(label_index)

    pitcher_paths = sorted(glob.glob(ipath+'/*'))
    for ppath in pitcher_paths:
      pitcher = ppath.split('/')[-1] # e.g. 'ZL'
      trial_paths = sorted(glob.glob(ppath+'/*'))
      np.random.shuffle(trial_paths) # shuffle all 10 trials, before travaltes arrangement
      #separate images to train, val, test (travaltes), 6/2/2
      train_trial_paths = trial_paths[:int(0.6*len(trial_paths))]
      val_trial_paths = trial_paths[int(0.6*len(trial_paths)):int(0.8*len(trial_paths))]
      test_trial_paths = trial_paths[int(0.8*len(trial_paths)):]
      for trnpath in train_trial_paths:
        trial = trnpath.split('/')[-1] # e.g. '201802071615_trial00'
        _, init_frmid = find_pitch_init(joint_path, intent, pitcher, trial)
        train_img_paths = sorted(glob.glob(trnpath+'/*.png'))[init_frmid:init_frmid+45]
        # summarize training data
        trainpaths += train_img_paths
        trainlabels += [label_index] * len(train_img_paths)
        assert len(trainpaths) == len(trainlabels)
        trainpitchers += [pitcher] * len(train_img_paths)
        assert len(trainpitchers) == len(trainpaths)
        traintrials += [trial] * len(train_img_paths)
        assert len(traintrials) == len(trainpaths)
        trainframes += ['_'.join(impath.split('.')[0].split('_')[-2:])
                        for impath in train_img_paths] # e.g. 'frame_0016'
        assert len(trainframes) == len(trainpaths)

      for valpath in val_trial_paths:
        trial = valpath.split('/')[-1]
        _, init_frmid = find_pitch_init(joint_path, intent, pitcher, trial)
        val_img_paths = sorted(glob.glob(valpath+'/*.png')[init_frmid:init_frmid+45])
        # summarize validating data
        validatepaths += val_img_paths
        validatelabels += [label_index] * len(val_img_paths)
        assert len(validatelabels) == len(validatepaths)
        validatepitchers += [pitcher] * len(val_img_paths)
        assert len(validatepitchers) == len(validatepaths)
        validatetrials += [trial] * len(val_img_paths)
        assert len(validatetrials) == len(validatepaths)
        validateframes += ['_'.join(impath.split('.')[0].split('_')[-2:])
                        for impath in val_img_paths]
        assert len(validateframes) == len(validatepaths)

      for tespath in test_trial_paths:
        trial = tespath.split('/')[-1]
        _, init_frmid = find_pitch_init(joint_path, intent, pitcher, trial)
        test_img_paths = sorted(glob.glob(tespath+'/*.png')[init_frmid:init_frmid+45])
        # summarize testing data
        testpaths += test_img_paths
        testlabels += [label_index] * len(test_img_paths)
        assert len(testlabels) == len(testpaths)
        testpitchers += [pitcher] * len(train_img_paths)
        assert len(testpitchers) == len(testpaths)
        testtrials += [trial] * len(train_img_paths)
        assert len(testtrials) == len(testpaths)
        testframes += ['_'.join(impath.split('.')[0].split('_')[-2:])
                        for impath in test_img_paths]
        assert len(testframes) == len(testpaths)

    # Construct the list of PNG files and labels
    print('Finished finding files in {}.'.format(intent))
    label_index += 1 # label index increase when investigating new intent

  print('Found {num_trn} images for training; \nFound {num_val} images for validating; \nFound {num_tes} images for testing.'.format(num_trn=len(trainpaths),
                                                                                                                                     num_val=len(validatepaths),
                                                                                                                                     num_tes=len(testpaths)))

  return trainpaths, trainlabels, trainpitchers, traintrials, trainframes, \
      validatepaths, validatelabels, validatepitchers, validatetrials, validateframes, \
      testpaths, testlabels, testpitchers, testtrials, testframes

def main():
    (trainpaths, trainlabels, trainpitchers, traintrials, trainframes,
    validatepaths, validatelabels, validatepitchers, validatetrials,
    validateframes, testpaths, testlabels, testpitchers, testtrials, testframes) = notate_image_files(data_dir)
    # write data info into text files
    today = datetime.datetime.today().strftime("%Y%m%d")

    filename = data_dir+'/dataset_config/travaltes_'+today+'/train_paths.txt'
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    with open(filename, 'w') as f:
        for item in trainpaths:
            f.write("{}\n".format(item))

    filename = data_dir+'/dataset_config/travaltes_'+today+'/train_labels.txt'
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    with open(filename, 'w') as f:
        for item in trainlabels:
            f.write("{}\n".format(item))

    filename = data_dir+'/dataset_config/travaltes_'+today+'/train_pitchers.txt'
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    with open(filename, 'w') as f:
        for item in trainpitchers:
            f.write("{}\n".format(item))

    filename = data_dir+'/dataset_config/travaltes_'+today+'/train_trials.txt'
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    with open(filename, 'w') as f:
        for item in traintrials:
            f.write("{}\n".format(item))

    filename = data_dir+'/dataset_config/travaltes_'+today+'/train_frames.txt'
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    with open(filename, 'w') as f:
        for item in trainframes:
            f.write("{}\n".format(item))

    filename = data_dir+'/dataset_config/travaltes_'+today+'/validate_paths.txt'
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    with open(filename, 'w') as f:
        for item in validatepaths:
            f.write("{}\n".format(item))

    filename = data_dir+'/dataset_config/travaltes_'+today+'/validate_labels.txt'
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    with open(filename, 'w') as f:
        for item in validatelabels:
            f.write("{}\n".format(item))

    filename = data_dir+'/dataset_config/travaltes_'+today+'/validate_pitchers.txt'
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    with open(filename, 'w') as f:
        for item in validatepitchers:
            f.write("{}\n".format(item))

    filename = data_dir+'/dataset_config/travaltes_'+today+'/validate_trials.txt'
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    with open(filename, 'w') as f:
        for item in validatetrials:
            f.write("{}\n".format(item))

    filename = data_dir+'/dataset_config/travaltes_'+today+'/validate_frames.txt'
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    with open(filename, 'w') as f:
        for item in validateframes:
            f.write("{}\n".format(item))

    filename = data_dir+'/dataset_config/travaltes_'+today+'/test_paths.txt'
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    with open(filename, 'w') as f:
        for item in testpaths:
            f.write("{}\n".format(item))

    filename = data_dir+'/dataset_config/travaltes_'+today+'/test_labels.txt'
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    with open(filename, 'w') as f:
        for item in testlabels:
            f.write("{}\n".format(item))

    filename = data_dir+'/dataset_config/travaltes_'+today+'/test_pitchers.txt'
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    with open(filename, 'w') as f:
        for item in testpitchers:
            f.write("{}\n".format(item))

    filename = data_dir+'/dataset_config/travaltes_'+today+'/test_trials.txt'
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    with open(filename, 'w') as f:
        for item in testtrials:
            f.write("{}\n".format(item))

    filename = data_dir+'/dataset_config/travaltes_'+today+'/test_frames.txt'
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    with open(filename, 'w') as f:
        for item in testframes:
            f.write("{}\n".format(item))

if __name__ == '__main__':
    main()
