from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import itertools

num_frames = 150

def detectInit(joint_vectors):
  # reshape(num_examples, 11250) to (num_examples, 150, 75)
  joint_matrix = joint_vectors.reshape(
    joint_vectors.shape[0], # trial
    num_frames, # frame
    75
  )
  start_id = np.zeros((joint_vectors.shape[0])).astype(int)
  for i in range(joint_matrix.shape[0]):
    d0 = 0
    inc_inarow = 0
    dist = []
    for j in range(joint_matrix.shape[1]):
      d = np.linalg.norm(joint_matrix[i,j,:] - joint_matrix[i,0,:])
      dist.append(d)
      if d > 4: # and d > d0:
        inc_inarow += 1
      else:
        inc_inarow = 0
      if inc_inarow > 20:
        start_id[i] = j - 20 + 4
        # in case pitch started too late
        if start_id[i] > 99:
          start_id[i] = 110
        break
      d0 = d

  return start_id

def vote(classes, num_frames, vote_opt="even"):
  """ Vote pitching trial by frame-wise classes using vote options

      Args:
        classes: classified results of the data frame-wise
        num_frames: scalar, indicate number of frames in a single trial
        vote_opt: vote options, 'even', 'disc', 'logr'
      Returns:
        prediction: scalar labels of the trials
  """
  # Convert classes into one-hot labels
  num_classes = np.max(classes)+1
  classes = np.eye(num_classes)[classes]
  num_examples = int(classes.shape[0]/num_frames)
  prediction = np.zeros(num_examples).astype(int)
  for i in range(num_examples):
    if vote_opt == "even":
      accumulation = np.sum(classes[i*num_frames:(i+1)*num_frames], axis=0)
      prediction[i] = np.argmax(accumulation, axis=0)
    elif vote_opt == "disc":
      gamma = 0.9 # discount
      accumulation = np.zeros(num_classes)
      for f in range(num_frames):
        accumulation += np.power(gamma, (num_frames-f))*classes[i*num_frames+f]
      prediction[i] = np.argmax(accumulation, axis=0)
    elif vote_opt == "logr":
      accumulation = np.zeros(num_classes)
      for f in range(num_frames):
        accumulation += np.log1p(f)*classes[i*num_frames+f]
      prediction[i] = np.argmax(accumulation, axis=0)

  return prediction

def prepJointData(raw_data, raw_labels, init_id, num_frames, shuffle=False):
  """Prepare joint data for training and testing

     Args:
       example_id: int scalar, example index to specify the trial
       raw_data: float array, original data in shape (num_examples, frames, num_joints*num_coords)
       raw_labels: int array, (num_examples,)
       init_id: int scalar, frame index mark the start of the pitch
       num_frames: int scalar, how many frames are used for each trial
       shuffle: boolean, shuffle the data or not
     Returns:
       X:
       y: (num_examples*num_frames, 1)
"""
  X = []
  y = []
  num_examples = raw_data.shape[0]
  for ie in range(num_examples): # loop all examples in raw data
    example_data = raw_data[ie, init_id[ie]:init_id[ie]+num_frames, :]
    X.append(example_data)
    example_labels = raw_labels[ie]*np.ones((num_frames,), dtype=int)
    y.append(example_labels)
  X = np.array(X).reshape(num_examples*num_frames, -1)
  y = np.array(y).reshape(X.shape[0], )
  if shuffle:
    p = np.random.permutation(X.shape[0]) # generate shuffled index
    X = X[p]
    y = y[p]

  return X, y

def plotConfusionMatrix(cm, classes, normalize=False, cmap=plt.cm.YlOrBr):
  """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    Args:
      cm: confution matrix
      classes: class names
      normalize:
      cmap: color map
  """
  # Print out confusion matrix
  if normalize:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Normalized confusion matrix")
  else:
    print('Confusion matrix, without normalization')
  print(cm)
  #
  # plt.figure(figsize=(6,5.5))
  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  # plt.title(title)
  plt.colorbar(ticks = np.linspace(0,4,5))
  plt.clim(0,4)
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=45)
  plt.yticks(tick_marks, classes)
  #
  fmt = '.2f' if normalize else 'd'
  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")
  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')

  plt.show()

def plotAccBar(train_acc, test_acc, num_frames):
  """Plot bar chart for training and testing accuracies
  """
  fig, ax = plt.subplots()
  #x_ticks = tuple([str(i) for i in num_frames])
  #y_ticks = tuple([str(j) for j in np.arange(0, 1, 0.1)])
  x_pos = np.arange(num_frames.shape[0])
  y_pos = np.linspace(0, 1, num=11, dtype=np.float16)
  x_ticks = num_frames.astype(str)
  y_ticks = y_pos.astype('|S4')
  
  bar_width = 0.35
  # Plot bars
  rects1 = plt.bar(x_pos, train_acc, bar_width, color = 'k', label = 'Train accuracy')
  rects2 = plt.bar(x_pos + bar_width, test_acc, bar_width, color = '#ff9b1a', label = 'Test accuracy')
  plt.xlabel('Frames')
  plt.ylabel('Accuracy')
  plt.xlim(0, 7+2*bar_width)
  plt.ylim(0, 1.01)
  plt.xticks(x_pos+bar_width, x_ticks)
  plt.yticks(y_pos, y_ticks)
  ax.legend(loc=9, bbox_to_anchor=(0.5, 1.1),
          ncol=2, fancybox=True, shadow=True) # upper center

  plt.show()
