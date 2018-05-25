from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

num_frames = 150

def detect_init(joint_vectors):
  # reshape(11250,) to (150,75)
  joint_matrix = joint_vectors.reshape(
    joint_vectors.shape[0], # trial
    num_frames, # frame
    75
  )
  start_id = np.zeros((joint_vectors.shape[0])).astype(int)
  for i in range(joint_matrix.shape[0]):
    inc_inarow = 0
    dist = []
    for j in range(joint_matrix.shape[1]):
      d = np.linalg.norm(joint_matrix[i,j,:] - joint_matrix[i,0,:])
      dist.append(d)
      if d > 4:
        inc_inarow += 1
      else:
        inc_inarow = 0
      if inc_inarow > 32:
        start_id[i] = j - 32 + 4
        # plt.plot(dist)
        # plt.show()
        # in case pitch started too late
        if start_id[i] > 99:
          start_id[i] = 100
        break

  return start_id
  
