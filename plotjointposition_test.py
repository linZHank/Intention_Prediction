""" This script plots joint position change for any pitch trial """

import numpy as np
import scipy.io as spio
import glob
import os
import matplotlib.pyplot as plt

allmatfile_dir = '/media/linzhank/DATA/Works/Intention_Prediction/Dataset/Ball pitch/pit2d9blk/joint'

def pick_intent(file_dir, intent_index):
    ittpath_list = sorted(glob.glob(os.path.join(file_dir, '*')))
    intent_path = ittpath_list[intent_index]
    print("You picked intent%.2d" %(intent_index+1))
    
    return intent_path 

def pick_pitcher(intent_path, name):
    pchrpath_list = sorted(glob.glob(os.path.join(intent_path, '*')))
    pchr_list = ['HJ', 'LSC', 'LYX', 'MY', 'XH', 'ZL']
    pitcher_path = pchrpath_list[pchr_list.index(name)]
    print("You are looking at %s's data" %name)

    return pitcher_path

def pick_trial(pitcher_path, trial_index):
    trlpath_list = sorted(glob.glob(os.path.join(pitcher_path, '*')))
    trial_path = trlpath_list[trial_index]
    print("You selected trial at %s" %trial_path)
    
    return trial_path


def main():
    intent_index = input("Specify intent(from 0 to 8): ")
    intent_path = pick_intent(allmatfile_dir, intent_index)
    pitcher_name = raw_input("Give me a name: ")
    pitcher_path = pick_pitcher(intent_path, pitcher_name)
    trial_index = input("Give me a number (from 0 to 9): ")
    trial_path = pick_trial(pitcher_path, trial_index)
    filename = glob.glob(os.path.join(trial_path, '*.mat'))

    joint_position = spio.loadmat(filename[0])['joint_positions_3d']
    dist = []
    for i in range(joint_position.shape[2]):
        d = np.linalg.norm(joint_position[:,:,i] - joint_position[:,:,0])
        dist.append(d)
    for idx, elem in enumerate(dist):
        print(idx, elem)

    plt.plot(dist)
    plt.show()
    

if __name__ == '__main__':
    main()
    
        
    

    
    
    
