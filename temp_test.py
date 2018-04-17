import os
import glob

def read_annotation_files(name, directory):
  files_location = os.path.join(directory,
                                'dataset_config', 'travaltes_20180415', name+'*.txt')
  info_files = sorted(glob.glob(files_location))
  annotations = {}
  keys = []
  
  for ifile in info_files:
    key = ifile.split('/')[-1].split('.')[0]
    keys.append(key)
    with open(ifile) as f:
      annotations[key] = f.read().split('\n')[:-1]
      
  return annotations

data_dir = '/media/linzhank/DATA/Works/Intention_Prediction/Dataset/Ball pitch/pit2d9blk'
name = 'train'
annotations = read_annotation_files(name, data_dir)
