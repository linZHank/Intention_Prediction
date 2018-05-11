import numpy as np
import cv2

fdir = '/media/linzhank/DATA/Works/Intention_Prediction/Dataset/Ball pitch/pit2d9blk/dataset_config/travaltes_20180415/test_paths.txt'

with open(fdir) as f:
  img_files = f.read().splitlines()

for i in range(450, 495):
  frame = cv2.imread(img_files[i])
  frame_crop = frame[:, frame.shape[1]/2-72:frame.shape[1]/2+72]
  
  cv2.imshow('croppedframe',frame_crop)
  if cv2.waitKey(10) & 0xFF == ord('q'):
    break

cv2.destroyAllWindows()
