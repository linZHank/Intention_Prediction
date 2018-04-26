import tensorflow as tf
# for displaying image
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
# for data operation
import pandas as pd
import numpy as np

np.set_printoptions(precision=3)
# display function
def display(alist, show = True):
    print('type:%s\nshape: %s' %(alist[0].dtype,alist[0].shape))
    if show:
        for i in range(3):
            print('example%s\n%s' %(i,alist[i]))

scalars = np.array([1,2,3],dtype=np.int32)
print('\nscalar')
display(scalars)

vectors = np.array([[0.1,0.1,0.1],
                   [0.2,0.2,0.2],
                   [0.3,0.3,0.3]],dtype=np.float32)
print('\nvector')
display(vectors)

matrices = np.array([np.array((vectors[0],vectors[0])),
                    np.array((vectors[1],vectors[1])),
                    np.array((vectors[2],vectors[2]))],dtype=np.float32)
print('\nmatrix')
display(matrices)

# shape of image:(806,806,3)
img=mpimg.imread('/home/linzhank/playground/toy_imgs/Hank_study.jpeg') 
tensors = np.array([img,img,img])
# show image
print('\n3D-tensor')
display(tensors, show = False)
plt.imshow(img)

