import os
import glob
import numpy as np
import random
import cv2


# # debug and experiment
# color_train_path = '/media/linzhank/DATA/Works/Action_Recognition/Data/train/color'
# # color_test_path = '/media/linzhank/DATA/Works/Action_Recognition/Data/test/color'
# # depth_train_path = '/media/linzhank/DATA/Works/Action_Recognition/Data/train/depth'
# # depth_test_path = '/media/linzhank/DATA/Works/Action_Recognition/Data/test/depth'
# img_size = 224
# classes = ['blk01', 'blk02', 'blk03', 'blk04', 'blk05', 'blk06', 'blk07', 'blk08', 'blk09']


# Load color image
def load_color(image_path, image_size, classes, frames): # frames range: 1-55
    images = []
    labels = []
    ids = []
    cls = []
    include_frames = []

    print('Reading color images')
    for fld in classes:   # assuming data directory has a separate folder for each class, and that each folder is named after the class
        index = classes.index(fld)
        print('Loading {} files (Index: {})'.format(fld, index))
        path = os.path.join(image_path, fld, '*', '*g')
        files = sorted(glob.glob(path))
        # frames need to be kept
        for frm in frames:
            include_frames.append('frame_' + '{:04d}'.format(frm))

        files = [inc for inc in files if any(names in inc for names in include_frames)]
        # # frames need to be excluded
        # exclude_frames = ['frame_0001', 'frame_0002', 'frame_0003', 'frame_0004', 'frame_0005'] # exclude first 5 frames or 200ms
        # files = [ex for ex in files if not any(names in ex for names in exclude_frames)]

        for fl in files:
            image = cv2.imread(fl)
            image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
            images.append(image)
            label = np.zeros(len(classes))
            label[index] = 1.0
            labels.append(label)
            flbase = os.path.basename(fl)
            ids.append(flbase)
            cls.append(fld)
    images = np.array(images)
    labels = np.array(labels)
    ids = np.array(ids)
    cls = np.array(cls)

    return images, labels, ids, cls


# Load depth image
def load_depth(image_path, image_size, classes, frames): # frames range: 1-55
    images = []
    labels = []
    ids = []
    cls = []
    include_frames = []

    print('Reading depth images')
    for fld in classes:   # assuming data directory has a separate folder for each class, and that each folder is named after the class
        index = classes.index(fld)
        print('Loading {} files (Index: {})'.format(fld, index))
        path = os.path.join(image_path, fld, '*', '*g')
        files = sorted(glob.glob(path))
        # frames need to be kept
        for frm in frames:
            include_frames.append('frame_' + '{:04d}'.format(frm))

        files = [inc for inc in files if any(names in inc for names in include_frames)]
        # # frames need to be excluded
        # exclude_frames = ['frame_0001', 'frame_0002', 'frame_0003', 'frame_0004', 'frame_0005'] # exclude first 5 frames or 200ms
        # files = [ex for ex in files if not any(names in ex for names in exclude_frames)]

        for fl in files:
            image = cv2.imread(fl, -1)
            image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
            images.append(image)
            label = np.zeros(len(classes))
            label[index] = 1.0
            labels.append(label)
            flbase = os.path.basename(fl)
            ids.append(flbase)
            cls.append(fld)

    images = np.array(images)
    labels = np.array(labels)
    ids = np.array(ids)
    cls = np.array(cls)

    return images, labels, ids, cls


# Load color trials
def load_color_vector(image_path, image_size, classes): # frames range: 1-55
    vectors = []
    labels = []
    ids = []
    cls = []

    print('Reading color trails')
    for fld in classes:   # assuming data directory has a separate folder for each class, and that each folder is named after the class
        index = classes.index(fld)
        print('Loading {} files (Index: {})'.format(fld, index))
        trial_path = os.path.join(image_path, fld, '*')
        trials = sorted(glob.glob(trial_path))
        for tr in trials:
            file_path = os.path.join(trial_path, tr, '*g')
            files = sorted(glob.glob(file_path))
            trial_vector = []
            for fl in files:
                image = cv2.imread(fl)
                image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
                flat_image = np.reshape(image, -1)
                trial_vector = np.concatenate((trial_vector, flat_image), axis=0)
                # trial_vector.append(flat_image)

            vectors.append(trial_vector)
            label = np.zeros(len(classes))
            label[index] = 1.0
            labels.append(label)
            trbase = os.path.basename(tr)
            ids.append(trbase)
            cls.append(fld)

    vectors = np.array(vectors)
    labels = np.array(labels)
    ids = np.array(ids)
    cls = np.array(cls)

    return vectors, labels, ids, cls
