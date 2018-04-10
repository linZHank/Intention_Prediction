import glob
import numpy as np

data_dir = '/media/linzhank/850EVO_1T/Works/Data/Ball pitch/pit2d9blk/color'
train_paths = []
val_paths = []
test_paths = []

train_labels = []
val_labels = []
test_labels = []

intents = []
labels = []
filenames = []

# Leave label index 0 empty as a background class.
label_index = 1

# Construct the list of PNG files and labels.
intent_paths = glob.glob(data_dir+'/*')
for ipath in intent_paths:
    intent = ipath.split('/')[-1]
    intents.append(intent)
    labels.append(label_index)

    pitcher_paths = glob.glob(ipath+'/*')
    for ppath in pitcher_paths:
        trial_paths = glob.glob(ppath+'/*')
        np.random.shuffle(trial_paths)
        #separate images to train, val, test (travaltes)
        train_trial_paths = trial_paths[:int(0.6*len(trial_paths))]
        val_trial_paths = trial_paths[int(0.6*len(trial_paths)):int(0.8*len(trial_paths))]
        test_trial_paths = trial_paths[int(0.8*len(trial_paths)):]
        for trntrlpath in train_trial_paths:
            train_img_paths = glob.glob(trntrlpath+'/*.png')
            train_paths += train_img_paths
            train_labels += [label_index] * len(train_img_paths)
        for valtrlpath in val_trial_paths:
            val_img_paths = glob.glob(valtrlpath+'/*.png')
            val_labels += val_img_paths
            val_labels += [label_index] * len(val_img_paths)
        for testrlpath in test_trial_paths:
            test_img_paths = glob.glob(testrlpath+'/*.png')
            test_labels += test_img_paths
            test_labels += [label_index] * len(test_img_paths)

    label_index += 1
