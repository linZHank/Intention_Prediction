#DATA_DIR = '/media/linzhank/DATA/Works/Intention_Prediction/Dataset/Ball pitch/pit2d9blk'
DATA_DIR = '/media/linzhank/850EVO_1T/Works/Data/Ball pitch/pit2d9blk'

trainpathsfile = DATA_DIR+'/dataset_config/travaltes_20180420/train_paths.txt'
validatepathsfile = DATA_DIR+'/dataset_config/travaltes_20180420/validate_paths.txt'
testpathsfile = DATA_DIR+'/dataset_config/travaltes_20180420/test_paths.txt'

trainlabelsfile = DATA_DIR+'/dataset_config/travaltes_20180420/train_labels.txt'
validatelabelsfile = DATA_DIR+'/dataset_config/travaltes_20180420/validate_labels.txt'
testlabelsfile = DATA_DIR+'/dataset_config/travaltes_20180420/test_labels.txt'

trainpaths = []
validatepaths = []
testpaths = []

trainlabels = []
validatelabels = []
testlabels = []

with open(trainpathsfile, 'r') as f:
  trainpaths = f.read().split('\n')[:-1]

with open(validatepathsfile, 'r') as f:
  validatepaths = f.read().split('\n')[:-1]

with open(testpathsfile, 'r') as f:
  testpaths = f.read().split('\n')[:-1]
    
with open(trainlabelsfile, 'r') as f:
  trainlabels = f.read().split('\n')[:-1]
  trainlabels = map(int, trainlabels)

with open(validatelabelsfile, 'r') as f:
  validatelabels = f.read().split('\n')[:-1]
  validatelabels = map(int, validatelabels)
  
with open(testlabelsfile, 'r') as f:
  testlabels = f.read().split('\n')[:-1]
  testlabels = map(int, testlabels)
  
# Re-allocate your train/test partition: 8/2
train_dict = {}
test_dict = {}

trainpaths_new = trainpaths + validatepaths
testpaths_new = testpaths

trainlabels_new = trainlabels + validatelabels
testlabels_new = testlabels

# Create dicts
for (idx, key) in enumerate(trainpaths_new):
  if not idx % 45:
    train_dict[key] = trainlabels_new[idx]

for (idx, key) in enumerate(testpaths_new):
  if not idx % 45:
    test_dict[key] = testlabels_new[idx]
