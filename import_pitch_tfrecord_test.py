import os
import glob
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
import pdb

# create dataset from multiple tfrecord files
tfrecords_dir = "/media/linzhank/850EVO_1T/Works/Data/Ball pitch/pit2d9blk/tfrecord_20180423"  
filenames = glob.glob(os.path.join(tfrecords_dir, 'test*'))
dataset = tf.data.TFRecordDataset(filenames)

# collect all data info
# data_info = pd.DataFrame({'name':['scalar','vector','matrix','matrix_shape','tensor','tensor_shape'],
#                          'type':['int32','float32','float32',tf.int64, 'uint8',tf.int64],
#                          'shape':[(), (1, 3), (2, 3), (2, ), (460, 460, 3), (3, )],
#                          'isbyte':[False,False,True,False,False,False],
#                          'length_type':['fixed','fixed','var','fixed','fixed','fixed']},
#                          columns=['name','type','shape','isbyte','length_type','default'])

def parse_function(example_proto):
    # example_proto, tf_serialized
    features = {'colorspace': tf.FixedLenFeature(shape=(), dtype=tf.string, default_value="RGB"),
                'channels': tf.FixedLenFeature(shape=(), dtype=tf.int64, default_value=3), 
                'format': tf.FixedLenFeature(shape=(), dtype=tf.string, default_value="PNG"), 
                'filename': tf.FixedLenFeature(shape=(), dtype=tf.string, default_value=""), 
                'image': tf.FixedLenFeature(shape=(), dtype=tf.string, default_value=""), 
                'class/label': tf.FixedLenFeature(shape=(), dtype=tf.int64, default_value=1),
                'height': tf.FixedLenFeature(shape=(), dtype=tf.int64, default_value=360),
                'width': tf.FixedLenFeature(shape=(), dtype=tf.int64, default_value=640),
                'pitcher': tf.FixedLenFeature(shape=(), dtype=tf.string, default_value=""),
                'trial': tf.FixedLenFeature(shape=(), dtype=tf.string, default_value=""),
                'frame': tf.FixedLenFeature(shape=(), dtype=tf.string, default_value="")}
    # parse all features in a single example according to the dics
    parsed_features = tf.parse_single_example(example_proto, features)
    # decode string
    parsed_features['image'] = tf.decode_raw(parsed_features['image'], tf.uint8)
    # reshape image
    parsed_features['image'] = tf.reshape(parsed_features['image'], (224, 224, 3))
    return parsed_features

new_dataset = dataset.map(parse_function)
iterator = new_dataset.make_one_shot_iterator()
next_element = iterator.get_next()

sess = tf.InteractiveSession()

i = 1
while True:
    # 
    try:
        image = sess.run([next_element['image']])
    except tf.errors.OutOfRangeError:
        print("End of dataset")
        break
    else:
        print('==============example %s ==============' %i)
        print('tensor shape: %s | type: %s' %(image.shape, image.dtype))
    i+=1
plt.imshow(image)
plt.show()
