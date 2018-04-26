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
    features = {'image/colorspace': tf.FixedLenFeature(shape=(), dtype=tf.string, default_value="RGB"),
                'image/channels': tf.FixedLenFeature(shape=(), dtype=tf.int64, default_value=3), 
                'image/format': tf.FixedLenFeature(shape=(), dtype=tf.string, default_value="PNG"), 
                'image/filename': tf.FixedLenFeature(shape=(), dtype=tf.string, default_value=""), 
                'image/encoded': tf.FixedLenFeature(shape=(), dtype=tf.string, default_value=""), 
                'image/class/label': tf.FixedLenFeature(shape=(), dtype=tf.int64, default_value=1),
                'image/height': tf.FixedLenFeature(shape=(), dtype=tf.int64, default_value=360),
                'image/width': tf.FixedLenFeature(shape=(), dtype=tf.int64, default_value=640),
                'image/pitcher': tf.FixedLenFeature(shape=(), dtype=tf.string, default_value=""),
                'image/trial': tf.FixedLenFeature(shape=(), dtype=tf.string, default_value=""),
                'image/frame': tf.FixedLenFeature(shape=(), dtype=tf.string, default_value="")}
    
    # parse all features in a single example according to the dics
    parsed_features = tf.parse_single_example(example_proto, features)
    # decode string
    decoded = tf.image.decode_image((parsed_features['image/encoded']))
    # reshape image
    reshaped = tf.reshape(decoded, [360, 640, 3])
    # resize image
    resized = tf.cast(tf.image.resize_images(reshaped, [224, 224]), tf.uint8)
    # parsed_features['resized'] = tf.image.resize_images(parsed_features['decoded'], [224, 224, 3])
    label = tf.cast(parsed_features['image/class/label'], tf.int32)
    return {"image_bytes": parsed_features['image/encoded'],
            "image_decoded": decoded,
            "image_reshaped": reshaped,
            "image_resized": resized}, label

dataset = dataset.map(parse_function)
dataset = dataset.batch(32)
iterator = dataset.make_one_shot_iterator()
features, labels = iterator.get_next()

sess = tf.InteractiveSession()

i = 1
while i < 4:
    # 
    try:
        image_reshaped = sess.run(features['image_reshaped'])
        image_resized = sess.run(features['image_resized'])
        
    except tf.errors.OutOfRangeError:
        print("End of dataset")
        break
    else:
        print('==============example %s ==============' %i)
        print('image shape: %s | type: %s' %(image_reshaped.shape, image_reshaped.dtype))
        print('image shape: %s | type: %s' %(image_resized.shape, image_resized.dtype))
    i+=1
plt.imshow(image_resized[0])
plt.show()
