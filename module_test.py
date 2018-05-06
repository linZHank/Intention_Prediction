from __future__ import absolute_import, division, print_function

import tensorflow as tf
import tensorflow.contrib.eager as tfe

import glob
import os
import pdb

tf.enable_eager_execution()

tfrecords_dir = "/media/linzhank/DATA/Works/Intention_Prediction/Dataset/Ball pitch/pit2d9blk/tfrecord_20180418"
filenames = glob.glob(os.path.join(tfrecords_dir, 'train*'))
# filenames = glob.glob(os.path.join(tfrecords_dir, 'validate*'))
dataset = tf.data.TFRecordDataset(filenames)
# Use `tf.parse_single_example()` to extract data from a `tf.Example`
# protocol buffer, and perform any additional per-record preprocessing.
def parse_function(example_proto):
    # example_proto, tf_serialized
    keys_to_features = {
        'image/colorspace': tf.FixedLenFeature(shape=(), dtype=tf.string, default_value="RGB"),
        'image/channels': tf.FixedLenFeature(shape=(), dtype=tf.int64, default_value=3), 
        'image/format': tf.FixedLenFeature(shape=(), dtype=tf.string, default_value="PNG"), 
        'image/filename': tf.FixedLenFeature(shape=(), dtype=tf.string, default_value=""), 
        'image/encoded': tf.FixedLenFeature(shape=(), dtype=tf.string, default_value=""), 
        'image/class/label': tf.FixedLenFeature(shape=(), dtype=tf.int64, default_value=1),
        'image/height': tf.FixedLenFeature(shape=(), dtype=tf.int64, default_value=360),
        'image/width': tf.FixedLenFeature(shape=(), dtype=tf.int64, default_value=640),
        'image/pitcher': tf.FixedLenFeature(shape=(), dtype=tf.string, default_value=""),
        'image/trial': tf.FixedLenFeature(shape=(), dtype=tf.string, default_value=""),
        'image/frame': tf.FixedLenFeature(shape=(), dtype=tf.string, default_value="")
    }
    
    # parse all features in a single example according to the dics
    parsed_features = tf.parse_single_example(example_proto, keys_to_features)
    # decode the encoded image to the (360, 640, 3) uint8 array
    decoded_image = tf.image.decode_image((parsed_features['image/encoded']))
    # reshape image
    reshaped_image = tf.reshape(decoded_image, [360, 640, 3])
    # resize decoded image
    resized_image = tf.cast(tf.image.resize_images(reshaped_image, [224, 224]), tf.float32)
    # label
    label = tf.cast(parsed_features['image/class/label']-1, tf.int32)
    label = tf.one_hot(indices=label-1, depth=9)
    
    return {"image_bytes": parsed_features['image/encoded'],
            "image_decoded": decoded_image,
            "image_reshaped": reshaped_image,
            "image_resized": resized_image}, label

dataset.map(parse_function)
batched_dataset = dataset.batch(4)
# iterator = batched_dataset.make_one_shot_iterator()
# nelem = iterator.get_next()
iterator = tfe.Iterator(batched_dataset)
features, labels = iterator.get_next()


# nelem = tfe.Iterator(dataset).get_next()
