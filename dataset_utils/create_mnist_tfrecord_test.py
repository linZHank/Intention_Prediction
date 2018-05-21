"""Create tfrecord for mnist dataset
"""

from __future__ import absolute_import, print_function

import tensorflow as tf
import numpy as np
import os
import glob
from matplotlib import pyplot as plt


def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.tostring()]))

def dict_to_example(img, label):
  feature_dict = {
      "image/encoded": bytes_feature(img),
      "image/label": int64_feature(label)
      }
  example = tf.train.Example(features=tf.train.Features(feature=feature_dict))

  return example

# Create tfrecord files for mnist dataset
mnist = tf.contrib.learn.datasets.load_dataset("mnist")
dataset_train = mnist.train
output_path = "/home/linzhank/playground/mnist_tfrecord/train/train"
if not os.path.exists(os.path.dirname(output_path)):
  os.makedirs(os.path.dirname(output_path))  
num_examples_per_record = 1000
num_so_far = 0
writer = tf.python_io.TFRecordWriter("{}{:04d}-{}.tfrecord".format(output_path, num_so_far, num_examples_per_record-1))
for i in np.arange(dataset_train.num_examples):
  example = dict_to_example(dataset_train.images[i], dataset_train.labels[i])
  writer.write(example.SerializeToString())
  if not i % num_examples_per_record and i:
    writer.close()
    num_so_far = i
    writer = tf.python_io.TFRecordWriter("{}{:04d}-{}.tfrecord".format(output_path, num_so_far, i+num_examples_per_record-1))
    print("saved {}{:04d}-{}.tfrecord".format(output_path, num_so_far, i+num_examples_per_record-1))
writer.close()

dataset_test = mnist.test
output_path = "/home/linzhank/playground/mnist_tfrecord/test/test"
if not os.path.exists(os.path.dirname(output_path)):
  os.makedirs(os.path.dirname(output_path))  
num_examples_per_record = 1000
num_so_far = 0
writer = tf.python_io.TFRecordWriter("{}{:04d}-{}.tfrecord".format(output_path, num_so_far, num_examples_per_record-1))
for i in np.arange(dataset_test.num_examples):
  example = dict_to_example(dataset_test.images[i], dataset_test.labels[i])
  writer.write(example.SerializeToString())
  if not i % num_examples_per_record and i:
    writer.close()
    num_so_far = i
    writer = tf.python_io.TFRecordWriter("{}{:04d}-{}.tfrecord".format(output_path, num_so_far, i+num_examples_per_record-1))
    print("saved {}{:04d}-{}.tfrecord".format(output_path, num_so_far, i+num_examples_per_record-1))
writer.close()


# Create tf Dataset objects based on tfrecord
tfrecords_dir = "~/playground/mnist_tfrecord"
# tfrecords_dir = "/media/linzhank/850EVO_1T/Works/Data/Ball pitch/pit2d9blk/tfrecord_20180423"  
train_filenames = glob.glob(os.path.join(tfrecords_dir, "train", "train*"))
test_filenames = glob.glob(os.path.join(tfrecords_dir, "test", "test*"))

def parse_function(example_proto):
  # example_proto, tf_serialized
  keys_to_features = {
    "image/encoded": tf.FixedLenFeature(shape=(), dtype=tf.string),
    "image/label": tf.FixedLenFeature(shape=(), dtype=tf.int64), 
  }
  # parse all features in a single example according to the dics
  parsed_features = tf.parse_single_example(example_proto, keys_to_features)
  # decode the encoded image to the (360, 640, 3) uint8 array
  decoded_image = tf.decode_raw(parsed_features["image/encoded"], tf.float32)
  # reshape image
  reshaped_image = tf.reshape(decoded_image, [28, 28])
  # label
  label = tf.cast(parsed_features["image/label"], tf.int32)

  parsed_example = {
    "image_bytes": parsed_features["image/encoded"],
    "image_decoded": decoded_image,
    "image_reshaped": reshaped_image,
    "label": label,
  }
  return parsed_example

dataset = tf.data.TFRecordDataset(test_filenames)
parsed_dataset = dataset.map(parse_function)
iterator = parsed_dataset.make_one_shot_iterator()
nexelem = iterator.get_next()

sess = tf.InteractiveSession()
image = sess.run(nexelem["image_reshaped"])
plt.imshow(image)
plt.show()



