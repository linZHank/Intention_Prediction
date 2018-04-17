"""Converts Pitch2d data to TFRecords file format with Example protos.

The raw Pitch2d data set is expected to reside in PNG files located in the
following directory structure.

  data_dir/intent##/ID/datetime_trial##/trial_datetime_frame_####.png
  e.g.: /data_dir/intent02/ZL/201801302254_trial08/trial_201801302254_frame0045.png
  ...

where 'intent##' is the unique label associated with
these images.

The training data set consists of 9 sub-directories (i.e. labels)
each containing 1080 PNG images for a total of 9720 PNG images.

The evaluation data set consists of 9 sub-directories (i.e. labels)
each containing 380 PNG images for a total of 3420 PNG images.

The testing data set consists of 9 sub-directories (i.e. labels)
each containing 380 PNG images for a total of 3420 PNG images.

This TensorFlow script converts the training, evaluation and testing data into
a sharded data set consisting of 16, 8 and 8 TFRecord files, respectively.

  train_directory/train-00001-of-00016
  train_directory/train-00002-of-00016
  ...
  train_directory/train-00016-of-00016

and

  validation_directory/validation-00-of-08
  validation_directory/validation-01-of-08
  ...
  validation_directory/validation-08-of-08

and

  test_directory/test-00-of-08
  test_directory/test-01-of-08
  ...
  test_directory/test-08-of-08


Each training TFRecord file contains ~608 records. Each validation TFRecord 
file contains ~428 records. Each testing TFRecord file contains ~428 records. 
Each record within the TFRecord file is a serialized Example proto. 
The Example proto contains the following fields:

  image/encoded: string containing PNG encoded image in RGB colorspace
  image/height: integer, image height in pixels
  image/width: integer, image width in pixels
  image/colorspace: string, specifying the colorspace, always 'RGB'
  image/channels: integer, specifying the number of channels, always 3
  image/format: string, specifying the format, always 'JPEG' or 'PNG'

  image/filename: string containing the basename of the image file
            e.g. 'trial_201801302254_frame0045.png'
  image/class/label: integer specifying the index in a classification layer.
    The label ranges from [1, 9] where 0 is not used.
  image/class/pid: string specifying the unique ID of the pitcher,
    e.g. 'ZL'

Running this script using 8 threads may take around ~? hours 
on an Alienware Aurora R6.

Running this script using 16 threads may take around ~? hours 
on an Threadripper 1900X.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import glob
import random
import sys

import numpy as np
import scipy.io as spio
import six
import tensorflow as tf


DATA_DIR_R6 = '/media/linzhank/DATA/Works/Intention_Prediction/Dataset/Ball pitch/pit2d9blk/'

tf.app.flags.DEFINE_string('data_dir', DATA_DIR_R6,
                           'data directory')
tf.app.flags.DEFINE_string('output_directory', DATA_DIR_R6+'tfrecord/',
                           'Output data directory')

tf.app.flags.DEFINE_integer('train_shards', 16,
                            'Number of shards in training TFRecord files.')
tf.app.flags.DEFINE_integer('validate_shards', 8,
                            'Number of shards in validation TFRecord files.')

tf.app.flags.DEFINE_integer('num_threads', 8,
                            'Number of threads to preprocess the images.')

FLAGS = tf.app.flags.FLAGS


def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
  """Wrapper for inserting float features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  if isinstance(value, six.string_types):           
    value = six.binary_type(value, encoding='utf-8') 
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to_example(filename, image_buffer, label, height, width,
                        pitcher, trial, frame):
  """Build an Example proto for an example.

  Args:
    filename: string, path to an image file, e.g., '/path/to/example.PNG'
    image_buffer: string, encoding of RGB image
    label: integer, identifier for the ground truth for the network
    pitcher: string, pitcher's name
    trial: string, trial name
    frame: integer, frame index between 1 and 90
    height: integer, image height in pixels
    width: integer, image width in pixels
  Returns:
    Example proto
  """
  colorspace = 'RGB'
  channels = 3
  image_format = 'PNG'

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/colorspace': _bytes_feature(colorspace),
      'image/channels': _int64_feature(channels),
      'image/format': _bytes_feature(image_format),
      'image/filename': _bytes_feature(os.path.basename(filename)),
      'image/encoded': _bytes_feature(image_buffer),
      'image/class/label': _int64_feature(label),
      'image/height': _int64_feature(height),
      'image/width': _int64_feature(width),
      'image/pitcher': _bytes_feature(pitcher),
      'image/trial': _bytes_feature(trial),
      'image/frame': _int64_feature(frame)}))

  return example


class ImageCoder(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Create a single Session to run all image coding calls.
    self._sess = tf.Session()

    # Initializes function that decodes RGB JPEG data.
    self._decode_image_data = tf.placeholder(dtype=tf.string)
    self._decode_image = tf.image.decode_image(self._decode_image_data, channels=3)

  def decode_image(self, image_data):
    image = self._sess.run(self._decode_image,
                           feed_dict={self._decode_image_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image


def process_image(filename, coder):
  """Process a single image file.

  Args:
    filename: string, path to an image file e.g., '/path/to/example.JPG'.
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
  Returns:
    image_buffer: string, encoding of RGB image.
    height: integer, image height in pixels.
    width: integer, image width in pixels.
  """
  # Read the image file.
  with tf.gfile.FastGFile(filename, 'rb') as f:
    image_data = f.read()

  # Decode the RGB image.
  image = coder.decode_image(image_data)

  # Check that image converted to RGB
  assert len(image.shape) == 3
  height = image.shape[0]
  width = image.shape[1]
  assert image.shape[2] == 3

  return image_data, height, width


def process_image_files_shards(name, coder, ranges, annotations, num_shards):
  """Processes and saves list of images as TFRecord in 1 shard.

  Args:
    name: string of dataset name specifying travaltes
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
    ranges: list of pairs of integers specifying ranges of each shard.
    annotations: dictionary of data; each key stands for 
    num_shards: integer number of shards for this data set.
  """
  for s in range(num_shards):
    # Generate a sharded version of the file name, e.g. 'train-0002-of-0016'
    output_filename = "{}-{:05d}-of-{:05d}".format(name, s+1, num_shards)
    output_file = os.path.join(FLAGS.output_directory, output_filename)
    writer = tf.python_io.TFRecordWriter(output_file)

    shard_counter = 0
    indices_in_shard = np.arange(ranges[s][0], ranges[s][1], dtype=int)
    counter = 0
    keys = sorted(annotations.keys())
    for i in indices_in_shard:
      filename = annotations[name+'_paths'][i]
      label = annotations[name+'_labels'][i]
      pitcher = annotations[name+'_pitchers'][i]
      trial = annotations[name+'_trials'][i]
      frame = annotations[name+'_frames'][i]

      image_buffer, height, width = process_image(filename, coder)

      example = convert_to_example(filename, image_buffer, label, height, width,
                                    pitcher, trial, frame)
      writer.write(example.SerializeToString())
      counter += 1

      print("{} [ shard-{:d}]: Processed {:d} of {:d} images in shard.".format
              (datetime.now(), s, counter, len(indices_in_shard)))
      sys.stdout.flush()

    writer.close()
    shard_counter += 1
    print("==== {} [shard-{:d}] wrote to {}".format
          (datetime.now(), shard_counter, output_file))
    sys.stdout.flush()


def process_annotated_files(name, annotations, num_shards):
  """Process and save list of images as TFRecord of Example protos.

  Args:
    name: string, unique identifier specifying the data set (train, validate, test).
    annotations: dictionary of annotations: 
      each key is one type of annotations to the image files
    num_shards: integer number of shards for this data set.
  """
  # Examing if annotations have same length
  keys = sorted(annotations.keys())
  for i in range(len(keys)-1):
    assert len(annotations[keys[i]]) == len(annotations[keys[i+1]])

  # Break all images into batches with [ranges[i][0], ranges[i][1]].
  spacing = np.linspace(0, len(annotations[keys[0]]), num_shards + 1).astype(np.int)
  ranges = []
  for i in range(len(spacing) - 1):
    ranges.append([spacing[i], spacing[i + 1]])

  # Create a generic TensorFlow-based utility for converting all image codings.
  coder = ImageCoder()

  # create tfrecord shard by shard
  process_image_files_shards(name, coder, ranges, annotations, num_shards)

  print('{} Finished writing all {} images in data set.'.
        format(datetime.now(), len(filenames)))
  sys.stdout.flush()

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

  
def process_dataset(name, directory, num_shards):
  """Process a complete data set and save it as a TFRecord.

  Args:
    name: string, unique identifier specifying the data set (train, validate, test).
    directory: string, root path to the data set.
    num_shards: integer number of shards for this data set.
  """
  annotations = read_annotation_files(name, directory)
  process_annotated_files(name, annotations, num_shards)

def main(unused_argv):
  print('Saving results to %s' % FLAGS.output_directory)

  # Run it!
  process_dataset('train', FLAGS.data_dir, FLAGS.train_shards)
  process_dataset('validation', FLAGS.data_dir, FLAGS.validate_shards)
  process_dataset('test', FLAGS.data_dir, FLAGS.test_shards)


if __name__ == '__main__':
  tf.app.run()
