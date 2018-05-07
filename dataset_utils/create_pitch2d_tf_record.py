"""Converts Pitch2d data to TFRecords file format with Example protos.

The raw Pitch2d data set is expected to reside in PNG files located in the
following directory structure.

  data_dir/intent##/ID/datetime_trial##/trial_datetime_frame_####.png
  e.g.: /data_dir/intent02/ZL/201801302254_trial08/trial_201801302254_frame0045.png
  ...

where 'intent##' is the unique label associated with
these images.

The training data set consists of 9 sub-directories (i.e. labels)
each containing 1620 PNG images for a total of 1458 0 PNG images.

The validation data set consists of 9 sub-directories (i.e. labels)
each containing 540 PNG images for a total of 4860 PNG images.

The testing data set consists of 9 sub-directories (i.e. labels)
each containing 540 PNG images for a total of 4860 PNG images.

This TensorFlow script converts the training, evaluation and testing data into
a sharded data set consisting of 16, 8 and 8 TFRecord files, respectively.

  train_directory/train-00001-of-00016
  train_directory/train-00002-of-00016
  ...
  train_directory/train-00016-of-00016

and

  validation_directory/validate-0001-of-0008
  validation_directory/validate-0002-of-0008
  ...
  validation_directory/validate-0008-of-0008

and

  test_directory/test-0001-of-0008
  test_directory/test-0002-of-0008
  ...
  test_directory/test-0008-of-0008


Each training TFRecord file contains ~911 records. Each validation TFRecord 
file contains ~608 records. Each testing TFRecord file contains ~608 records. 
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

Running this script may take around 863 seconds on an Alienware Aurora R6.

Running this script may take around 445 seconds on an Threadripper 1900X.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time
import os
import glob
import random
import sys

import numpy as np
import scipy.io as spi
import tensorflow as tf

import pdb

DATA_DIR = '/media/linzhank/DATA/Works/Intention_Prediction/Dataset/Ball pitch/pit2d9blk'
# DATA_DIR = '/media/linzhank/850EVO_1T/Works/Data/Ball pitch/pit2d9blk'
TODAY = datetime.today().strftime("%Y%m%d")

tf.app.flags.DEFINE_string('data_dir', DATA_DIR, 'The original dataset lies here')
tf.app.flags.DEFINE_string('output_directory',
                           os.path.join(DATA_DIR, 'tfrecord_'+TODAY),
                           'Generated TFRecords shards goes to here')

tf.app.flags.DEFINE_integer('train_shards', 16,
                            'Number of shards in training TFRecord files.')
tf.app.flags.DEFINE_integer('validate_shards', 8,
                            'Number of shards in validation TFRecord files.')

tf.app.flags.DEFINE_integer('test_shards', 8,
                            'Number of shards test TFRecord files.')

FLAGS = tf.app.flags.FLAGS


def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def bytes_feature(value):
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
      "colorspace": bytes_feature(colorspace),
      "channels": int64_feature(channels),
      "format": bytes_feature(image_format),
      "filename": bytes_feature(os.path.basename(filename)),
      "encoded_image": bytes_feature(image_buffer),
      "label": int64_feature(label),
      "height": int64_feature(height),
      "width": int64_feature(width),
      "pitcher": bytes_feature(pitcher),
      "trial": bytes_feature(trial),
      "frame": bytes_feature(frame)
  }))

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
  shard_counter = 0
  for s in range(num_shards):
    # Generate a sharded version of the file name, e.g. 'train-0002-of-0016'
    output_filename = "{}-{:05d}-of-{:05d}.tfrecord".format(name, s+1, num_shards)
    output_file = os.path.join(FLAGS.output_directory, output_filename)
    if not os.path.exists(FLAGS.output_directory):
      try:
        os.makedirs(FLAGS.output_directory)
      except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
          raise
    writer = tf.python_io.TFRecordWriter(output_file)

    indices_in_shard = np.arange(ranges[s][0], ranges[s][1], dtype=int)
    counter = 0
    keys = sorted(annotations.keys())
    for i in indices_in_shard:
      filename = annotations[name+'_paths'][i]
      label = map(int, annotations[name+'_labels'])[i]
      pitcher = annotations[name+'_pitchers'][i]
      trial = annotations[name+'_trials'][i]
      frame = annotations[name+'_frames'][i]

      image_buffer, height, width = process_image(filename, coder)

      example = convert_to_example(filename, image_buffer, label, height, width,
                                    pitcher, trial, frame)
      writer.write(example.SerializeToString())
      counter += 1

      print("{}: [{} shard-{:d}]: Processed {:d} of {:d} images in shard.".format
              (datetime.now(), name, s+1, counter, len(indices_in_shard)))
      sys.stdout.flush()

    writer.close()
    shard_counter += 1
    print("===> {}: [{} shard-{:d}] wrote to {}".format
          (datetime.now(), name, shard_counter, output_file))
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

  print('{}: Finished writing all {} images in {} dataset.'.format(datetime.now(),
                                                                   len(annotations[keys[0]]),
                                                                   name))
  sys.stdout.flush()

def read_annotation_files(name, directory):
  """Read annotated and sliced dataset information files
  Args: 
    name: string, locates the dataset storage directory.
    directory: string, root path to the data set.
  """
  files_dir = glob.glob(os.path.join(directory, 'dataset_config', '*'))[-1] # pick the newest files
  files_location = os.path.join(files_dir, name+'*.txt')
  info_files = sorted(glob.glob(files_location))
  assert info_files, "No file was loaded!" # check if files are loaded
  annotations = {}
  keys = []
  
  for ifile in info_files:
    key = ifile.split('/')[-1].split('.')[0]
    keys.append(key)
    with open(ifile) as f:
      annotations[key] = f.read().split('\n')[:-1]
      
  return annotations

  
def process_dataset(name, directory, num_shards):
  """Process a complete data set and save it as TFRecord shards.

  Args:
    name: string, specifies dataset storing location.
    directory: string, root path to the data set.
    num_shards: integer number of shards for this data set.
  """
  annotations = read_annotation_files(name, directory)
  process_annotated_files(name, annotations, num_shards)

def main(unused_argv):
  start_time = time.time()
  print('Saving results to %s' % FLAGS.output_directory)

  # Run it!
  process_dataset('train', FLAGS.data_dir, FLAGS.train_shards)
  process_dataset('validate', FLAGS.data_dir, FLAGS.validate_shards)
  process_dataset('test', FLAGS.data_dir, FLAGS.test_shards)
  end_time = time.time()
  print("It took {:g} seconds to create all tfrecord files".format(end_time - start_time))
if __name__ == '__main__':
  tf.app.run()
