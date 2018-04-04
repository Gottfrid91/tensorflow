# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Converts MNIST data to TFRecords file format with Example protos."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import numpy as np
import scipy.io
from array import array


import tensorflow as tf

FLAGS = None


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(data_set, name):
  """Converts a dataset to tfrecords."""
  images = data_set.images
  labels = data_set.labels
  num_examples = data_set.num_examples
  print(num_examples)

  if images.shape[0] != num_examples:
    raise ValueError('Images size %d does not match label size %d.' %
                     (images.shape[0], num_examples))
  rows = images.shape[1]
  cols = images.shape[2]
  depth = images.shape[3]

  filename = os.path.join(FLAGS.directory, name + '.tfrecords')
  print('Writing', filename)
  with tf.python_io.TFRecordWriter(filename) as writer:
    for index in range(num_examples):
        if labels[index] == 10:
            labels[index] = 0
        image_raw = images[index].tostring()
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'height': _int64_feature(rows),
                    'width': _int64_feature(cols),
                    'depth': _int64_feature(depth),
                    'label': _int64_feature(int(labels[index])),
                    'image_raw': _bytes_feature(image_raw)

                }))
        writer.write(example.SerializeToString())


def main(unused_argv):
  # Get the data.
  class data_set(object):
      pass

  read_input = scipy.io.loadmat('/home/olle/PycharmProjects/digits/models/resnet/data/test/test_32x32.mat')

  train_images = read_input['X']
  train_labels = read_input['y']

  train_images = train_images.transpose()
  train_images = np.swapaxes(train_images, 1, 3)

  data_set.images = train_images
  data_set.labels = train_labels
  data_set.num_examples = data_set.images.shape[0]

  # Convert to Examples and write the result to TFRecords.
  convert_to(data_set, 'test')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--directory',
      type=str,
      default='./data/test/',
      help='Directory to download data files and write the converted result'
  )
  parser.add_argument(
      '--validation_size',
      type=int,
      default=5000,
      help="""\
      Number of examples to separate from the testing data for the validation
      set.\
      """
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)