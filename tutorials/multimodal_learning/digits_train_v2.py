"""A binary to train CIFAR-10 using a single GPU.
Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.
Speed: With batch_size 128.
System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)
Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.
http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time
import unicodedata

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import digits

parser = digits.parser

parser.add_argument('--train_dir', type=str, default='./output/digits_train',
                    help='Directory where to write event logs and checkpoint.')

parser.add_argument('--max_steps', type=int, default=100000,
                    help='Number of batches to run.')

parser.add_argument('--log_device_placement', type=bool, default=False,
                    help='Whether to log device placement.')

parser.add_argument('--log_frequency', type=int, default=10,
                    help='How often to log results to the console.')

parser.add_argument('--n_residual_blocks', type=int, default=2,
                    help='Number of residual blocks in network')


def train():
    """Train CIFAR-10 for a number of steps."""
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)

        # Get images and labels for CIFAR-10.
        images, labels, ages = digits.distorted_inputs()

        # Build a Graph that computes the logits predictions from the
        # inference model.

        #here age represents any additional tabular data  (here numeric) to be added in the common embedding
        logits, reshape, local3, local4 = digits.inference(images, ages)

        # calculate predictions
        predictions = tf.cast(tf.argmax(logits, axis=1), tf.int32)

        #cast labels to ints
        labels = tf.cast(labels, tf.int32)

        # ops for batch accuracy calcultion
        correct_prediction = tf.equal(predictions, labels)
        batch_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # calculate training accuracy
        # Calculate loss.
        loss = digits.loss(logits, labels)

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = digits.train(loss, global_step)

        # Create a saver.
        saver = tf.train.Saver(tf.global_variables())

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()

        # Start running operations on the Graph.
        sess = tf.Session(config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement))
        # sess.run(init)

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

        step_start = 0

        try:
            print("Trying to restore last checkpoint ...")
            save_dir = FLAGS.train_dir
            # Use TensorFlow to find the latest checkpoint - if any.
            last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=save_dir)
            # Try and load the data in the checkpoint.
            saver.restore(sess, save_path=last_chk_path)

            # If we get to this point, the checkpoint was successfully loaded.
            print("Restored checkpoint from:", last_chk_path)
            # get the step integer from restored path to start step from there
            step_start = int(
                filter(str.isdigit, unicodedata.normalize('NFKD', last_chk_path).encode('ascii', 'ignore')))

        except:
            # If the above failed for some reason, simply
            # initialize all the variables for the TensorFlow graph.
            print("Failed to restore checkpoint. Initializing variables instead.")
            sess.run(init)

        accuracy_dev = []
        for step in xrange(step_start, FLAGS.max_steps):
            start_time = time.time()

            if step == 0:
                reshape, local3, local4 = sess.run([reshape, local3, local4])

                print('shape of reshape is {}'.format(reshape.shape))
                print('shape of local3 is {}'.format(local3.shape))
                print('shape of local4 is {}'.format(local4.shape))

            _, loss_value, accuracy= sess.run([train_op, loss, batch_accuracy])


            #print('%s: the shape of reshape is:%d' % (reshape.shape))
            # append the next accuray to the development list
            accuracy_dev.append(accuracy)

            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 10 == 0:
                num_examples_per_step = FLAGS.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, loss = %.2f, avg_batch_accuracy = %.2f, (%.1f examples/sec; %.3f '
                              'sec/batch)')
                # take averages of all the accuracies from the previous bathces
                print(format_str % (datetime.now(), step, loss_value, np.mean(accuracy_dev),
                                    examples_per_sec, sec_per_batch))

            if step % 100 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % 100 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)


def main(argv=None):  # pylint: disable=unused-argument
    train()


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    tf.app.run()