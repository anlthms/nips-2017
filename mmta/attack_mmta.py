"""Implementation of sample attack."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os

import numpy as np
from scipy.misc import imread
from scipy.misc import imsave

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import inception_resnet_v2
import inception_v3
from tensorflow.contrib.slim.nets import inception
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.python.ops.losses.losses import softmax_cross_entropy as sce

slim = tf.contrib.slim


tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string(
    'checkpoint_path', '', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'input_dir', '', 'Input directory with images.')

tf.flags.DEFINE_string(
    'output_dir', '', 'Output directory with images.')

tf.flags.DEFINE_float(
    'max_epsilon', 16.0, 'Maximum size of adversarial perturbation.')

tf.flags.DEFINE_float(
    'iter_alpha', 1.0, 'Step size for one iteration.')

tf.flags.DEFINE_integer(
    'num_iter', 24, 'Number of iterations.')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size', 16, 'How many images process at one time.')

FLAGS = tf.flags.FLAGS


def load_target_class(input_dir):
  """Loads target classes."""
  with tf.gfile.Open(os.path.join(input_dir, 'target_class.csv')) as f:
    return {row[0]: int(row[1]) for row in csv.reader(f) if len(row) >= 2}


def load_images(input_dir, batch_shape):
  """Read png images from input directory in batches.

  Args:
    input_dir: input directory
    batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]

  Yields:
    filenames: list file names without path of each image
      Lenght of this list could be less than batch_size, in this case only
      first few images of the result are elements of the minibatch.
    images: array with all images from this batch
  """
  images = np.zeros(batch_shape)
  filenames = []
  idx = 0
  batch_size = batch_shape[0]
  for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
    with tf.gfile.Open(filepath) as f:
      image = imread(f, mode='RGB').astype(np.float) / 255.0
    # Images for inception classifier are normalized to be in [-1, 1] interval.
    images[idx] = image * 2.0 - 1.0
    filenames.append(os.path.basename(filepath))
    idx += 1
    if idx == batch_size:
      yield filenames, images
      filenames = []
      images = np.zeros(batch_shape)
      idx = 0
  if idx > 0:
    yield filenames, images


def save_images(images, filenames, output_dir):
  """Saves images to the output directory.

  Args:
    images: array with minibatch of images
    filenames: list of filenames without path
      If number of file names in this list less than number of images in
      the minibatch then only first len(filenames) images will be saved.
    output_dir: directory where to save images
  """
  for i, filename in enumerate(filenames):
    # Images for inception classifier are normalized to be in [-1, 1] interval,
    # so rescale them back to [0, 1].
    with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
      imsave(f, (images[i] + 1.0) * 0.5, format='png')


class EnsModel(object):
  """Model class for CleverHans library."""

  def __init__(self, num_classes):
    self.num_classes = num_classes
    self.built = False

  def __call__(self, x_input):
    """Constructs model and return probabilities for given input."""
    reuse = True if self.built else None

    scopes = [
        inception.inception_v3_arg_scope(), inception_v3.inception_v3_arg_scope(),
        inception_resnet_v2.inception_resnet_v2_arg_scope()]
    models = [
        inception.inception_v3, inception_v3.inception_v3,
        inception_resnet_v2.inception_resnet_v2]
    weights = [0.2, 0.2, 0.6]
    for idx, scope in enumerate(scopes):
      with slim.arg_scope(scope):
        model = models[idx]
        weight = weights[idx]
        result = model(x_input, num_classes=self.num_classes, is_training=False, reuse=reuse)
        if idx == 0:
          logits = result[0] * weight
          aux_logits = result[1]['AuxLogits'] * weight
        else:
          logits += result[0] * weight
          aux_logits += result[1]['AuxLogits'] * weight

    self.built = True
    predictions = layers_lib.softmax(logits, scope='Predictions')
    return logits, aux_logits, predictions


def main(_):
  # Images for inception classifier are normalized to be in [-1, 1] interval,
  # eps is a difference between pixels so it should be in [0, 2] interval.
  # Renormalizing epsilon from [0, 255] to [0, 2].
  eps = 2.0 * FLAGS.max_epsilon / 255.0
  alpha = 100.0 * FLAGS.iter_alpha
  num_iter = FLAGS.num_iter
  batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
  num_classes = 1001

  tf.logging.set_verbosity(tf.logging.INFO)

  all_images_taget_class = load_target_class(FLAGS.input_dir)

  # Prepare graph
  x_adv = tf.Variable(tf.zeros(shape=batch_shape))

  x_init = tf.placeholder(tf.float32, shape=batch_shape)
  x_min = tf.Variable(tf.zeros(shape=batch_shape))
  x_max = tf.Variable(tf.zeros(shape=batch_shape))
  assign_op = tf.assign(x_adv, x_init)
  min_step = tf.assign(x_min, tf.clip_by_value(x_init - eps, -1.0, 1.0))
  max_step = tf.assign(x_max, tf.clip_by_value(x_init + eps, -1.0, 1.0))
  target_class_input = tf.placeholder(tf.int32, shape=[FLAGS.batch_size])
  one_hot_target_class = tf.one_hot(target_class_input, num_classes)

  model = EnsModel(num_classes)
  logits, aux_logits, predictions = model(x_adv)
  learning_rate = tf.placeholder(tf.float32, ())
  loss = sce(one_hot_target_class, logits)
  loss += sce(one_hot_target_class, aux_logits, weights=0.8)

  opt = tf.train.AdamOptimizer(learning_rate)
  optim_step = opt.minimize(loss, var_list=[x_adv])
  clip_step = tf.assign(x_adv, tf.clip_by_value(x_adv, x_min, x_max))

  sess = tf.InteractiveSession()
  init = tf.global_variables_initializer()
  sess.run(init)

  saver = tf.train.Saver(slim.get_model_variables('InceptionResnetV2'))
  saver.restore(sess, 'ens_adv_inception_resnet_v2.ckpt')
  saver = tf.train.Saver(slim.get_model_variables('InceptionV3_renamed'))
  saver.restore(sess, 'inception_v3_renamed.ckpt')
  saver = tf.train.Saver(slim.get_model_variables('InceptionV3'))
  saver.restore(sess, 'adv_inception_v3.ckpt')

  for filenames, images in load_images(FLAGS.input_dir, batch_shape):
    sess.run(assign_op, feed_dict={x_init: images})
    sess.run(min_step, feed_dict={x_init: images})
    sess.run(max_step, feed_dict={x_init: images})
    target_class_for_batch = (
      [all_images_taget_class[n] for n in filenames]
      + [0] * (FLAGS.batch_size - len(filenames)))
    for i in range(num_iter):
      sess.run(
          [optim_step, loss],
          feed_dict={learning_rate: alpha, target_class_input: target_class_for_batch})
      sess.run(clip_step)
    adv_images = x_adv.eval()
    save_images(adv_images, filenames, FLAGS.output_dir)
  sess.close()


if __name__ == '__main__':
  tf.app.run()
