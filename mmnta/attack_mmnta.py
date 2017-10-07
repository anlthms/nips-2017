"""Implementation of sample attack."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
from PIL import Image

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
    'num_iter', 26, 'Number of iterations.')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size', 16, 'How many images process at one time.')

FLAGS = tf.flags.FLAGS


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
      image = np.array(Image.open(f).convert('RGB')).astype(np.float) / 255.0
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
      img = np.round(((images[i] + 1.0) * 0.5) * 255.0).astype(np.uint8)
      Image.fromarray(img).save(f, format='PNG')


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
    weights = [0.1, 0.2, 0.7]
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


def validate(images, adv_images, filenames, eps, in_dir, out_dir):
  print(in_dir, out_dir)
  for i in range(images.shape[0]):
    img = images[i]
    adv_img = adv_images[i]
    assert img.min() >= -1.0
    assert img.max() <= 1.0
    assert adv_img.min() >= -1.0
    assert adv_img.max() <= 1.0
    img = np.round(((img + 1.0) * 0.5) * 255.0).astype(np.uint8)
    adv_img = np.round(((adv_img + 1.0) * 0.5) * 255.0).astype(np.uint8)
    diff = np.int32(img) - np.int32(adv_img)
    print(filenames[i], 'diff min', diff.min(), 'max', diff.max())
    assert diff.min() >= -eps
    assert diff.max() <= eps


def main(_):
  # Images for inception classifier are normalized to be in [-1, 1] interval,
  # eps is a difference between pixels so it should be in [0, 2] interval.
  # Renormalizing epsilon from [0, 255] to [0, 2].
  eps = 2.0 * FLAGS.max_epsilon / 255.0
  alpha = 40000.0 * FLAGS.iter_alpha
  num_iter = FLAGS.num_iter
  batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
  num_classes = 1001

  tf.logging.set_verbosity(tf.logging.INFO)

  # Prepare graph
  graph = tf.Graph()
  with graph.as_default():
      x_adv = tf.Variable(tf.zeros(shape=batch_shape))

      x_init = tf.placeholder(tf.float32, shape=batch_shape)
      assign_op = tf.assign(x_adv, x_init)

      model = EnsModel(num_classes)
      logits, aux_logits, predictions = model(x_adv)
      preds_ph = tf.placeholder(tf.int32, shape=[FLAGS.batch_size])
      one_hot_preds = tf.one_hot(preds_ph, num_classes)

      target_ph = tf.placeholder(tf.int32, shape=[FLAGS.batch_size])
      one_hot_targets = tf.one_hot(target_ph, num_classes)

      learning_rate = tf.placeholder(tf.float32, ())
      aux_logits_weight = 0.4
      reg_coeff = 0.1

      # Discourage the most likely class
      loss = -0.1 * sce(one_hot_preds, logits)
      loss -= 0.1 * sce(one_hot_preds, aux_logits, weights=aux_logits_weight)

      # Encourage the next likely class
      loss += sce(one_hot_targets, logits)
      loss += sce(one_hot_targets, aux_logits, weights=aux_logits_weight)

      opt = tf.train.AdamOptimizer(learning_rate)
      optim_op = opt.minimize(loss, var_list=[x_adv])
      diff = x_adv - x_init
      clamp_op = tf.assign(x_adv, tf.clip_by_value(x_init + eps * tf.tanh(diff), -1.0, 1.0))
      init = tf.global_variables_initializer()
      inc_res_saver = tf.train.Saver(slim.get_model_variables('InceptionResnetV2'))
      inc_saver = tf.train.Saver(slim.get_model_variables('InceptionV3_renamed'))
      adv_saver = tf.train.Saver(slim.get_model_variables('InceptionV3'))

  sess = tf.Session(graph=graph)
  sess.run(init)

  inc_saver.restore(sess, 'inception_v3_renamed.ckpt')
  adv_saver.restore(sess, 'adv_inception_v3.ckpt')
  inc_res_saver.restore(sess, 'ens_adv_inception_resnet_v2.ckpt')

  for filenames, images in load_images(FLAGS.input_dir, batch_shape):
    sess.run(assign_op, feed_dict={x_init: images})
    preds_vals = sess.run(predictions)
    preds_for_batch = np.argmax(preds_vals, axis=1).ravel()
    # Stamp out the most likely class
    preds_vals[range(FLAGS.batch_size), preds_for_batch] = 0.0
    # Get argmax again to determine the next likely class
    targets_for_batch = np.argmax(preds_vals, axis=1).ravel()
    for i in range(num_iter):
      sess.run(
          [optim_op, loss],
          feed_dict={learning_rate: alpha,
                     target_ph: targets_for_batch,
                     preds_ph: preds_for_batch})
      sess.run(clamp_op, feed_dict={x_init: images})
    adv_images = x_adv.eval(session=sess)
    save_images(adv_images, filenames, FLAGS.output_dir)
    if False:
      validate(images, adv_images, filenames, FLAGS.max_epsilon, FLAGS.input_dir, FLAGS.output_dir)
  sess.close()


if __name__ == '__main__':
  tf.app.run()
