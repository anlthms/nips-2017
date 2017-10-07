"""Implementation of sample defense.

This defense loads inception v3 checkpoint and classifies all images
using loaded checkpoint.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
from scipy.misc import imread
from scipy.ndimage import interpolation
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import inception_resnet_v2
from tensorflow.contrib.slim.nets import inception
from tensorflow.contrib.slim.nets import vgg, resnet_v2, resnet_utils
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
    'output_file', '', 'Output file to save labels.')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size', 16, 'How many images process at one time.')

FLAGS = tf.flags.FLAGS

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

_MAX_SCALE = 1.1
_MIN_SCALE = 0.75
_VGG_MAX_SCALE = 0.8
_MAX_ANGLE = 4
_MAX_NOISE = 4


# The function below was copied from tensorflow code
def transform_matrix_offset_center(matrix, x, y):
  o_x = float(x) / 2 + 0.5
  o_y = float(y) / 2 + 0.5
  offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
  reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
  transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
  return transform_matrix


# The function below was copied from tensorflow code
def apply_transform(x,
                    transform_matrix,
                    channel_axis=0,
                    fill_mode='constant',
                    cval=0.):
  """Apply the image transformation specified by a matrix.

  Arguments:
      x: 2D numpy array, single image.
      transform_matrix: Numpy array specifying the geometric transformation.
      channel_axis: Index of axis for channels in the input tensor.
      fill_mode: Points outside the boundaries of the input
          are filled according to the given mode
          (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
      cval: Value used for points outside the boundaries
          of the input if `mode='constant'`.

  Returns:
      The transformed version of the input.
  """
  x = np.rollaxis(x, channel_axis, 0)
  final_affine_matrix = transform_matrix[:2, :2]
  final_offset = transform_matrix[:2, 2]
  channel_images = [
      interpolation.affine_transform(
          x_channel,
          final_affine_matrix,
          final_offset,
          order=0,
          mode=fill_mode,
          cval=cval) for x_channel in x
  ]
  x = np.stack(channel_images, axis=0)
  x = np.rollaxis(x, 0, channel_axis + 1)
  return x


def load_images(input_dir, batch_shape, vgg_batch_shape):
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
  ens_images = np.zeros(batch_shape)
  inc_images = np.zeros(batch_shape)
  tcd_images = np.zeros(batch_shape)
  vgg_images = np.zeros(vgg_batch_shape)
  filenames = []
  idx = 0
  batch_size = batch_shape[0]
  for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
    with tf.gfile.Open(filepath) as f:
      image = imread(f, mode='RGB')

    tcd_image = transcode(image).astype(np.float)
    image = image.astype(np.float)
    vgg_image = vgg_distort(tcd_image, vgg_batch_shape[1:3])
    # Images for inception classifier are normalized to be in [-1, 1] interval.
    image = (image / 255.0) * 2.0 - 1.0
    ens_images[idx] = ens_distort(image)
    # Resize and mean subtract for VGG
    vgg_image -= np.array((_R_MEAN, _G_MEAN, _B_MEAN)).reshape((1, 1, 3))
    vgg_images[idx] = vgg_image
    inc_images[idx] = inc_distort(image)
    tcd_images[idx] = (tcd_image / 255.0) * 2.0 - 1.0
    filenames.append(os.path.basename(filepath))
    idx += 1
    if idx == batch_size:
      yield filenames, ens_images, vgg_images, inc_images, tcd_images
      filenames = []
      idx = 0
  if idx > 0:
    yield filenames, ens_images, vgg_images, inc_images, tcd_images


def inc_distort(image):
  image += np.random.randn(*image.shape).reshape(image.shape)*_MAX_NOISE/255.
  np.clip(image, -1, 1, out=image)
  angle = np.random.randint(_MAX_ANGLE - 2, _MAX_ANGLE)
  if np.random.randint(2) == 0:
    angle = -angle

  rotated = interpolation.rotate(image, angle, reshape=True)
  zoom = (np.random.uniform(_MAX_SCALE - 0.05, _MAX_SCALE), np.random.uniform(_MAX_SCALE - 0.05, _MAX_SCALE), 1)
  zoomed = interpolation.zoom(rotated, zoom)

  starts = [(zoomed.shape[i] - image.shape[i]) // 2 for i in range(2)]
  ends = [starts[i] + image.shape[i] for i in range(2)]
  result = zoomed[starts[0]:ends[0], starts[1]:ends[1]]
  return result


def vgg_distort(image, out_shape):
  angle = np.random.randint(_MAX_ANGLE - 2, _MAX_ANGLE)
  if np.random.randint(2) == 0:
    angle = -angle
  rotated = interpolation.rotate(image, angle, reshape=True)

  zoom = (np.random.uniform(_MIN_SCALE, _VGG_MAX_SCALE),
          np.random.uniform(_MIN_SCALE, _VGG_MAX_SCALE), 1)
  zoomed = interpolation.zoom(rotated, zoom)

  starts = [(zoomed.shape[i] - out_shape[i]) // 2 for i in range(2)]
  ends = [starts[i] + out_shape[i] for i in range(2)]
  result = zoomed[starts[0]:ends[0], starts[1]:ends[1]]
  return result


def ens_distort(image):
  h = image.shape[0]
  w = image.shape[1]
  channel_axis = 2
  shear = jitter(np.pi/128)
  theta = jitter(-np.pi/128)
  tx = int(jitter(w*0.02))
  ty = int(jitter(h*0.02))
  zx = jitter(0.95)
  zy = jitter(0.95)

  # shear
  transform_matrix = np.array(
    [[1, -np.sin(shear), 0], [0, np.cos(shear), 0], [0, 0, 1]])
  transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
  image = apply_transform(image, transform_matrix, channel_axis)

  # shift
  transform_matrix = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
  image = apply_transform(image, transform_matrix, channel_axis)

  # zoom
  transform_matrix = np.array([[zx, 0, 0], [0, zy, 0], [0, 0, 1]])
  transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
  image = apply_transform(image, transform_matrix, channel_axis)

  # rotation
  transform_matrix = np.array(
    [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
  transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
  image = apply_transform(image, transform_matrix, channel_axis)

  return image


def jitter(num):
  scale = np.random.randint(95, 106) / 100.
  return num*scale


def transcode(image):
  im = Image.fromarray(image)
  im.save('/dev/shm/tmp.jpg', quality=50)
  transcoded = Image.open('/dev/shm/tmp.jpg')
  return np.asarray(transcoded)


class EnsModel(object):
  """Model class for CleverHans library."""

  def __init__(self, num_classes):
    self.num_classes = num_classes
    self.built = False

  def __call__(self, ens_x_input, vgg_x_input, inc_x_input, tcd_x_input):
    """Constructs model and return probabilities for given input."""
    reuse = True if self.built else None
    logits = None
    aux_logits = None
    weights = [[0.7, 0.1], [0.2, 0.1]]
    all_inputs = [[ens_x_input, tcd_x_input], [inc_x_input, tcd_x_input]]
    scopes = [inception_resnet_v2.inception_resnet_v2_arg_scope(), inception.inception_v3_arg_scope()]
    reuse_flags = [reuse, True]
    for model_idx, model in enumerate([inception_resnet_v2.inception_resnet_v2, inception.inception_v3]):
      with slim.arg_scope(scopes[model_idx]):
        for idx, inputs in enumerate(all_inputs[model_idx]):
          result = model(inputs, num_classes=self.num_classes, is_training=False, reuse=reuse_flags[idx])
          weight = weights[model_idx][idx]
          # :1 is for slicing out the background class
          if logits == None:
            logits = result[0][:, 1:] * weight
            aux_logits = result[1]['AuxLogits'][:, 1:] * weight
          else:
            logits += result[0][:, 1:] * weight
            aux_logits += result[1]['AuxLogits'][:, 1:] * weight

    with slim.arg_scope(vgg.vgg_arg_scope()):
      weight = 0.1
      result = vgg.vgg_16(vgg_x_input, num_classes=1000, is_training=False)
      logits += result[0] * weight

    with slim.arg_scope(resnet_utils.resnet_arg_scope()):
      weight = 0.05
      result = resnet_v2.resnet_v2_152(vgg_x_input, num_classes=self.num_classes, reuse=reuse)
      logits += tf.squeeze(result[0])[:, 1:] * weight

    self.built = True
    aux_weight = 0.8
    logits += aux_logits * aux_weight

    predictions = layers_lib.softmax(logits)
    return predictions


def main(_):
  np.random.seed(0)
  batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
  vgg_batch_shape = [FLAGS.batch_size, 224, 224, 3]
  num_classes = 1001

  tf.logging.set_verbosity(tf.logging.INFO)

  # Prepare graph
  graph = tf.Graph()
  with graph.as_default():
    ens_x_input = tf.placeholder(tf.float32, shape=batch_shape)
    vgg_x_input = tf.placeholder(tf.float32, shape=vgg_batch_shape)
    inc_x_input = tf.placeholder(tf.float32, shape=batch_shape)
    tcd_x_input = tf.placeholder(tf.float32, shape=batch_shape)

    model = EnsModel(num_classes)
    predictions = model(ens_x_input, vgg_x_input, inc_x_input, tcd_x_input)

    # Add the background class back
    predicted_labels = tf.argmax(predictions, 1) + 1

    init = tf.global_variables_initializer()
    inc_res_saver = tf.train.Saver(slim.get_model_variables('InceptionResnetV2'))
    adv_inc_saver = tf.train.Saver(slim.get_model_variables('InceptionV3'))
    res_saver = tf.train.Saver(slim.get_model_variables('resnet_v2_152'))
    vgg_saver = tf.train.Saver(slim.get_model_variables('vgg_16'))

  sess = tf.Session(graph=graph)
  sess.run(init)

  inc_res_saver.restore(sess, 'ens_adv_inception_resnet_v2.ckpt')
  adv_inc_saver.restore(sess, 'adv_inception_v3.ckpt')
  res_saver.restore(sess, 'resnet_v2_152.ckpt')
  vgg_saver.restore(sess, 'vgg_16.ckpt')

  res = []
  all_files = []
  for filenames, ens_images, vgg_images, inc_images, tcd_images in load_images(FLAGS.input_dir, batch_shape, vgg_batch_shape):
    labels = sess.run(
      predicted_labels, feed_dict={
          ens_x_input: ens_images,
          vgg_x_input: vgg_images,
          inc_x_input: inc_images,
          tcd_x_input: tcd_images})
    res.extend(labels)
    all_files.extend(filenames)

  with tf.gfile.Open(FLAGS.output_file, 'w') as out_file:
      for filename, label in zip(all_files, res):
        out_file.write('{0},{1}\n'.format(filename, label))
  sess.close()


if __name__ == '__main__':
  tf.app.run()
