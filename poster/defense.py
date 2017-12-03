import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
from tensorflow.python.ops.losses.losses import softmax_cross_entropy as sce
import os
from scipy.ndimage import interpolation
from PIL import Image
import numpy as np


tf.logging.set_verbosity(tf.logging.ERROR)

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

def inception(img, reuse):
    arg_scope = nets.inception.inception_v3_arg_scope()
    with slim.arg_scope(arg_scope):
        logits, result = nets.inception.inception_v3(
            img, 1001, is_training=False, reuse=reuse)
        logits = logits[:,1:] # ignore background class
        aux_logits = result['AuxLogits'][:, 1:]
        probs = tf.nn.softmax(logits) # probabilities
    return logits, aux_logits, probs

def classify(img):
    img = img.reshape((1, 299, 299, 3))
    g, p = sess.run([logits, probs], feed_dict={x_adv: img})
    g = np.squeeze(g)
    p = np.squeeze(p)
    topk = list(p.argsort()[-10:][::-1])
    topprobs = p[topk]
    toplogits = g[topk]
    #print 'probs', zip(topk, topprobs, toplogits)
    print 'probs 920, 94:', p[920], p[94]

def get_logits(img, labels):
    g, _ = sess.run([logits, probs], feed_dict={x_adv: img})
    g = np.squeeze(g)
    return [g[label] for label in labels]

def get_probs(img, labels):
    _, p = sess.run([logits, probs], feed_dict={x_adv: img})
    p = np.squeeze(p)
    return [p[label] for label in labels]

def open_image(filename):
    img = Image.open(filename)
    img = img.resize((299, 299))
    img = (np.asarray(img) / 255.0).astype(np.float32)
    img = img * 2.0 - 1.0
    return img

def save_image(img, filename):
    with tf.gfile.Open(filename, 'w') as f:
      img = np.round(((img + 1.0) * 0.5) * 255.0).astype(np.uint8)
      Image.fromarray(img.squeeze()).save(f, format='PNG')


img_shape = (1, 299, 299, 3)
eps = 2.0 * 16 / 255.0
alpha = 0.003
n_steps = 100
target = 94 # hummingbird

x_adv = tf.Variable(tf.zeros(img_shape))
x_init = tf.placeholder(tf.float32, shape=img_shape)
x_min = tf.Variable(tf.zeros(shape=img_shape))
x_max = tf.Variable(tf.zeros(shape=img_shape))
assign_op = tf.assign(x_adv, x_init)
min_step = tf.assign(x_min, tf.clip_by_value(x_init - eps, -1.0, 1.0))
max_step = tf.assign(x_max, tf.clip_by_value(x_init + eps, -1.0, 1.0))
target_class_input = tf.placeholder(tf.int32, shape=[1])
one_hot_target_class = tf.one_hot(target_class_input, 1000)

logits, aux_logits, probs = inception(x_adv, reuse=False)
learning_rate = tf.placeholder(tf.float32, ())

sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)

saver = tf.train.Saver(slim.get_model_variables('InceptionV3'))
saver.restore(sess, 'inception_v3.ckpt')
print('restored inceptionv3')

img = open_image('adv-90.png')
print('going to classify img')
classify(img)

h = img.shape[0]
w = img.shape[1]
channel_axis = 2
shear = 5 * np.pi/180
theta = 5 * np.pi/180
tx = int(w*0.05)
ty = int(h*0.05)
zx = 0.95
zy = 0.95

# add noise
noisy = img + np.random.randn(*img.shape)*0.05
np.clip(noisy, -1, 1, out=noisy)
print('going to classify noisy img')
classify(noisy)
save_image(noisy, 'noisy.png')

# rotate
transform_matrix = np.array(
    [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
rotated = apply_transform(img, transform_matrix, channel_axis)
print('going to classify rotated img')
classify(rotated)
save_image(rotated, 'rotated.png')

 # shift
transform_matrix = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
shifted = apply_transform(img, transform_matrix, channel_axis)
print('going to classify shifted img')
classify(shifted)
save_image(shifted, 'shifted.png')

# shear
transform_matrix = np.array(
    [[1, -np.sin(shear), 0], [0, np.cos(shear), 0], [0, 0, 1]])
transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
sheared = apply_transform(img, transform_matrix, channel_axis)
print('going to classify sheared img')
classify(sheared)
save_image(sheared, 'sheared.png')

# zoom
transform_matrix = np.array([[zx, 0, 0], [0, zy, 0], [0, 0, 1]])
transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
zoomed = apply_transform(img, transform_matrix, channel_axis)
print('going to classify zoomed img')
classify(zoomed)
save_image(zoomed, 'zoomed.png')

# jpeg compress
im = Image.fromarray(np.around((img + 1.0) * 255 / 2).astype(np.uint8))
im.save('/dev/shm/tmp.jpg', quality=50)
compressed = np.asarray(Image.open('/dev/shm/tmp.jpg'))
print('going to classify compressed img')
compressed = (np.float32(compressed) / 255) * 2 - 1.0
classify(compressed)
save_image(compressed, 'compressed.png')

