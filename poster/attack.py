import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
from tensorflow.python.ops.losses.losses import softmax_cross_entropy as sce
import os
from PIL import Image
import numpy as np


tf.logging.set_verbosity(tf.logging.ERROR)
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
    g, p = sess.run([logits, probs], feed_dict={x_adv: img})
    g = np.squeeze(g)
    p = np.squeeze(p)
    topk = list(p.argsort()[-10:][::-1])
    topprobs = p[topk]
    toplogits = g[topk]
    print 'probs', zip(topk, topprobs, toplogits)

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
loss = sce(one_hot_target_class, logits)
loss += sce(one_hot_target_class, aux_logits, weights=0.8)

opt = tf.train.AdamOptimizer(learning_rate)
optim_step = opt.minimize(loss, var_list=[x_adv])
clip_step = tf.assign(x_adv, tf.clip_by_value(x_adv, x_min, x_max))

sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)

saver = tf.train.Saver(slim.get_model_variables('InceptionV3'))
saver.restore(sess, 'inception_v3.ckpt')
print('restored inceptionv3')

img = open_image('traffic-light-small.png')
img = img.reshape((1, 299, 299, 3))
print('going to classify img')
classify(img)

sess.run(assign_op, feed_dict={x_init: img})
sess.run(min_step, feed_dict={x_init: img})
sess.run(max_step, feed_dict={x_init: img})

chart_logits = np.zeros((n_steps, 5), dtype=np.float32)
for i in range(n_steps):
  if True:
      adv_img = x_adv.eval()
      chart_logits[i, 0] = i
      chart_logits[i, 1:] = get_logits(adv_img, [11, 94, 811, 920])
      if i % 10 == 0:
          adv_img = x_adv.eval()
          diff_img = adv_img - img
          save_image(diff_img / np.abs(diff_img.max()), 'diff-' + str(i) + '.png')
          print('max diff', (abs(diff_img.max()) * 255) / 2)
          save_image(adv_img, 'adv-' + str(i) + '.png')
          classify(adv_img)
  sess.run(
      [optim_step, loss],
      feed_dict={learning_rate: alpha, target_class_input: [target]})
  sess.run(clip_step)

np.savetxt('logits.csv', chart_logits, delimiter=',', header='epoch,goldfinch,hummingbird,space heater,traffic light', comments='')
save_image(diff_img / np.abs(diff_img.max()), 'diff-' + str(i) + '.png')
print 'classifying adv example'
classify(adv_img)
