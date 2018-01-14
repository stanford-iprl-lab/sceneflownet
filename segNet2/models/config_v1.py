from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

#### RESTORE
tf.app.flags.DEFINE_string(
  'train_dir','./output/mask_rcnn',
  'Directory where checkpoints and event logs are written to.')
 
tf.app.flags.DEFINE_string(
  'pretrained_model', './data/pretrained_models/resnet_v1_50.ckpt',
  'Path to pretrained model.')

#### network
tf.app.flags.DEFINE_string(
  'network', 'resnet50',
  'name of backbone network')

#### optimization flags
tf.app.flags.DEFINE_float(
  'weight_decay',0.00005,'The weight decay on the model weights.')

tf.app.flags.DEFINE_float(
  'adam_beta1', 0.9,
  'The exponential decay rate for the 1st moment estimates')

tf.app.flags.DEFINE_float(
  'adam_beta2', 0.999,
  'The exponential decay rate for the 2nd moment estimates.')

#### learning rate flags
tf.app.flags.DEFINE_string(
  'learning_rate_decay_type','exponential',
  'specifies how the learning rate is decayed. One of "fixed", "exponential", "polynomial"')

tf.app.flags.DEFINE_float('learning_rate',0.002,'Initial learning rate.')

tf.app.flags.DEFINE_float(
  'end_learning_rate', 0.00001,
  'The minimal end learning rate used by a polynomial decay learning rate')

tf.app.flags.DEFINE_float(
  'learning_rate_decay_factor',0.94,'Learning rate decay factor.')

tf.app.flags.DEFINE_float(
  'learning_rate_decay',2.0,
  'Number of epochs after which learning rate decays.')

tf.app.flags.DEFINE_float(
  'num_epochs_per_decay', 2.0,
  'Number of epochs after which learning rate decays.')

tf.app.flags.DEFINE_bool(
  'sync_replicas', False,
  'Whether or not to synchronize the replicas during training.')

tf.app.flags.DEFINE_integer(
  'replicas_to_aggregate',1,
  'The Number of gradients to collect before updating params.')

tf.app.flags.DEFINE_float(
  'moving_average_decay',None,
  'The decay to use for the moving average.'
  'If left as None, then moving averages are not used.')

FLAGS = tf.app.flags.FLAGS
