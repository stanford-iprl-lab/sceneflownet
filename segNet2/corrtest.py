import tensorflow as tf
import numpy as np

with tf.Session('') as sess:
  with tf.device('/gpu:0'):
    fmA = tf.ones((BATCH_SIZE, HEIGHT, WIDTH, CHANNELS), dtype=tf.int32)
    fmB = tf.convert_to_tensor(np.random.randint(5, size=(BATCH_SIZE, HEIGHT, WIDTH, CHANNELS)), dtype=tf.int32)

