import tensorflow as tf
import numpy as np
import sys
import math
from correlation import correlation

BATCH_SIZE = 1
HEIGHT = 2
WIDTH = 2
CHANNELS = 1

NEIGHBORHOOD_SIZE = 2
MAX_DISPLACEMENT = int(math.ceil(NEIGHBORHOOD_SIZE / 2.0))
STRIDE_2 = 1

with tf.Session('') as sess:
  with tf.device('/gpu:0'):
    fmA = tf.ones((BATCH_SIZE, HEIGHT, WIDTH, CHANNELS), dtype=tf.float32)
    fmB = tf.convert_to_tensor(np.random.randint(1,5, size=(BATCH_SIZE, HEIGHT, WIDTH, CHANNELS)), dtype=tf.float32)
    corr = correlation(fmA,fmB,1,1,1,1,1) # input_a, input_b, kernel_size, max_displacement, stride_1, stride_2, padding
    sess.run(tf.initialize_all_variables())
    corr_, fmA_, fmB_= sess.run([corr,fmA,fmB])
    print(corr_[0])
    print(fmA_[0][:,:,0])
    print(fmB_[0][:,:,0])                       
