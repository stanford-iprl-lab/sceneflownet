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

test_forward = False

if test_forward:
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
else:
  with tf.Session('') as sess:
    fmA1 = np.ones((BATCH_SIZE, HEIGHT, WIDTH, CHANNELS)).astype('float32')
    fmB1 = np.random.randn(BATCH_SIZE, HEIGHT, WIDTH, CHANNELS).astype('float32')
    with tf.device('/gpu:0'):
      fmA = tf.constant(fmA1)   
      fmB = tf.Variable(fmB1)
      corr = correlation(fmA,fmB,1,1,1,1,1)
      loss = tf.reduce_sum(corr)
      train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)
      sess.run(tf.initialize_all_variables())
      grad_corr = tf.gradients(loss,[corr])[0]
      grad_fmB = tf.gradients(loss,[fmB])[0]
      grad_fmA = tf.gradients(loss,[fmA])[0] 
      for i in xrange(1):
        fmB_ = sess.run(fmB)
        print(fmB_[0][:,:,0])
        trainloss, _, corr_, fmA_, fmB_, grad_corr_, grad_fmB_, grad_fmA_ = sess.run([loss,train,corr,fmA,fmB,grad_corr, grad_fmB, grad_fmA])
        print(trainloss)
        print(corr_[0])
        print(grad_corr_[0])   
        print(fmB_[0][:,:,0]) 
        print(grad_fmB_[0][:,:,0])   
        print(fmA_[0][:,:,0]) 
        print(grad_fmA_[0][:,:,0])   
