import tensorflow as tf
import numpy as np
from pyquaternion import Quaternion


np.random.seed(100)

with tf.Session('') as sess:
  xyz1 = np.random.randn(5,3).astype('float32')
  xyz2 = np.zeros((5,3)).astype('float32')
  quatern = np.random.randn(5,4).astype('float32')
  for i in xrange(5):
    a,b,c,d = np.random.randn(4)
    q = Quaternion(a=a,b=b,c=c,d=d).normalised
    R = q.rotation_matrix
    xyz2[i] = R.dot(xyz1[i])
    w1, x1, y1, z1 = q.elements
    x2, y2, z2  = xyz1[i]#tf.unstack(inp, axis=-1)

    wm =         - x1 * x2 - y1 * y2 - z1 * z2
    xm = w1 * x2           + y1 * z2 - z1 * y2
    ym = w1 * y2           + z1 * x2 - x1 * z2
    zm = w1 * z2           + x1 * y2 - y1 * x2

    x = -wm * x1 + xm * w1 - ym * z1 + zm * y1
    y = -wm * y1 + ym * w1 - zm * x1 + xm * z1
    z = -wm * z1 + zm * w1 - xm * y1 + ym * x1
    print("compare")
    print(np.array([x,y,z]))
    print(xyz2[i])   
 
  with tf.device('/gpu:0'):
      inp = tf.constant(xyz1)
      gt  = tf.constant(xyz2)
      rot_quaternion = tf.Variable(quatern)
      print(rot_quaternion)
      quaternion_norm = tf.norm(rot_quaternion,axis=1) * tf.sign(rot_quaternion[:,0])
      quaternion_norm = tf.expand_dims(quaternion_norm,-1)
      rot_quaternion /= quaternion_norm

      w1, x1, y1, z1 = tf.unstack(rot_quaternion, axis=-1)
      x2, y2, z2  = tf.unstack(inp, axis=-1)


      wm =         - x1 * x2 - y1 * y2 - z1 * z2
      xm = w1 * x2           + y1 * z2 - z1 * y2
      ym = w1 * y2           + z1 * x2 - x1 * z2
      zm = w1 * z2           + x1 * y2 - y1 * x2

      x = -wm * x1 + xm * w1 - ym * z1 + zm * y1
      y = -wm * y1 + ym * w1 - zm * x1 + xm * z1
      z = -wm * z1 + zm * w1 - xm * y1 + ym * x1
 
      x_flow = tf.squeeze(tf.stack((x,y,z),axis=-1))
      loss = tf.reduce_mean(tf.norm(x_flow-gt,2))
      train = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)
      sess.run(tf.initialize_all_variables())
      for i in xrange(2000000):
        trainloss,_ = sess.run([loss,train])
        up = np.mean(np.linalg.norm(xyz1-xyz2,axis=1))
        print i, trainloss, up
