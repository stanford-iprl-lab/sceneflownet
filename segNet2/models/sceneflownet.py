import numpy as np
import tensorflow as tf
import tflearn
import sys
from nets_factory import get_network
import resnet_v1 as resnet_v1

def encoder(x,reuse=False):
  with tf.variable_scope("encodexyz"):
    x = tflearn.layers.conv.conv_2d(x,16,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2',reuse=reuse,scope="conv1_1")
    x = tflearn.layers.conv.conv_2d(x,16,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2',reuse=reuse,scope="conv1_2")
    x = tflearn.layers.conv.conv_2d(x,16,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2',reuse=reuse,scope="conv1_3")
##120,160
    x = tflearn.layers.conv.conv_2d(x,16,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2',reuse=reuse,scope="conv2_1")
    x = tflearn.layers.conv.conv_2d(x,16,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2',reuse=reuse,scope="conv2_2")
    x = tflearn.layers.conv.conv_2d(x,32,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2',reuse=reuse,scope="conv2_3")
##60,80
    x = tflearn.layers.conv.conv_2d(x,32,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2',reuse=reuse,scope="conv3_1")
    x = tflearn.layers.conv.conv_2d(x,32,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2',reuse=reuse,scope="conv3_2")
    x = tflearn.layers.conv.conv_2d(x,64,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2',reuse=reuse,scope="conv3_3")
##30,40
  return x

def decoder(x):
  with tf.variable_scope("decode") as scope:
    x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x2 = x
    
    x=tflearn.layers.conv.conv_2d(x,128,(5,5),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
    #15, 20
    x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x3 = x
    x=tflearn.layers.conv.conv_2d(x,256,(5,5),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')

    #8, 10
    x=tflearn.layers.conv.conv_2d(x,256,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x=tflearn.layers.conv.conv_2d(x,256,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x4 = x
    x=tflearn.layers.conv.conv_2d(x,512,(5,5),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')

    #4, 5
    x=tflearn.layers.conv.conv_2d(x,512,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x=tflearn.layers.conv.conv_2d(x,512,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x=tflearn.layers.conv.conv_2d_transpose(x,256,[5,5],[8,10],strides=2,activation='linear',weight_decay=1e-5,regularizer='L2')

    #8,10 
    x4=tflearn.layers.conv.conv_2d(x4,256,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
    x = tf.nn.relu(tf.add(x,x4))
    x=tflearn.layers.conv.conv_2d(x,512,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')  
    x=tflearn.layers.conv.conv_2d(x,256,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x=tflearn.layers.conv.conv_2d_transpose(x,128,[5,5],[15,20],strides=2,activation='linear',weight_decay=1e-5,regularizer='L2')

    #15,20
    x3=tflearn.layers.conv.conv_2d(x3,128,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
    x = tf.nn.relu(tf.add(x,x3))
    x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')  
    x=tflearn.layers.conv.conv_2d_transpose(x,64,[5,5],[30,40],strides=2,activation='linear',weight_decay=1e-5,regularizer='L2')
    
    #30,40
    x2=tflearn.layers.conv.conv_2d(x2,64,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
    x = tf.nn.relu(tf.add(x,x2))
    x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2') 
    x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x=tflearn.layers.conv.conv_2d_transpose(x,64,[5,5],[60,80],strides=2,activation='linear',weight_decay=1e-5,regularizer='L2')
   
    #60,80
    x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
    x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x=tflearn.layers.conv.conv_2d_transpose(x,64,[5,5],[120,160],strides=2,activation='linear',weight_decay=1e-5,regularizer='L2')

    #120,160
    x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')  
    x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
    x=tflearn.layers.conv.conv_2d_transpose(x,64,[5,5],[240,320],strides=2,activation='linear',weight_decay=1e-5,regularizer='L2')

    #240,320
    x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')  
 
    #### success
    x_s = tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
  return x_s

rad = 10
dia = 2 * rad + 1

def quaternion_multiply(a,b):
  w1, x1, y1, z1 = tf.unstack(a, axis=-1)
  w2, x2, y2, z2 = tf.unstack(b, axis=-1)
  w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
  x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
  y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
  z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
  return tf.squeeze(tf.stack((w,x,y,z),axis=-1))

def cnnmodel(frame1_xyz,frame1_rgb,frame2_xyz,frame2_rgb):
  x = tf.concat([frame1_xyz,frame1_rgb,frame2_xyz,frame2_rgb],3)
  x = encoder(x)
  x_s = decoder(x)
  x_transl = tflearn.layers.conv.conv_2d(x_s,3,(3,3),strides=1,activation='linear',weight_decay=1e-3,regularizer='L2')
  rot_quaternion = tflearn.layers.conv.conv_2d(x_s,4,(3,3),strides=1,activation='linear',weight_decay=1e-3,regularizer='L2')
  
  ### quaternion normalize 
  quaternion_norm = tf.norm(rot_quaternion,axis=3) * tf.sign(rot_quaternion[:,:,:,0])
  quaternion_norm = tf.expand_dims(quaternion_norm,-1)
  x_quaternion = rot_quaternion / (quaternion_norm)
 
  w1, x1, y1, z1 = tf.unstack(x_quaternion, axis=-1)
  x2, y2, z2  = tf.unstack(frame2_xyz, axis=-1)

  wm =         - x1 * x2 - y1 * y2 - z1 * z2
  xm = w1 * x2           + y1 * z2 - z1 * y2
  ym = w1 * y2           + z1 * x2 - x1 * z2
  zm = w1 * z2           + x1 * y2 - y1 * x2

  x = -wm * x1 + xm * w1 - ym * z1 + zm * y1
  y = -wm * y1 + ym * w1 - zm * x1 + xm * z1
  z = -wm * z1 + zm * w1 - xm * y1 + ym * x1

  x_flow = tf.stack((x,y,z),axis=-1)
  #x_flow = tf.squeeze(tf.stack((x,y,z),axis=-1))
  x_flow = x_flow + x_transl - frame2_xyz

  x_center = tflearn.layers.conv.conv_2d(x_s,3,(3,3),strides=1,activation='linear',weight_decay=1e-3,regularizer='L2')
  x_score = tflearn.layers.conv.conv_2d(x_s,2,(3,3),strides=1,activation='linear',weight_decay=1e-3,regularizer='L2')
  x_mask = tflearn.layers.conv.conv_2d(x_s,2,(3,3),strides=1,activation='linear',weight_decay=1e-3,regularizer='L2')
  x_boundary = tflearn.layers.conv.conv_2d(x_s,2,(3,3),strides=1,activation='linear',weight_decay=1e-3,regularizer='L2')
 
  x_center = tf.add(x_center,frame2_xyz)
  xc, yc, zc  = tf.unstack(x_center, axis=-1)

  wmc =         - x1 * xc - y1 * yc - z1 * zc
  xmc = w1 * xc           + y1 * zc - z1 * yc
  ymc = w1 * yc           + z1 * xc - x1 * zc
  zmc = w1 * zc           + x1 * yc - y1 * xc

  xc = -wmc * x1 + xmc * w1 - ymc * z1 + zmc * y1
  yc = -wmc * y1 + ymc * w1 - zmc * x1 + xmc * z1
  zc = -wmc * z1 + zmc * w1 - xmc * y1 + ymc * x1

  x_center_p = tf.stack((xc,yc,zc),axis=-1)
  
  x_traj = tf.concat([x_center,x_center],3)#x_center_p],3)
  return rot_quaternion, x_transl, x_traj, x_flow, x_center, x_mask, x_score, x_boundary
