import numpy as np
import tensorflow as tf
import tflearn
import sys
sys.path.append('/home/linshaonju/interactive-segmentation/segNet2/src')
from correlation import correlation

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

def cnnmodel(frame1_xyz,frame1_rgb,frame2_xyz,frame2_rgb):
  frame1_feat_rgb,_ = get_network('resnet50',frame1_rgb,weight_decay=1e-5, is_training=True)
  frame2_feat_rgb,_ = get_network('resnet50',frame2_rgb,weight_decay=1e-5, is_training=True, reuse=True)

  print(frame1_feat_rgb)
  frame1_feat_o = encoder(frame1_xyz)
  frame2_feat = encoder(frame2_xyz,reuse=True)
  
  cc_o = correlation(frame2_feat_rgb,frame1_feat_rgb,1,rad,1,1,rad)
  cc = tf.reshape(cc_o,[-1, 30*40, dia * dia, 1]) 
  cc_relu = tf.nn.relu(cc)

  frame1_feat = tf.transpose(frame1_feat_o,[0,3,1,2]) 
  frame1_feat_padded = tf.pad(frame1_feat,paddings=[[0,0],[0,0],[rad,rad],[rad,rad]])
  frame1_list = []
  for i in xrange(30):
    for j in xrange(40):
      tmp = frame1_feat_padded[:,:,0+i:2*rad+1+i,0+j:2*rad+1+j]
      tmp = tf.reshape(tmp,[-1,64,dia * dia])
      frame1_list.append(tmp) 
  frame1_list = tf.stack(frame1_list,axis=2)
  frame1_list = tf.transpose(frame1_list,[0,2,3,1])
  print(cc_relu) 
  print(frame1_list) 

  frame1_list = frame1_list * cc_relu
  
  frame1_list = tf.nn.max_pool(frame1_list,ksize=[1,1,dia * dia,1],strides=[1,1,dia * dia,1],padding='VALID')
  frame1_list = tf.reshape(frame1_list,(-1,30,40,64))

  x = tf.concat([frame2_feat,frame1_feat_o,frame1_list],3)

  x_s = decoder(x)
  x_transl = tflearn.layers.conv.conv_2d(x_s,3,(3,3),strides=1,activation='linear',weight_decay=1e-3,regularizer='L2')
  rot_quaternion = tflearn.layers.conv.conv_2d(x_s,4,(3,3),strides=1,activation='linear',weight_decay=1e-3,regularizer='L2')

  quaternion_norm = tf.norm(rot_quaternion,axis=3) * tf.sign(rot_quaternion[:,:,:,0])
  quaternion_norm = tf.expand_dims(quaternion_norm,-1)
  rot_quaternion /= quaternion_norm

  w1, x1, y1, z1 = tf.unstack(rot_quaternion, axis=-1)
  x2, y2, z2  = tf.unstack(frame2_xyz, axis=-1)

  wm =         - x1 * x2 - y1 * y2 - z1 * z2
  xm = w1 * x2           + y1 * z2 - z1 * y2
  ym = w1 * y2           + z1 * x2 - x1 * z2
  zm = w1 * z2           + x1 * y2 - y1 * x2

  x = -wm * x1 + xm * w1 - ym * z1 + zm * y1
  y = -wm * y1 + ym * w1 - zm * x1 + xm * z1
  z = -wm * z1 + zm * w1 - xm * y1 + ym * x1

  x_flow = tf.squeeze(tf.stack((x,y,z),axis=-1))
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

  x_traj = tf.concat([x_center,x_center_p],3)

  return x_traj, rot_quaternion, x_transl, x_flow, x_center, x_mask, x_score, x_boundary
