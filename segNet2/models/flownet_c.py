import numpy as np
import tensorflow as tf
import tflearn
import sys
sys.path.append('/home/lins/interactive-segmentation/segNet2/src')
from correlation import correlation

def flownet_c_encoder(x,reuse=False):
  with tf.name_scope("model"):
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

def cnnmodel(frame1_xyz,frame1_rgb,frame2_xyz,frame2_rgb):
  frame1_input = tf.concat([frame1_xyz,frame1_rgb],3)
  frame2_input = tf.concat([frame2_xyz,frame2_rgb],3)

  frame1_x = flownet_c_encoder(frame1_input)
  frame2_x = flownet_c_encoder(frame2_input,reuse=True)

  cc = correlation(frame2_x,frame1_x,1,20,1,2,20)
  cc_relu = tf.nn.relu(cc)

  x = tf.concat([frame2_x,cc],3)

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

  pred_frame2_xyz =tflearn.layers.conv.conv_2d(x_s,3,(3,3),strides=1,activation='linear',weight_decay=1e-3,regularizer='L2')
  pred_frame2_r = tflearn.layers.conv.conv_2d(x_s,1,(3,3),strides=1,activation='linear',weight_decay=1e-3,regularizer='L2')
  pred_frame2_mask = tflearn.layers.conv.conv_2d(x_s,2,(3,3),strides=1,activation='linear',weight_decay=1e-3,regularizer='L2')
  pred_frame2_score = tflearn.layers.conv.conv_2d(x_s,2,(3,3),strides=1,activation='linear',weight_decay=1e-3,regularizer='L2')

  pred_frame2_xyz = tf.add(pred_frame2_xyz,frame2_xyz)
 
  x_transl = tflearn.layers.conv.conv_2d(x_s,3,(3,3),strides=1,activation='linear',weight_decay=1e-3,regularizer='L2')
  x_rot = tflearn.layers.conv.conv_2d(x_s,3,(3,3),strides=1,activation='linear',weight_decay=1e-3,regularizer='L2')

  return pred_frame2_mask, pred_frame2_r, pred_frame2_xyz, pred_frame2_score, x_transl, x_rot
