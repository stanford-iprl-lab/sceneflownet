import numpy as np
import tensorflow as tf
import tflearn

def cnnmodel(inp_xyz):
  x = inp_xyz
  x=tflearn.layers.conv.conv_2d(x,16,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
  x=tflearn.layers.conv.conv_2d(x,16,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
  x00 = x
  x=tflearn.layers.conv.conv_2d(x,16,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')

#120, 160
  x=tflearn.layers.conv.conv_2d(x,16,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
  x=tflearn.layers.conv.conv_2d(x,16,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
  x0 = x
  x=tflearn.layers.conv.conv_2d(x,16,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')

#60, 80
  x=tflearn.layers.conv.conv_2d(x,32,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
  x=tflearn.layers.conv.conv_2d(x,32,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2') 
  x1 = x
  x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')

#30, 40
  x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
  x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
  x2 = x
  x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')

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
  x=tflearn.layers.conv.conv_2d(x,256,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
  x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')  
  x=tflearn.layers.conv.conv_2d_transpose(x,64,[5,5],[30,40],strides=2,activation='linear',weight_decay=1e-5,regularizer='L2')

#30,40
  x2=tflearn.layers.conv.conv_2d(x2,64,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
  x = tf.nn.relu(tf.add(x,x2))
  x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2') 
  x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
  x=tflearn.layers.conv.conv_2d_transpose(x,64,[5,5],[60,80],strides=2,activation='linear',weight_decay=1e-5,regularizer='L2')


#60,80
  x1=tflearn.layers.conv.conv_2d(x1,64,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')
  x = tf.nn.relu(tf.add(x,x1))
  x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
  x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
  x=tflearn.layers.conv.conv_2d_transpose(x,64,[5,5],[120,160],strides=2,activation='linear',weight_decay=1e-5,regularizer='L2')

#120,160
  x0=tflearn.layers.conv.conv_2d(x0,64,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')  
  x_t =tf.nn.relu(tf.add(x,x0))
  x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
  x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
  x=tflearn.layers.conv.conv_2d_transpose(x,64,[5,5],[240,320],strides=2,activation='linear',weight_decay=1e-5,regularizer='L2')

#240,320
  x00=tflearn.layers.conv.conv_2d(x00,64,(3,3),strides=1,activation='linear',weight_decay=1e-5,regularizer='L2')  
  x_t =tf.nn.relu(tf.add(x,x00))
 
#### success
  x_s = tflearn.layers.conv.conv_2d(x_t,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
  x_xyz =tflearn.layers.conv.conv_2d(x_s,3,(3,3),strides=1,activation='linear',weight_decay=1e-3,regularizer='L2')
  x_r = tflearn.layers.conv.conv_2d(x_s,1,(3,3),strides=1,activation='linear',weight_decay=1e-3,regularizer='L2')
  mask = tflearn.layers.conv.conv_2d(x_s,2,(3,3),strides=1,activation='linear',weight_decay=1e-3,regularizer='L2')
  x_score = tflearn.layers.conv.conv_2d(x_s,1,(3,3),strides=1,activation='linear',weight_decay=1e-3,regularizer='L2')

  x_xyz = tf.add(x_xyz,inp_xyz)
 
  return mask, x_r, x_xyz, x_score
