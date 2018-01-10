import numpy as np
import tensorflow as tf
import tflearn
import sys
sys.path.append('/home/linshaonju/interactive-segmentation/segNet2/src')
from correlation import correlation
from utils import LeakyReLU

def encoder_rgb(x,reuse=False):
  with tf.name_scope("model_rgb"):
    x = tflearn.layers.conv.conv_2d(x,16,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2',reuse=reuse,scope="conv1_1_rgb")
    x = tflearn.layers.conv.conv_2d(x,16,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2',reuse=reuse,scope="conv1_2_rgb")
    x = tflearn.layers.conv.conv_2d(x,16,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2',reuse=reuse,scope="conv1_3_rgb")
##120,160
    x = tflearn.layers.conv.conv_2d(x,16,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2',reuse=reuse,scope="conv2_1_rgb")
    x = tflearn.layers.conv.conv_2d(x,16,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2',reuse=reuse,scope="conv2_2_rgb")
    x = tflearn.layers.conv.conv_2d(x,32,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2',reuse=reuse,scope="conv2_3_rgb")
##60,80
    x = tflearn.layers.conv.conv_2d(x,32,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2',reuse=reuse,scope="conv3_1_rgb")
    x = tflearn.layers.conv.conv_2d(x,32,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2',reuse=reuse,scope="conv3_2_rgb")
    x = tflearn.layers.conv.conv_2d(x,64,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2',reuse=reuse,scope="conv3_3_rgb")
##30,40
  return x

def encoder(x,reuse=False):
  with tf.name_scope("model_xyz"):
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


rad = 10
dia = 2 * rad + 1

def cnnmodel(frame1_xyz,frame1_rgb,frame2_xyz,frame2_rgb):
  frame1_feat_rgb = encoder_rgb(frame1_rgb)
  frame2_feat_rgb = encoder_rgb(frame2_rgb,reuse=True)

  frame1_feat = encoder(frame1_xyz)
  frame2_feat = encoder(frame2_xyz,reuse=True)
  
  frame2_feat_rgb = tf.pad(frame2_feat_rgb,paddings=[[0,0],[1,1],[0,0],[0,0]])
  frame1_feat_rgb = tf.pad(frame1_feat_rgb,paddings=[[0,0],[1,1],[0,0],[0,0]])

  cc = correlation(frame2_feat_rgb,frame1_feat_rgb,1,rad,1,1,rad)[:,1:-1,:,:]
  cc_relu = tf.nn.relu(cc)

  frame1_feat = tf.transpose(frame1_feat,[0,3,1,2]) 
  frame1_feat_padded = tf.pad(frame1_feat,paddings=[[0,0],[0,0],[rad,rad],[rad,rad]])
  
  frame1_list = []

  for i in xrange(30):
    for j in xrange(40):
      tmp = frame1_feat_padded[:,:,0+i:2*rad+1+i,0+j:2*rad+1+j]
      tmp = tf.reshape(tmp,[-1,64,dia * dia])
      frame1_list.append(tmp)
 
  frame1_list = tf.stack(frame1_list,axis=2)
  frame1_list = tf.transpose(frame1_list,[0,2,3,1])
  cc_relu = tf.reshape(cc_relu,[-1,30*40,dia * dia,1])
  
  frame1_list = frame1_list * cc_relu
  frame1_list = tf.nn.max_pool(frame1_list,ksize=[1,1,dia * dia,1],strides=[1,1,dia * dia,1],padding='VALID')
  frame1_list = tf.reshape(frame1_list,(-1,30,40,64))
  
  x = tf.concat([frame2_feat,frame1_list],3)

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
