import numpy as np
import tensorflow as tf
import tflearn
import sys
sys.path.append('/home/linshaonju/interactive-segmentation/segNet2/src')
from correlation import correlation

from nets_factory import get_network
import resnet_v1 as resnet_v1

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
  frame1_rgb = tf.image.resize_images(frame1_rgb,[480,640])
  frame2_rgb = tf.image.resize_images(frame2_rgb,[480,640])

  frame1_feat_rgb,_ = get_network('resnet50',frame1_rgb,weight_decay=1e-5, is_training=True)
  frame2_feat_rgb,_ = get_network('resnet50',frame2_rgb,weight_decay=1e-5, is_training=True, reuse=True)

  frame1_feat = encoder(frame1_xyz)
  frame2_feat = encoder(frame2_xyz,reuse=True)
  
  cc_o = correlation(frame2_feat_rgb,frame1_feat_rgb,1,rad,1,1,rad)
  return cc_o
