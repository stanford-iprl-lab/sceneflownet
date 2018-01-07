from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import time
import sys


def loss(frame2_input_xyz, gt_frame1_pred_xyz, pred_xyz, pred_r, pred_mask, pred_score, pred_transl,pred_rot, gt_rot, gt_transl, gt_xyz, gt_r, gt_score, batch_size, dim=3, h=240, w=320): 
  obj_mask_origin = tf.greater(gt_xyz[:,:,:,2],tf.zeros_like(gt_xyz[:,:,:,2]))
  obj_mask_origin = tf.cast(obj_mask_origin,tf.float32)
  obj_mask_1  = tf.reshape(obj_mask_origin,[-1,h,w,1])
  obj_mask_dim = tf.tile(obj_mask_1,[1,1,1,dim])
  
  loss_mask = tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(obj_mask_origin,dtype=tf.int32),logits=pred_mask)) 
  score_weight = obj_mask_origin + 0.001
  
  loss_score = tf.reduce_sum( (score_weight) * tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(gt_score,dtype=tf.int32), logits=pred_score)) / tf.reduce_sum(score_weight + 0.000001)

  loss_elem = tf.reduce_sum(tf.squared_difference(pred_xyz,gt_xyz)*obj_mask_dim)/tf.reduce_sum(obj_mask_1 + 0.000001)

  loss_transl = tf.reduce_sum(obj_mask_dim * tf.squared_difference(pred_transl,gt_transl))/tf.reduce_sum(obj_mask_1 + 0.000001)
  loss_rot = tf.reduce_sum(obj_mask_dim * tf.squared_difference(pred_rot,gt_rot))/tf.reduce_sum(obj_mask_1 + 0.000001)

  loss_boundary = tf.reduce_sum(tf.squared_difference(gt_r, pred_r)*obj_mask_1 )/tf.reduce_sum(obj_mask_1 + 0.000001)
  loss_variance = 0.0
  loss_violation = 0.0
  loss_variance_transl = 0.0
  loss_flow = 0.0

  for b_i in xrange(batch_size):
    tmp = gt_transl[b_i,:,:,2]
    tmp = tf.reshape(tmp,(-1,))
    y, idx = tf.unique(tmp)
    idx = tf.reshape(idx,(h,w,1)) 
   
    ins_tmp = tf.ones_like(idx)
    ones = tf.ones_like(gt_transl[b_i,:,:,2]) 
    
    obj_mask = obj_mask_origin[b_i]
 
    def flow_loss(z):
      #idx_mask = tf.logical_and(tf.equal(gt_transl[b_i,:,:,2], ones * z) , tf.not_equal(gt_transl[b_i,:,:,2],tf.zeros_like(ones)))
      idx_mask = tf.equal(gt_transl[b_i,:,:,2], ones * z)
      idx_mask = tf.reshape(idx_mask,[h,w,1]) 
      idx_mask = tf.cast(idx_mask,tf.float32)
      idx_mask = idx_mask * obj_mask_1[b_i]
      idx_mask_3d = tf.tile(idx_mask,[1,1,3])  

      transl_tmp_prd = idx_mask_3d * pred_transl[b_i]
      transl_tmp_prd = tf.reshape(transl_tmp_prd,(-1,3))
      transl_tmp_mean = tf.reduce_sum(transl_tmp_prd,axis=0)/(tf.reduce_sum(idx_mask)+0.000001)
      transl_tmp_mean = tf.reshape(transl_tmp_mean,(1,3))

      center_tmp_mean = idx_mask_3d * gt_xyz[b_i]
      center_tmp_mean = tf.reshape(center_tmp_mean,(-1,3))
      center_tmp_mean = tf.reduce_sum(center_tmp_mean,axis=0)/(tf.reduce_sum(idx_mask)+0.000001)
      center_tmp_mean = tf.reshape(center_tmp_mean,(1,3))
      center_tmp_mean = center_tmp_mean + transl_tmp_mean
 
      rot_tmp_prd = idx_mask_3d * pred_rot[b_i]
      rot_tmp_prd = tf.reshape(rot_tmp_prd,(-1,3))
      rot_tmp_mean = tf.reduce_sum(rot_tmp_prd,axis=0)/(tf.reduce_sum(idx_mask)+0.000001)
    
      angle = tf.norm(rot_tmp_mean+0.0000000001)
      axis = rot_tmp_mean / (angle)
      c = tf.cos(angle)
      v = 1 - c
      s = tf.sin(angle)
      
      rot00 = axis[0] ** 2 * v + c
      rot01 = axis[0] * axis[1] * v - axis[2] * s
      rot02 = axis[0] * axis[2] * v + axis[1] * s
      rot10 = axis[0] * axis[1] * v + axis[2] * s
      rot11 = axis[1] ** 2 * v + c
      rot12 = axis[1] * axis[2] * v - axis[0] * s
      rot20 = axis[0] * axis[2] * v - axis[1] * s
      rot21 = axis[1] * axis[2] * v + axis[0] * s
      rot22 = axis[2] ** 2 * v + c
      rot_matrix = tf.stack([rot00, rot10, rot20, rot01, rot11, rot21, rot02, rot12, rot22]) 
      rot_matrix = tf.reshape(rot_matrix,(3,3)) 
     
      pred_frame1_xyz = tf.reshape(frame2_input_xyz[b_i],(-1,3)) 
      pred_frame1_xyz = pred_frame1_xyz + transl_tmp_mean - center_tmp_mean
      pred_frame1_xyz = tf.matmul(pred_frame1_xyz,rot_matrix) + center_tmp_mean

      pred_frame1_xyz = tf.reshape(pred_frame1_xyz,(h,w,3)) 
      loss_tmp = tf.reduce_sum(idx_mask_3d * tf.squared_difference(pred_frame1_xyz, gt_frame1_pred_xyz[b_i])) / (tf.reduce_sum(idx_mask)+0.000001) 
      return loss_tmp
  
    loss_flow += tf.reduce_mean(tf.map_fn(flow_loss,y)) 
   
   
  for b_i in xrange(batch_size):
    tmp = gt_xyz[b_i,:,:,2]
    tmp = tf.reshape(tmp,(-1,))
    y, idx = tf.unique(tmp)
    idx = tf.reshape(idx,(h,w,1))
  
    ins_tmp = tf.ones_like(idx)
    ones = tf.ones_like(gt_xyz[b_i,:,:,dim-1])
    
    obj_mask = obj_mask_origin[b_i]

    def instance_variance_loss(z):
      idx_mask = tf.equal(gt_xyz[b_i,:,:,2], ones * z)
      idx_mask = tf.reshape(idx_mask, [h,w,1])
      idx_mask = tf.cast(idx_mask, tf.float32)
      idx_mask_3d  = tf.tile(idx_mask, [1,1,dim])
      tmp_prd = idx_mask_3d * pred_xyz[b_i]
      tmp_prd = tf.reshape(tmp_prd,(-1,dim))
      tmp_mean = tf.reduce_sum(tmp_prd,axis=0)/(tf.reduce_sum(idx_mask)+0.000001)
      tmp_mean = tf.reshape(tmp_mean,(1,1,3))
      tmp_mean_final = tf.tile(tmp_mean,[h,w,1])
      loss_variance_instance = tf.reduce_sum(idx_mask_3d * tf.squared_difference(tmp_mean_final,pred_xyz[b_i])) / (tf.reduce_sum(idx_mask)+0.000001)
      return loss_variance_instance
  
    #loss_variance += tf.reduce_mean(tf.map_fn(instance_variance_loss,y)) 

    def instance_variance_transl_loss(z):
      idx_mask = tf.equal(gt_xyz[b_i,:,:,2], ones * z)
      idx_mask = tf.reshape(idx_mask, [h,w,1])
      idx_mask = tf.cast(idx_mask, tf.float32)
      idx_mask_3d  = tf.tile(idx_mask, [1,1,dim])
      tmp_prd = idx_mask_3d * pred_transl[b_i]
      tmp_prd = tf.reshape(tmp_prd,(-1,dim))
      tmp_mean = tf.reduce_sum(tmp_prd,axis=0)/(tf.reduce_sum(idx_mask)+0.000001)
      tmp_mean = tf.reshape(tmp_mean,(1,1,3))
      tmp_mean_final = tf.tile(tmp_mean,[h,w,1])
      loss_variance_transl_instance = tf.reduce_sum(idx_mask_3d * tf.squared_difference(tmp_mean_final, pred_transl[b_i])) / (tf.reduce_sum(idx_mask)+0.000001)
      return loss_variance_transl_instance
  
    #loss_variance_transl += tf.reduce_mean(tf.map_fn(instance_variance_transl_loss,y)) 
 
  
    def instance_violation_loss(z):
      idx_mask = tf.equal(gt_xyz[b_i,:,:,dim-1],ones * z)
      idx_mask = tf.logical_and(idx_mask,tf.cast(obj_mask,tf.bool))
      idx_mask = tf.reshape(idx_mask,[h,w,1])
      idx_mask = tf.cast(idx_mask,tf.float32)
      idx_mask_3d  = tf.tile(idx_mask,[1,1,dim])

      tmp_prd = idx_mask_3d * pred_xyz[b_i,:,:,0:dim]
      tmp_prd = tf.reshape(tmp_prd,(-1,dim))

      tmp_r = idx_mask_3d[:,:,0:1] * pred_r[b_i]
      tmp_r_mean = tf.reduce_sum(tmp_r)/(tf.reduce_sum(idx_mask)+0.000001)
      r = tmp_r_mean * 0.5

      friend_mask = idx_mask_3d[:,:,0]
      l2_error = tf.reduce_sum(tf.squared_difference(pred_xyz[b_i], gt_xyz[b_i]),2)
      dist = tf.sqrt(l2_error)
      pull_mask =  tf.less(r * ones, dist)
      pull_mask = tf.cast(pull_mask,tf.float32)
      pos = tf.reduce_sum(friend_mask * pull_mask * l2_error)/(tf.reduce_sum(friend_mask * pull_mask) + 0.000001)
      return pos

    #loss_violation += tf.reduce_mean(tf.map_fn(instance_violation_loss,y))
   
    return loss_flow, loss_rot, loss_transl#, loss_variance_transl, loss_mask, loss_score, loss_elem, loss_variance, loss_boundary, loss_violation     
