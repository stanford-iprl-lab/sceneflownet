from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import time
import sys


h = 240
w = 320

def read_and_decode(filename_queue):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
    serialized_example,
    # Defaults are not specified since both keys are required.
    features = {
      'in_1frame_xyz':tf.FixedLenFeature([],tf.string),
      'in_2frame_xyz':tf.FixedLenFeature([],tf.string),
      'in_1frame_rgb':tf.FixedLenFeature([],tf.string),
      'in_2frame_rgb':tf.FixedLenFeature([],tf.string), 
      'outs_2frame_xyz':tf.FixedLenFeature([],tf.string),
      'outs_1frame_id':tf.FixedLenFeature([],tf.string),
      'outs_2frame_id':tf.FixedLenFeature([],tf.string), 
      'outs_2frame_r': tf.FixedLenFeature([],tf.string),
      'outs_2frame_score':tf.FixedLenFeature([],tf.string),  
      'outs_trans_translation':tf.FixedLenFeature([],tf.string),
      'outs_trans_rot':tf.FixedLenFeature([],tf.string),
      'outs_1frame_pred_xyz':tf.FixedLenFeature([],tf.string),
      'outs_cc':tf.FixedLenFeature([],tf.string),
      'instance_id':tf.FixedLenFeature([],tf.int64),
    })

  in_1frame_xyz = tf.decode_raw(features['in_1frame_xyz'],tf.float32)
  in_2frame_xyz= tf.decode_raw(features['in_2frame_xyz'],tf.float32)
  in_1frame_rgb= tf.decode_raw(features['in_1frame_rgb'],tf.float32)
  in_2frame_rgb= tf.decode_raw(features['in_2frame_rgb'],tf.float32)
 
  outs_2frame_xyz = tf.decode_raw(features['outs_2frame_xyz'],tf.float32)
  outs_1frame_id = tf.decode_raw(features['outs_1frame_id'],tf.float32)
  outs_2frame_id = tf.decode_raw(features['outs_2frame_id'],tf.float32)

  outs_2frame_score = tf.decode_raw(features['outs_2frame_score'],tf.float32)

  outs_2frame_r = tf.decode_raw(features['outs_2frame_r'],tf.float32)

  outs_trans_translation = tf.decode_raw(features['outs_trans_translation'],tf.float32)
  outs_trans_rot = tf.decode_raw(features['outs_trans_rot'],tf.float32)

  outs_1frame_pred_xyz = tf.decode_raw(features['outs_1frame_pred_xyz'],tf.float32)
  outs_cc = tf.decode_raw(features['outs_cc'],tf.float32)

  in_1frame_xyz = tf.reshape(in_1frame_xyz,[h,w,3])
  in_2frame_xyz = tf.reshape(in_2frame_xyz,[h,w,3])
  in_1frame_rgb = tf.reshape(in_1frame_rgb,[h,w,3])
  in_2frame_rgb = tf.reshape(in_2frame_rgb,[h,w,3])
 
  outs_2frame_xyz = tf.reshape(outs_2frame_xyz,[h,w,3])
  outs_1frame_id = tf.reshape(outs_1frame_id,[h,w]) 
  outs_2frame_id = tf.reshape(outs_2frame_id,[h,w]) 

  outs_2frame_r = tf.reshape(outs_2frame_r,[h,w,1])

  outs_2frame_score = tf.reshape(outs_2frame_score,[h,w])
 
  outs_trans_translation = tf.reshape(outs_trans_translation,[h,w,3]) 
  outs_trans_rot = tf.reshape(outs_trans_rot,[h,w,3]) 

  outs_1frame_pred_xyz = tf.reshape(outs_1frame_pred_xyz,[h,w,3])
  outs_cc = tf.reshape(outs_cc,[30,40,441])
 
  instance_id = tf.cast(features['instance_id'],tf.int32)

  return in_1frame_xyz, \
         in_1frame_rgb, \
         in_2frame_xyz, \
         in_2frame_rgb, \
         outs_1frame_id,\
         outs_2frame_id,\
         outs_2frame_xyz, \
         outs_2frame_r,\
         outs_2frame_score, \
         outs_trans_translation, \
         outs_trans_rot,\
         outs_1frame_pred_xyz, \
         outs_cc,\
         instance_id


def inputs(batch_size,num_epochs,tfrecords_filename):
  with tf.name_scope('inputf'):
    filename_queue = tf.train.string_input_producer(
      tfrecords_filename, num_epochs=num_epochs)

    in_1frame_xyz_, \
      in_1frame_rgb_, \
      in_2frame_xyz_, \
      in_2frame_rgb_, \
      gt_1frame_id_, \
      gt_2frame_id_, \
      gt_2frame_xyz_, \
      gt_2frame_r_, \
      gt_2frame_score_, \
      gt_translation_, \
      gt_rot_, \
      gt_1frame_pred_xyz_, \
      gt_cc_,\
      instance_id_ = read_and_decode(filename_queue)
      
    in_1frame_xyz,\
      in_1frame_rgb,\
      in_2frame_xyz, \
      in_2frame_rgb,\
      gt_1frame_id, \
      gt_2frame_id,\
      gt_2frame_xyz, \
      gt_2frame_r, \
      gt_2frame_score, \
      gt_translation, \
      gt_rot, \
      gt_1frame_pred_xyz, \
      gt_cc,\
      instance_id = tf.train.shuffle_batch(
        [in_1frame_xyz_,\
         in_1frame_rgb_,\
         in_2frame_xyz_,\
         in_2frame_rgb_,\
         gt_1frame_id_,\
         gt_2frame_id_,\
         gt_2frame_xyz_, \
         gt_2frame_r_, \
         gt_2frame_score_, \
         gt_translation_, \
         gt_rot_,\
         gt_1frame_pred_xyz_,\
         gt_cc_,\
         instance_id_],\
      batch_size=batch_size, num_threads=10,
      capacity = 10 + 3 * batch_size,
      min_after_dequeue=10)
   
    return in_1frame_xyz, \
           in_1frame_rgb, \
           in_2frame_xyz, \
           in_2frame_rgb,\
           gt_1frame_id, \
           gt_2frame_id,\
           gt_2frame_xyz,\
           gt_2frame_r, \
           gt_2frame_score,\
           gt_translation,\
           gt_rot,\
           gt_1frame_pred_xyz,\
           gt_cc,\
           instance_id 
