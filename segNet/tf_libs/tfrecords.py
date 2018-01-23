from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import time
import sys


def read_and_decode(filename_queue):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
    serialized_example,
    # Defaults are not specified since both keys are required.
    features = {
      'in_xyz_raw':tf.FixedLenFeature([],tf.string),
       'out_r_raw':tf.FixedLenFeature([],tf.string),
       'out_feat_raw':tf.FixedLenFeature([],tf.string),
       'out_score_raw':tf.FixedLenFeature([],tf.string),
       'instance_id':tf.FixedLenFeature([],tf.int64),
    })

  in_xyz_raw = tf.decode_raw(features['in_xyz_raw'],tf.float32)
  out_feat_raw = tf.decode_raw(features['out_feat_raw'],tf.float32)
  out_r_raw = tf.decode_raw(features['out_r_raw'],tf.float32)
  out_score_raw = tf.decode_raw(features['out_score_raw'],tf.float32)

  out_feat_raw = tf.reshape(out_feat_raw,tf.stack([240,320,9]))
  out_r_raw = tf.reshape(out_r_raw,tf.stack([240,320,1]))
  in_raw = tf.reshape(in_xyz_raw,[240,320,3])
  out_score_raw = tf.reshape(out_score_raw,[240,320])
  instance_id_raw = tf.cast(features['instance_id'],tf.int32)

  return in_raw, out_feat_raw, out_r_raw, instance_id_raw, out_score_raw


def inputs(batch_size,num_epochs,tfrecords_filename):
  with tf.name_scope('inputf'):
    filename_queue = tf.train.string_input_producer(
      [tfrecords_filename], num_epochs=num_epochs)

    ins_, gt_feat_, gt_r_, instance_id_r_ , gt_score_ = read_and_decode(filename_queue)
    
    ins, gt_feat,  gt_r, instance_id_r, gt_score = tf.train.shuffle_batch(
      [ins_, gt_feat_, gt_r_, instance_id_r_, gt_score_], batch_size=batch_size, num_threads=10,
      capacity = 1000 + 3 * batch_size,
      min_after_dequeue=1000)
  
    return ins, gt_feat, gt_r, instance_id_r, gt_score
