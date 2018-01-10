from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import numpy
from multiprocessing import Pool
import shutil
import cv2
import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
from Loader import *
from scipy.ndimage import rotate

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR,os.pardir))
from local_variables import *
import tensorflow as tf

from matplotlib import pyplot as plt


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

class Dataset(object):
  def __init__(self, ids, top_sub=True, ins=None, outs=None, ins_dir=None, ins_extension=None, outs_dir=None, outs_extension=None, flag_rotation_aug=False,tfrecords_filename=None):
    self._num_examples = ids.shape[0]
    self.tfrecords_filename = tfrecords_filename
    self._ids = ids
    self._ins_dir = ins_dir
    self._ins_path = {}
    self._outs_path = {}
    self.num_instance =0

    self.generate_path()
    self._tfrecords()

  def generate_path(self):
    self._ins_path['1framexyz'] = []
    self._ins_path['2framexyz'] = []
    self._ins_path['1framergb'] = []
    self._ins_path['2framergb'] = []
    self._outs_path['1framexyz'] = []
    self._outs_path['2framexyz'] = []
    self._outs_path['1frameid'] = [] 
    self._outs_path['2frameid'] = []
    self._outs_path['transformation'] = []
    self._outs_path['cc'] = []

    ins_= {}
    outs_ = {}
    for id_line in xrange(self._num_examples):
      ins_sub_dir = os.path.join(self._ins_dir, self._ids[id_line])
     
      ins_['1framexyz'] = [line for line in os.listdir(ins_sub_dir) if line.endswith('.pgm') and line.startswith('frame20')]
      ins_['2framexyz'] = [line for line in os.listdir(ins_sub_dir) if line.endswith('.pgm') and line.startswith('frame80')]
      ins_['1framergb'] = [line for line in os.listdir(ins_sub_dir) if line.endswith('.png') and line.startswith('frame20')]
      ins_['2framergb'] = [line for line in os.listdir(ins_sub_dir) if line.endswith('.png') and line.startswith('frame80')]
      
      outs_['1framexyz'] = [line for line in os.listdir(ins_sub_dir) if line.endswith('labeling.npz') and line.startswith('frame20')]
      outs_['2framexyz'] = [line for line in os.listdir(ins_sub_dir) if line.endswith('labeling.npz') and line.startswith('frame80')]
      outs_['1frameid'] = [line for line in os.listdir(ins_sub_dir) if line.endswith('model_id.npz') and line.startswith('frame20')]
      outs_['2frameid'] = [line for line in os.listdir(ins_sub_dir) if line.endswith('model_id.npz') and line.startswith('frame80')]
      outs_['cc'] = [line for line in os.listdir(ins_sub_dir) if line.endswith('cc.npz')]

      num_list = []
      for key in ins_:
        num_list.append(len(ins_[key]))
      for key in outs_:
        num_list.append(len(outs_[key]))
      num_list = np.array(num_list)
      
      if np.all(num_list == num_list[0]) and num_list[0] == 1:
        self._ins_path['1framexyz'].append(os.path.join(ins_sub_dir,ins_['1framexyz'][0]))
        self._ins_path['2framexyz'].append(os.path.join(ins_sub_dir,ins_['2framexyz'][0]))
        self._ins_path['1framergb'].append(os.path.join(ins_sub_dir,ins_['1framergb'][0]))
        self._ins_path['2framergb'].append(os.path.join(ins_sub_dir,ins_['2framergb'][0]))
        self._outs_path['2framexyz'].append(os.path.join(ins_sub_dir,outs_['2framexyz'][0])) 
        self._outs_path['1framexyz'].append(os.path.join(ins_sub_dir,outs_['1framexyz'][0]))
        self._outs_path['1frameid'].append(os.path.join(ins_sub_dir,outs_['1frameid'][0]))
        self._outs_path['2frameid'].append(os.path.join(ins_sub_dir,outs_['2frameid'][0]))
        self._outs_path['transformation'].append(ins_sub_dir)
        self._outs_path['cc'].append(os.path.join(ins_sub_dir,outs_['cc'][0]))

    self.num_instance = len(self._ins_path['1framexyz'])

  def _tfrecords(self):
    total_path = os.path.join(DATA_DIR,'Tfrecords_SegNet2',self.tfrecords_filename)
    writer =  tf.python_io.TFRecordWriter(total_path)
  
    ins_ = {}
    outs_ = {}
    
    for idx in xrange(self.num_instance):
      ins_['1framexyz'] = load_xyz(self._ins_path['1framexyz'][idx]).astype(np.float32)
      ins_['2framexyz'] = load_xyz(self._ins_path['2framexyz'][idx]).astype(np.float32)
      ins_['1framergb'] = load_rgb(self._ins_path['1framergb'][idx]).astype(np.float32)
      ins_['2framergb'] = load_rgb(self._ins_path['2framergb'][idx]).astype(np.float32)
      outs_['1framexyz'] = load_seg(self._outs_path['1framexyz'][idx]).astype(np.float32)
      outs_['2framexyz'] = load_seg(self._outs_path['2framexyz'][idx]).astype(np.float32)
      outs_['1framer'] = load_r(self._outs_path['1framexyz'][idx]).astype(np.float32)
      outs_['1framescore'] = load_score(self._ins_path['1framexyz'][idx],self._outs_path['1framexyz'][idx]).astype(np.float32)
      outs_['2framer'] = load_r(self._outs_path['2framexyz'][idx]).astype(np.float32)
      outs_['2framescore'] = load_score(self._ins_path['2framexyz'][idx],self._outs_path['2framexyz'][idx]).astype(np.float32)
 
      outs_['1frameid'] = load_labeling(self._outs_path['1frameid'][idx]).astype(np.float32)
      outs_['2frameid'] = load_labeling(self._outs_path['2frameid'][idx]).astype(np.float32)

      outs_['trans_translation'], outs_['trans_rot'] = load_transformation(self._outs_path['transformation'][idx])
      outs_['trans_translation'] =  outs_['trans_translation'].astype(np.float32)
      outs_['trans_rot'] =  outs_['trans_rot'].astype(np.float32)

      outs_['pred_1frame_xyz'] = load_predicted_frame1_feat(ins_['2framexyz'], outs_['2framexyz'], (outs_['trans_translation'], outs_['trans_rot']), outs_['1frameid'], outs_['2frameid']).astype(np.float32)
      instance_id = idx
      outs_['cc'] = load_cc(self._outs_path['cc'][idx]).astype(np.float32)
      print(outs_['cc'].shape)
      print(idx)

      if 0:
        plt.figure(0)
        plt.imshow(ins_['2framexyz'][:,:,2])
        plt.figure(1)
        plt.imshow(ins_['2framergb'])
        plt.figure(2) 
        plt.imshow(ins_['1framexyz'][:,:,2])
        plt.figure(3)
        plt.imshow(ins_['1framergb'])
        plt.figure(4)
        plt.imshow(outs_['pred_1frame_xyz'][:,:,2])
  #      plt.imshow(outs_['score'])
  #      plt.figure(5)
  #      plt.imshow(outs_['2framer'][:,:,0])
        #plt.figure(5)
        #plt.imshow(outs_['trans_rot']) 
        plt.show()
        
      if 0:
        img = cv2.imread(self._ins_path['1framergb'][idx])
        cv2.imshow('image',ins_['1framergb'])
        cv2.waitKey()
        
      ins_1frame_rgb = ins_['1framergb'].tostring()
      ins_2frame_rgb = ins_['2framergb'].tostring()
      ins_1frame_xyz = ins_['1framexyz'].tostring()
      ins_2frame_xyz = ins_['2framexyz'].tostring()
      outs_1frame_xyz = outs_['1framexyz'].tostring()
      outs_2frame_xyz = outs_['2framexyz'].tostring()
      outs_1frame_r = outs_['2framer'].tostring()
      outs_2frame_r = outs_['2framer'].tostring()
      
      outs_1frame_score = outs_['1framescore'].tostring()
      outs_2frame_score = outs_['2framescore'].tostring()
  
      outs_trans_translation = outs_['trans_translation'].tostring()
      outs_trans_rot = outs_['trans_rot'].tostring()
   
      outs_1frame_id = outs_['1frameid'].tostring()
      outs_2frame_id = outs_['2frameid'].tostring()

      outs_1frame_pred_xyz =  outs_['pred_1frame_xyz'].tostring()
      outs_cc = outs_['cc'].tostring()


      example = tf.train.Example(features=tf.train.Features(feature={
          'instance_id':_int64_feature(instance_id),
          'in_1frame_xyz':_bytes_feature(ins_1frame_xyz),
          'in_2frame_xyz':_bytes_feature(ins_2frame_xyz),
          'in_1frame_rgb':_bytes_feature(ins_1frame_rgb),
          'in_2frame_rgb':_bytes_feature(ins_2frame_rgb), 
          'outs_1frame_xyz':_bytes_feature(outs_1frame_xyz),
          'outs_2frame_xyz':_bytes_feature(outs_2frame_xyz),
          'outs_1frame_id':_bytes_feature(outs_1frame_id),
          'outs_2frame_id':_bytes_feature(outs_2frame_id), 
          'outs_1frame_r':_bytes_feature(outs_1frame_r), 
          'outs_2frame_r':_bytes_feature(outs_2frame_r),
          'outs_1frame_score':_bytes_feature(outs_1frame_score),
          'outs_2frame_score':_bytes_feature(outs_2frame_score),
          'outs_1frame_pred_xyz':_bytes_feature(outs_1frame_pred_xyz),
          'outs_trans_translation':_bytes_feature(outs_trans_translation),
          'outs_trans_rot':_bytes_feature(outs_trans_rot),
          'outs_cc':_bytes_feature(outs_cc),
        }))

      writer.write(example.SerializeToString())

    writer.close()

if __name__ == '__main__':
  from data_preparing import train_val_test_list 
  #train_dataset = Dataset(train_val_test_list._val, ins_dir=os.path.join(DATA_DIR,'BlensorResult_2frame'), ins_extension='.pgm',flag_rotation_aug=True,tfrecords_filename='val.tfrecords') 
  train_dataset = Dataset(train_val_test_list._train, ins_dir=os.path.join(DATA_DIR,'BlensorResult_2frame'), ins_extension='.pgm',flag_rotation_aug=True,tfrecords_filename='train.tfrecords') 
