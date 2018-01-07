from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import numpy
from multiprocessing import Pool
import shutil

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
    #print(self._ins_path)
    #print(self._outs_path)

    self._tfrecords()

  def generate_path(self):
    self._ins_path['1framexyz'] = []
    self._ins_path['2framexyz'] = []
    self._outs_path['1framexyz'] = []
    self._outs_path['2framexyz'] = []
    self._outs_path['1frameid'] = [] 
    self._outs_path['2frameid'] = []
    self._outs_path['transformation'] = []
    ins_= {}
    outs_ = {}
    for id_line in xrange(self._num_examples):
      ins_sub_dir = os.path.join(self._ins_dir, self._ids[id_line])
     
      ins_['1framexyz'] = [line for line in os.listdir(ins_sub_dir) if line.endswith('.pgm') and line.startswith('frame20')]
      ins_['2framexyz'] = [line for line in os.listdir(ins_sub_dir) if line.endswith('.pgm') and line.startswith('frame80')]
      outs_['1framexyz'] = [line for line in os.listdir(ins_sub_dir) if line.endswith('labeling.npz') and line.startswith('frame20')]
      outs_['2framexyz'] = [line for line in os.listdir(ins_sub_dir) if line.endswith('labeling.npz') and line.startswith('frame80')]
      outs_['1frameid'] = [line for line in os.listdir(ins_sub_dir) if line.endswith('model_id.npz') and line.startswith('frame20')]
      outs_['2frameid'] = [line for line in os.listdir(ins_sub_dir) if line.endswith('model_id.npz') and line.startswith('frame80')]
      
      num_list = []
      for key in ins_:
        num_list.append(len(ins_[key]))
      for key in outs_:
        num_list.append(len(outs_[key]))
      num_list = np.array(num_list)
      
      if np.all(num_list == num_list[0]) and num_list[0] == 1:
        self._ins_path['1framexyz'].append(os.path.join(ins_sub_dir,ins_['1framexyz'][0]))
        self._ins_path['2framexyz'].append(os.path.join(ins_sub_dir,ins_['2framexyz'][0]))
        self._outs_path['2framexyz'].append(os.path.join(ins_sub_dir,outs_['2framexyz'][0])) 
        self._outs_path['1framexyz'].append(os.path.join(ins_sub_dir,outs_['1framexyz'][0]))
        self._outs_path['1frameid'].append(os.path.join(ins_sub_dir,outs_['1frameid'][0]))
        self._outs_path['2frameid'].append(os.path.join(ins_sub_dir,outs_['2frameid'][0]))
        self._outs_path['transformation'].append(ins_sub_dir)

    #print('path ')
    #print(self._ins_path['1framexyz'])
    self.num_instance = len(self._ins_path['1framexyz'])
    #print('num instance %d' % self.num_instance)

  def _tfrecords(self):
    total_path = os.path.join(DATA_DIR,'Tfrecords_SegNet2',self.tfrecords_filename)
    writer =  tf.python_io.TFRecordWriter(total_path)
  
    ins_ = {}
    outs_ = {}
    
    for idx in xrange(self.num_instance):
      ins_['1framexyz'] = load_xyz(self._ins_path['1framexyz'][idx]).astype(np.float32)
      ins_['2framexyz'] = load_xyz(self._ins_path['2framexyz'][idx]).astype(np.float32)

      outs_['1framexyz'] = load_seg(self._outs_path['1framexyz'][idx]).astype(np.float32)
      outs_['2framexyz'] = load_seg(self._outs_path['2framexyz'][idx]).astype(np.float32)
      outs_['2framer'] = load_r(self._outs_path['2framexyz'][idx]).astype(np.float32)

      outs_['trans_translation'], outs_['trans_rot'] = load_transformation(self._outs_path['transformation'][idx])

      instance_id = idx
      print(idx)

      if False:
        plt.figure(0)
        plt.imshow(ins_[:,:,2])
        plt.figure(1)
        plt.imshow(out_label[:,:,2])
        plt.figure(2) 
        plt.imshow(out_r[:,:,0])
        plt.show()

      ins_1frame_xyz = ins_['1framexyz'].tostring()
      ins_2frame_xyz = ins_['2framexyz'].tostring()
      outs_1frame_xyz = outs_['1framexyz'].tostring()
      outs_2frame_xyz = outs_['2framexyz'].tostring()
      outs_2frame_r = outs_['2framer'].tostring()
      outs_trans_translation = outs_['trans_translation'].tostring()
      outs_trans_rot = outs_['trans_rot'].tostring()

      example = tf.train.Example(features=tf.train.Features(feature={
          'instance_id':_int64_feature(instance_id),
          'in_1frame_xyz':_bytes_feature(ins_1frame_xyz),
          'in_2frame_xyz':_bytes_feature(ins_2frame_xyz),
          'outs_1frame_xyz':_bytes_feature(outs_1frame_xyz),
          'outs_2frame_xyz':_bytes_feature(outs_2frame_xyz),
          'outs_2frame_r':_bytes_feature(outs_2frame_r),
          'outs_trans_translation':_bytes_feature(outs_trans_translation),
          'outs_trans_rot':_bytes_feature(outs_trans_rot)}
          ))

      writer.write(example.SerializeToString())

    writer.close()

if __name__ == '__main__':
  from data_preparing import train_val_test_list 
  #train_dataset = Dataset(train_val_test_list._val, ins_dir=os.path.join(DATA_DIR,'BlensorResult_2frame'), ins_extension='.pgm',flag_rotation_aug=True,tfrecords_filename='val.tfrecords') 
  train_dataset = Dataset(train_val_test_list._train, ins_dir=os.path.join(DATA_DIR,'BlensorResult_2frame'), ins_extension='.pgm',flag_rotation_aug=True,tfrecords_filename='train.tfrecords') 
