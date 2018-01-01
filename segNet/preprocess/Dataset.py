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
  def __init__(self, ids, top_sub=True, ins=None, outs=None, ins_dir=None, ins_extension=None, outs_dir=None, outs_extension=None, flag_rotation_aug=False,tfrecords_filename=None, secondmoment=True):
    self._num_examples = ids.shape[0]
    self.tfrecords_filename = tfrecords_filename
    self._ids = ids
    self._ins_dir = ins_dir
    self.generate_path()
    self._2m = secondmoment 

    self._tfrecords()

  def generate_path(self):
    self._ins_path = []
    self._outs_path = []
    self._outs_2m_path = []
    for id_line in xrange(self._num_examples):
      ins_sub_dir = os.path.join(self._ins_dir, self._ids[id_line])
      ins_ = [line for line in os.listdir(ins_sub_dir) if line.endswith('.pgm') and line.startswith('frame80')]
      out_ = [line for line in os.listdir(ins_sub_dir) if line.endswith('_labeling.npz') and line.startswith('frame80')]
      out_2m_ = [line for line in os.listdir(ins_sub_dir) if line.endswith('_2moment.npz') and line.startswith('frame80')]
      if len(ins_) > 0 and len(out_) > 0 and len(ins_) == len(out_):
        self._ins_path.append(os.path.join(ins_sub_dir,ins_[0]))
        self._outs_path.append(os.path.join(ins_sub_dir,out_[0]))
        self._outs_2m_path.append(os.path.join(ins_sub_dir,out_2m_[0]))
 
  def _tfrecords(self):
    total_path = os.path.join(DATA_DIR,'Tfrecords_SegNet',self.tfrecords_filename)
    writer =  tf.python_io.TFRecordWriter(total_path)

    for idx in xrange(len(self._ins_path)):
      ins_ = load_xyz(self._ins_path[idx]).astype(np.float32)
      print(self._ins_path[idx])
      print(ins_.shape)

      out_label = load_seg(self._outs_path[idx]).astype(np.float32)
      out_r = load_r(self._outs_path[idx],self._outs_2m_path[idx],self._2m).astype(np.float32)
      out_score = load_score(self._ins_path[idx],self._outs_path[idx]).astype(np.float32)
      out_2m = load_2m(self._outs_2m_path[idx]).astype(np.float32) * 100.0

      instance_id = idx

      if self._2m:
        out_feat = np.concatenate((out_label,out_2m),axis=2)
      else:
        out_feat = out_label

      print(out_feat.shape)
      print(out_r.shape)
      print(idx)

      if 0:
        plt.figure(0)
        plt.imshow(ins_[:,:,2])
        plt.figure(1)
        plt.imshow(out_feat[:,:,2])
        plt.figure(2) 
        plt.imshow(out_r[:,:,0])
        plt.figure(3)
        plt.imshow(out_score)
        plt.figure(4)
        plt.imshow(out_2m[:,:,0:3]*5000.0)
        plt.show()

      ins_raw = ins_.tostring()
      out_feat_raw = out_feat.tostring()
      out_r_raw = out_r.tostring()
      out_score_raw = out_score.tostring()
      out_2m_raw = out_2m.tostring()

      example = tf.train.Example(features=tf.train.Features(feature={
          'instance_id':_int64_feature(instance_id),
          'in_xyz_raw':_bytes_feature(ins_raw),
          'out_r_raw':_bytes_feature(out_r_raw),
          'out_score_raw':_bytes_feature(out_score_raw),
          'out_2m_raw':_bytes_feature(out_2m_raw),
          'out_feat_raw':_bytes_feature(out_feat_raw)}))

      writer.write(example.SerializeToString())
    print('num instances : %d' % (len(self._ins_path)))

    writer.close()

if __name__ == '__main__':
  from data_preparing import train_val_test_list 
  train_dataset = Dataset(train_val_test_list._train, ins_dir=os.path.join(DATA_DIR,'BlensorResult_2frame'), ins_extension='.pgm',flag_rotation_aug=True,tfrecords_filename='train.tfrecords',secondmoment=True) 
