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

#from matplotlib import pyplot as plt


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

class Dataset(object):
  def __init__(self, base, ids, top_sub=True, ins=None, outs=None, ins_dir=None, ins_extension=None, outs_dir=None, outs_extension=None, flag_rotation_aug=False,tfrecords_filename=None):
    self._num_examples = ids.shape[0]
    self.tfrecords_filename = tfrecords_filename
    self._ids = ids
    self._ins_dir = ins_dir
    self._ins_path = {}
    self._outs_path = {}
    self.num_instance =0

    self.generate_path()
    self.base = int(base) 
    self.base_step = int(500)

  def generate_path(self):
    self._ins_ids = []
    self._ins_path['1framexyz'] = []
    self._ins_path['2framexyz'] = []
    self._ins_path['1framergb'] = []
    self._ins_path['2framergb'] = []
    self._outs_path['2framexyz'] = []
    self._outs_path['top_dir'] = []
    self._outs_path['2framescore'] = []
    self._outs_path['boundary'] = []
    self._outs_path['flow'] = []
    self._outs_path['end_center'] = []
    self._outs_path['transl'] = []
    self._outs_path['rot'] = []

    ins_= {}
    outs_ = {}
    for id_line in xrange(self._num_examples):
      ins_sub_dir = os.path.join(self._ins_dir, self._ids[id_line])
     
      ins_['1framexyz'] = [line for line in os.listdir(ins_sub_dir) if line.endswith('.pgm') and line.startswith('frame20')]
      ins_['2framexyz'] = [line for line in os.listdir(ins_sub_dir) if line.endswith('.pgm') and line.startswith('frame80')]
      ins_['1framergb'] = [line for line in os.listdir(ins_sub_dir) if line.endswith('.png') and line.startswith('frame20')]
      ins_['2framergb'] = [line for line in os.listdir(ins_sub_dir) if line.endswith('.png') and line.startswith('frame80')]
      
      outs_['2framexyz'] = [line for line in os.listdir(ins_sub_dir) if line.endswith('labeling.npz') and line.startswith('frame80')]
      outs_['1frameid'] = [line for line in os.listdir(ins_sub_dir) if line.endswith('model_id.npz') and line.startswith('frame20')]
      outs_['2frameid'] = [line for line in os.listdir(ins_sub_dir) if line.endswith('model_id.npz') and line.startswith('frame80')]

      outs_['2frame_score'] = [line for line in os.listdir(ins_sub_dir) if line.endswith('frame80_score.npz')]
      outs_['boundary'] = [line for line in os.listdir(ins_sub_dir) if line.startswith('boundary.npz')]
      outs_['flow'] = [line for line in os.listdir(ins_sub_dir) if line.startswith('flow')]
      outs_['end_center'] = [line for line in os.listdir(ins_sub_dir) if line.startswith('end_center.npz')] 
      outs_['transl'] = [line for line in os.listdir(ins_sub_dir) if line.startswith('translation.npz')] 
      outs_['rot'] = [line for line in os.listdir(ins_sub_dir) if line.startswith('rotation.npz')] 
       
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
        self._outs_path['top_dir'].append(ins_sub_dir)
        self._outs_path['2framescore'].append(os.path.join(ins_sub_dir,outs_['2frame_score'][0]))
        self._outs_path['boundary'].append(os.path.join(ins_sub_dir,outs_['boundary'][0]))
        self._outs_path['flow'].append(os.path.join(ins_sub_dir,outs_['flow'][0]))
        self._outs_path['end_center'].append(os.path.join(ins_sub_dir,outs_['end_center'][0]))
        self._outs_path['transl'].append(os.path.join(ins_sub_dir,outs_['transl'][0]))
        self._outs_path['rot'].append(os.path.join(ins_sub_dir,outs_['rot'][0]))
 
        self._ins_ids.append(ins_sub_dir.split('/')[-1])

    self.num_instance = len(self._ins_path['1framexyz'])
    print("self.num_instances")
    print(self.num_instance)
 	

def tfrecords_single(db):
    total_path = os.path.join(DATA_DIR,'Tfrecords_train',str(db.base)+db.tfrecords_filename)
    
    writer =  tf.python_io.TFRecordWriter(total_path)
    print(total_path)  
    ins_ = {}
    outs_ = {}
   
    low = int(db.base * db.base_step)
    high = min((db.base+1) * db.base_step,db.num_instance)
    for idx in xrange(low,high):
      print(db._ins_path['1framexyz'][idx])
      ins_['1framexyz'] = load_xyz(db._ins_path['1framexyz'][idx]).astype(np.float32)
      ins_['2framexyz'] = load_xyz(db._ins_path['2framexyz'][idx]).astype(np.float32)
      ins_['1framergb'] = load_rgb(db._ins_path['1framergb'][idx]).astype(np.float32)
      ins_['2framergb'] = load_rgb(db._ins_path['2framergb'][idx]).astype(np.float32)
      outs_['2framexyz'] = load_seg(db._outs_path['2framexyz'][idx]).astype(np.float32)
      outs_['2framer'] = load_boundary(db._outs_path['boundary'][idx]).astype(np.float32)
      outs_['2framescore'] = load_score(db._outs_path['2framescore'][idx]).astype(np.float32) 
      outs_['flow'] = load_flow(db._outs_path['top_dir'][idx]).astype(np.float32)
      outs_['end_center'] = load_end_center(db._outs_path['end_center'][idx]).astype(np.float32)
      outs_['transl'] = load_transl(db._outs_path['transl'][idx]).astype(np.float32)
      outs_['rot'] = load_rot(db._outs_path['rot'][idx]).astype(np.float32)

      instance_id = int(db._ins_ids[idx])
      print('instance_id %d ' % (instance_id))
      print('%d / %d' % (idx,high))

       
      ins_1frame_rgb = ins_['1framergb'].tostring()
      ins_2frame_rgb = ins_['2framergb'].tostring()
      ins_1frame_xyz = ins_['1framexyz'].tostring()
      ins_2frame_xyz = ins_['2framexyz'].tostring()
      outs_2frame_xyz = outs_['2framexyz'].tostring()
      outs_2frame_r = outs_['2framer'].tostring()
      outs_2frame_score = outs_['2framescore'].tostring()
      outs_flow =  outs_['flow'].tostring()
      outs_end_center = outs_['end_center'].tostring()
      outs_transl = outs_['transl'].tostring()
      outs_rot = outs_['rot'].tostring()

      example = tf.train.Example(features=tf.train.Features(feature={
          'instance_id':_int64_feature(instance_id),
          'in_1frame_xyz':_bytes_feature(ins_1frame_xyz),
          'in_2frame_xyz':_bytes_feature(ins_2frame_xyz),
          'in_1frame_rgb':_bytes_feature(ins_1frame_rgb),
          'in_2frame_rgb':_bytes_feature(ins_2frame_rgb), 
          'outs_2frame_xyz':_bytes_feature(outs_2frame_xyz),
          'outs_2frame_r':_bytes_feature(outs_2frame_r),
          'outs_2frame_score':_bytes_feature(outs_2frame_score),
          'outs_end_center':_bytes_feature(outs_end_center),
          'outs_transl':_bytes_feature(outs_transl),
          'outs_rot':_bytes_feature(outs_rot),
          'outs_flow':_bytes_feature(outs_flow)
        }))

      writer.write(example.SerializeToString())

    writer.close()


if __name__ == '__main__':
  from data_preparing import train_val_test_list 
  def data_base(i):
    print("receving %d" % i)
    train_dataset = Dataset(i,train_val_test_list._train, ins_dir=os.path.join(DATA_DIR,'BlensorResult_val'), ins_extension='.pgm',flag_rotation_aug=True,tfrecords_filename='val.tfrecords')
    print("base %d" % train_dataset.base)
    return train_dataset
 
  tflist = [None] * 8

  print("starting") 
  pool = Pool(17)
  idlist = [i for i in range(8)]
  print(idlist) 
  for i, data in enumerate(pool.imap(data_base,idlist)):
    tflist[i] = data  
  pool.close()
  
  pool = Pool(17)
  
  for i, data in enumerate(pool.imap(tfrecords_single,tflist)):
    print(i)
  pool.close()
