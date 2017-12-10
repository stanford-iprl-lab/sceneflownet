from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import sys
import time
import shutil

from tf_libs.tf_logging import LOG
from tf_libs.save_result import generate_result_folder, save_gt_segments, save_pred_segments

class Experiment:
  def __init__(self,flags_para,inputf,model,lossf,log):
    self.flags = flags_para
    self.inputf = inputf
    self.model = model
    self.lossf = lossf
    self.sess = None
    self.log = log
    self.saver = None
    self.loss= {}
    self.input = {}
    self.gt = {}
    self.pred = {}
    self.epoch = 0 
    self.batch_size = 1
    self.num_instance = 0
    self.num_batch = 0 
    self.lossv = {}
    self.predv = {}
    self.inputv = {}
    self.gtv = {}
 
  def build_sess(self, restore_epoch):
    self.saver = tf.train.Saver(max_to_keep=self.flags.max_model_to_keep) 
    init_op = tf.group(tf.global_variables_initializer(),
                            tf.local_variables_initializer())
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    self.sess = tf.Session(config=config)
    self.sess.run(init_op)

    if restore_epoch >= 0:
      print('restore epoch %d' % (restore_epoch))
      new_saver = tf.train.import_meta_graph(os.path.join(self.flags.model_save_dir,'-'+str(restore_epoch)+'.meta'))
      new_saver.restore(self.sess,os.path.join(self.flags.model_save_dir,'-'+str(restore_epoch)))
    else:
      self.clean_model_save_dir()

      
  def build_model(self, tfrecords_filename, num_epochs):
    dim = 3
    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    self.input['xyz'], self.gt['xyz'], self.gt['r'], self.instance_id = self.inputf(batch_size = self.batch_size, num_epochs = num_epochs, tfrecords_filename = tfrecords_filename)
    self.pred['mask'], self.pred['r'], self.pred['xyz'], self.pred['score'] = self.model(self.input['xyz'])
    pred_seg = tf.sigmoid(self.pred['mask'])[:,:,:,1]

    pred_seg_flag = pred_seg > 0.5
    pred_seg_flag_float = tf.cast(pred_seg_flag,tf.float32)
    pred_seg_flag_1 = tf.reshape(pred_seg_flag_float,[-1,120,160,1])
    pred_seg_flag_dim = tf.tile(pred_seg_flag_1,[1,1,1,dim])

    pred_c_infer = pred_seg_flag_dim * self.pred['xyz']
    pred_r_infer = pred_seg_flag_1 * self.pred['r']
    pred_score_infer = pred_seg_flag_1 * self.pred['score']


  def loss_op(self):
    self.loss['mask'], self.loss['score'], self.loss['elem'], self.loss['variance'], self.loss['boundary'], self.loss['violation'] = self.lossf(self.pred['xyz'], self.pred['r'], self.pred['mask'], self.pred['score'], self.gt['xyz'], self.gt['r'], global_step=self.global_step,batch_size=self.batch_size) 

    self.cost = self.loss['mask'] #+ loss_elem * 10 + loss_variance * 20 + loss_boundary * 100 #+ loss_violation * 20


  def build_framework(self,restore_epoch,train_val_test):
    if restore_epoch >= 0:
      tf.reset_default_graph()

    if train_val_test == 'train':
      self.batch_size = self.flags.train_batch_size
      self.num_instance = self.flags.num_train_model
      self.build_model(tfrecords_filename=self.flags.train_tfrecords_filename,num_epochs=self.flags.num_epochs)
    elif train_val_test == 'val':
      self.batch_size = self.flags.val_batch_size
      self.num_instance = self.flags.num_val_model
      self.build_model(tfrecords_filename=self.flags.val_tfrecords_filename,num_epochs=1)
    else:
      self.batch_size = self.flags.test_batch_size
      self.num_instance = self.flags.num_test_model
      self.build_model(tfrecords_filename=self.flags.val_tfrecords_filename,num_epochs=1)

    self.num_batch = int(self.num_instance / self.batch_size)
    self.loss_op()

    if train_val_test == 'train':
      self.train_op = tf.train.AdamOptimizer(learning_rate=self.flags.learning_rate).minimize(self.cost,global_step=self.global_step)

    self.build_sess(restore_epoch=restore_epoch)

  def align_variable_value(self):
    for key in self.input.keys():
      self.inputv[key] = None
    for key in self.gt.keys():
      self.gtv[key] = None
    for key in self.pred.keys():
      self.predv[key] = None
    for key in self.loss.keys():
      self.lossv[key] = None

  def loss_value_init(self):
    self.loss_dict = {'total_loss':0.0,'mask':0.0,'score':0.0,'elem':0.0,'variance':0.0,'boundary':0.0,'violation':0.0}


  def loss_value_add(self,loss_dict):
    for key, value in loss_dict.iteritems():
       self.loss_dict[key] += value

 
  def loss_value_average(self):
    for key, value in self.loss_dict.iteritems():
      self.loss_dict[key] /= self.num_batch 


  def report_loss_value(self):
    string_tmp = 'epoch %d |' % self.epoch
    for key, value in self.loss_dict.iteritems():
      string_tmp += ' %s %f |' % (key,value)
    return string_tmp  


  def train(self,restore_epoch=-1):
    self.build_framework(restore_epoch,train_val_test='train')

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=self.sess,coord=coord)

    self.loss_value_init()
    print('%d batch_size ' % (self.batch_size))

    if restore_epoch >= 0:
      self.epoch = restore_epoch + 1
    else:
      self.epoch = 0
    try:
      step=0
      while not coord.should_stop():
        _ ,loss_value, loss_value_mask, loss_value_elem, loss_value_boundary, loss_value_variance, loss_value_violation = self.sess.run([self.train_op, self.cost, self.loss['mask'], self.loss['elem'], self.loss['boundary'], self.loss['variance'], self.loss['violation']])
        self.loss_value_add({'total_loss':loss_value,'mask':loss_value_mask})
        step += 1
        if step % self.num_batch == 0:
          self.loss_value_average()
          self.log.log_string(self.report_loss_value())
          self.saver.save(self.sess,self.flags.model_save_dir,global_step=self.epoch)
          self.epoch += 1
          self.loss_value_init()
        if self.epoch >= self.flags.num_epochs:
          break     
    except tf.errors.OutOfRangeError:
      print('Training is Done')
    finally:
      coord.request_stop()

    coord.join(threads)
    self.sess.close()


  def validate_epoch(self,restore_epoch=0):
    self.build_framework(restore_epoch=restore_epoch,train_val_test='val') 
     
    self.epoch = restore_epoch
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=self.sess,coord=coord)
 
    print('num of batch in validate %d' % (self.num_batch))
    self.loss_value_init() 
    for ii in xrange(self.num_batch):
       loss_value,loss_value_mask = self.sess.run([self.cost,self.loss['mask']])
       #print('%f loss value' % loss_value_mask)
       self.loss_value_add({'total_loss':loss_value,'mask':loss_value_mask})
 
    print('num_batch %d' % (self.num_batch))
    self.loss_value_average()
    print(self.report_loss_value())

    coord.request_stop()
    coord.join(threads)
    self.sess.close()
    return self.loss_dict['total_loss']


  def validate(self,start_epoch=0,end_epoch=1):
    val_loss_list = []
    for epoch_i in xrange(start_epoch,end_epoch):
      val_loss_list.append( ( epoch_i, self.validate_epoch(epoch_i) ) )
    from operator import itemgetter
    best_epoch = min(val_loss_list,key=itemgetter(1))[0] 
    print("best epoch %d" % (best_epoch)) 
    return best_epoch 


  def test(self,restore_epoch):
    self.build_framework(restore_epoch=restore_epoch,train_val_test='test') 

    self.epoch = restore_epoch

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=self.sess,coord=coord)

    self.loss_value_init() 
    print('num %d batch in test ' % (self.num_batch)) 
    for ii in xrange(self.num_batch):
       loss_value,loss_value_mask = self.sess.run([self.cost,self.loss['mask']])
       #print('%f loss value' % loss_value_mask) 
       self.loss_value_add({'total_loss':loss_value,'mask':loss_value_mask})
       
        
    print('num_batch %d' % (self.num_batch))

    self.loss_value_average()
    print(self.report_loss_value())

    coord.request_stop()
    coord.join(threads)
    self.sess.close()

  def save_instance_result(self,instance_id):
    for i in xrange(self.batch_size):
      tmp_path = os.path.join(self.flags.result_save_dir,str(instance_id[i]))
      np.savez(os.path.join(tmp_path,'pred'),pred=self.predv['xyz'][i])

  def save_result(self,restore_epoch):
    self.build_framework(restore_epoch=restore_epoch,train_val_test='test') 
 
    self.clean_result_save_dir([str(i) for i in xrange(self.num_instance)])
    self.epoch = restore_epoch
    
    self.align_variable_value()
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=self.sess,coord=coord)
 
    for ii in xrange(self.num_batch):
      self.predv['xyz'], self.gtv['xyz'], self.inputv['xyz'], instance_id = self.sess.run([self.pred['xyz'],self.gt['xyz'],self.input['xyz'],self.instance_id])
      self.save_instance_result(instance_id) 

    print('num_batch %d' % (self.num_batch))

    coord.request_stop()
    coord.join(threads)
    self.sess.close()

  def whole_process(self):
    self.train()
    best_epoch = self.validate(0,self.flags.num_epochs)
    self.test(best_epoch)
    self.save_result(best_epoch) 
    self.analysis()


  def analysis(self):
    pass


  def clean_model_save_dir(self):
    if not os.path.exists(self.flags.model_save_dir):
      os.mkdir(self.flags.model_save_dir)
    else:
      shutil.rmtree(self.flags.model_save_dir)
      os.mkdir(self.flags.model_save_dir)

  
  def clean_result_save_dir(self,id_list):
    if not os.path.exists(self.flags.result_save_dir):
      os.mkdir(self.flags.result_save_dir)
    else:
      shutil.rmtree(self.flags.result_save_dir)
      os.mkdir(self.flags.result_save_dir)

    for model_id in id_list:
       tmp_path = os.path.join(self.flags.result_save_dir,model_id)
       if not os.path.exists(tmp_path):
         os.mkdir(tmp_path)
       else:
         shutil.rmtree(tmp_path)
         os.mkdir(tmp_path)
 
