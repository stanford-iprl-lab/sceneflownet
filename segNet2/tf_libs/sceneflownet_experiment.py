from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import sys
import time
import shutil
import datetime
import pytz

from tf_libs.tf_logging import LOG
from tf_libs.save_result import generate_result_folder, save_gt_segments, save_pred_segments
from inference.infer import infer_seg,nms
from evaluation.metric import m_AP50,m_AP75,m_AP90, m_AP
from mayavi import mlab as mayalab
#from matplotlib import pyplot as plt
from preprocess.Loader import angleaxis_rotmatrix 

from tf_libs.train_utils import get_var_list_to_restore,get_var_list_to_restore_by_name

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
    self.tfrecord_train_list = [os.path.join(self.flags.train_tfrecords_filename,line) for line in os.listdir(self.flags.train_tfrecords_filename)]
    print("tfrecord_train_list")
    print(self.tfrecord_train_list)
    self.tfrecord_val_list = [os.path.join(self.flags.val_tfrecords_filename, line) for line in os.listdir(self.flags.val_tfrecords_filename)]
    print("tfrecord_val_list")
    print(self.tfrecord_val_list)
    self.tfrecord_test_list = [os.path.join(self.flags.val_tfrecords_filename, line) for line in os.listdir(self.flags.val_tfrecords_filename)]
    print("tfrecord_test_list")
    #print(self.tfrecord_test_list)

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
      restore_var = False
      if restore_var:
        vars_to_restore = get_var_list_to_restore()
        checkpoint_path = os.path.join(self.flags.model_save_dir,'-'+str(restore_epoch))
        restorer = tf.train.Saver(vars_to_restore)
        restorer.restore(self.sess, checkpoint_path)      
      else:
        new_saver = tf.train.import_meta_graph(os.path.join(self.flags.model_save_dir,'-0.meta'))
        new_saver.restore(self.sess,os.path.join(self.flags.model_save_dir,'-'+str(restore_epoch)))
    else:
      self.clean_model_save_dir()
 
 
  def build_model(self, tfrecords_filename, num_epochs):
    dim = 3
    print("using tfrecord list")
    print(tfrecords_filename)
    self.global_step = tf.Variable(0, name='global_step', trainable=False)

    self.input['frame1_xyz'], \
    self.input['frame1_rgb'], \
    self.input['frame2_xyz'], \
    self.input['frame2_rgb'], \
    self.gt['frame2_xyz'], \
    self.gt['frame2_r'],\
    self.gt['frame2_score'], \
    self.gt['end_center'], \
    self.gt['flow'],\
    self.instance_id = self.inputf(batch_size = self.batch_size, num_epochs = num_epochs, tfrecords_filename = tfrecords_filename)
    print("self")
    print(self.gt['end_center']) 

    imagenet_mean = np.array([123.68, 116.779, 103.939])
    imagenet_mean = np.reshape(imagenet_mean,(1,1,1,3))
    self.input['frame1_rgb'] = self.input['frame1_rgb'] - imagenet_mean
    self.input['frame2_rgb'] = self.input['frame2_rgb'] - imagenet_mean
 

    self.pred['traj'], self.pred['quaternion'],self.pred['transl'],self.pred['flow'],self.pred['frame2_xyz'], self.pred['frame2_mask'], self.pred['frame2_score'], self.pred['frame2_r']  = self.model(self.input['frame1_xyz'],self.input['frame1_rgb'],self.input['frame2_xyz'],self.input['frame2_rgb'])

    
    self.gt['traj'] =  tf.concat([self.gt['frame2_xyz'],self.gt['end_center']],3)

    self.pred['frame2_mask_positive'] = tf.sigmoid(self.pred['frame2_mask'])[:,:,:,1]

    self.pred['frame2_mask_truncated'] = self.pred['frame2_mask_positive'] > 0.5
    self.pred['frame2_mask_truncated'] = tf.cast(self.pred['frame2_mask_truncated'],tf.float32)
    self.pred['frame2_mask_truncated_1'] = tf.reshape(self.pred['frame2_mask_truncated'],[-1,240,320,1])
    self.pred['frame2_mask_truncated_6'] = tf.tile(self.pred['frame2_mask_truncated_1'],[1,1,1,6])
    self.pred['frame2_mask_truncated_3'] = tf.tile(self.pred['frame2_mask_truncated_1'],[1,1,1,3])


    self.pred['frame2_xyz_masked'] = self.pred['frame2_mask_truncated_3'] * self.pred['frame2_xyz']
    self.pred['frame2_boundary_masked'] = self.pred['frame2_mask_truncated_1'] * self.pred['frame2_r']
    self.pred['frame2_score_masked'] = self.pred['frame2_mask_truncated_1'] * self.pred['frame2_score']
    self.pred['traj_masked'] = self.pred['frame2_mask_truncated_6'] * self.pred['traj']


    self.pred['frame2_score_positive'] = tf.sigmoid(self.pred['frame2_score'])[:,:,:,1]
    self.pred['frame2_score_positive_masked'] = self.pred['frame2_score_positive'] * self.pred['frame2_mask_truncated']

  
  def loss_op(self):
    #self.loss['violation'],\
    self.loss['boundary'],\
    self.loss['flow'],self.loss['elem'],self.loss['mask'],self.loss['score'] = self.lossf(\
      self.pred['frame2_mask'],\
      self.pred['traj'],\
      self.gt['traj'],\
      self.pred['frame2_score'],\
      self.gt['frame2_score'],\
      self.pred['frame2_r'],\
      self.gt['frame2_r'],\
      self.pred['flow'],\
      self.gt['flow'],\
      batch_size=self.batch_size)

    self.cost = self.loss['mask'] * 0.01+ self.loss['flow']  #+ self.loss['score'] + self.loss['flow'] * 100.0 #self.loss['boundary'] * 100.0 + self.loss['flow'] * 1000.0# + self.loss['violation'] * 0.1 + self.loss['variance'] * 0.1 

  def build_framework(self,restore_epoch,train_val_test):
    if restore_epoch >= 0:
      tf.reset_default_graph()

    if train_val_test == 'train':
      self.batch_size = self.flags.train_batch_size
      self.num_instance = self.flags.num_train_model
      self.build_model(tfrecords_filename=self.tfrecord_train_list,num_epochs=self.flags.num_epochs)
    elif train_val_test == 'val':
      self.batch_size = self.flags.val_batch_size
      self.num_instance = self.flags.num_val_model
      self.build_model(tfrecords_filename=self.tfrecord_val_list,num_epochs=1)
    else:
      self.batch_size = self.flags.test_batch_size
      self.num_instance = self.flags.num_test_model
      self.build_model(tfrecords_filename=self.tfrecord_val_list,num_epochs=1)

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
    self.loss_dict = {'flow':0.0,'total_loss':0.0,'mask':0.0,'score':0.0,'elem':0.0,'variance':0.0,'boundary':0.0,'violation':0.0}
    self.log.init_keys(self.loss_dict.keys())


  def loss_value_add(self,loss_dict):
    for key, value in loss_dict.iteritems():
       self.loss_dict[key] += value

 
  def loss_value_average(self):
    for key, value in self.loss_dict.iteritems():
      self.loss_dict[key] /= self.num_batch 


  def report_loss_value(self,train_val_test):
    string_tmp = '%s | epoch %d |' % (train_val_test,self.epoch)
    for key, value in self.loss_dict.iteritems():
      string_tmp += ' %s %.10f |' % (key,value)
      self.log.add_item(train_val_test,self.epoch,key,value)
    self.log.log_string(string_tmp)
    return string_tmp


  def train(self,restore_epoch=-1):
    self.build_framework(restore_epoch,train_val_test='train')

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=self.sess,coord=coord)
    start_time = time.time()
    self.loss_value_init()

    if restore_epoch >= 0:
      self.epoch = restore_epoch + 1
    else:
      self.epoch = 0
    try:
      current_time = datetime.datetime.now(pytz.utc).astimezone(pytz.timezone('US/Pacific-New'))
      print('At %s training starts from epoch %d !!! %d batch_size ' % (str(current_time),self.epoch,self.batch_size))
      step=0
      while not coord.should_stop():
        _,self.lossv['total_loss'],\
            self.gtv['traj'],\
            self.gtv['frame2_xyz'],\
            self.lossv['boundary'],\
            self.lossv['elem'],\
            self.lossv['mask'],\
            self.lossv['score'],\
            self.lossv['flow']= self.sess.run([\
            self.train_op,\
            self.cost,\
            self.gt['traj'],\
            self.gt['frame2_xyz'],\
            self.loss['boundary'],\
            self.loss['elem'],\
            self.loss['mask'],\
            self.loss['score'],\
            self.loss['flow']])
        #print(self.gtv['traj'][:,:,:,0:3].shape)
        #print(self.gtv['frame2_xyz'].shape)
        diff = np.abs(self.gtv['traj'][:,:,:,3:]-self.gtv['frame2_xyz'])
        diff = diff.reshape((-1,))
        #print(np.max(diff))
        self.loss_value_add({'flow':self.lossv['flow'],'boundary':self.lossv['boundary'],'total_loss':self.lossv['total_loss'],'mask':self.lossv['mask'], 'score':self.lossv['score'], 'elem':self.lossv['elem']})
        step += 1
        if step % self.num_batch == 0:
          self.loss_value_average()
          self.report_loss_value('train')
          self.saver.save(self.sess,self.flags.model_save_dir,global_step=self.epoch)
          end_time = time.time()
          print('Epoch %d last %f minutes' % (self.epoch, (end_time-start_time)/60.0))
          start_time = time.time()
          self.epoch += 1
          self.loss_value_init()
          current_time = datetime.datetime.now(pytz.utc).astimezone(pytz.timezone('US/Pacific-New')) 
          print('At %s training starts from epoch %d' % (str(current_time),self.epoch))
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
    current_time = datetime.datetime.now(pytz.utc).astimezone(pytz.timezone('US/Pacific-New'))
    print('At %s validate starts from epoch %d !!! %d batch_size ' % (str(current_time),self.epoch,self.batch_size))
    start_time = time.time()
 
    self.loss_value_init() 
    for ii in xrange(self.num_batch):
        self.lossv['total_loss'],\
            self.gtv['frame2_xyz'],\
            self.predv['frame2_mask_truncated'],\
            self.gtv['traj'],\
            self.lossv['boundary'],\
            self.lossv['elem'],\
            self.lossv['mask'],\
            self.lossv['score'],\
            self.lossv['flow']= self.sess.run([\
            self.cost,\
            self.gt['frame2_xyz'],\
            self.pred['frame2_mask_truncated'],\
            self.gt['traj'],\
            self.loss['boundary'],\
            self.loss['elem'],\
            self.loss['mask'],\
            self.loss['score'],\
            self.loss['flow']])
        #plt.figure(0)
        #print(self.predv['frame2_mask_truncated'][0].shape)
        #plt.imshow(self.predv['frame2_mask_truncated'][0])
        #plt.figure(1)
        #plt.imshow(self.gtv['frame2_xyz'][0][:,:,2])
        #plt.show()
        self.loss_value_add({'flow':self.lossv['flow'],'boundary':self.lossv['boundary'],'total_loss':self.lossv['total_loss'],'mask':self.lossv['mask'], 'score':self.lossv['score'], 'elem':self.lossv['elem']})
    print('num_batch %d' % (self.num_batch))
    self.loss_value_average()
    self.report_loss_value('val')
    end_time = time.time()
    print('Epoch %d last %f minutes' % (self.epoch, (end_time-start_time)/60.0))
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
       loss_value = self.sess.run([self.cost])
       #print('%f loss value' % loss_value_mask) 
       self.loss_value_add({'total_loss':loss_value})
       
        
    print('num_batch %d' % (self.num_batch))

    self.loss_value_average()
    self.report_loss_value('test')

    coord.request_stop()
    coord.join(threads)
    self.sess.close()

  def result_op(self,save=True):  
    self.gtv['frame2_xyz'],\
    self.predv['frame2_boundary_masked'], \
    self.predv['frame2_score_positive_masked'], \
    self.predv['traj_masked'], \
    self.instance_idv = self.sess.run([\
      self.gt['frame2_xyz'],\
      self.pred['frame2_boundary_masked'], \
      self.pred['frame2_score_positive_masked'], \
      self.pred['traj_masked'], \
      self.instance_id])
    print('dine')

    for i in range(self.batch_size):
      self.predv['instance_center'], self.predv['instance_boundary'], self.predv['instance_score'] = nms(self.predv['traj_masked'][i], self.predv['frame2_boundary_masked'][i], self.predv['frame2_score_positive_masked'][i])
      self.predv['final_seg'], self.predv['final_instance'], self.predv['final_score'] = infer_seg(self.predv['instance_center'],self.predv['instance_boundary'],self.predv['instance_score'],self.predv['traj_masked'][i])
 
        #plt.figure(2)
      save = True
      if save:
        tmp_path = os.path.join(self.result_save_epoch_top_dir,str(self.instance_idv[i]))
        print(tmp_path)
        np.savez(os.path.join(tmp_path,'gt'),seg=self.gtv['frame2_xyz'][i][:,:,2])
        np.savetxt(os.path.join(tmp_path,'pred.txt'),self.predv['final_score'],fmt='%.8f')
        final_seg = np.zeros((240,320,1))
        print("num of %d objecct" % (len(self.predv['final_instance'])))
        for j in range(len(self.predv['final_instance'])):
          final_seg += self.predv['final_instance'][j] * float(j+1)
        np.savez(os.path.join(tmp_path,'pred'),seg=final_seg)



  def save_result(self,restore_epoch):
    self.build_framework(restore_epoch=restore_epoch,train_val_test='test') 
 
    self.clean_result_save_dir([str(i) for i in xrange(self.num_instance)],restore_epoch)  
    self.epoch = restore_epoch
    
    self.align_variable_value()
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=self.sess,coord=coord)
 
    for ii in xrange(self.num_batch):
      self.result_op()
    print('num_batch %d' % (self.num_batch))

    coord.request_stop()
    coord.join(threads)
    self.sess.close()


  def whole_process(self):
    #self.train(6)
    #print("finishing traini)
    best_epoch = self.validate(6,30)#self.flags.num_epochs)
    #self.log.log_plotting(['transl','rot','total_loss','flow'])
    #self.test(best_epoch)
    best_epoch = 47#best_epoch #self.flags.num_epochs - 1
    self.save_result(best_epoch) 
    self.analysis(best_epoch)


  def analysis(self):
    self.result_save_epoch_top_dir = os.path.join(self.flags.result_save_dir,str(epoch))
    id_lists = [os.path.join(self.result_save_epoch_top_dir,line) for line in  os.listdir(self.result_save_epoch_top_dir)]
    m_AP50(id_lists)


  def clean_model_save_dir(self):
    if not os.path.exists(self.flags.model_save_dir):
      os.mkdir(self.flags.model_save_dir)
    else:
      shutil.rmtree(self.flags.model_save_dir)
      os.mkdir(self.flags.model_save_dir)

  
  def clean_result_save_dir(self,id_list,epoch):
    if not os.path.exists(self.flags.result_save_dir):
      os.mkdir(self.flags.result_save_dir)

    self.result_save_epoch_top_dir = os.path.join(self.flags.result_save_dir,str(epoch))
    if not os.path.exists(self.result_save_epoch_top_dir):
      os.mkdir(self.result_save_epoch_top_dir)
    else:
      shutil.rmtree(self.result_save_epoch_top_dir)
      os.mkdir(self.result_save_epoch_top_dir)

    for model_id in id_list:
       tmp_path = os.path.join(self.result_save_epoch_top_dir,str(model_id))
       if not os.path.exists(tmp_path):
         os.mkdir(tmp_path)
       else:
         shutil.rmtree(tmp_path)
         os.mkdir(tmp_path)
