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
    print(self.tfrecord_train_list) 

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
      vars_to_restore = get_var_list_to_restore()
      for var in vars_to_restore:
        print("restoring %s" % var.name) 
      checkpoint_path = '/home/linshaonju/interactive-segmentation/Data/pretrained_models/resnet_v1_50.ckpt'
      restorer = tf.train.Saver(vars_to_restore)
      restorer.restore(self.sess, checkpoint_path)     
 
      #vars_to_restore_1 = get_var_list_to_restore_by_name('encode,decode')
      #checkpoint_path_1 = '/home/linshaonju/interactive-segmentation/segNet2/saved_models/ub/-20'
      #restorer = tf.train.Saver(vars_to_restore_1)
      #restorer.restore(self.sess, checkpoint_path_1) 
      #for var in vars_to_restore_1:
      #  print("restoring %s" % var.name)
      for var in tf.trainable_variables():
        print(var.name)
 
  def build_model(self, tfrecords_filename, num_epochs):
    dim = 3

    self.global_step = tf.Variable(0, name='global_step', trainable=False)

    self.input['frame1_xyz'], \
    self.input['frame1_rgb'], \
    self.input['frame2_xyz'], \
    self.input['frame2_rgb'], \
    self.gt['frame1_id'], \
    self.gt['frame2_id'], \
    self.gt['frame2_xyz'], \
    self.gt['frame2_r'],\
    self.gt['frame2_score'], \
    self.gt['transl'], \
    self.gt['rot'], \
    self.gt['frame1_pred_xyz'], \
    self.gt['cc'],\
    self.instance_id = self.inputf(batch_size = self.batch_size, num_epochs = num_epochs, tfrecords_filename = self.tfrecord_train_list)
    print("self.tfrecord_train_list")
    print(self.tfrecord_train_list)
    imagenet_mean = np.array([123.68, 116.779, 103.939])
    imagenet_mean = np.reshape(imagenet_mean,(1,1,1,3))
    self.input['frame1_rgb'] = self.input['frame1_rgb'] - imagenet_mean
    self.input['frame2_rgb'] = self.input['frame2_rgb'] - imagenet_mean

    self.pred['transl'], \
     self.pred['rot'], self.pred['frame2_xyz'], self.pred['frame2_mask'], self.pred['frame2_score'], self.pred['frame2_r']  = self.model(self.input['frame1_xyz'],self.input['frame1_rgb'],self.input['frame2_xyz'],self.input['frame2_rgb'])

    self.pred['objfeat'] = tf.concat([self.pred['frame2_xyz'],self.pred['transl']+self.pred['frame2_xyz'],self.pred['rot']+self.pred['frame2_xyz']],3)
    self.gt['objfeat'] = tf.concat([self.gt['frame2_xyz'],self.gt['transl']+self.gt['frame2_xyz'],self.gt['rot']+self.gt['frame2_xyz']],3)

    self.gt['mask_cc'] = tf.nn.max_pool(self.gt['frame2_xyz'][:,:,:,2:],ksize=[1,8,8,1],strides=[1,8,8,1],padding='VALID')
    self.gt['mask_cc'] = tf.cast(self.gt['mask_cc'] > 0.0,tf.float32)
    
    #self.pred['frame2_mask_positive'] = tf.sigmoid(self.pred['frame2_mask'])[:,:,:,1]

    #self.pred['frame2_mask_truncated'] = self.pred['frame2_mask_positive'] > 0.5
    #self.pred['frame2_mask_truncated'] = tf.cast(self.pred['frame2_mask_truncated'],tf.float32)
    #self.pred['frame2_mask_truncated_1'] = tf.reshape(self.pred['frame2_mask_truncated'],[-1,240,320,1])
    #self.pred['frame2_mask_truncated_dim'] = tf.tile(self.pred['frame2_mask_truncated_1'],[1,1,1,dim])

    #self.gt['frame2_mask'] = self.gt['frame2_xyz'][:,:,:,2] > 0.0
    #self.gt['frame2_mask'] = tf.cast(self.gt['frame2_mask'],tf.float32)
    #self.gt['frame2_mask_truncated_1'] = tf.reshape(self.gt['frame2_mask'],[-1,240,320,1])
    #self.gt['frame2_mask_truncated_dim'] = tf.tile(self.gt['frame2_mask_truncated_1'],[1,1,1,dim])

    #self.pred['frame2_xyz_masked'] = self.pred['frame2_mask_truncated_dim'] * self.pred['frame2_xyz']
    #self.pred['frame2_boundary_masked'] = self.pred['frame2_mask_truncated_1'] * self.pred['frame2_r']
    #self.pred['frame2_score_masked'] = self.pred['frame2_mask_truncated_1'] * self.pred['frame2_score']
    #self.pred['transl_masked'] = self.gt['frame2_mask_truncated_dim'] * self.pred['transl']
    #self.pred['rot_masked'] = self.gt['frame2_mask_truncated_dim'] * self.pred['rot']


    #self.gt['frame2_mask'] = self.gt['frame2_xyz'][:,:,:,2] > 0.0
    #self.gt['frame2_mask'] = tf.cast(self.gt['frame2_mask'],tf.float32)

    #self.pred['frame2_score_positive'] = tf.sigmoid(self.pred['frame2_score'])[:,:,:,1]
    #self.pred['frame2_score_positive_masked'] = self.pred['frame2_score_positive'] * self.pred['frame2_mask_truncated']

  
  def loss_op(self):
    self.loss['variance'],\
    self.loss['violation'],\
    self.loss['boundary'],\
    self.loss['flow'], \
    self.loss['rot'],\
    self.loss['elem'],\
    self.loss['mask'],\
    self.loss['score'],\
    self.loss['rot'], \
    self.loss['transl'] = self.lossf(
      self.pred['objfeat'],\
      self.gt['objfeat'],\
      self.pred['frame2_r'],\
      self.gt['frame2_r'],\
      self.pred['frame2_xyz'],\
      self.pred['frame2_mask'],\
      self.gt['frame2_score'],\
      self.pred['frame2_score'],\
      self.input['frame2_xyz'],\
      self.gt['frame1_pred_xyz'],\
      self.pred['transl'], \
      self.pred['rot'], \
      self.gt['rot'], \
      self.gt['transl'], self.gt['frame2_xyz'], batch_size=self.batch_size)

    self.cost =  self.loss['rot'] + self.loss['transl'] + self.loss['mask'] + self.loss['score'] + self.loss['elem'] * 100.0 + self.loss['boundary'] * 100.0 + self.loss['flow'] * 100.0 + self.loss['violation'] * 0.1 + self.loss['variance'] * 0.1

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
    self.loss_dict = {'flow':0.0, 'rot':0.0, 'transl':0.0, 'total_loss':0.0,'mask':0.0,'score':0.0,'elem':0.0,'variance':0.0,'boundary':0.0,'violation':0.0}
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
          self.lossv['variance'],\
          self.lossv['violation'],\
          self.lossv['boundary'],\
          self.lossv['flow'],\
          self.lossv['rot'],\
          self.lossv['transl'],\
          self.lossv['mask'],\
          self.lossv['score'],\
          self.lossv['elem']= self.sess.run([\
            self.train_op,\
            self.cost,\
            self.loss['variance'],\
            self.loss['violation'],\
            self.loss['boundary'],\
            self.loss['flow'],\
            self.loss['rot'],\
            self.loss['transl'],\
            self.loss['mask'],\
            self.loss['score'],\
            self.loss['elem']])
        self.loss_value_add({'violation':self.lossv['violation'],'variance':self.lossv['variance'],'transl':self.lossv['transl'],'flow':self.lossv['flow'],'boundary':self.lossv['boundary'],'total_loss':self.lossv['total_loss'],'rot':self.lossv['rot'],'mask':self.lossv['mask'], 'score':self.lossv['score'], 'elem':self.lossv['elem']})
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
#        self.lossv['transl_variance'], self.lossv['transl'], self.lossv['total_loss'], self.lossv['mask'], self.lossv['elem'], self.lossv['boundary'], self.lossv['variance'], self.lossv['violation'], self.lossv['score'] = self.sess.run([self.loss['transl_variance'], self.loss['transl'], self.cost, self.loss['mask'], self.loss['elem'], self.loss['boundary'], self.loss['variance'], self.loss['violation'], self.loss['score']])
#        self.loss_value_add({'transl_variance':self.lossv['transl_variance'],'transl':self.lossv['transl'],'total_loss':self.lossv['total_loss'],'mask':self.lossv['mask'],'elem':self.lossv['elem'],'boundary':self.lossv['boundary'], 'violation':self.lossv['violation'],'variance':self.lossv['variance'], 'score':self.lossv['score']}) 
        self.lossv['cc'], self.lossv['flow'], self.lossv['total_loss'], self.lossv['rot'], \
         self.lossv['transl'] = self.sess.run([ self.loss['cc'], self.loss['flow'], \
           self.cost, self.loss['rot'], self.loss['transl']])
        self.loss_value_add({'cc':self.lossv['cc'], 'flow':self.lossv['flow'] * 100.0, 'total_loss':self.lossv['total_loss'],'rot':self.lossv['rot'],'transl':self.lossv['transl']*100.0})
#       loss_value = self.sess.run([self.cost])
       #print('%f loss value' % loss_value_mask)
#       self.loss_value_add({'total_loss':loss_value})
 
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
    self.gtv['frame1_id'], self.gtv['frame2_id'], \
    self.gtv['frame1_xyz'], \
    self.gtv['rot'], \
    self.predv['rot_masked'], \
    self.predv['transl_masked'], \
    self.inputv['frame1_xyz'], \
    self.gtv['frame2_xyz'], \
    self.predv['frame2_xyz_masked'], \
    self.predv['frame2_boundary_masked'], \
    self.predv['frame2_score_positive_masked'], \
    self.gtv['frame2_score'], \
    self.gtv['frame2_mask'], \
    self.inputv['frame1_xyz'], \
    self.inputv['frame1_rgb'], \
    self.inputv['frame2_rgb'], \
    self.predv['transl'], \
    self.gtv['transl'], \
    self.inputv['frame2_xyz'], \
    self.instance_idv = self.sess.run([\
      self.gt['frame1_id'], self.gt['frame2_id'], \
      self.gt['frame1_xyz'], \
      self.gt['rot'], \
      self.pred['rot_masked'], \
      self.pred['transl_masked'], \
      self.input['frame1_xyz'], \
      self.gt['frame2_xyz'], \
      self.pred['frame2_xyz_masked'], \
      self.pred['frame2_boundary_masked'], \
      self.pred['frame2_score_positive_masked'], \
      self.gt['frame2_score'], \
      self.gt['frame2_mask'], \
      self.input['frame1_xyz'], \
      self.input['frame1_rgb'], \
      self.input['frame2_rgb'], \
      self.pred['transl'], \
      self.gt['transl'], \
      self.input['frame2_xyz'], \
      self.instance_id])
    print('dine')

    for i in range(self.batch_size):
      #self.predv['instance_center'], self.predv['instance_boundary'], self.predv['instance_score'] = nms(self.predv['xyz_masked'][i], self.predv['boundary_masked'][i], self.predv['score_positive_masked'][i])
      #self.predv['final_seg'], self.predv['final_instance'], self.predv['final_score'] = infer_seg(self.predv['instance_center'],self.predv['instance_boundary'],self.predv['instance_score'],self.predv['xyz_masked'][i])

      #error_dist = self.predv['xyz_masked'][i] - self.gtv['xyz'][i]
      #error_dist = np.linalg.norm(error_dist,axis=2) * 100.0 * self.gtv['mask'][i]
      #error_dist[error_dist > 2.0] = 0.0
      
      gtxyz = self.gtv['frame2_xyz'][0]
      gtxyz = gtxyz.reshape((-1,3))
      gtxyz = np.unique(gtxyz,axis=0)
      
  
      input_frame1_xyz = self.inputv['frame1_xyz'][0]
      input_frame1_xyz = input_frame1_xyz.reshape((-1,3)) 
      input_frame2_xyz = self.inputv['frame2_xyz'][0]
      input_frame2_xyz = input_frame2_xyz.reshape((-1,3))

      gt_transl = self.gtv['transl'][0]
      gt_transl = gt_transl.reshape((-1,3))

      predxyz = self.predv['frame2_xyz_masked'][0]
      predxyz = predxyz.reshape((-1,3))

      if 1:
        pred_frame1_xyz = self.inputv['frame2_xyz'][0] 
        frame1_id = np.unique(self.gtv['frame1_id'][0])
        frame2_id = np.unique(self.gtv['frame2_id'][0]) 

        for frame_id_  in frame2_id:
          if frame_id_ > 0 and frame_id_ in frame1_id:
            frame2_model_id = self.gtv['frame2_id'][0] == frame_id_
            frame1_model_id = self.gtv['frame1_id'][0] == frame_id_

            pred_transl = self.predv['transl_masked'][0][frame2_model_id]
            pred_transl = np.mean(pred_transl,axis=0)
            pred_frame1_xyz_ = pred_frame1_xyz[frame2_model_id]

            pred_frame1_xyz_ += pred_transl
             #print("pred_transl")
             #print(pred_transl)

            gt_transl = self.gtv['transl'][0][frame2_model_id]
            gt_transl = np.mean(gt_transl,axis=0)
             #print("gt_transl")
             #print(gt_transl)
 
            frame1_center = self.gtv['frame1_xyz'][0][frame1_model_id]
            frame1_center = np.mean(frame1_center,axis=0)
            pred_frame1_xyz_ -= frame1_center
             
            pred_rot = self.predv['rot_masked'][0][frame2_model_id] 
            pred_rot = np.mean(pred_rot,axis=0)

            gt_rot = self.gtv['rot'][0][frame2_model_id] 
            gt_rot = np.mean(gt_rot,axis=0)
            print('gt_rot')
            print(gt_rot)
            print(np.linalg.norm(gt_rot))

            pred_rot = angleaxis_rotmatrix(pred_rot)        
            pred_frame1_xyz_ = pred_rot.dot(pred_frame1_xyz_.T).T + frame1_center
                
            frame1_model_id = self.gtv['frame1_id'][0] == frame_id_
            input_frame1_xyz = self.inputv['frame1_xyz'][0][frame1_model_id]
            input_frame2_xyz = self.inputv['frame2_xyz'][0][frame2_model_id]  
            mayalab.points3d(pred_frame1_xyz_[:,0], pred_frame1_xyz_[:,1], pred_frame1_xyz_[:,2], color=(0,1,0),mode='sphere',scale_factor=0.01)
        
            mayalab.points3d(input_frame1_xyz[:,0],input_frame1_xyz[:,1],input_frame1_xyz[:,2], color=(1,0,0),mode='sphere',scale_factor=0.01) 
            mayalab.points3d(input_frame2_xyz[:,0],input_frame2_xyz[:,1],input_frame2_xyz[:,2], color=(0,0,1),mode='sphere',scale_factor=0.01) 
 
        mayalab.show()
 
         
        #mayalab.points3d(input_frame2_xyz[:,0],input_frame2_xyz[:,1],input_frame2_xyz[:,2],color=(0,1,0),mode='point')
        #mayalab.points3d(predxyz[:,0],predxyz[:,1],predxyz[:,2],mode='point')
        #mayalab.points3d(gtxyz[:,0],gtxyz[:,1],gtxyz[:,2],color=(0,0,1),mode='point')
        #for jj in xrange(num_):
        #  mayalab.points3d(center[jj:jj+1,0],center[jj:jj+1,1],center[jj:jj+1,2],color=(1,0,0),mode='sphere',opacity=0.1,scale_mode='vector',scale_factor=r[jj])
          #mayalab.points3d(center[jj:jj+1,0],center[jj:jj+1,1],center[jj:jj+1,2],color=(1,0,0),mode='sphere',opacity=1.0,scale_mode='vector',scale_factor=0.001)

        #mayalab.quiver3d(flow_xyz[:,0], flow_xyz[:,1], flow_xyz[:,2], flow_dir[:,0], flow_dir[:,1], flow_dir[:,2],scale_mode='vector',scale_factor=0.9)
         #direction = predxyz-inputxyz
      #norm_dir = np.linalg.norm(direction,axis=1)
      #flow_norm = norm_dir[norm_dir < 0.1]
      #flow_xyz = inputxyz[norm_dir < 0.1]
      #flow_dir = direction[norm_dir < 0.1]
      #center = self.predv['instance_center']
      #r = self.predv['instance_boundary']
      #num_ = len(r)

      if 0:
        plt.figure(0)
        plt.imshow(self.gtv['transl'][0])
        plt.figure(1)
        plt.imshow(self.predv['transl_masked'][0])
        plt.figure(2)
        plt.imshow(self.inputv['frame1_xyz'][0][:,:,2])
        plt.figure(3)
        plt.imshow(self.inputv['frame2_xyz'][0][:,:,2])
        plt.figure(4)
        plt.imshow((self.inputv['frame1_rgb'][0]+0.5)*255.0)
        plt.figure(5)
        plt.imshow((self.inputv['frame2_rgb'][0]+0.5)*255.0)
        plt.figure(6)
        plt.imshow(self.gtv['rot'][0])
        plt.figure(7)
        plt.imshow(self.predv['rot_masked'][0])
        plt.show()


       #plt.figure(0)
        #plt.imshow(self.predv['mask_truncated'][0])
        #plt.figure(1)
        #plt.imshow(self.predv['mask_positive'][0])
        #plt.figure(2)
      save = False
      if save:
        tmp_path = os.path.join(self.result_save_epoch_top_dir,str(self.instance_idv[i]))
        np.savez(os.path.join(tmp_path,'gt'),seg=self.gtv['xyz'][i][:,:,2])
        np.savetxt(os.path.join(tmp_path,'pred.txt'),self.predv['final_score'],fmt='%.8f')
        for j in range(len(self.predv['final_instance'])):
          np.savez(os.path.join(tmp_path,'pred'+str(j)),seg=self.predv['final_instance'][j])



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
    self.train(23)
    print("finishing trainin")
    best_epoch = self.validate(0,self.flags.num_epochs)
    #self.log.log_plotting(['transl','rot','total_loss','flow'])
    #self.test(best_epoch)
    #best_epoch = 36#best_epoch #self.flags.num_epochs - 1
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
       tmp_path = os.path.join(self.result_save_epoch_top_dir,model_id)
       if not os.path.exists(tmp_path):
         os.mkdir(tmp_path)
       else:
         shutil.rmtree(tmp_path)
         os.mkdir(tmp_path)
