from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import os
import sys
import time
import shutil

from tf_libs.tf_logging import LOG
from tf_libs.save_result import generate_result_folder, save_gt_segments, save_pred_segments
from inference.infer import infer_seg,nms
from evaluation.metric import m_AP50,m_AP75,m_AP90, m_AP
from mayavi import mlab as mayalab

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
    self._2m = True
    self.dim = 9
 
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
      new_saver = tf.train.import_meta_graph(os.path.join(self.flags.model_save_dir,'-0.meta'))
      new_saver.restore(self.sess,os.path.join(self.flags.model_save_dir,'-'+str(restore_epoch)))
    else:
      self.clean_model_save_dir()

      
  def build_model(self, tfrecords_filename, num_epochs):
    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    self.input['xyz'], self.gt['feat'], self.gt['r'], self.instance_id, self.gt['score'] = self.inputf(batch_size = self.batch_size, num_epochs = num_epochs, tfrecords_filename = tfrecords_filename)
    self.pred['mask'], self.pred['r'], self.pred['feat'], self.pred['score'] = self.model(self.input['xyz'])
    self.pred['mask_positive'] = tf.sigmoid(self.pred['mask'])[:,:,:,1]
    self.pred['score_positive']  = tf.sigmoid(self.pred['score'])[:,:,:,1]

    self.pred['mask_truncated'] = self.pred['mask_positive'] > 0.5
    self.pred['mask_truncated'] = tf.cast(self.pred['mask_truncated'],tf.float32)
    self.pred['mask_truncated_1'] = tf.reshape(self.pred['mask_truncated'],[-1,240,320,1])
    self.pred['mask_truncated_dim'] = tf.tile(self.pred['mask_truncated_1'],[1,1,1,self.dim])

    self.pred['feat_masked'] = self.pred['mask_truncated_dim'] * self.pred['feat']
    self.pred['boundary_masked'] = self.pred['mask_truncated_1'] * self.pred['r']

    self.gt['mask'] = self.gt['feat'][:,:,:,2] > 0.0
    self.gt['mask'] = tf.cast(self.gt['mask'],tf.float32)
  
    self.pred['score_positive_masked'] = self.pred['score_positive'] * self.pred['mask_truncated']

  
  def loss_op(self):
    self.loss['mask'], self.loss['score'], self.loss['elem'], self.loss['variance'], self.loss['boundary'], self.loss['violation'] = self.lossf(self.pred['feat'], self.pred['r'], self.pred['mask'], self.pred['score'], self.gt['feat'], self.gt['r'], self.gt['score'], dim=self.dim, batch_size=self.batch_size) 

    self.cost = self.loss['mask'] + self.loss['elem'] * 100.0 + self.loss['boundary'] * 1000.0 + self.loss['score']  + self.loss['violation'] + self.loss['variance'] 


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
      string_tmp += ' %s %f |' % (key,value)
      self.log.add_item(train_val_test,self.epoch,key,value)
    self.log.log_string(string_tmp)
    return string_tmp  


  def train(self,restore_epoch=-1):
    self.build_framework(restore_epoch,train_val_test='train')

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=self.sess,coord=coord)

    self.loss_value_init()

    if restore_epoch >= 0:
      self.epoch = restore_epoch + 1
    else:
      self.epoch = 0
    try:
      step=0
      while not coord.should_stop():
        _ , self.lossv['total_loss'], \
        self.lossv['mask'], \
        self.lossv['elem'], \
        self.lossv['boundary'], \
        self.lossv['variance'], \
        self.lossv['violation'], \
        self.lossv['score'] = self.sess.run([self.train_op, self.cost, \
          self.loss['mask'], \
          self.loss['elem'], \
          self.loss['boundary'], \
          self.loss['variance'], \
          self.loss['violation'], \
          self.loss['score']])
        self.loss_value_add({'total_loss':self.lossv['total_loss'],'mask':self.lossv['mask'],'elem':self.lossv['elem'],'boundary':self.lossv['boundary'], 'violation':self.lossv['violation'],'variance':self.lossv['variance'], 'score':self.lossv['score']})
        step += 1
        if step % self.num_batch == 0:
          self.loss_value_average()
          self.report_loss_value('train')
          if self.epoch > 0:
            self.saver.save(self.sess,self.flags.model_save_dir,global_step=self.epoch,write_meta_graph=False)
          else:
            print('saving the epoch %d' % self.epoch)
            self.saver.save(self.sess,self.flags.model_save_dir,global_step=self.epoch)
          self.epoch += 1
          self.loss_value_init()
        if self.epoch > self.flags.num_epochs:
          print('break because self.epoch')
          break     
    except tf.errors.OutOfRangeError:
      print('Training is Done')
    finally:
      print('tfrecord is running out')
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
        self.lossv['total_loss'], self.lossv['mask'], self.lossv['elem'], self.lossv['boundary'], self.lossv['variance'], self.lossv['violation'], self.lossv['score'] = self.sess.run([self.cost, self.loss['mask'], self.loss['elem'], self.loss['boundary'], self.loss['variance'], self.loss['violation'], self.loss['score']])
        self.loss_value_add({'total_loss':self.lossv['total_loss'],'mask':self.lossv['mask'],'elem':self.lossv['elem'],'boundary':self.lossv['boundary'], 'violation':self.lossv['violation'],'variance':self.lossv['variance'], 'score':self.lossv['score']})

     
    self.loss_value_average()
    self.report_loss_value('val')

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
    for ii in xrange(self.num_batch):
      loss_value, loss_value_mask, loss_value_elem, loss_value_boundary, loss_value_variance, loss_value_violation, loss_value_score = self.sess.run([self.cost, self.loss['mask'], self.loss['elem'], self.loss['boundary'], self.loss['variance'], self.loss['violation'], self.loss['score']])
      self.loss_value_add({'total_loss':loss_value,'mask':loss_value_mask,'elem':loss_value_elem,'boundary':loss_value_boundary, 'violation':loss_value_violation})
 
    self.loss_value_average()
    self.report_loss_value('test')

    coord.request_stop()
    coord.join(threads)
    self.sess.close()

  
  def result_op(self,save=True):
    self.gtv['score'], \
    self.predv['score_positive'], \
    self.inputv['xyz'], \
    self.gtv['mask'], \
    self.predv['score_positive_masked'], \
    self.predv['mask_positive'], \
    self.predv['mask_truncated'], \
    self.predv['feat_masked'], \
    self.predv['boundary_masked'], \
    self.predv['feat_masked'], \
    self.gtv['feat'], \
    self.instance_idv = self.sess.run([ \
        self.gt['score'], \
        self.pred['score_positive'], \
        self.input['xyz'], \
        self.gt['mask'], \
        self.pred['score_positive_masked'], \
        self.pred['mask_positive'], \
        self.pred['mask_truncated'], \
        self.pred['feat_masked'], \
        self.pred['boundary_masked'], \
        self.pred['feat_masked'], \
        self.gt['feat'], \
        self.instance_id]) 

    for i in range(self.batch_size):
      self.predv['feat_masked'][i][:,:,3:6] += self.predv['feat_masked'][i][:,:,0:3]
      self.predv['feat_masked'][i][:,:,6:9] += self.predv['feat_masked'][i][:,:,0:3]

      self.predv['instance_center'], self.predv['instance_boundary'], self.predv['instance_score'] = nms(self.predv['feat_masked'][i], self.predv['boundary_masked'][i], self.predv['score_positive_masked'][i],self.dim)
      self.predv['final_seg'], self.predv['final_instance'], self.predv['final_score'] = infer_seg(self.predv['instance_center'],self.predv['instance_boundary'],self.predv['instance_score'],self.predv['feat_masked'][i])

      print(self.predv['final_seg'].shape) 
      if 1:
        plt.figure(0)
        plt.imshow(self.gtv['score'][0])
        #plt.figure(1)
        #plt.imshow(self.predv['feat_positive'][0][:,:,2])
        plt.figure(2)
        plt.imshow(self.gtv['feat'][0][:,:,2]) 
        plt.figure(4)
        plt.imshow(self.predv['score_positive_masked'][0])
        plt.figure(5)
        plt.imshow(self.predv['final_seg'])
        #for feat_i in xrange(3,self.dim):
        #  plt.figure(6+feat_i*2)
        #  plt.imshow(self.predv['feat_masked'][0][:,:,feat_i])
        #  plt.figure(6+feat_i*2+1)
        #  plt.imshow(self.gtv['feat'][0][:,:,feat_i])
        plt.show() 


      if 0:
        inputxyz = self.inputv['xyz'][0]
        inputxyz = inputxyz.reshape((-1,3))
        predxyz = self.predv['feat_masked'][0]
        predxyz = predxyz.reshape((-1,3+6))
        gtxyz = self.gtv['feat'][0]
        gtxyz = gtxyz.reshape((-1,3+6))
        s = 1
        
        predxx = predxyz[:,3] * s + gtxyz[:,0]
        predyy = predxyz[:,4] * s + gtxyz[:,1]
        predzz = predxyz[:,5] * s + gtxyz[:,2]
        predxy = predxyz[:,6] * s + gtxyz[:,0]
        predyz = predxyz[:,7] * s + gtxyz[:,1]
        predzx = predxyz[:,8] * s + gtxyz[:,2]
        
        gtxx = gtxyz[:,3] * s + gtxyz[:,0]
        gtyy = gtxyz[:,4] * s + gtxyz[:,1]
        gtzz = gtxyz[:,5] * s + gtxyz[:,2]
        gtxy = gtxyz[:,6] * s + gtxyz[:,0]
        gtyz = gtxyz[:,7] * s + gtxyz[:,1]
        gtzx = gtxyz[:,8] * s + gtxyz[:,2]
 
        ugtxyz = np.unique(gtxyz,axis=0)
        gtxxyyzz = np.vstack([gtxx,gtyy,gtzz]).T
        gtxxyyzz = np.unique(gtxxyyzz,axis=0)
        ugtxx= gtxxyyzz[:,0]
        ugtyy = gtxxyyzz[:,1]
        ugtzz = gtxxyyzz[:,2]
        center = self.predv['instance_center']
        r = self.predv['instance_boundary']

        mayalab.points3d(inputxyz[:,0],inputxyz[:,1],inputxyz[:,2],color=(0,1,0),mode='point')
        mayalab.points3d(predxyz[:,0] , predxyz[:,1] , predxyz[:,2], mode='point')
        mayalab.points3d(predxx , predyy , predzz, mode='point', color=(0,0,1))

        gt_id = np.unique(self.gtv['feat'][0][:,:,2])
        for inst_id in gt_id:
          if inst_id > 0:
            tmp_id = self.gtv['feat'][0][:,:,2] == inst_id
            print(tmp_id.shape)
            predxyz_ = self.predv['feat_masked'][0][tmp_id]
            gtxyz_ = self.gtv['feat'][0][tmp_id] 
            dfxyz_ = predxyz_ - gtxyz_            
            print(np.std(dfxyz_,axis=0))
        for jj in xrange(len(ugtxx)):
          #mayalab.points3d(center[jj:jj+1,0],center[jj:jj+1,1],center[jj:jj+1,2],color=(1,0,0),mode='sphere',opacity=0.1,scale_mode='vector',scale_factor=r[jj])
          mayalab.points3d(ugtxx, ugtyy, ugtzz,color=(1,0,0),mode='sphere',opacity=0.1,scale_mode='vector',scale_factor=0.01)
          mayalab.points3d(ugtxyz[:,0], ugtxyz[:,1], ugtxyz[:,2],color=(1,0,0),mode='sphere',opacity=0.1,scale_mode='vector',scale_factor=0.01)
          mayalab.quiver3d(ugtxyz[:,0], ugtxyz[:,1], ugtxyz[:,2], ugtxx - ugtxyz[:,0], ugtyy - ugtxyz[:,1], ugtzz - ugtxyz[:,2], scale_mode='vector',scale_factor=0.9)
        
        mayalab.show() 
        #plt.figure(0)
        #plt.imshow(self.predv['mask_truncated'][0])
        #plt.figure(1)
        #plt.imshow(self.predv['mask_positive'][0])
        #plt.figure(2)
      save = True 
      if save:
        tmp_path = os.path.join(self.result_save_epoch_top_dir,str(self.instance_idv[i]))
        np.savez(os.path.join(tmp_path,'gt'),seg=self.gtv['feat'][i][:,:,2])
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
    #self.train()
    #best_epoch = self.validate(0,self.flags.num_epochs)
    #self.log.log_plotting(['mask','score'])
    #self.test(best_epoch)
    best_epoch = 43#self.flags.num_epochs - 1
    self.save_result(best_epoch) 
    self.analysis(best_epoch)

  def analysis(self,epoch):
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
 
