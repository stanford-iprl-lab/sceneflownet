import re
import numpy as np
#import matplotlib.pyplot as plt
import os

class LOG(object):
    '''
    A class records information during deep net experiment
    '''
    def __init__(self, log_path, log_file): 
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        self.log_path = log_path
        self.log_file_path = os.path.join(log_path, log_file)
        if not os.path.isfile(self.log_file_path): 
            self.log_fout = open(self.log_file_path, 'w') 
        else:
            self.log_fout = open(self.log_file_path, 'a+')      
        self.loss = {}
        self.loss['train'] = {}
        self.loss['val'] = {}
        self.loss['test'] = {}  
        self.train_epoch = []
        self.val_epoch = []
        self.loss_keys = []
        self.img_path = os.path.join(self.log_path,'plotting.png')

    def init_keys(self,key_list): 
        for key in key_list:
          if key not in self.loss_keys:
            self.loss_keys.append(key)
            self.loss['train'][key] = []
            self.loss['val'][key] = []
            self.loss['test'][key] = []

    def log_string(self, out_str):
        self.log_fout.write(out_str+'\n')
        self.log_fout.flush()
        print(out_str)

    def add_item(self,train_val_test,epoch,key,value):
        self.loss[train_val_test][key].append(value)
        if train_val_test == 'train' and epoch not in self.train_epoch:
          self.train_epoch.append(epoch)
        if train_val_test == 'val' and epoch not in self.val_epoch:
          self.val_epoch.append(epoch)
     
    def log_plotting(self,key_list=None,show=True):
        if key_list == None:
          key_list = self.loss_keys
        for key in key_list:
          label_train = 'train_%s' % key
	  plt.plot(self.train_epoch, self.loss['train'][key], label=label_train)
          label_val = 'val_%s' % key
          plt.plot(self.val_epoch, self.loss['val'][key], label=label_val)
        plt.legend(loc='upper right',fontsize='x-small')
        plt.title('train val loss')
        plt.savefig(self.img_path)
        if show:
          plt.show() 
