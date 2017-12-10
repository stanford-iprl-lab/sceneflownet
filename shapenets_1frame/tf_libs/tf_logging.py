import re
import numpy as np
import matplotlib.pyplot as plt
import os

class LOG(object):
    '''
    A class records information during deep net experiment
    '''
    def __init__(self, log_path, log_file): 
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        self.log_path = os.path.join(log_path, log_file)
        if not os.path.isfile(self.log_path): 
            self.log_fout = open(self.log_path, 'w') 
        else:
            self.log_fout = open(self.log_path, 'a+')           
        self.train_loss = []
        self.val_loss = []
        self.test_loss = []
        self.train_epoch = []
        self.val_epoch = []
        self.test_epoch = []
        self.epoch_saving = []
 
    def log_string(self, out_str):
        self.log_fout.write(out_str+'\n')
        self.log_fout.flush()
        print(out_str)
     
    def log_train_loss(self, train_loss, train_epoch):
        self.log_string(str(train_epoch) + ' Epoch Train Loss: ' + str(train_loss))

    def log_val_loss(self, val_loss, val_epoch):
        self.log_string(str(val_epoch) + ' Epoch Val Loss: ' + str(val_loss))

    def log_test_loss(self, test_loss, test_epoch):
        self.log_string(str(test_epoch) + ' Epoch Test Loss: ' + str(test_loss))

    def log_save_model(self, epoch):
        self.log_string('Saving Epoch '+ str(epoch))

    def log_parsing(self):
        lines = [line for line in open(self.log_path)]
        for line in lines:
            if 'Epoch Train Loss' in line:
                self.train_epoch.append(int(line.strip().split()[0]))
                self.train_loss.append(float(line.strip().split()[-1]))
            if 'Epoch Test Loss'  in line:
                self.test_epoch.append(int(line.strip().split()[0]))
                self.test_loss.append(float(line.strip().split()[-1]))
            if 'Epoch Val Loss'  in line:
                self.val_epoch.append(int(line.strip().split()[0]))
                self.val_loss.append(float(line.strip().split()[-1]))
            if 'Saving Epoch' in line:
                self.epoch_saving.append(int(line.strip().split()[-1]))

    def log_plotting(self, img_path, plot_train=True, plot_val=True, plot_test=True):
        if plot_train:
	    plt.plot(self.train_epoch, self.train_loss)
        if plot_val:
            plt.plot(self.val_epoch, self.val_loss)
        if plot_test:
            plt.plot(self.test_epoch, self.test_loss)
        plt.savefig(img_path)         
