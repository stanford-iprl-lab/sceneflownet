from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import time
import sys
import argparse
import shutil

from local_variables import *

sys.path.append(BASE_DIR)

from models.sceneflownet_corr import cnnmodel

seed = 42
np.random.seed(seed)
tf.set_random_seed(seed)

from tf_libs.tf_logging import LOG
from tf_libs.tfrecords import inputs
from evaluation.metric import IoU
from lossf.loss_corr_no_loss import loss
from inference.infer import nms,infer_seg
experiment_name = 'sceneflownet_corr_no_loss'

MODEL_SAVE_DIR = os.path.join(BASE_DIR,'saved_models')
RESULT_SAVE_DIR = os.path.join(BASE_DIR,'saved_results')
LOGGING_DIR = os.path.join(BASE_DIR,'logging')

# Basic model parameters 
parser = argparse.ArgumentParser()
parser.add_argument('--model_save_dir',default=os.path.join(MODEL_SAVE_DIR,experiment_name)+'/',help='Directory to save the trained model')
parser.add_argument('--learning_rate', type=float,default=0.0001,help='Initial learning rate.')
parser.add_argument('--num_epochs',type=int,default=400,help='Number of epochs to run trainer.')
parser.add_argument('--train_batch_size',type=int,default=12,help='Number of models within a batch.')
parser.add_argument('--val_batch_size',type=int,default=1,help='Number of models within a batch.')
parser.add_argument('--test_batch_size',type=int,default=1,help='Number of models within a batch.')
parser.add_argument('--num_train_model',type=int,default=24994,help='Number of models within a batch.')#29172
parser.add_argument('--num_val_model',type=int,default=1000,help='Number of models within a batch.')#8160#3360
parser.add_argument('--num_test_model',type=int,default=8300,help='Number of models within a batch.')
parser.add_argument('--max_model_to_keep',type=int,default=400,help='max saved models')
parser.add_argument('--log_dir',default=os.path.join(LOGGING_DIR,experiment_name),help='folder to save logging infor')
parser.add_argument('--train_tfrecords_filename',default=os.path.join(DATA_DIR,'Tfrecords_train'),help='directory to contain train tfrecord files')
parser.add_argument('--val_tfrecords_filename',default=os.path.join(DATA_DIR,'Tfrecords_test'),help='directory to contain train tfrecord files')
parser.add_argument('--test_tfrecords_filename',default=os.path.join(DATA_DIR,'Tfrecords_test'),help='directory to contain train tfrecord files')
parser.add_argument('--result_save_dir',default=os.path.join(RESULT_SAVE_DIR,experiment_name)+'/',help='Directory to save the result')

FLAGS = parser.parse_args()
dim = 3

from tf_libs.sceneflownet_experiment_corr_no_loss import Experiment

if not os.path.exists(FLAGS.log_dir):
  os.mkdir(FLAGS.log_dir)

log = LOG(FLAGS.log_dir,'log.txt')

ex = Experiment(FLAGS,inputs,cnnmodel,loss,log)
ex.whole_process()
