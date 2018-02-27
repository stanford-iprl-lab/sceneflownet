import numpy as np
import glob
import os
from multiprocessing import Pool
import shutil

pred_top_dir = '/home/linshaonju/pd_flow_results'
gt_top_dir = '/home/linshaonju/interactive-segmentation/Data/BlensorResult_test/'

def EPE_(i,match=True):
  gt_dir = os.path.join(gt_top_dir,str(i))
  pred_dir = os.path.join(pred_top_dir,str(i)) 
  for file_ in glob.glob(os.path.join(pred_dir,'*.txt')):
    pred_flow = np.loadtxt(file_,comments="#",delimiter=" ",unpack=False)
    pred_flow = pred_flow[:,2:5].reshape((240,320,3))
  pred_flow = np.zeros((240,320,3))
  gt_flow = np.load(os.path.join(gt_dir,'flow.npz'))['flow']
  if match:
    frame80_id = np.load(os.path.join(gt_dir,'frame80_labeling_model_id.npz'))['labeling']
    frame20_id = np.load(os.path.join(gt_dir,'frame20_labeling_model_id.npz'))['labeling']
    u_frame20_id = np.unique(frame20_id) 
    frame80_id = frame80_id.reshape((-1,))
    frame80_id_flag = np.array([line in u_frame20_id and line > 0 for line in frame80_id]).astype(np.float)
  diff = gt_flow - pred_flow
  diff = diff.reshape((-1,3))
  dist = np.linalg.norm(diff,axis=1)
  if match:
    if np.sum(frame80_id_flag) == 0:
      print(gt_dir)
      shutil.rmtree(gt_dir)
      epe = 0.0
    else:
      epe = np.sum(dist * frame80_id_flag)/np.sum(frame80_id_flag) 
  else:
    epe = np.mean(dist)
  return epe

def AAE_(i,match=True):
  gt_dir = os.path.join(gt_top_dir,str(i))
  pred_dir = os.path.join(pred_top_dir,str(i)) 
  for file_ in glob.glob(os.path.join(pred_dir,'*.txt')):
    pred_flow = np.loadtxt(file_,comments="#",delimiter=" ",unpack=False)
    pred_flow = pred_flow[:,2:5].reshape((240,320,3))
  if match:
    frame80_id = np.load(os.path.join(gt_dir,'frame80_labeling_model_id.npz'))['labeling']
    frame20_id = np.load(os.path.join(gt_dir,'frame20_labeling_model_id.npz'))['labeling']
    u_frame20_id = np.unique(frame20_id)
    frame80_id = frame80_id.reshape((-1,))
    frame80_id_flag = np.array([line in u_frame20_id and line > 0 for line in frame80_id]).astype(np.float)
  gt_flow = np.load(os.path.join(gt_dir,'flow.npz'))['flow']
  gt_flow = gt_flow.reshape((-1,3))
  pred_flow = pred_flow.reshape((-1,3))
  gt_a = np.ones((240*320,4))
  pred_a = np.ones((240*320,4))
  gt_a[:,0:3] = gt_flow
  pred_a[:,0:3] = pred_flow
  gt_a = gt_a / np.expand_dims(np.linalg.norm(gt_a,axis=1),1)
  pred_a = pred_a / np.expand_dims(np.linalg.norm(pred_a,axis=1),1) 
  aae = np.arccos(np.sum(gt_a * pred_a,axis=1)/np.linalg.norm(gt_a,axis=1)/np.linalg.norm(gt_a,axis=1)) * (180.0/np.pi) 
  if match:
    aae_ = np.sum(frame80_id_flag * aae)/np.sum(frame80_id_flag)
  else:
    aae_ = np.mean(aae)
  return aae_

filelist = []

for i in xrange(8500):
  if os.path.exists(os.path.join(gt_top_dir,str(i))):
    filelist.append(i)

num_ = len(filelist)
print(num_)
res = np.zeros((num_,))
pool = Pool(180)

for i, data in enumerate(pool.imap(EPE_,filelist)):
  res[i] = data

print(np.mean(res))
