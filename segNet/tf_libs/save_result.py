import tensorflow as tf
import numpy as np
import os
import shutil

def generate_result_folder(id_lists,top_dir):
  for idd in id_lists:
    id_dir = os.path.join(top_dir,idd)
    if not os.path.exists(id_dir):
      os.mkdir(id_dir)
    else:
      shutil.rmtree(id_dir)
      os.mkdir(id_dir)

def save_gt_segments(save_dir,final_seg,save_id):
  save_dir = os.path.join(save_dir,save_id,'gt')
  np.savez(save_dir,seg=final_seg)

def save_pred_segments(save_dir,seg_list,scores,save_id):
  save_dir = os.path.join(save_dir,save_id,'pred')
  scores = np.array(scores)
  np.savetxt(save_dir+'.txt',scores,fmt='%.8f')
  for i in xrange(len(seg_list)):
    np.savez(save_dir+str(i),seg=seg_list[i])

