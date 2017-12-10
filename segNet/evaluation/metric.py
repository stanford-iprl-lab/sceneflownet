import numpy as np
import os

def IoU(x, y, h=120,w=160):
    # segmentation IoU of 2 binary map
    x = x.reshape((h,w))
    y = y.reshape((h,w))
    return 1.0 * np.logical_and(x, y).sum() / (np.logical_or(x, y).sum()+0.000001)

def tp_fp(x, y, thres=0.5):
    #  x are the list of predicted segments, each entry contains id_map and confidence score (id_map,score) 
    #  y are the list of gt segments, each entry contains id_map (id_map)
    num_pred = len(x)
    num_gt = len(y)
    tp = np.zeros(num_pred,)
    fp = np.zeros(num_pred,)
    gt_detected = np.ones((num_gt,))*-1.0
    for x_i in xrange(num_pred):
      pred_id_map = x[x_i]
      max_iou = -1.0
      max_gt_id = 0
      for y_i in xrange(num_gt):
        tmp_iou = IoU(pred_id_map, y[y_i])
        if tmp_iou > max_iou:
          max_iou = tmp_iou
          max_gt_id = y_i
      if max_iou > thres:
        if gt_detected[max_gt_id] < 0:
          tp[x_i] = 1.0
          gt_detected[max_gt_id] = x_i
        else:
          fp[x_i] = 1.0
      else:
        fp[x_i] = 1.0
    return tp,fp


def tp_fp_scores(path_dir,thres=0.5):
  gt_seg = np.load(os.path.join(path_dir,'gt.npz'))['seg']
  idx = np.unique(gt_seg)  
  gt_seg_list = []
  pred_seg_list = []
  for i in idx:
    if i > 0:
      gt_seg_list.append(gt_seg == i)
  pred_scores = np.loadtxt(os.path.join(path_dir,'pred.txt')) 
  pred_seg_list = [np.load(os.path.join(path_dir,'pred'+str(i)+'.npz'))['seg'] for i in xrange(len(pred_scores))]
  pred_scores = np.loadtxt(os.path.join(path_dir,'pred.txt'))
  tp_id, fp_id = tp_fp(pred_seg_list,gt_seg_list,thres=thres) 
  return tp_id, fp_id, pred_scores, len(gt_seg_list)    

def m_AP_(dir_lists,thres): 
  tp = []
  fp = []
  scores = []
  num_gt = 0
  for dir_id in dir_lists:
    tp_tmp,fp_tmp,scores_tmp,num_gt_tmp = tp_fp_scores(dir_id,thres=thres)
    num_gt += num_gt_tmp
    for i in xrange(len(tp_tmp)):
      tp.append(tp_tmp[i])
      fp.append(fp_tmp[i])
      scores.append(scores_tmp[i])
  return m_AP__(tp,fp,scores,num_gt)


def m_AP50(dir_lists):
  return m_AP_(dir_lists,thres=0.5)

def m_AP75(dir_lists):
  return m_AP_(dir_lists,thres=0.75)

def m_AP90(dir_lists):
  return m_AP_(dir_lists,thres=0.9)

def m_AP(dir_lists):
  AP = [m_AP_(dir_lists,thres=i) for i in np.arange(0.5,0.95+0.05,0.05)]
  print(AP)
  return np.mean(np.array(AP))
    
def m_AP__(tp,fp,scores,num_gt):
    idx = sorted(range(len(scores)), key=lambda k: scores[k])
    idx = np.array(idx).astype(np.int32)
    tp = np.array(tp)
    fp = np.array(fp)
    tp = tp[idx]
    fp = fp[idx]
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    rec = tp/num_gt
    prec = tp/(fp+tp)
    ap = 0.0
    ap_step = 0.1
    for t in np.arange(ap_step,1.0+ap_step,ap_step):
      num_ = np.sum(rec < t)
      if num_ == 0.0:
        p = 0.0
      else:
        p=np.mean(prec[rec < t])
      ap=ap+p * ap_step;
    print(ap)
    return ap



if __name__ == "__main__":
    x = np.array([
        [1, 1, 1, 1, 2, 2],
        [1, 1, 2, 2, 2, 2],
        [1, 3, 3, 2, 2, 2],
        [3, 3, 3, 3, 3, 3]
        ])
    y = np.array([
        [3, 3, 7, 7, 7, 7],
        [3, 3, 3, 7, 7, 7],
        [3, 5, 5, 5, 7, 7],
        [5, 5, 5, 5, 5, 5]
        ])
    thres_list = np.linspace(0, 1, 21)
    x = [('ABE',0.5),('ddd',0.1),('DDD',0.2)]
    x = sorted(x, key=lambda x: x[1],reverse=True)
    print(x)
