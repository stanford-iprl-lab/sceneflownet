import numpy as np
import scipy.ndimage
import math
#import matplotlib.pyplot as plt
from utils import *
from math import *
import os
from scipy.ndimage import imread
from mayavi import mlab as mayalab
import skimage.measure
from multiprocessing import Pool
import shutil

np.set_printoptions(precision=4,suppress=True,linewidth=300)

h = 240
w = 320

def quaternion_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion.
    >>> M = quaternion_matrix([0.99810947, 0.06146124, 0, 0])
    >>> numpy.allclose(M, rotation_matrix(0.123, [1, 0, 0]))
    True
    >>> M = quaternion_matrix([1, 0, 0, 0])
    >>> numpy.allclose(M, numpy.identity(4))
    True
    >>> M = quaternion_matrix([0, 1, 0, 0])
    >>> numpy.allclose(M, numpy.diag([1, -1, -1, 1]))
    True
    """
    _EPS = np.finfo(float).eps * 4.0
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0], 0.0],
        [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0], 0.0],
        [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2], 0.0],
        [                0.0,                 0.0,                 0.0, 1.0]])

def load_rgb(filepath):
  tmp = imread(filepath)
  
  zoom_scale = 0.5
  r = scipy.ndimage.zoom(tmp[:,:,0], zoom_scale, order=1)
  g = scipy.ndimage.zoom(tmp[:,:,1], zoom_scale, order=1)
  b = scipy.ndimage.zoom(tmp[:,:,2], zoom_scale, order=1)
  image = np.dstack((r,g,b))
  return image 


def load_labeling(filepath):
  label_id = np.load(filepath)['labeling']
  return label_id


def tran_rot(filepath):
  rot = np.zeros((3,3))
  tran = np.zeros((3,))
  lines = [line.strip() for line in open(filepath)]
  for idx, line in enumerate(lines):
    tmp = str(line).split('(')[1].split(')')[0].split()
    tmp = [float(x.split(',')[0]) for x in tmp]
    if idx < 3:
      rot[idx,:] = np.array(tmp[0:3])
      tran[idx] = tmp[3]
  return tran,rot


def rotmatrix_angleaxis(rot):
  angleaxis = np.zeros((3,)) 
  angleaxis[0] = rot[2,1] - rot[1,2]
  angleaxis[1] = rot[0,2] - rot[2,0]
  angleaxis[2] = rot[1,0] - rot[0,1]
  angleaxis = angleaxis / (np.linalg.norm(angleaxis) + 0.000001)
  tmp = (rot[0,0] + rot[1,1] + rot[2,2] - 1) * 0.5
  if tmp > 1.0:
    tmp = 1.0
  elif tmp < -1.0:
    tmp = -1.0
  angle = np.arccos( tmp )
  angleaxis *= angle
  assert(np.all(np.logical_not(np.isnan(angleaxis))))
  return angleaxis


def angleaxis_rotmatrix(angleaxis):
  angle = np.linalg.norm(angleaxis)
  axis = angleaxis / (angle + 0.000001)
  c = np.cos(angle)
  v = 1 - c
  s = np.sin(angle)
  rot = np.zeros((3,3))
  rot[0,0] = axis[0] ** 2 * v + c
  rot[0,1] = axis[0] * axis[1] * v - axis[2] * s
  rot[0,2] = axis[0] * axis[2] * v + axis[1] * s
  rot[1,0] = axis[0] * axis[1] * v + axis[2] * s
  rot[1,1] = axis[1] ** 2 * v + c
  rot[1,2] = axis[1] * axis[2] * v - axis[0] * s
  rot[2,0] = axis[0] * axis[2] * v - axis[1] * s
  rot[2,1] = axis[1] * axis[2] * v + axis[0] * s
  rot[2,2] = axis[2] ** 2 * v + c
  return rot

def load_cc(cc_file):
  cc = np.load(cc_file)['cc']
  return cc

def load_transformation(top_dir):
  transl_file = os.path.join(top_dir,'translation.npz')
  rot_file = os.path.join(top_dir,'rotation.npz')
  if not os.path.exists(transl_file):
    return None
  transl = np.load(transl_file)['transl']
  rot = np.load(rot_file)['rot']
  return transl, rot


def cal_transformation(top_dir):
  pgm_filepath = [line for line in os.listdir(top_dir) if line.endswith('.pgm') and line.startswith('frame80')]
  if len(pgm_filepath) < 1:
    return
  else:
    pgm_filepath = pgm_filepath[0] 
  tmp = pgm_filepath.split('.pgm')[0].split('_')
  
  azimuth_deg = float(tmp[2].split('azi')[1])
  elevation_deg = float(tmp[3].split('ele')[1]) 
  theta_deg = float(tmp[4].split('theta')[1])
  rho = float(tmp[1].split('rho')[1])
    
  cx, cy, cz = obj_centened_camera_pos(rho, azimuth_deg, elevation_deg)
  q1 = camPosToQuaternion(cx , cy , cz)
  q2 = camRotQuaternion(cx, cy , cz, theta_deg)
  q = quaternionProduct(q2, q1)
  R = quaternion_matrix(q)[0:3,0:3]
  C = np.zeros((3,))
  C[0] = cx
  C[1] = cy
  C[2] = cz

  if not os.path.exists(os.path.join(top_dir,'frame80_labeling_model_id.npz')):
    return
  if not os.path.exists(os.path.join(top_dir,'frame20_labeling_model_id.npz')):
    return
  if not os.path.exists(os.path.join(top_dir,'frame80_labeling.npz')):
    return
  if not os.path.exists(os.path.join(top_dir,'frame20_labeling.npz')):
    return
  
  frame2_id = load_labeling(os.path.join(top_dir,'frame80_labeling_model_id.npz')) 
  frame1_id = load_labeling(os.path.join(top_dir,'frame20_labeling_model_id.npz'))   

  frame2_center = load_seg(os.path.join(top_dir,'frame80_labeling.npz'))
  frame1_center = load_seg(os.path.join(top_dir,'frame20_labeling.npz'))

  frame2_xyz_name = [line for line in os.listdir(top_dir) if line.startswith('frame80') and line.endswith('.pgm')][0]
  frame1_xyz_name = [line for line in os.listdir(top_dir) if line.startswith('frame20') and line.endswith('.pgm')][0]
  frame2_xyz = load_xyz(os.path.join(top_dir,frame2_xyz_name))
  frame1_xyz = load_xyz(os.path.join(top_dir,frame1_xyz_name))
  
  frame2_id_list = np.unique(frame2_id)
  frame1_id_list = np.unique(frame1_id)

  model_ids = [line.split('frame80_')[1] for line in os.listdir(top_dir) if line.endswith('.txt') and line.startswith('frame80')]

  model_ids.sort()
  transformation_rot = np.zeros((h,w,3))
  transformation_translation = np.zeros((h,w,3))

  for instance_id in frame2_id_list:
    frame2_pid = frame2_id == instance_id
    frame2_pid = frame2_pid.reshape((240,320))
    frame1_pid = frame1_id == instance_id
    frame1_pid = frame1_pid.reshape((240,320))

    if instance_id > 0: 
      if instance_id in frame1_id_list:
        frame1_tran, frame1_rot = tran_rot(os.path.join(top_dir,'frame20_'+model_ids[int(instance_id)-1]))
        frame2_tran, frame2_rot = tran_rot(os.path.join(top_dir,'frame80_'+model_ids[int(instance_id)-1]))
        R12 = frame1_rot.dot(np.linalg.inv(frame2_rot))
        rot = R.T.dot(R12.dot(R))
        tran = R.T.dot(frame1_tran-C) + R.T.dot(R12.dot(C-frame2_tran))
  
        tran[2] *= -1.0
        rot[0,2] *= -1.0 
        rot[1,2] *= -1.0
        rot[2,0] *= -1.0
        rot[2,1] *= -1.0
      else:    
        tran = -np.mean(frame2_center[frame2_pid],0)
        rot = np.identity(3)
        print("yes") 
      angle_axis = rotmatrix_angleaxis(rot)
      transformation_translation[frame2_pid] = tran
      transformation_rot[frame2_pid] = angle_axis
  transformation_file = os.path.join(top_dir,'translation.npz')
  rotation_file = os.path.join(top_dir,'rotation.npz')
  np.savez(transformation_file,transl=transformation_translation)
  np.savez(rotation_file,rot=transformation_rot)


def load_seg(filepath):
  try:
    seg = np.load(filepath)['labeling']
    seg[:,:,2] *= -1.0
  except:
    print(filepath)
    print('sth is wrong!')
    return np.zeros((h,w,3))
  return seg


def load_xyz(filename):
    """Return image data from a PGM file generated by blensor. """
    fx = 472.92840576171875
    fy = fx 
    with open(filename, 'rb') as f:
        f.readline()
        f.readline()
        width_height = f.readline().strip().split()
        if len(width_height) > 1:
          width, height = map(int,width_height)
          value_max_range = float(f.readline())
          image_ = [float(line.strip()) for line in f.readlines()]
          if len(image_) == height * width:
            nx,ny = (width,height)
            x_index = np.linspace(0,width-1,width)
            y_index = np.linspace(0,height-1,height)
            xx,yy = np.meshgrid(x_index,y_index)
            xx -= float(width)/2
            yy -= float(height)/2
            xx /= fx
            yy /= fy

            cam_z = np.reshape(image_,(height, width))
            cam_z = cam_z / value_max_range * 1.5
            cam_x = xx * cam_z 
            cam_y = yy * cam_z
            image_z = np.flipud(cam_z)
            image_y = np.flipud(cam_y)
            image_x = np.flipud(cam_x)

            zoom_scale = 0.5
            image_x = scipy.ndimage.zoom(image_x, zoom_scale, order=1)
            image_y = scipy.ndimage.zoom(image_y, zoom_scale, order=1)
            image_z = scipy.ndimage.zoom(image_z, zoom_scale, order=1)
            image = np.dstack((image_x,image_y,image_z))
            return image
    return np.zeros((h,w,3))


def load_flow(top_dir):
  tmp = os.path.join(top_dir,'flow.npz')
  result = np.load(tmp)
  result = result['flow']
  return result


def cal_flow(top_dir,frame2_input_xyz_file, transformation_file, frame1_id_file, frame2_id_file):
  frame1_id_file = load_labeling(frame1_id_file)
  frame2_id_file = load_labeling(frame2_id_file)
  frame1_id = np.squeeze(frame1_id_file)
  frame2_id = np.squeeze(frame2_id_file)
  transl, rot = load_transformation(transformation_file)
  frame2_id_unique = np.unique(frame2_id)
  frame1_id_unique = np.unique(frame1_id)
  pred_frame1_xyz = np.zeros((h,w,3)) 
  frame2_input_xyz = load_xyz(frame2_input_xyz_file)
  for frame_id in frame2_id_unique:
    if frame_id > 0:
       model_id = frame2_id == frame_id
       transl_model = np.mean(transl[model_id],axis=0)
       rot_model = np.mean(rot[model_id],axis=0)
       rot_matrix = angleaxis_rotmatrix(rot_model)
       pred_frame1_model = frame2_input_xyz[model_id]
       pred_frame1_model = rot_matrix.dot(pred_frame1_model.T).T + transl_model - pred_frame1_model
       pred_frame1_xyz[model_id] = pred_frame1_model
  
  pred_frame1_xyz_file = os.path.join(top_dir,'flow.npz')
  np.savez(pred_frame1_xyz_file,flow=pred_frame1_xyz)

  if 0:
    post_p = frame2_input_xyz.reshape((-1,3)) 
    p1 = pred_frame1_xyz.reshape((-1,3)) + post_p
    prev_p = [line for line in os.listdir(top_dir) if line.startswith('frame20') and line.endswith('.pgm')][0]
    prev_p = os.path.join(top_dir,prev_p)
    prev_p = load_xyz(prev_p)
    p2 = prev_p.reshape((-1,3))
    mayalab.points3d(p1[:,0],p1[:,1],p1[:,2],color=(0,1,0),mode='point')
    mayalab.points3d(p2[:,0],p2[:,1],p2[:,2],color=(1,0,0),mode='point')
    mayalab.show()


def raw_cal_flow(total):
  top_dir, frame2_input_xyz_file, frame1_id_file, frame2_id_file = total.split('#')
  cal_flow(top_dir,frame2_input_xyz_file, top_dir, frame1_id_file, frame2_id_file)

def load_predicted_frame1_feat(top_d):
  pred_file = os.path.join(top_d,'pred_frame1_xyz.npz') 
  pred = np.load(pred_file)['flow']
  return pred 

def cal_score(top_dir,inputfilename,gtfilename):
  xyz = load_xyz(inputfilename)[:,:,0:2]
  seg = load_seg(gtfilename)[:,:,0:2]
  score = np.zeros((h,w))
  score_tmp = score.reshape((-1,1))
  xyz_tmp = xyz.reshape((-1,2))
  seg_tmp = seg.reshape((-1,2))
  idx_c = np.unique(seg_tmp,axis=0)
  diff = xyz_tmp - seg_tmp
  diff_norm = np.linalg.norm(diff,axis=1)
  for idx in idx_c:
    if idx[0] != 0.0:
      tmp = np.where(seg_tmp == idx)[0]
      dist = diff_norm[tmp]
      top_k = min(len(dist),300)
      tmp_indx = dist.argsort()[:top_k]
      index = tmp[tmp_indx]
      score_tmp[index] = 1.0
  score = score_tmp.reshape((h,w))
  score_file = os.path.join(top_dir,'frame80_score.npz')
  np.savez(score_file,score=score)


def load_score(score_file):
  tmp = np.load(score_file)['score']
  return tmp


def raw_cal_score(total):
  top_dir,inputfilename, gtfilename = total.split('#')
  cal_score(top_dir,inputfilename,gtfilename)


def cal_boundary(top_dir):
   dist_image = np.zeros((240,320,1))
   transl, rot = load_transformation(top_dir)
   filepath = os.path.join(top_dir,'frame80_labeling.npz')
   if not os.path.exists(filepath):
     return
   seg = load_seg(filepath)
   feat = np.zeros((240,320,3))
   feat = seg
   
   d2_image = np.reshape(feat,(-1,3))
   idx_c = np.unique(d2_image,axis=0)
   idx_c = [idx_c[i] for i in xrange(len(idx_c)) if idx_c[i][0] != 0.0 and idx_c[i][1] != 0.0 and idx_c[i][2] != 0.0]
   print(idx_c)
   d2_list = [i for i in xrange(len(idx_c))]
   if len(idx_c) == 1:  
     dist_image[seg[:,:,2] == idx_c[0][2]] = 0.02
   elif len(idx_c) > 1:
     for i_c in xrange(len(idx_c)):
       dist = np.min(np.array([np.linalg.norm(idx_c[i_c] - idx_c[i]) for i in d2_list if i != i_c]))
       print(dist)
       dist_image[seg[:,:,2] == idx_c[i_c][2]] = dist / 4
   boundary_file = os.path.join(top_dir,'boundary.npz')
   np.savez(boundary_file,boundary=dist_image)


def load_boundary(boundary_file):
  tmp = np.load(boundary_file)['boundary']
  return tmp


def cal_ending_traj(top_dir):
  if not os.path.exists(os.path.join(top_dir,'frame80_labeling_model_id.npz')):
    return
  if not os.path.exists(os.path.join(top_dir,'frame20_labeling_model_id.npz')):
    return
  if not os.path.exists(os.path.join(top_dir,'frame80_labeling.npz')):
    return
  if not os.path.exists(os.path.join(top_dir,'frame20_labeling.npz')):
    return
  start_pos = load_seg(os.path.join(top_dir,'frame80_labeling.npz'))
  end_pos = load_seg(os.path.join(top_dir,'frame20_labeling.npz'))
  start_id = load_labeling(os.path.join(top_dir,'frame80_labeling_model_id.npz'))
  end_id = load_labeling(os.path.join(top_dir,'frame20_labeling_model_id.npz'))
  end_center = np.zeros((240,320,3))
  u_start_id = np.unique(start_id)
  u_end_id = np.unique(end_id)
  for u_i in u_start_id:
    if u_i in u_end_id:
      tmp_e = end_pos[(end_id == u_i)[:,:,0]]
      end_center[(start_id == u_i)[:,:,0]] = np.mean(tmp_e,axis=0)
 
if __name__ == '__main__':
  filelist = []
  top_dir = '/home/linshaonju/interactive-segmentation/Data/BlensorResult_train/'


  if 0:
    filelist = []
    for i in xrange(0,30000):
      top_d = os.path.join(top_dir,str(i))
      transfile = os.path.join(top_d,'translation.npz')
      if os.path.exists(top_d):
        filelist.append(top_d)
    
    pool = Pool(100)
    for i, data in enumerate(pool.imap(cal_transformation,filelist)):
      print(i)
    pool.close()
    pool.join()   


  if 0:
    for i in xrange(0,30000):
      top_d = os.path.join(top_dir,str(i))
      if os.path.exists(top_d):
        frame1_id_file = os.path.join(top_d,'frame20_labeling_model_id.npz')
        frame2_id_file = os.path.join(top_d,'frame80_labeling_model_id.npz')
        frame2_input_xyz_file = [line for line in os.listdir(top_d) if line.startswith('frame80') and line.endswith('.pgm')] 
        if len(frame2_input_xyz_file) > 0:
          frame2_input_xyz_file = frame2_input_xyz_file[0]
          frame2_input_xyz_file = os.path.join(top_d,frame2_input_xyz_file)
          total = top_d + '#' + frame2_input_xyz_file + '#' +frame1_id_file + '#' + frame2_id_file 
          if os.path.exists(frame1_id_file) and os.path.exists(frame2_id_file):
            filelist.append(total)
    pool = Pool(100)
    for i, data in enumerate(pool.imap(raw_cal_flow,filelist)):
      print(i)
 
    pool.close()
    pool.join()
    print("pred scene flow")


  if 0: 
    filelist = []
    for i in xrange(0,30000):
      top_d = os.path.join(top_dir,str(i))
      if os.path.exists(top_d):
        filelist.append(top_d)
    pool = Pool(100)
    for i , data in enumerate(pool.imap(cal_ending_traj,filelist)):
      print(i)
    pool.close()


   
  if 0:
    for i in xrange(0,30000):
      top_d = os.path.join(top_dir,str(i))
      if os.path.exists(top_d):
        frame1_id_file = os.path.join(top_d,'frame20_labeling_model_id.npz')
        frame2_id_file = os.path.join(top_d,'frame80_labeling_model_id.npz')
        frame2_input_xyz_file = [line for line in os.listdir(top_d) if line.startswith('frame80') and line.endswith('.pgm')] 
        if len(frame2_input_xyz_file) > 0:
          frame2_input_xyz_file = frame2_input_xyz_file[0]
          frame2_input_xyz_file = os.path.join(top_d,frame2_input_xyz_file)
          total = top_d + '#' + frame2_input_xyz_file + '#' +frame1_id_file + '#' + frame2_id_file 
          if os.path.exists(frame1_id_file) and os.path.exists(frame2_id_file):
            filelist.append(total)
            #raw_cal_flow(total) 
    pool = Pool(100)
    for i, data in enumerate(pool.imap(raw_cal_flow,filelist)):
      print(i)
 
    pool.close()
    pool.join()
    print("pred scene flow")


  if 0:
    for i in xrange(0,4000):
      top_d = os.path.join(top_dir,str(i))
      if os.path.exists(top_d):
        frame2_input_xyz_file = [line for line in os.listdir(top_d) if line.startswith('frame80') and line.endswith('.pgm')] 
        frame2_gt_file = os.path.join(top_d,'frame80_labeling.npz')
        if len(frame2_input_xyz_file) > 0:
          frame2_input_xyz_file = frame2_input_xyz_file[0]
          frame2_input_xyz_file = os.path.join(top_d,frame2_input_xyz_file)
          total = top_d + '#' + frame2_input_xyz_file + '#' +frame2_gt_file
          print(total)
          filelist.append(total)
          raw_cal_score(total)
    #pool = Pool(10)
    #for i, data in enumerate(pool.imap(raw_cal_score,filelist)):
    #  print(i)
 
    #pool.close()
    #pool.join()


  if 0:
    filelist = []
    for i in xrange(0,10):
      top_d = os.path.join(top_dir,str(i))
      if os.path.exists(top_d):
        if not os.path.exists(os.path.join(top_d,"translation.npz")):
          print(top_d)
        else:
          filelist.append(top_d)
          #cal_boundary(top_d)
 
    pool = Pool(1)
    for i, data in enumerate(pool.imap(cal_boundary,filelist)):
      print(i)
      print(filelist[i])

    pool.close()
    #pool.join() 
