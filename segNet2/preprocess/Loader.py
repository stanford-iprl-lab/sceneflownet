import numpy as np
import scipy.ndimage
import math
import matplotlib.pyplot as plt
from utils import *
from math import *
import os
from mayavi import mlab as mayalab

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


def load_transformation(top_dir):
  pgm_filepath = [line for line in os.listdir(top_dir) if line.endswith('.pgm') and line.startswith('frame80')][0]
  print(pgm_filepath)
  
  tmp = pgm_filepath.split('.pgm')[0].split('_')
  
  azimuth_deg = float(tmp[2].split('azi')[1])
  elevation_deg = float(tmp[3].split('ele')[1]) 
  theta_deg = float(tmp[4].split('theta')[1])
  rho = float(tmp[1].split('rho')[1])
  
  print('azi %f ele %f the %f rho %f' % (azimuth_deg, elevation_deg, theta_deg, rho))
  
  cx, cy, cz = obj_centened_camera_pos(rho, azimuth_deg, elevation_deg)
  q1 = camPosToQuaternion(cx , cy , cz)
  q2 = camRotQuaternion(cx, cy , cz, theta_deg)
  q = quaternionProduct(q2, q1)
  R = quaternion_matrix(q)[0:3,0:3]
  C = np.zeros((3,))
  C[0] = cx
  C[1] = cy
  C[2] = cz
  
  frame2_id = load_labeling(os.path.join(top_dir,'frame80_labeling_model_id.npz')) 
  frame1_id = load_labeling(os.path.join(top_dir,'frame20_labeling_model_id.npz'))   

  frame2_xyz = load_xyz(os.path.join(top_dir,'frame80_rho0.240233_azi11.000000_ele54.641796_theta0.01388500000.pgm'))
  frame1_xyz = load_xyz(os.path.join(top_dir,'frame20_rho0.240233_azi11.000000_ele54.641796_theta0.01388500000.pgm'))
  
  frame2_id_list = np.unique(frame2_id)
  frame1_id_list = np.unique(frame1_id)
  
  plt.figure(0)
  plt.imshow(frame1_id[:,:,0])
  plt.figure(1)
  plt.imshow(frame2_id[:,:,0])
  plt.figure(2)
  plt.imshow(frame1_xyz[:,:,2])
  plt.figure(3)
  plt.imshow(frame2_xyz[:,:,2])
  plt.show()

  model_ids = [line.split('80_')[1] for line in os.listdir(top_dir) if line.endswith('.txt') and line.startswith('frame80')]
  model_ids.sort()
  print(frame2_id_list)
  print(frame1_id_list)

  for instance_id in frame2_id_list:

    if instance_id in frame1_id_list and instance_id > 0:
      #print(instance_id)
      #print(model_ids[int(instance_id)-1])
      frame2_tran, frame2_rot = tran_rot(os.path.join(top_dir,'frame80_'+model_ids[int(instance_id)-1]))             
      frame1_tran, frame1_rot = tran_rot(os.path.join(top_dir,'frame20_'+model_ids[int(instance_id)-1]))
      R12 = frame1_rot.dot(np.linalg.inv(frame2_rot))
      rot = R.T.dot(R12.dot(R))
      tran = R.T.dot(frame1_tran-C) + R.T.dot(R12.dot(C-frame2_tran))
      
      tran[2] *= -1.0
      rot[0,2] *= -1.0 
      rot[1,2] *= -1.0
      rot[2,0] *= -1.0
      rot[2,1] *= -1.0
      #print(np.linalg.det(frame1_rot))
      #print(np.linalg.det(rot))
      #print(np.linalg.det(R))
      #print(np.linalg.det(R1R2T))
      #print(np.linalg.det(frame1_rot))
      
      
      if int(instance_id) == 16:
        frame2_pid = frame2_id == instance_id
        frame2_pid = frame2_pid.reshape((240,320))
        frame2_pid_xyz = frame2_xyz[frame2_pid]
        frame1_pid = frame1_id == instance_id
        frame1_pid = frame1_pid.reshape((240,320))
        frame1_pid_xyz = frame1_xyz[frame1_pid]
        frame21_xyz = rot.dot(frame2_pid_xyz.T).T + tran
        mayalab.points3d(frame21_xyz[:,0],frame21_xyz[:,1],frame21_xyz[:,2])
        mayalab.points3d(frame1_pid_xyz[:,0],frame1_pid_xyz[:,1],frame1_pid_xyz[:,2],color=(1,0,0))
      
        
    if instance_id not in frame1_id_list and instance_id > 0:
      tran = np.zeros((3,))
      rot = np.identity(3)
 

def load_seg(filepath):
  try:
    seg = np.load(filepath)['labeling']
    seg[:,:,2] *= -1.0
  except:
    print(filepath)
    print('sth is wrong!')
    return np.zeros((h,w,3))
  return seg

def load_r(filepath):
   height = h
   width = w
   dist_image = np.zeros((height,width,1))
   seg = np.load(filepath)['labeling']
   d2_image = np.reshape(seg,(-1,3))
   idx_c = np.unique(d2_image,axis=0)
   d2_list = [i for i in xrange(len(idx_c))]
   if len(idx_c) == 1:
     dist_image = np.ones((height,width,1))
   else:
     for i_c in xrange(len(idx_c)):
       dist = np.min(np.array([np.linalg.norm(idx_c[i_c] - idx_c[i]) for i in d2_list if i != i_c]))
       dist_image[seg[:,:,2] == idx_c[i_c][2]] = dist / 4
   return dist_image
 
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

if __name__ == '__main__':
  load_transformation('/Users/lins/interactive-segmentation/segNet2/preprocess/300') 
