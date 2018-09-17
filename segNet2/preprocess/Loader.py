import numpy as np
import scipy.ndimage
import math
from utils import *
from math import *
import os
from scipy.ndimage import imread
from mayavi import mlab as mayalab
import skimage.measure
from multiprocessing import Pool
import shutil
import numpy
from symmetry_issue import *
from quaternionlib import *

np.set_printoptions(precision=4,suppress=True,linewidth=300)

h = 240
w = 320

def quaternion_from_matrix(matrix,isprecise=False):
    M = numpy.array(matrix, dtype=numpy.float64, copy=False)[:4, :4]
    if isprecise:
        q = numpy.empty((4, ))
        t = numpy.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 0, 1, 2
            if M[1, 1] > M[0, 0]:
                i, j, k = 1, 2, 0
            if M[2, 2] > M[i, i]:
                i, j, k = 2, 0, 1
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
            q = q[[3, 0, 1, 2]]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = numpy.array([[m00-m11-m22, 0.0,         0.0,         0.0],
                         [m01+m10,     m11-m00-m22, 0.0,         0.0],
                         [m02+m20,     m12+m21,     m22-m00-m11, 0.0],
                         [m21-m12,     m02-m20,     m10-m01,     m00+m11+m22]])
        K /= 3.0
        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = numpy.linalg.eigh(K)
        q = V[[3, 0, 1, 2], numpy.argmax(w)]
    if q[0] < 0.0:
        numpy.negative(q, q)
    return q


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

fa = open('symmetry_example.txt','a+')

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
  transformation_rot = np.zeros((h,w,4))
  transformation_rot[:,:,0] = 1
  transformation_translation = np.zeros((h,w,3))
  
  symmetry_top_dir = '/home/linshaonju/Symmetry'
  for instance_id in frame2_id_list:
    frame2_pid = frame2_id == instance_id
    frame2_pid = frame2_pid.reshape((240,320))
    frame1_pid = frame1_id == instance_id
    frame1_pid = frame1_pid.reshape((240,320))

    if instance_id > 0:
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
      
      quater = quaternion_from_matrix(rot)

      instance_center = np.mean(frame2_center[frame2_pid],axis=0)
      tran1 = quaternion_rotation(quater,instance_center)

      cate,md5 = model_ids[int(instance_id)-1].split('_')[0:2]
      if cate in cate_symmetry and md5 not in cate_except[cate]:
        symmetry_file = os.path.join(symmetry_top_dir,cate,md5+'.generator')
        if os.path.exists(symmetry_file): 
          symmetry_line = [line for line in open(symmetry_file) if line.startswith('C')]
          if len(symmetry_line) > 0: 
            print(cate+' '+md5)
            print(symmetry_line)    
            for sline in symmetry_line:
              ssline = sline.strip().split() 
              if len(ssline) > 1:
                Cname,Cn,Rx,Ry,Rz = ssline
                Cn = float(Cn)
                Raxis = np.array([float(Rx),float(Ry),float(Rz)]).astype(np.float64)
                Raxis = frame2_rot.dot(Raxis) 
                Raxis = R.T.dot(Raxis)
                Raxis_norm = np.linalg.norm(Raxis)
                Raxis = Raxis / Raxis_norm
                Raxis[2] *= -1.0
                print(Raxis)
                quater,quater_3 = quaternion_shrink(quater,Raxis,Cn)
                if Cn >= 20:
                  print("c20 quater changed!")
                  quater = quater_3 
              else:
                assert 'Cylinder' in ssline
                _, Rc2 = angle_axis_from_quaternion(quater)
                quater,quater_3 = quaternion_shrink(quater,Rc2,2)
      
            tran2 = quaternion_rotation(quater,instance_center)
            tran = tran + tran1 - tran2

            if 0:
              objf20 = frame1_xyz[frame1_pid]
              objf80 = frame2_xyz[frame2_pid]
              p20 = objf20
              p80 = objf80
              if len(p20) > 0:
                p80_n = quaternion_rotation(quater,p80)
                p80_n = p80_n + tran
                mayalab.points3d(p20[:,0],p20[:,1],p20[:,2],color=(0,1,0),mode='sphere')
                mayalab.points3d(p80_n[:,0],p80_n[:,1],p80_n[:,2],color=(0,0,1),mode='sphere')
                mayalab.points3d(p80[:,0],p80[:,1],p80[:,2],color=(1,0,0),mode='sphere') 
                mayalab.show()

      transformation_translation[frame2_pid] = tran
      transformation_rot[frame2_pid] = quater
  
  transformation_file = os.path.join(top_dir,'translation.npz')
  rotation_file = os.path.join(top_dir,'rotation.npz')
  print(transformation_file)
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

def load_transl(filename):
  tmp = np.load(filename)['transl']
  return tmp

def load_rot(filename):
  tmp = np.load(filename)['rot']
  return tmp

def cal_flow(top_dir,frame2_input_xyz_file, transformation_file, frame1_id_file, frame2_id_file):
  frame1_id_file = load_labeling(frame1_id_file)
  frame2_id_file = load_labeling(frame2_id_file)
  frame1_id = np.squeeze(frame1_id_file)
  frame2_id = np.squeeze(frame2_id_file)
  transl = load_transl(os.path.join(transformation_file,'translation.npz'))
  quater = load_rot(os.path.join(transformation_file,'rotation.npz'))
  frame2_id_unique = np.unique(frame2_id)
  frame1_id_unique = np.unique(frame1_id)
  flow = np.zeros((h,w,3)) 
  frame2_input_xyz = load_xyz(frame2_input_xyz_file)

  w1, x1, y1, z1 = quater[:,:,0], quater[:,:,1], quater[:,:,2], quater[:,:,3]#rot_quaternion, axis=-1)
  x2, y2, z2  = frame2_input_xyz[:,:,0],frame2_input_xyz[:,:,1],frame2_input_xyz[:,:,2]

  wm =         - x1 * x2 - y1 * y2 - z1 * z2
  xm = w1 * x2           + y1 * z2 - z1 * y2
  ym = w1 * y2           + z1 * x2 - x1 * z2
  zm = w1 * z2           + x1 * y2 - y1 * x2

  x = -wm * x1 + xm * w1 - ym * z1 + zm * y1
  y = -wm * y1 + ym * w1 - zm * x1 + xm * z1
  z = -wm * z1 + zm * w1 - xm * y1 + ym * x1

  flow = np.stack((x,y,z),axis=-1)
  flow = flow + transl - frame2_input_xyz

  flow_file = os.path.join(top_dir,'flow.npz')
  np.savez(flow_file,flow=flow)

  if 0:
    post_p = frame2_input_xyz.reshape((-1,3)) 
    p1 = flow.reshape((-1,3)) + post_p
    prev_p = [line for line in os.listdir(top_dir) if line.startswith('frame20') and line.endswith('.pgm')][0]
    prev_p = os.path.join(top_dir,prev_p)
    prev_p = load_xyz(prev_p)
    p2 = prev_p.reshape((-1,3))
    mayalab.points3d(p1[:,0],p1[:,1],p1[:,2],color=(0,1,0),mode='point')
    mayalab.points3d(p2[:,0],p2[:,1],p2[:,2],color=(0,0,1),mode='point')
    mayalab.show()


def raw_cal_flow(total):
  top_dir, frame2_input_xyz_file, frame1_id_file, frame2_id_file = total.split('#')
  cal_flow(top_dir,frame2_input_xyz_file, top_dir, frame1_id_file, frame2_id_file)


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
  score_file = os.path.join(top_dir,'frame20_score.npz')
  np.savez(score_file,score=score)


def load_score(score_file):
  tmp = np.load(score_file)['score']
  return tmp

def raw_cal_score(total):
  top_dir,inputfilename, gtfilename = total.split('#')
  cal_score(top_dir,inputfilename,gtfilename)

def cal_boundary(top_dir):
   dist_image = np.zeros((240,320,1))
   filepath = os.path.join(top_dir,'frame80_labeling.npz')
   if not os.path.exists(filepath):
     return
   if not os.path.exists(os.path.join(top_dir,'end_center.npz')):
     return
   seg = load_seg(filepath)
   end_center = np.load(os.path.join(top_dir,'end_center.npz'))['end_center']
   feat = np.zeros((240,320,6))
   feat[:,:,0:3] = seg
   feat[:,:,3:6] = end_center

   d2_image = np.reshape(feat,(-1,6))
   idx_c = np.unique(d2_image,axis=0)
   idx_c = [idx_c[i] for i in xrange(len(idx_c)) if idx_c[i][0] != 0.0 and idx_c[i][1] != 0.0 and idx_c[i][2] != 0.0]
   d2_list = [i for i in xrange(len(idx_c))]
   if len(idx_c) == 1:  
     dist_image[seg[:,:,2] == idx_c[0][2]] = 0.02
   elif len(idx_c) > 1:
     for i_c in xrange(len(idx_c)):
       dist = np.min(np.array([np.linalg.norm(idx_c[i_c] - idx_c[i]) for i in d2_list if i != i_c]))
       dist_image[seg[:,:,2] == idx_c[i_c][2]] = dist / 10
   boundary_file = os.path.join(top_dir,'boundary.npz')
   np.savez(boundary_file,boundary=dist_image)

cateid_cate = {'02876657':1, # bottle
               '02691156':2, # toy airplane
               '02747177':3, # trash can
               '02773838':4, # bag 
               '02808440':5, # bowl
               '02924116':6, # toy bus
               '02942699':7, # camera
               '02946921':8, # can
               '02954340':9, # cap
               '02958343':10,# toy car
               '03001627':11,# toy chair
               '03046257':12,#clocks
               '03085013':13,#key boards
               '03211117':14,#display
               '03261776':15,#earphone
               '03624134':16,#knife
               '03642806':17,#laptop
               '03790512':18,#toy motorcycle
               '03797390':19,#mug
               '03948459':20,#pistol
               '04074963':21,#remote control
               '04401088':22,#telephone
               '04530566':23,#toy boat
               '04468005':24,#toy train
               '04099429':25,#toy rocket
               '04256520':26,#toy sofa
               '03513137':27,#helmet
               '04379243':28,#toy table
               }

def load_boundary(boundary_file):
  tmp = np.load(boundary_file)['boundary']
  return tmp

def cal_ending_traj(top_dir):
  frame2_center = load_seg(os.path.join(top_dir,'frame80_labeling.npz'))
  frame1_center = load_seg(os.path.join(top_dir,'frame20_labeling.npz'))

  frame2_id = load_labeling(os.path.join(top_dir,'frame80_labeling_model_id.npz')) 
  frame1_id = load_labeling(os.path.join(top_dir,'frame20_labeling_model_id.npz'))   

  end_center = np.zeros((240,320,3))

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

  frame2_xyz_name = [line for line in os.listdir(top_dir) if line.startswith('frame80') and line.endswith('.pgm')][0]
  frame1_xyz_name = [line for line in os.listdir(top_dir) if line.startswith('frame20') and line.endswith('.pgm')][0]
  frame2_xyz = load_xyz(os.path.join(top_dir,frame2_xyz_name))
  frame1_xyz = load_xyz(os.path.join(top_dir,frame1_xyz_name))
  
  frame2_id_list = np.unique(frame2_id)
  frame1_id_list = np.unique(frame1_id)

  model_ids = [line.split('frame80_')[1] for line in os.listdir(top_dir) if line.endswith('.txt') and line.startswith('frame80')]

  model_ids.sort()

  for instance_id in frame2_id_list:
    frame2_pid = frame2_id == instance_id
    frame2_pid = frame2_pid.reshape((240,320))
    frame1_pid = frame1_id == instance_id
    frame1_pid = frame1_pid.reshape((240,320))

    if instance_id > 0:
      if instance_id not in frame1_id_list:
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

        tmp_e = np.mean(frame2_center[frame2_pid],axis=0)
        end_center[frame2_pid] = rot.dot(tmp_e) + tran
      else:
        tmp_e = np.mean(frame1_center[frame1_pid],axis=0)
        end_center[frame2_pid] = tmp_e

  ending_traj_file = os.path.join(top_dir,'end_center.npz')
  np.savez(ending_traj_file,end_center=end_center)

def load_end_center(end_center_file):
  tmp = np.load(end_center_file)['end_center']
  return tmp  

def cal_rigidflowmask(top_dir):
  frame2_id = load_labeling(os.path.join(top_dir,'frame80_labeling_model_id.npz'))
  frame1_id = load_labeling(os.path.join(top_dir,'frame20_labeling_model_id.npz'))
  mask = np.zeros((240,320))
  
  frame2_id_list = np.unique(frame2_id)
  frame1_id_list = np.unique(frame1_id)
  count = 0
  for instance_id in frame2_id_list:
    if instance_id in frame1_id_list and instance_id > 0:
       frame2_pid = frame2_id == instance_id
       frame2_pid = frame2_pid.reshape((240,320))
       mask[frame2_pid] = 1.0#instance_id
    elif not instance_id in frame1_id_list:
       count += 1
  if count > 2:
    print(count)
    print(top_dir)
    shutil.rmtree(top_dir)
    return 1
  else:
    np.savez(os.path.join(top_dir,'rigidflowmask.npz'),rigidflowmask=mask)
    return 0

def load_rigidflowmask(top_dir):
  tmp = np.load(os.path.join(top_dir,'rigidflowmask.npz'))['rigidflowmask']
  return tmp

if __name__ == '__main__':
  "Annotate the ground truth dataset. Follow the order
   Step 1. Calculate translation and rotation
   Step 2. Calculate the trajectory ending point
   Step 3. Calculate the distances between trajectories and score"

  top_dir = ''
  
  num = 10000
  if 1:
    filelist = []
    for i in xrange(0,num):
      top_d = os.path.join(top_dir,str(i))
      if os.path.exists(top_d):
        if not os.path.exists(os.path.join(top_d,'frame80_labeling_model_id.npz')) or not os.path.exists(os.path.join(top_d,'frame20_labeling_model_id.npz')) or not os.path.exists(os.path.join(top_d,'frame80_labeling.npz')) or not os.path.exists(os.path.join(top_d,'frame20_labeling.npz')):
          pass
        else:
          filelist.append(top_d)
        flow_file = os.path.join(top_d,'rotation.npz')
        print(flow_file)
        flow = load_rot(flow_file) 

  if 0:
    filelist = []
    for i in xrange(0,num):
      top_d = os.path.join(top_dir,str(i))
      if os.path.exists(top_d):
        filelist.append(top_d)

    pool = Pool(100)
    for i, data in enumerate(pool.imap(cal_transformation,filelist)):
      print(i)
    pool.close()

  if 0:
    filelist = []
    for i in xrange(0,num):
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
    pool = Pool(150)
    for i, data in enumerate(pool.imap(raw_cal_flow,filelist)):
      print(i)
 
    pool.close()
    print("pred scene flow")
  
  if 0: 
    filelist = []
    for i in xrange(0,num):
      top_d = os.path.join(top_dir,str(i))
      if os.path.exists(top_d):
        print(top_d)
        filelist.append(top_d)
    pool = Pool(150)
    for i , data in enumerate(pool.imap(cal_ending_traj,filelist)):
      print(i)
    pool.close()
    print("cal ending traj")

  if 0:
    filelist=[]
    for i in xrange(0,num):
      top_d = os.path.join(top_dir,str(i))
      if os.path.exists(top_d):
        frame2_input_xyz_file = [line for line in os.listdir(top_d) if line.startswith('frame20') and line.endswith('.pgm')] 
        frame2_gt_file = os.path.join(top_d,'frame20_labeling.npz')
        if len(frame2_input_xyz_file) > 0:
          frame2_input_xyz_file = frame2_input_xyz_file[0]
          frame2_input_xyz_file = os.path.join(top_d,frame2_input_xyz_file)
          total = top_d + '#' + frame2_input_xyz_file + '#' +frame2_gt_file
          print(total)
          filelist.append(total)
    pool = Pool(100)
    for i, data in enumerate(pool.imap(raw_cal_score,filelist)):
      print(i) 
    pool.close()

  if 0:
    filelist = []
    for i in xrange(0,num):
      top_d = os.path.join(top_dir,str(i))
      if os.path.exists(top_d):
        filelist.append(top_d)
    pool = Pool(150)
    for i, data in enumerate(pool.imap(cal_boundary,filelist)):
      print(i)
      print(filelist[i])

    pool.close()
