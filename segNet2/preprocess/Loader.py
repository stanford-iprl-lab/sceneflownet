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

cate_axis = ['02876657',\
             #bottle
            '02747177',\
             #trash can
            '02808440',\
             #bowl
            '02946921',\
             #can
            '04099429',\
             #toy rocket
            '04379243',\
             ]

cate_except = {}
cate_except['04379243'] = []
cate_except['02946921'] =[\
'baaa4b9538caa7f06e20028ed3cb196e',\
'bf974687b678c66e93fb5c975e8de2b7',\
'3a7d8f866de1890bab97e834e9ba876c',\
'343287cd508a798d38df439574e01b2',\
'38dd2a8d2c984e2b6c1cd53dbc9f7b8e',\
'3c8af6b0aeaf13c2abf4b6b757f4f768',\
'5bd768cde93ec1acabe235874aea9b9b',\
'4a6ba57aa2b47dfade1831cbcbd278d4',\
'2eeefdfc9b70b89eeb153e9a37e99fa5',\
'4cc3601af4a09418b459058f42771eff',\
'90d40359197c648b23e7e4bd2944793',\
'd052c17866cf5cf8387e8ce4aad01a52',\
'10c9a321485711a88051229d056d81db',\
'129880fda38f3f2ba1ab68e159bfb347',\
'f4108f92f3f12f99e3ecb6fd6ed1dd90',\
'f4ad0b7f82c36051f51f77a6d7299806',\
'295be2a01cb9f29b716714dd1fd945b7',\
'85fa7911905e932bf485d100eb31d589',\
'91483776d1930de7515bc9246d80fdcc',\
'd3e24e7712e1e82dece466fd8a3f2b40',\
'990a058fbb51c655d773a8448a79e14c',\
'd3e24e7712e1e82dece466fd8a3f2b40',\
'91483776d1930de7515bc9246d80fdcc',\
'fe6be0860c63aa1d8b2bf9f4ef8234',\
'4d4fc73864844dad1ceb7b8cc3792fd',\
'9effd38015b7e5ecc34b900bb2492e',\
'788094fbf1a523f768104c9df46104ca',\
'a70947df1f1490c2a81ec39fd9664e9b',\
'7b643c8136a720d9db4a36333be9155',\
'29bc4b2e86b91d392e06d87a0fadf00',\
'dc815e056c71e2ed7c8ed5da8582ce91',\
'a5bab9546d6a1baa33ff264b2ec3aaa9',\
'203c5e929d588d07c6754428123c8a7b',\
'b1980d6743b7a98c12a47018402419a2',\
'fac6341f9e5bfddaf5aaab5ed17143d6',\
'a087f6b5ea424ccc785f06f424b9d06',\
'2b08d2c26d1fce3afe34061aca66f702',\
'28c17225887339bd6193d9e76bb15876',\
'f6316c6702c49126193d9e76bb15876',\
'6b2c6961ad0891936193d9e76bb15876',\
'bea7315d4410d0ce83b1cdcee646c9a4',\
]

cate_except['02808440']=[\
'4eefe941048189bdb8046e84ebdc62d2',\
'd28f7a7a8fbc5fc925b5a13384fa548b',\
'ce48ffb418b99996912a38ce5826ebb8',\
]

cate_except['02747177']=[\
'fbf7021503a2a11fce41b639931f9ca1',\
'7a73f3cf362ef4fa619b1cc3b6756f94',\
]

cate_except['04099429']=[\
'3e34a0b61d3678f9646343ecf02768e7',\
'bb07f8aea99dd2cd533e0927d26599e2',\
]

cate_except['02876657'] = [\
'e101cc44ead036294bc79c881a0e818b',\
'9f2bb4a157164af19a7c9976093a710d',\
'908e85e13c6fbde0a1ca08763d503f0e',\
'1ef68777bfdb7d6ba7a07ee616e34cd7',\
'3c6d6ff143483efaebb19cf38af396e6',\
'd8b6c270d29c58c55627157b31e16dc2',\
'62451f0ab130709ef7480cb1ee830fb9',\
'fa44223c6f785c60e71da2487cb2ee5b',\
'a86d587f38569fdf394a7890920ef7fd',\
'3dbd66422997d234b811ffed11682339',\
'5ad47181a9026fc728cc22dce7529b69',\
'd297d1b0e4f0c244f61150ce90be197a',\
'621e786d6343d3aa2c96718b14a4add9',\
'af3dda1cfe61d0fc9403b0d0536a04af',\
'3f91158956ad7db0322747720d7d37e8',\
'dc0926ce09d6ce78eb8e919b102c6c08',\
'd9aee510fd5e8afb93fb5c975e8de2b7',\
'e8b48d395d3d8744e53e6e0633163da8',\
'8309e710832c07f91082f2ea630bf69e',\
'799397068de1ae1c4587d6a85176d7a0',\
'81b2ce7d719326c8fd54367fe37b16',\
'f83c3b75f637241aebe67d9b32c3ddf8',\
'b45d6cadaba5b6a72d20e9f11baa5f8f',\
'47ede0c10e36fe309029fc9335eeb05c',\
'831918158307c1eef4757ae525403621',\
'f4851a2835228377e101b7546e3ee8a7',\
'ab6792cddc7c4c83afbf338b16b43f53',\
'd74bc917899133e080c257afea181fa2',\
'7980922e83b5461febe67d9b32c3ddf8',\
'8a23e8ae357fa2b71920da6870de352',\
'c771267feb7ee16761d12ece735ab44',\
'158634b1d7d010eeebe67d9b32c3ddf8',\
'9012b03ddb6d9a3dfbe67b89c7bdca4f',\
'523cddb320608c09a37f3fc191551700',\
'32074e5642bad0e12c16495e79df12c1',\
'8a980192662f95b84f42eadbd9a0b820',\
'22d18e34097ec57a80b49bbcfa357c86',\
'9f50b2ddbcc2141cfa20324e30e0bf40',\
'ee74f5bfb0d7c8a5bd288303be3d57e7',\
'c13219fac28e722edd6a2f6a8ecad52d',\
'c4b6121d162a3cb216ae07d515c8c56e',\
'b1d75ad18d986ec760005b40a079e2d3',\
'7778c06ab2af1121b4bfcf9b3e6ed915',\
'7b1fc86844257f8fa54fd40ef3a8dfd0',\
'aec6aa917d61d1faebe67d9b32c3ddf8',\
'5566f264a6fa08cd2a68e506fbd6eecf',\
'6ebe74793197919e93f2361527e0abe5',\
'cb3ff04a607ea9651b22d29e47ec3f2',\
'545225eda0e900f6d4a7a4c7d6e9bdc3',\
'26e6f23bf6baea05fe5c8ffd0f5eba47',\
'4185c4eb651bd7e03c752b66cc923fdb',\
'4301fe10764677dcdf0266d76aa42ba',\
'24feb92770933b1663995fb119e59971',\
'cc399bb619ddddf7c13f8623d10d3404',\
'6da7fa9722b2a12d195232a03d04563a',\
'6da7fa9722b2a12d195232a03d04563a',\
'56c23ba1699f6294435b5a0263ddd2e2',\
'77699e08e3824eab47765668cf4023ed',\
'134c723696216addedee8d59893c8633',\
'4d4fc73864844dad1ceb7b8cc3792fd',\
'634c59bf37676ca64c3a35cee92bb95b',\
'40e5d2c6e9e9cbbf5cafd3b1501bc74',\
'bcbc2c637bed89e4d1b69ad96276e132',\
'5979870763de5ced4c8b72e8da0e65c5',\
'5979870763de5ced4c8b72e8da0e65c5',\
'ca210c6696357f98e8ec08b84f068b50',\
'e56e77c6eb21d9bdf577ff4de1ac394c',\
'29b6f9c7ae76847e763c517ce709a8cc',\
'58279e870f4aec6963b58f539d58d6d5',\
'ed55f39e04668bf9837048966ef3fcb9',\
'e90e0506fd00fe93f42d6bd378df1c70',\
'541c4dc1d1b42978648ed39e926e682f',\
'b38eb308649c8c5de7de92adde5735ef',\
'412d5f41b5fda0b271dbcf1c4061c69b',\
'216adefa94f25d7968a3710932407607',\
'48202d357e79315e45891653421dc140',\
'3e7d7a9637e385f2fd1efdcc788bb066',\
'c3767df815e0e43e4c3a35cee92bb95b',\
'c46bfae78beaa4a7988abef1fd117e7',\
'59763dfe99084f0440ba17ccf542984c',\
'5561fe6ad83a5000fc0eb7bdb358f874',\
'3b0e35ff08f09a85f0d11ae402ef940e',\
'ff13595434879bba557ef92e2fa0ccb2',\
'73632ddb4a5684503594b3be653e6bff',\
'33f6ca7ec9d3f1e6940806ade53ef2f',\
'77be452b59bebfd3940806ade53ef2f',\
'b7ffc4d34ffbd449940806ade53ef2f',\
'77c9a8391c708ae5940806ade53ef2f',\
'9a777a5f3701fb6b940806ade53ef2f',\
'dc005c019fbfb32c90071898148dca0e',\
]


def load_transformation(top_dir):
  transl_file = os.path.join(top_dir,'translation.npz')#[line for line in os.listdir(top_dir) if line.endswith('_transl.npz')][0]
  rot_file = os.path.join(top_dir,'rotation.npz')#[line for line in os.listdir(top_dir) if line.endswith('_rot.npz')][0]
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
  #print(top_dir) 

  if 0:
    plt.figure(0)
    plt.imshow(frame1_xyz[:,:,2])
    plt.figure(1)
    plt.imshow(frame2_xyz[:,:,2])
    plt.show()

  model_ids = [line.split('frame80_')[1] for line in os.listdir(top_dir) if line.endswith('.txt') and line.startswith('frame80')]

  model_ids.sort()
  transformation_rot = np.zeros((h,w,3))
  transformation_translation = np.zeros((h,w,3))

  symmetry_top_dir = '/home/linshaonju/Symmetry'

  for instance_id in frame2_id_list:
    frame2_pid = frame2_id == instance_id
    frame2_pid = frame2_pid.reshape((240,320))
    frame1_pid = frame1_id == instance_id
    frame1_pid = frame1_pid.reshape((240,320))
    if instance_id > 0: 
      if instance_id in frame1_id_list:
        frame2_tran, frame2_rot = tran_rot(os.path.join(top_dir,'frame80_'+model_ids[int(instance_id)-1]))             
        frame1_tran, frame1_rot = tran_rot(os.path.join(top_dir,'frame20_'+model_ids[int(instance_id)-1]))
        cate_id, md5 = model_ids[int(instance_id)-1].split('_')[0:2]
        symmetry_file = os.path.join(symmetry_top_dir,cate_id,md5+'.generator') 
        symmetry_lines = []
        symmetry_lines_20 = []
        if os.path.exists(symmetry_file):
          symmetry_lines = [line for line in open(symmetry_file) if line.startswith('C')] 
          symmetry_types = []
          c_types_list =[]
          c_vectors_list = []
          for line in symmetry_lines:
            line_tmp = line.strip().split()
            c_types = int(line_tmp[1])
            c_vectors = np.array([float(line_tmp[2]),float(line_tmp[3]),float(line_tmp[4])])
            c_types_list.append(c_types)
            c_vectors_list.append(c_vectors)

 
        if cate_id in cate_axis and md5 not in cate_except[cate_id] and len(symmetry_lines) > 0:
          print(model_ids[int(instance_id)-1])
          print(symmetry_lines)       
          frame2_tran, frame2_rot = tran_rot(os.path.join(top_dir,'frame80_'+model_ids[int(instance_id)-1]))             
          frame1_tran, frame1_rot = tran_rot(os.path.join(top_dir,'frame20_'+model_ids[int(instance_id)-1]))
          R12 = frame1_rot.dot(np.linalg.inv(frame2_rot))
          angle_axis = rotmatrix_angleaxis(R12)
          deter = np.cbrt(np.linalg.det(frame2_rot))
          frame2_rot_norm = frame2_rot / deter  
 
          for idx in xrange(len(c_types_list)):
            if c_types_list[idx] >= 10:
              c_vector = frame2_rot_norm.dot(c_vectors_list[idx])
              c_vector /= np.linalg.norm(c_vector)
              angle_axis = angle_axis - np.dot(angle_axis,c_vector)*c_vector
            else:
              c_interval = np.pi * 2 / float(c_types_list[idx])
              c_vector = frame2_rot_norm.dot(c_vectors_list[idx])
              c_vector /= np.linalg.norm(c_vector)
              current_angle = np.dot(angle_axis,c_vector) 
              if abs(current_angle) > c_interval / 2:
                angle_diff = math.floor(current_angle / c_interval) 
                angle_axis = angle_axis - angle_diff * c_vector

          #frame2_vector = frame2_rot[:,1]
          #frame1_vector = frame1_rot[:,1]
          #frame2_vector /= (np.linalg.norm(frame2_vector)+0.000001)
          #frame1_vector /= (np.linalg.norm(frame1_vector)+0.000001)
          #Raxis = np.cross(frame2_vector,frame1_vector)
          #Raxis = Raxis / (np.linalg.norm(Raxis) + 0.000001)
          #angle = np.arccos(np.dot(frame2_vector,frame1_vector))
          #angle_axis = angle * Raxis
          R12 = angleaxis_rotmatrix(angle_axis)
          rot = R.T.dot(R12.dot(R)) 
          tran = R.T.dot(frame1_tran-C) + R.T.dot(R12.dot(C-frame2_tran))
        else:
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
      else:        
        tran = np.zeros((3,))
        rot = np.identity(3)
    
      angle_axis = rotmatrix_angleaxis(rot)
        
          
      transformation_translation[frame2_pid] = tran
      transformation_rot[frame2_pid] = angle_axis #rot.reshape((9))
  transformation_file = os.path.join(top_dir,'translation.npz')
  rotation_file = os.path.join(top_dir,'rotation.npz')
  print(transformation_file)
  print(rotation_file)
  np.savez(transformation_file,transl=transformation_translation)
  np.savez(rotation_file,rot=transformation_rot)
  print("finish")
  return "good"


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

def load_cc(cc_file):
  tmp = np.load(cc_file)['cc']
  return tmp

def cal_cc(frame1_id_file,frame2_id_file,cc_file, rad=10):
  frame1_id = np.squeeze(frame1_id_file)
  frame2_id = np.squeeze(frame2_id_file)
  frame1_tmp = skimage.measure.block_reduce(frame1_id, (8,8), np.max)
  frame2_tmp = skimage.measure.block_reduce(frame2_id,(8,8), np.max)

  dia = rad * 2 + 1
  cc_value = np.zeros((30,40,dia ** 2))
  for i in xrange(30):
    for j in xrange(40):
      if frame2_tmp[i,j] > 0:
        for i_shift in xrange(-rad,rad+1):
          for j_shift in xrange(-rad,rad+1):
            i_idx = i + i_shift
            j_idx = j + j_shift
            if i_idx >= 0 and i_idx < 30 and j_idx >= 0 and j_idx < 40:
              ch = (i_shift + rad) * dia + (j_shift+rad)
              if frame1_tmp[i_idx,j_idx] == frame2_tmp[i,j]:
                cc_value[i,j,ch] = 1.0
          
  if 0:   
    plt.figure(0)
    plt.imshow(frame1_tmp)
    plt.figure(1)
    plt.imshow(frame2_tmp)
    plt.figure(2)
    plt.imshow(cc_value[:,:,221])
    plt.show()

  np.savez(cc_file,cc=cc_value)

def raw_cal_cc(total):
  frame1_id_file, frame2_id_file, cc_file = total.split('#')
  frame1_id = load_labeling(frame1_id_file)
  frame2_id = load_labeling(frame2_id_file)
  cal_cc(frame1_id,frame2_id,cc_file)
  return "good"

def load_predicted_frame1_feat(top_dir):
  tmp = os.path.join(top_dir,'pred_frame1_xyz.npz')
  result = np.load(tmp)
  result = result['flow']
  return result

def cal_predicted_frame1_feat(top_dir,frame2_input_xyz_file, transformation_file, frame1_id_file, frame2_id_file):
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
    if frame_id > 0:# and frame_id in frame1_id_unique:
       model_id = frame2_id == frame_id
       transl_model = np.mean(transl[model_id],axis=0)
       rot_model = np.mean(rot[model_id],axis=0)
       rot_matrix = angleaxis_rotmatrix(rot_model)
       pred_frame1_model = frame2_input_xyz[model_id]
       pred_frame1_model = rot_matrix.dot(pred_frame1_model.T).T + transl_model
       pred_frame1_xyz[model_id] = pred_frame1_model
  #return pred_frame1_xyz
  pred_frame1_xyz_file = os.path.join(top_dir,'pred_frame1_xyz.npz')
  np.savez(pred_frame1_xyz_file,flow=pred_frame1_xyz)

def raw_cal_pred_frame1_xyz(total):
  top_dir, frame2_input_xyz_file, frame1_id_file, frame2_id_file = total.split('#')
  cal_predicted_frame1_feat(top_dir,frame2_input_xyz_file, top_dir, frame1_id_file, frame2_id_file)


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
   feat = np.zeros((240,320,6))
   feat[:,:,0:3] = seg
   seg_tmp = seg
   seg_tmp_ = np.reshape(seg_tmp,(-1,3))
   seg_uni = np.unique(seg_tmp_,axis=0)
   for i in xrange(len(seg_uni)):
     inds = feat[:,:,0:3] == seg_uni[i]
     rot_tmp = rot[inds].reshape(-1,3)
     rot_mat = angleaxis_rotmatrix(rot_tmp[0])
     transl_tmp = transl[inds].reshape(-1,3)[0]
     seg_single = seg[inds]
     seg_single = seg_single.reshape(-1,3)[0]
     new_xyz = rot_mat.dot(seg_single) + transl_tmp
     feat[:,:,3:6][seg[:,:,2] == seg_single[2]] = new_xyz

   d2_image = np.reshape(feat,(-1,9))
   idx_c = np.unique(d2_image,axis=0)
   idx_c = [idx_c[i] for i in xrange(len(idx_c)) if idx_c[i][0] != 0.0 and idx_c[i][1] != 0.0 and idx_c[i][2] != 0.0]
   d2_list = [i for i in xrange(len(idx_c))]
   if len(idx_c) == 1:
     dist_image = np.zeros((240,320,1))
     dist_image[seg[:,:,2] == idx_c[0][2]] = 0.07
   else:
     for i_c in xrange(len(idx_c)):
       dist = np.min(np.array([np.linalg.norm(idx_c[i_c] - idx_c[i]) for i in d2_list if i != i_c]))
       dist_image[seg[:,:,2] == idx_c[i_c][2]] = dist / 10
   boundary_file = os.path.join(top_dir,'boundary.npz')
   np.savez(boundary_file,boundary=dist_image)

def load_boundary(boundary_file):
  tmp = np.load(boundary_file)['boundary']
  return tmp

if __name__ == '__main__':
  filelist = []
  top_dir = '/home/linshaonju/interactive-segmentation/Data/BlensorResult_train/'
  
  cc_flag = False
  if cc_flag:
    for i in xrange(0,4000):
      top_d = os.path.join(top_dir,str(i))
      cc_file = os.path.join(top_d,'cc.npz')
      frame1_id_file = os.path.join(top_d,'frame20_labeling_model_id.npz')
      frame2_id_file = os.path.join(top_d,'frame80_labeling_model_id.npz')
      total = frame1_id_file + '#' + frame2_id_file + '#' + cc_file
      if os.path.exists(frame1_id_file) and os.path.exists(frame2_id_file):
        filelist.append(total)
        print(total)
 
    pool = Pool(200)
    for i, data in enumerate(pool.imap(raw_cal_cc,filelist)):
      print(i)
 
    pool.close()
    pool.join()


  if 0:
    filelist = []
    for i in xrange(0,8500):
      top_d = os.path.join(top_dir,str(i))
      transfile = os.path.join(top_d,'translation.npz')
      if os.path.exists(top_d):
        filelist.append(top_d)
      print(top_d)
  
    pool = Pool(100)
    for i, data in enumerate(pool.imap(cal_transformation,filelist)):
      print(i)

    pool.close()
    pool.join()   

  if 0:
    for i in xrange(0,8500):
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
 
    pool = Pool(200)
    for i, data in enumerate(pool.imap(raw_cal_pred_frame1_xyz,filelist)):
      print(i)
 
    pool.close()
    pool.join()
    print("pred scene flow")
 
  if 0:
    for i in xrange(2000,4000):
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
 
    pool = Pool(100)
    for i, data in enumerate(pool.imap(raw_cal_score,filelist)):
      print(i)
 
    pool.close()
    pool.join()

  if 1:
    filelist = []
    for i in xrange(0,30000):
      top_d = os.path.join(top_dir,str(i))
      if os.path.exists(top_d):
        if not os.path.exists(os.path.join(top_d,"translation.npz")):
          print(top_d)
        else:
          filelist.append(top_d)
  
 
    pool = Pool(200)
    for i, data in enumerate(pool.imap(cal_boundary,filelist)):
      print(i)
      print(filelist[i])

    pool.close()
    pool.join()   


