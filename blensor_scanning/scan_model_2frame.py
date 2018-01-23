# data distribution is specified by a distribution file containing samples
import bpy
import math
import sys
import os
import numpy as np
import random
import struct
from numpy.linalg import inv
from math import *
import mathutils
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from utils import *
import blensor
from random import uniform

"""---------- main -----------"""
model_top_dir = '/home/linshaonju/GraspNet3.0/Data/ShapenetManifold'
outImagePath = sys.argv[-1]
modelids = sys.argv[-2]

import sys
import shutil

model_path_lists = []
model_ids = modelids.split('#')[:-1]
for line in model_ids:
  cate, model_id = line.split('_')
  model_path_lists.append(os.path.join(model_top_dir,cate,model_id,'model.obj'))

if not os.path.exists(outImagePath):
  os.mkdir(outImagePath)
else:
  shutil.rmtree(outImagePath) 
  os.mkdir(outImagePath)

#table_path = os.path.join(BASE_DIR,'table.obj')
#bpy.ops.import_scene.obj(filepath=table_path)

for modelPath in model_path_lists:
  bpy.ops.import_scene.obj(filepath=modelPath)

count = 2
for c in range(0,count):
  x = 0.0 #uniform(-1.0,1.0)
  y = 0.0 #uniform(-1.0,1.0)
  z = 10.0 #uniform(1,3)
  bpy.ops.mesh.primitive_cube_add(location=(x,y,z),radius=1)

#floor = bpy.data.objects['mesh1_mesh1-geometry_table']
#floor.scale = (100,100,0.001)
#floor.location = (0,0,-0.1-floor.dimensions.y/2)
#floor.location = (0,0,-0.1)
# set lights
bpy.ops.object.select_all(action='DESELECT')
candidate_list = [item.name for item in bpy.data.objects if item.type == "MESH"]

for object_name in candidate_list:
  if object_name != 'mesh1_mesh1-geometry_table' and object_name != 'Cube':
    bpy.data.objects[object_name].select = True

bpy.ops.rigidbody.objects_add(type='ACTIVE')

candidate_list = [item.name for item in bpy.data.objects if item.type == "MESH"]
for object_name in candidate_list:
  if object_name != 'Cube' and object_name != 'mesh1_mesh1-geometry_table':
    bpy.data.objects[object_name].rigid_body.linear_damping = 0.9
    bpy.data.objects[object_name].rigid_body.angular_damping = 0.8
    bpy.data.objects[object_name].rigid_body.friction = 1.0
    modelscale = np.random.uniform(0.07,0.3)
    bpy.data.objects[object_name].scale = (modelscale,modelscale,modelscale)
    bpy.data.objects[object_name].rigid_body.collision_margin=0.0
    bpy.data.objects[object_name].rigid_body.deactivate_linear_velocity = 0.04
    bpy.data.objects[object_name].rigid_body.deactivate_angular_velocity = 0.05

camObj = bpy.data.objects['Camera']

bpy.data.objects['Cube'].location=(0,0,-0.2)
bpy.data.objects['Cube'].scale = (1000,1000,0.0001)
#bpy.ops.import_scene.obj(filepath=table_path)
#table = bpy.data.objects['mesh1_mesh1-geometry_table']
#floor.scale = (100,100,0.001)
#table.location = (0,0,-0.2-table.dimensions.y/2)
#floor.location = (0,0,-0.1)
#table.scale = (100000,1000000,1)
#bpy.ops.object.select_all(action='DESELECT')
#for object_name in candidate_list:
#  if object_name == 'Cube':
#    bpy.data.objects[object_name].select = True
#  if object_name == 'mesh1_mesh1-geometry_table':
#    bpy.data.objects[object_name].select = True
#bpy.ops.rigidbody.objects_add(type='PASSIVE')

bpy.ops.object.select_all(action='DESELECT')
candidate_list = [item.name for item in bpy.data.objects if item.type== "MESH"]
for object_name in candidate_list:
  if object_name == 'Cube':
    bpy.data.objects[object_name].select = True
  if object_name == 'mesh1_mesh1-geometry_table':
    bpy.data.objects[object_name].select = True#
bpy.ops.rigidbody.objects_add(type='PASSIVE')   
bpy.data.objects['Cube'].rigid_body.collision_shape='MESH'
#bpy.data.objects['Cube'].rigid_body.linear_damping = 1.0
#bpy.data.objects['Cube'].rigid_body.angular_damping = 1.0
bpy.data.objects['Cube'].rigid_body.friction = 1.0
bpy.data.objects['Cube'].rigid_body.collision_margin = 0.0
bpy.data.objects['Cube.001'].location = (0,0,-1000)
#bpy.ops.object.select_all(action='TOGGLE')
#if 'Cube' in list(bpy.data.objects.keys()):
#  bpy.data.objects['Cube'].select = True
#if 'Cube.001' in list(bpy.data.objects.keys()):
#  bpy.data.objects['Cube.001'].select = True
#bpy.ops.object.delete()

view_params = []
for i in range(1):
  tilt_deg = np.random.uniform(-1,1) #np.random.normal(0, 4)
  dist = np.random.uniform(0.2, 0.8) 
  elevation_deg = np.random.uniform(20.0, 90.0)
  azimuth_deg = (i) * 30.0 + np.random.uniform(-15,15)
  if azimuth_deg < 0:
    azimuth_deg = 360 + azimuth_deg
  azimuth_deg = float(int(azimuth_deg) % 360) 
  rho = dist
  view_params.append([azimuth_deg, elevation_deg, tilt_deg, rho])

for param in view_params: 
  rho = param[3]
  azimuth_deg = param[0]
  elevation_deg = param[1]
  theta_deg = param[2]

  ### ### u:horizontal v: vertical 
  u_max = math.tan(68/2/180.0*np.pi) * rho
  v_max = math.tan(54/2/180.0*np.pi) * rho

  y_cam = 0.0 #np.random.uniform(v_max * -0.6, v_max * 0.6)
  x_cam = 0.0 #np.random.uniform(u_max * -0.6, u_max * 0.6)
  z_cam = rho
 
  # mesh
  cx, cy, cz = obj_centened_camera_pos(rho, azimuth_deg, elevation_deg)
  q1 = camPosToQuaternion(cx , cy , cz)
  q2 = camRotQuaternion(cx, cy , cz, theta_deg)
  q = quaternionProduct(q2, q1)
 
  camObj.rotation_mode = 'QUATERNION'
  camObj.rotation_quaternion[0] = q[0]
  camObj.rotation_quaternion[1] = q[1]
  camObj.rotation_quaternion[2] = q[2]
  camObj.rotation_quaternion[3] = q[3]

  cam_rot_mat = np.array(camObj.rotation_quaternion.to_matrix())
  
  u_vector = np.zeros((3,1))
  v_vector = np.zeros((3,1))
  principal_axis = np.zeros((3,1))

  u_vector = cam_rot_mat[:, 0]
  v_vector = cam_rot_mat[:, 1]
  principal_axis = cam_rot_mat[2, :]

  loc_x = u_vector[0] * x_cam + v_vector[0] * y_cam
  loc_y = u_vector[1] * x_cam + v_vector[1] * y_cam
  loc_z = u_vector[2] * x_cam + v_vector[2] * y_cam
 
  realsense_baseline = 0.048
  
  camObj.location[0] = cx - u_vector[0] * realsense_baseline - loc_x
  camObj.location[1] = cy - u_vector[1] * realsense_baseline - loc_y
  camObj.location[2] = cz - u_vector[2] * realsense_baseline - loc_z
 
scene = bpy.context.scene
scene.frame_start = 0
frame_mid = 20
scene.frame_end = 80

############### 1 frame
for i in range(scene.frame_start,frame_mid):
  scene.frame_set(i)
  #if i % 100 == 0:
  #  ob = bpy.data.objects['model']
    
outputname = 'frame%d_rho%f_azi%f_ele%f_theta%f' % (frame_mid,rho,azimuth_deg,elevation_deg, theta_deg)
outputFile_pgm = os.path.join(outImagePath, outputname+ '.pgm')
candidate_list = [item.name for item in bpy.data.objects if item.type== "MESH"]

for idx,modelid in enumerate(model_ids):
  if idx == 0:
    ob = bpy.data.objects['model']
  else:
    name_id = '{num:03d}'.format(num=idx)
    model_name = 'model.'+name_id
    ob = bpy.data.objects[model_name]
  fname = os.path.join(outImagePath,'frame%d_'+modelid+'_matrix_wolrd.txt') % frame_mid
  f = open(fname, "w" )
  f.write( str( ob.matrix_world ) )
  f.close()

blensor.realsense.scan_advanced(camObj, evd_file=outputFile_pgm)

################ 2 frame
for i in range(frame_mid,scene.frame_end):
  scene.frame_set(i)
  #if i % 100 == 0:
  #  ob = bpy.data.objects['model']
    
outputname = 'frame%d_rho%f_azi%f_ele%f_theta%f' % (scene.frame_end,rho,azimuth_deg,elevation_deg, theta_deg)
outputFile_pgm = os.path.join(outImagePath, outputname+ '.pgm')
candidate_list = [item.name for item in bpy.data.objects if item.type== "MESH"]
for idx,modelid in enumerate(model_ids):
  if idx == 0:
    ob = bpy.data.objects['model']
  else:
    name_id = '{num:03d}'.format(num=idx)
    model_name = 'model.'+name_id
    ob = bpy.data.objects[model_name]
  fname = os.path.join(outImagePath,'frame%d_'+modelid+'_matrix_wolrd.txt') % (scene.frame_end)
  f = open(fname, "w" )
  f.write( str( ob.matrix_world ) )
  f.close()

blensor.realsense.scan_advanced(camObj, evd_file=outputFile_pgm)
