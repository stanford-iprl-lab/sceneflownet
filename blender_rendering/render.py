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
from loader import azi_ele_the_rho, model_para_list
from rendering_variables import *

light_num_lowbound = g_syn_light_num_lowbound
light_num_highbound = g_syn_light_num_highbound 
light_dist_lowbound = g_syn_light_dist_lowbound 
light_dist_highbound = g_syn_light_dist_highbound

"""---------- main -----------"""
texture_path = sys.argv[-2]
result_path = sys.argv[-1]
print("result_path %s " % (result_path))
print(texture_path)

import shutil

def makeMaterial(name, diffuse, diffuse_intensity, specular, specular_intensity, alpha):
  mat = bpy.data.materials.new(name)
  mat.diffuse_color = diffuse
  mat.diffuse_shader = 'LAMBERT'
  mat.diffuse_intensity = diffuse_intensity
  mat.specular_color = specular
  mat.specular_shader = 'COOKTORR'
  mat.specular_intensity = specular_intensity
  mat.alpha = alpha
  mat.ambient = 1
  return mat

def setMaterial(ob,mat):
  me = ob.data
  me.materials.append(mat)

diffuse = np.random.uniform(0,1,(3))
diffuse_intensity = np.random.uniform(0.3,1,1)

specular = np.random.uniform(0,1,(3))
specular_intensity = np.random.uniform(0,0.5,1)

color = makeMaterial('color',diffuse,diffuse_intensity,specular,specular_intensity,1)
bpy.ops.mesh.primitive_cube_add(location=(0,0,0),radius=1)
bpy.data.objects['Cube'].location=(0,0,-0.2)
bpy.data.objects['Cube'].scale = (10,10,0.0001)
bpy.data.objects['Cube'].name = 'floor'

#setMaterial(bpy.data.objects['floor'],color)
texturepath = os.path.expanduser(texture_path)

def texture(texturepath,obj_bpy):
  img = bpy.data.images.load(texturepath)
   
  # create image texture from image
  cTex = bpy.data.textures.new('ColorTex',type='IMAGE')
  cTex.image = img

  # create materials
  mat = bpy.data.materials.new('TexMat')
  
  # Add texture slot for color texture
  mtex = mat.texture_slots.add()
  mtex.texture = cTex
  mtex.texture_coords = 'UV'
  mtex.use_map_color_diffuse = True
  mtex.use_map_color_emission = True
  mtex.emission_color_factor = 0.5
  mtex.use_map_density = True
  mtex.mapping = 'FLAT'

  me = obj_bpy.data
  me.materials.append(mat)

texture(texturepath,bpy.data.objects['floor'])

frame_id = '20'
model_path_list, transl_list, rot_list = model_para_list(result_path,frame_id)
print(model_path_list)
print(rot_list)

for idx,model_path in enumerate(model_path_list):
    tmp = bpy.ops.import_scene.obj(filepath=model_path)
    candidate_list = [item.name for item in bpy.data.objects if item.type == "MESH"]
    print(candidate_list)
    for candidate in candidate_list:
      if not candidate.startswith('MMMM') and candidate != 'floor':
        mw = bpy.data.objects[candidate].matrix_world
        transl = transl_list[idx]
        rot = rot_list[idx]
        for i in range(3):
          for j in range(3):
            mw[i][j] = rot[i,j]
        for i in range(3):
          mw[i][3] = transl[i]
        bpy.data.objects[candidate].name = 'MMMM'+str(idx) 
          
bpy.context.scene.render.alpha_mode = 'TRANSPARENT'
bpy.data.objects['Lamp'].data.energy = 0
bpy.context.scene.render.use_shadows = False
bpy.context.scene.render.use_raytrace = True
bpy.context.scene.render.resolution_x = 640
bpy.context.scene.render.resolution_y = 480
bpy.context.scene.render.resolution_percentage = 100

camObj = bpy.data.objects['Camera']

# set lights
bpy.ops.object.select_all(action='TOGGLE')
if 'Lamp' in list(bpy.data.objects.keys()):
    bpy.data.objects['Lamp'].select = True
bpy.ops.object.delete()

scene = bpy.context.scene
for obj in scene.objects:
    if obj.type == 'MESH':
        scene.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT',toggle=False)
    bpy.ops.mesh.reveal()
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.normals_make_consistent()
    bpy.ops.object.mode_set(mode='OBJECT',toggle=False)
 
view_params = azi_ele_the_rho(result_path) 

rho = view_params['rho']
azimuth_deg = view_params['azi']
elevation_deg = view_params['ele']
theta_deg = view_params['the']
    
cx, cy, cz = obj_centened_camera_pos(rho, azimuth_deg, elevation_deg)
q1 = camPosToQuaternion(cx, cy, cz )
q2 = camRotQuaternion(cx, cy, cz, theta_deg)
q = quaternionProduct(q2, q1)
camObj.location[0] = cx
camObj.location[1] = cy
camObj.location[2] = cz
camObj.rotation_mode = 'QUATERNION'
camObj.rotation_quaternion[0] = q[0]
camObj.rotation_quaternion[1] = q[1]
camObj.rotation_quaternion[2] = q[2]
camObj.rotation_quaternion[3] = q[3]

camObj.data.lens_unit = 'FOV'
camObj.data.angle = 68.1673296/180.0 * np.pi
camObj.data.sensor_height = camObj.data.sensor_width * 480 / 640

# clear default lights
bpy.ops.object.select_by_type(type='LAMP')
bpy.ops.object.delete(use_global=False)        
    
#clear environment lighting
bpy.context.scene.world.light_settings.use_environment_light = True
bpy.context.scene.world.light_settings.environment_energy = np.random.uniform(g_syn_light_environment_energy_lowbound, g_syn_light_environment_energy_highbound)
bpy.context.scene.world.light_settings.environment_color = 'PLAIN'

# set point lights
for i in range(random.randint(light_num_lowbound,light_num_highbound)):
  light_azimuth_deg = np.random.uniform(g_syn_light_azimuth_degree_lowbound, g_syn_light_azimuth_degree_highbound)
  light_elevation_deg  = np.random.uniform(g_syn_light_elevation_degree_lowbound, g_syn_light_elevation_degree_highbound)
  light_dist = np.random.uniform(light_dist_lowbound, light_dist_highbound)
  lx, ly, lz = obj_centened_camera_pos(light_dist, light_azimuth_deg, light_elevation_deg)
  bpy.ops.object.lamp_add(type='POINT', view_align = False, location=(lx, ly, lz))
  bpy.data.objects['Point'].data.energy = np.random.normal(g_syn_light_energy_mean, g_syn_light_energy_std)

outputname = 'frame'+ frame_id +'_rho%f_azi%f_ele%f_theta%f_' % (rho, azimuth_deg, elevation_deg, theta_deg)
outputFile_png = os.path.join(result_path, outputname+ '.png')
bpy.data.scenes['Scene'].render.filepath = outputFile_png
bpy.ops.render.render( write_still=True )
print(outputFile_png)

bpy.ops.object.select_all(action='TOGGLE')
candidate_list = [item.name for item in bpy.data.objects if item.type == "MESH"]
for candidate in candidate_list:
  if candidate.startswith('MMMM'):
    bpy.data.objects[candidate].select = True
bpy.ops.object.delete()

frame_id = '80'
model_path_list, transl_list, rot_list = model_para_list(result_path,frame_id)

for idx,model_path in enumerate(model_path_list):
    tmp = bpy.ops.import_scene.obj(filepath=model_path)
    candidate_list = [item.name for item in bpy.data.objects if item.type == "MESH"]
    print(candidate_list)
    for candidate in candidate_list:
      if not candidate.startswith('MMMM') and candidate != 'floor':
        mw = bpy.data.objects[candidate].matrix_world
        transl = transl_list[idx]
        rot = rot_list[idx]
        for i in range(3):
          for j in range(3):
            mw[i][j] = rot[i,j]
        for i in range(3):
          mw[i][3] = transl[i]
        bpy.data.objects[candidate].name = 'MMMM'+str(idx)

outputname = 'frame'+ frame_id +'_rho%f_azi%f_ele%f_theta%f_' % (rho, azimuth_deg, elevation_deg, theta_deg)
outputFile_png = os.path.join(result_path, outputname+ '.png')
bpy.data.scenes['Scene'].render.filepath = outputFile_png
bpy.ops.render.render( write_still=True )
print(outputFile_png)
