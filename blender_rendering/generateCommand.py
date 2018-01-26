import numpy as np
import sys
import os
from rendering_variables import *
from local_variables import *

command_file = open('commands.txt','w')

mesh_top_dir = '/home/linshaonju/interactive-segmentation/Data/ShapenetManifold'

top_dir = '/home/linshaonju/interactive-segmentation/Data/BlensorResult'

texture_path_top_dir = '/home/linshaonju/interactive-segmentation/Data/SUN397'

texture_path_list = []
alpha_dir = [line for line in os.listdir(texture_path_top_dir) if os.path.isdir(os.path.join(texture_path_top_dir,line))]
for a_dir in alpha_dir:
  texture_dir = os.path.join(texture_path_top_dir,a_dir)
  texture_sub_dir = [line for line in os.listdir(texture_dir)]
  for sub_dir in texture_sub_dir:
    texture_sub_sub_dir = os.path.join(texture_dir,sub_dir)
    texture_list = [line for line in os.listdir(texture_sub_sub_dir)]
    for line in texture_list:
      texture_path_list.append(os.path.join(texture_sub_sub_dir,line))
print(len(texture_path_list))

for i in xrange(25000,25000+10):
  result_path = os.path.join(top_dir,str(i)) 
  blender = os.path.join(ROOT_DIR,'3rd_parties','blensor','blender')
  blank_blend = os.path.join(ROOT_DIR,'blensor_scanning','blank.blend')
  render_file = os.path.join(ROOT_DIR,'blender_rendering','render.py')
  texture_file = texture_path_list[i]
  command = '%s %s --background --python %s %s %s ' % (blender, blank_blend, render_file, texture_file, result_path)
  command_file.write(command+'\n')

command_file.close()
