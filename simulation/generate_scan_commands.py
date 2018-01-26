import numpy as np
import sys
import os
from local_variables import *

command_file = open('commands.txt','w')
top_dir = '/home/linshaonju/interactive-segmentation/Data/BlensorResult'

for i in xrange(10000,15000):
  path_tmp = os.path.join(top_dir,str(i))
  len_npz = len([line for line in os.listdir(path_tmp) if line.endswith('npz')])
  print(len_npz)
  if len_npz < 4:
    command = 'python sim_2frame_sys.py  %s' % (path_tmp)
    command_file.write(command+'\n')

command_file.close() 
