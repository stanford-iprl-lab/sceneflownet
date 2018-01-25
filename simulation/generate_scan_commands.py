import numpy as np
import sys
import os
from local_variables import *

command_file = open('commands.txt','w')
top_dir = '/home/linshaonju/interactive-segmentation/Data/BlensorResult'

for i in xrange(5000):
  path_tmp = os.path.join(top_dir,str(i))
  command = 'python sim_2frame_sys.py  %s' % (path_tmp)
  command_file.write(command+'\n')

command_file.close() 
