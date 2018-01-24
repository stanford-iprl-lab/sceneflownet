import numpy as np
import sys
import os
from local_variables import *

command_file = open('commands.txt','w')
mesh_top_dir = '/home/linshaonju/GraspNet3.0/Data/ShapenetManifold'
cates = [line for line in os.listdir(mesh_top_dir) if os.path.isdir(os.path.join(mesh_top_dir,line))]

train_id = [line.strip() for line in open('/home/linshaonju/new_trainid_md5.txt')] 
val_id = [line.strip() for line in open('/home/linshaonju/new_valid_md5.txt')]
test_id = [line.strip() for line in open('/home/linshaonju/new_testid_md5.txt')]

cates = ['02876657',\
         # bottle 
         '02691156',\
         # toy airplane 
         '02747177',\
         # trash can
         '02773838',\
         # bag 
         '02808440',\
         # bowl
         '02924116',\
         #toy bus
         '02942699',\
         #camera
         '02946921',\
         #can
         '02954340',\
         # cap
         '02958343',\
         #toy car
         '03001627',\
         #chair
         '03046257',\
         #clocks
         '03085013',\
         #key boards
         '03211117',\
         #display
         '03261776',\
         #earphone
         '03624134',\
         #knife
         '03642806',\
         #laptop
         '03790512',\
         #toy motorcycle
         '03797390',\
         #mug
         '03948459',\
         #pistol
         '04074963',\
         #remote control
         '04401088',\
         #telephone
         '04530566',\
         #toy boat
         '04468005',\
         #toy train
         '04099429',\
         #toy rocket
         '04256520',\
         #sofa
         '03513137',\
         #helmet
         '04379243',\
         #table
         ]
### the origin CAD model bbox diagonal has length of 1
 
cates_sizes = {'02876657':[0.1,0.27],\
         # bottle 
         '02691156':[0.1,0.27],\
         # toy airplane 
         '02747177':[0.3,0.5],\
         # trash can
         '02773838':[0.3,0.4],\
         # bag 
         '02808440':[0.2,0.35],\
         # bowl
         '02924116':[0.1,0.25],\
         #toy bus
         '02942699':[0.1,0.23],\
         #camera
         '02946921':[0.12,0.15],\
         #can
         '02954340':[0.25,0.35],\
         # cap
         '02958343':[0.1,0.25],\
         #toy car
         '03001627':[0.1,0.20],\
         #chair
         '03046257':[0.1,0.2],\
         #clocks
         '03085013':[0.3,0.5],\
         #key boards
         '03211117':[0.5,0.7],\
         #display
         '03261776':[0.2,0.3],\
         #earphone
         '03624134':[0.1,0.2],\
         #knife
         '03642806':[0.3,0.5],\
         #laptop
         '03790512':[0.15,0.23],\
         #toy motorcycle
         '03797390':[0.13,0.17],\
         #mug
         '03948459':[0.12,0.20],\
         #pistol
         '04074963':[0.13,0.20],\
         #remote control
         '04401088':[0.13,0.17],\
         #telephone
         '04530566':[0.1,0.3],\
         #toy boat
         '04468005':[0.07,0.17],\
         #toy train
         '04099429':[0.08,0.23],\
         #toy rocket
         '04256520':[0.1,0.20],\
         #sofa
         '03513137':[0.3,0.4],\
         #helmet
         '04379243':[0.08,0.2],\
         #table
         }

model_ids_per_cate = {}
model_ids = []
max_num = 4882
model_sizes = []
for cate in cates:
  models = [line for line in os.listdir(os.path.join(mesh_top_dir,cate)) if cate+' '+line in train_id]
  model_ids_per_cate[cate] = models
  print(len(models))
  inds = np.random.choice(len(models),max_num)
  low_size,high_size = cates_sizes[cate]
  model_size = np.random.uniform(low=low_size,high=high_size,size=max_num)
  for model_id in inds:
    model_ids.append(cate+'_'+models[model_id]+'_'+str(model_size[model_id]))

print(len(model_ids))

count = 0
res_top = scan_result_dir

max_num_instances = 30
num_pairs = 40000

instances_in_pairs = np.random.choice(len(model_ids),max_num_instances * num_pairs)
instances_in_pairs = np.array(instances_in_pairs).reshape((num_pairs,max_num_instances))

print(instances_in_pairs.shape)


for i in xrange(num_pairs):
  num_instances = np.random.randint(1,max_num_instances)
  #inds = np.random.choice(len(model_ids),num_instances)
  inds = instances_in_pairs[i][0:num_instances]
  print(inds)
  tmp = ''
  for ids in inds:
    tmp += model_ids[ids] + '#'
  model_scan_path = os.path.join(scan_result_dir, str(i))
  if not os.path.exists(model_scan_path):
    os.mkdir(model_scan_path)
  blender = os.path.join(ROOT_DIR,'3rd_parties','blensor','blender')
  blank_blend = os.path.join(ROOT_DIR,'blensor_scanning','blank.blend')
  scan_file = os.path.join(ROOT_DIR,'blensor_scanning','scanner.py')
  command = '%s %s --background --python %s %s %s' % (blender, blank_blend, scan_file, tmp, model_scan_path)
  command_file.write(command+'\n')

command_file.close() 
