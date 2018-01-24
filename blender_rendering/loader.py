import os
import sys
import numpy as np

def azi_ele_the_rho(top_dir):
  pgm_filepath = [line for line in os.listdir(top_dir) if line.endswith('.pgm') and line.startswith('frame80')][0]
  
  tmp = pgm_filepath.split('.pgm')[0].split('_')
  
  azimuth_deg = float(tmp[2].split('azi')[1])
  elevation_deg = float(tmp[3].split('ele')[1]) 
  theta_deg = float(tmp[4].split('theta')[1])
  rho = float(tmp[1].split('rho')[1])
  
  print('azi %f ele %f the %f rho %f' % (azimuth_deg, elevation_deg, theta_deg, rho))
  view_params = {}
  view_params['azi'] = azimuth_deg
  view_params['ele'] = elevation_deg
  view_params['rho'] = rho
  view_params['the'] = theta_deg
  return view_params

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

def model_para_list(top_dir,frame_id):
  model_top_dir = '/home/linshaonju/GraspNet3.0/Data/ShapeNetCore'  
  model_list = [line.strip().split('_') for line in os.listdir(top_dir) if line.startswith('frame'+frame_id) and line.endswith('_matrix_wolrd.txt')]
  cate_model_list= [(line[1],line[2]) for line in model_list]
  model_path_list = [os.path.join(model_top_dir,line[0],line[1],'model.obj') for line in cate_model_list]
  print(model_path_list)
  rot_list = []
  tran_list = []
  for cate_model in cate_model_list:
    cate, model = cate_model
    tran_rot_file = os.path.join(top_dir,'frame'+frame_id+'_'+cate+'_'+model+'_matrix_wolrd.txt')
    tran, rot = tran_rot(tran_rot_file)
    rot_list.append(rot)
    tran_list.append(tran)
  return model_path_list, tran_list, rot_list

if __name__ == '__main__':
  azi_ele_the_rho('/home/linshaonju/interactive-segmentation/Data/BlensorResult_2frame/0')
