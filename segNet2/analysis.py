import os 
import numpy as np
from local_variables import *
from evaluation.metric import m_AP50,m_AP75,m_AP90,m_AP

RESULT_SAVE_DIR = os.path.join(BASE_DIR,'saved_results')
experiment_name = 'seg_std_4d_2sigma_new'

top_dir = os.path.join(RESULT_SAVE_DIR,experiment_name)
id_lists = [os.path.join(top_dir,line) for line in os.listdir(top_dir)]

m_AP50(id_lists)
m_AP75(id_lists)
m_AP90(id_lists)
m_AP(id_lists)
