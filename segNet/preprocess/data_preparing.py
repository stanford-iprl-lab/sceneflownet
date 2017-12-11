import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR,os.pardir))
from local_variables import DATA_DIR

from Train_Val_Test import *
import numpy as np
np.random.seed(42)
scanning_data_top_dir = os.path.join(DATA_DIR,'BlensorResult_2frame')
total_ids = []

for line in os.listdir(scanning_data_top_dir):
  total_ids.append(line)

train_val_test_list = Train_Val_Test(np.array(total_ids),splitting=[0.8,0.2,0.0])

print("train num %d , val num %d , test num %d" % (len(train_val_test_list._train),len(train_val_test_list._val),len(train_val_test_list._test)))

