import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, os.pardir)
DATA_DIR = os.path.join(ROOT_DIR,'Data')
mesh_top_dir = os.path.join(DATA_DIR,'GraspNet')
scan_result_dir = os.path.join(DATA_DIR,'BlensorResult')
CommandFile_dir = os.path.join(DATA_DIR,'CommandFiles')
