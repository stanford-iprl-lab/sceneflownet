from __future__ import print_function
import numpy as np
import random
import tensorflow as tf
import sys
from local_variables import SIM_DIR
sys.path.append(SIM_DIR)
from sim_env import SIM_ENV 
from dqn.agent import Agent
from config import get_config

import argparse 

### seed
seed = 42
np.random.seed(seed)
tf.set_random_seed(seed)
random.seed(seed)

### Basic parameters
parser = argparse.ArgumentParser()
parser.add_argument('--train',type=bool,default=True,help='Wether to train the agent or test')

FLAGS = parser.parse_args()

model_top_dir = '/home/lins/interactive-segmentation/simulation'
def main():
  with tf.Session() as sess:    
    env = SIM_ENV([os.path.join(model_top_dir,'model1.obj'),os.path.join(model_top_dir,'model2.obj'),os.path.join(model_top_dir,'model3.obj')])

    agent = Agent(FLAGS,env,sess)
    
    if FLAGS.is_train:
      agent.train()
    else:
      agent.play()

if __name__ == '__main__':
  main()
