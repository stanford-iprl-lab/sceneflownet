import pybullet as p
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage


class SIM_ENV:
  def __init__(self,obj_list,segNet=None,segNet2=None,gui=True):
    self.p = p
    self.seed = 42
    np.random.seed(self.seed)

    #### vision   
    self.screen_height = 120
    self.screen_width = 160
    self.screen_dim = 3

    self.action_scale = 8
    self.action_height = int(self.screen_height / self.action_scale) # 20
    self.action_width = int(self.screen_width / self.action_scale) # 15
    self.num_action = self.action_height * self.action_width # 300
  
    self.fx = 472.92840576171875 
    self.fy = self.fx
    nx,ny = (self.screen_width,self.screen_height)
    x_index = np.linspace(0,self.screen_width-1,self.screen_width)
    y_index = np.linspace(0,self.screen_height-1,self.screen_height)
    xx,yy = np.meshgrid(x_index,y_index)
    xx -= float(self.screen_width)/2
    yy -= float(self.screen_height)/2
    self.xx = xx/self.fx
    self.yy = yy/self.fy

    ee_zz = np.ones((self.screen_height, self.screen_width))
    ee_xx = self.xx * ee_zz
    ee_yy = self.yy * ee_zz

    zoom_scale = float(1.0 / self.action_scale)
    ee_x = scipy.ndimage.zoom(ee_xx, zoom_scale, order=1)
    ee_y = scipy.ndimage.zoom(ee_yy, zoom_scale, order=1)
    ee_z = scipy.ndimage.zoom(ee_zz, zoom_scale, order=1)
    self.ee_pos_map = np.dstack((ee_x,ee_y,ee_z))
    print(self.ee_pos_map.shape)
    
    #### Sim
    self.rgb = None
    self.depth = None
    self.seg = None
    self._screen = None
    self.pred = None
    self.gt = None
   
    self.reward = 0
    self.terminal = True
    self.gui = True

    self.objList = obj_list
    self.num_obj = len(self.objList)

    # init segNet
   
  def new_game(self):
    if self.gui:
      self.p.connect(self.p.GUI)
    else:
      self.p.connect(self.p.DIRECT)
    #### base table

    self.baseTableId = self.p.createCollisionShape(p.GEOM_BOX,halfExtents=[100,100,0.0001])
    self.baseTablePos = [0,0,-0.0001]
    self.p.createMultiBody(baseMass=0,baseCollisionShapeIndex=self.baseTableId, baseVisualShapeIndex=0, basePosition = self.baseTablePos)
    self.massList = np.random.uniform(0.5,2.5,self.num_obj)
    self.visualShapeIdList = [-1] * self.num_obj
    self.modelIdList = []   
    self.posList = []
    self.ornList = [] 
    self.EulerList = np.random.uniform(0,3.1415926,(self.num_obj,3))

    for i in xrange(self.num_obj):
      mass = self.massList[i]
      visualShapeId = self.visualShapeIdList[i]
      modelId = self.p.createCollisionShape(shapeType=p.GEOM_MESH,fileName=self.objList[i],meshScale=[0.3,0.3,0.3])
      self.modelIdList.append(modelId)
      pos = [0,0,i*0.6+0.3]
      orn = self.p.getQuaternionFromEuler(self.EulerList[i])
      self.p.createMultiBody(baseMass = mass, baseCollisionShapeIndex = modelId, baseVisualShapeIndex = visualShapeId, basePosition = pos, baseOrientation = orn) 
      self.p.changeDynamics(modelId,-1,spinningFriction = 0.001, rollingFriction = 0.001, linearDamping = 0.3)

    self.endEffectorId = self.p.createCollisionShape(p.GEOM_BOX,halfExtents=[0.01,0.01,0.2])
    self.endEffectorPos = [0,0,0.05]
    self.endEffectorOrn = self.p.getQuaternionFromEuler([0,0.7853979,0])
    self.p.createMultiBody(baseMass = 0, baseCollisionShapeIndex = self.endEffectorId, basePosition = self.endEffectorPos, baseOrientation = self.endEffectorOrn)    

    self.viewMatrix = self.p.computeViewMatrix(cameraEyePosition=[1,0,1],cameraTargetPosition=[0,0,0],cameraUpVector=[0,0,1])
    self.projectionMatrix = self.p.computeProjectionMatrixFOV(fov=57,aspect=1.0,nearVal=0.05,farVal=2.0)

    self.p.setGravity(0,0,-10)
 
    self.p.setRealTimeSimulation(0)
    frame_id = 0.0
    
    while(frame_id < 100):
      width, height,self.rgb, self.depth, self.seg = p.getCameraImage(width = self.screen_width, height = self.screen_height, viewMatrix=self.viewMatrix,projectionMatrix=self.projectionMatrix)
      self.p.stepSimulation()
      time.sleep(0.01) 
      frame_id += 1.0

      if False:
        plt.figure(0)
        plt.imshow(self.rgb)
        plt.figure(1)
        plt.imshow(self.depth)
        plt.figure(2)
        plt.imshow(self.seg)
        plt.show()
      
  def diff(self):
    if self.pred == self.gt:
      return True
    else:
      return False

  def act(self,action=[0,0]):    
    self.endEffectorPos = self.ee_pos_map[action[0],action[1],:]
    self.p.resetBasePositionAndOrientation(self.endEffectorId,self.endEffectorPos,self.endEffectorOrn)
    self.terminal = self.diff() 
    if self.terminal:
      self.p.disconnect()
      self.new_game()
    return self.screen, self.reward, self.terminal
 
  @property
  def screen(self):
    cam_z = self.depth
    cam_x = self.xx * cam_z
    cam_y = self.yy * cam_z
    return  np.dstack((cam_x,cam_y,cam_z))

  @property
  def state(self):
    return self.screen, self.reward, self.terminal

  
if __name__ == "__main__":    
  test = SIM_ENV(['./model1.obj','./model2.obj','./model3.obj'],gui=True)
  test.new_game()
  print('starting new game')
  test.act()
