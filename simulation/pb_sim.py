import pybullet as p
import time
import numpy as np

class pybullet_objects:
  def __init__(self,obj_list,gui=True):
    self.p = p
    if gui:
      self.p.connect(self.p.GUI)
    else:
      self.p.connect(self.p.DIRECT)
    self.p.createCollisionShape(p.GEOM_PLANE)
    self.p.createMultiBody(baseMass=0,baseCollisionShapeIndex=0)
    self.objList = obj_list
    self.num_obj = len(self.objList)
    self.seed = 42
    np.random.seed(self.seed)
    self.massList = np.random.uniform(0.5,2.5,self.num_obj)
    self.visualShapeIdList = [-1] * self.num_obj
    self.modelIdList = []   
    self.posList = []
    self.ornList = [] 
    self.EulerList = np.random.uniform(0,3.1415926,(self.num_obj,3))

    for i in xrange(self.num_obj):
      mass = self.massList[i]
      visualShapeId = self.visualShapeIdList[i]
      modelId = self.p.createCollisionShape(shapeType=p.GEOM_MESH,fileName=self.objList[i])
      self.modelIdList.append(modelId)
      pos = [0,0,i*1.0+0.3]
      orn = self.p.getQuaternionFromEuler(self.EulerList[i])
      self.p.createMultiBody(baseMass = mass, baseCollisionShapeIndex = modelId, baseVisualShapeIndex = visualShapeId, basePosition = pos, baseOrientation = orn) 
      self.p.changeDynamics(modelId,-1,spinningFriction = 0.001, rollingFriction = 0.001, linearDamping = 0.3)

    self.p.setGravity(0,0,-10)

    self.endEffectorId = self.p.createCollisionShape(p.GEOM_BOX,halfExtents=[0.05,0.05,0.3])
    self.endEffectorPos = [0,0,0.3]
    self.endEffectorOrn = self.p.getQuaternionFromEuler([0.5,0.5,0])
    self.p.createMultiBody(baseMass = 0, baseCollisionShapeIndex = self.endEffectorId, basePosition = self.endEffectorPos, baseOrientation = self.endEffectorOrn)    
   
  def simulation(self):
    self.p.setRealTimeSimulation(1)
    frame_id = 0.0
    while(1):
      self.p.getKeyboardEvents()
      image = p.getCameraImage(320,320)
      self.endEffectorOrn = self.p.getQuaternionFromEuler([0.5,0.5,0.1*frame_id])
      self.p.resetBasePositionAndOrientation(self.endEffectorId,self.endEffectorPos,self.endEffectorOrn)
      time.sleep(0.01) 
      frame_id += 1.0

if __name__ == "__main__":    
  test = pybullet_objects(['./model1.obj','./model2.obj','./model3.obj'],gui=True)
  test.simulation()
