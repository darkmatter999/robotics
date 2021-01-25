import pybullet as p
import time
import pybullet_data
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,0)
planeId = p.loadURDF("plane.urdf")
startPos = [0,0,1]
startOrientation1 = p.getQuaternionFromEuler([0,2,0])
startOrientation2 = p.getQuaternionFromEuler([0,5,0])
boxId = p.loadURDF("biped/biped2d_pybullet.urdf",startPos, startOrientation1)
boxId = p.loadURDF("biped/biped2d_pybullet.urdf",startPos, startOrientation2)
#set the center of mass frame (loadURDF sets base link frame) startPos/Ornp.resetBasePositionAndOrientation(boxId, startPos, startOrientation)
for i in range (10000):
    p.stepSimulation()
    time.sleep(1./240.)
cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
print(cubePos,cubeOrn)
p.disconnect()
