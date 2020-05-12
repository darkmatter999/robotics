import math
import numpy as np 
from scipy.linalg import expm

'''
This algorithm finds the forward kinematics of given screw axes, end-effector transformation matrix and list of angles
The reference frame is the space frame ('FK in Space')
The goal is to find the transformation matrix T (here called FK) of the end-effector frame 
given a particular set of joint coordinates **theta**, i.e. the position and orientation of the end-effector frame 
given a certain pose of the robot.
'''

set_S_orig=np.array([[0,0,0,0,0,0],[0,1,1,1,0,0],[1,0,0,0,0,1],[0,0,1,-0.73,0,0],[-1,0,0,0,0,-3.73],[0,1,2.73,3.73,1,0]])
#set_S_orig=np.array([[0,0,0,0,0,0],[0,1,1,1,0,0],[1,0,0,0,0,1], [0,0,1,-0.73,0,0], [0,1,2.73,3.73,1,0]])
#set_S_orig=np.array([[0,0,0,0,0,0],[0,1,1,1,0,0],[1,0,0,0,0,1]])
#set_S_orig=np.array([[0,0,0,0,0,0],[0,1,1,1,0,0],[1,0,0,0,0,1], [0,0,1,-0.73,0,0], [0,1,2.73,3.73,1,0], [1,0,0,0,0,1],
#[0,1,2.73,3.73,1,0]])

set_S=np.transpose(set_S_orig)
set_S=np.reshape(set_S, (len(set_S_orig), 6))

thetalist=np.array([-math.pi/2, math.pi/2, math.pi/3, -math.pi/4, 1, math.pi/6])
#thetalist=np.array([-math.pi/2, math.pi/2, math.pi/3, 1, 0.524])
#thetalist=np.array([-math.pi/2, math.pi/2, math.pi/3, 1, 0.524, math.pi/4, math.pi/2])
#thetalist=np.array([-math.pi/2, math.pi/4, 0.524])

M=np.array([[1,0,0,3.73],[0,1,0,0],[0,0,1,2.73],[0,0,0,1]])

#first step:
#building the 4x4 (R --> se3) matrix [S] (here called S_se3) from each S (screw axis)
#take your R6 vector S (one for each joint / screw axis) and re-arrange its first three elements (i.e. the R3 vector **w**) into the 
#rows of the to-be-created skew-symmetric matrix as follows:
#R1 --> 0, -x3, x2 *** R2 --> x3, 0, -x1 *** R3 --> -x2, x1, 0
#then add the linear component as the first three rows of the last column
#finally add a row of all zeros

#iteratively build a flattened array from the *w* (angular velocity) and *v* (linear velocity) vectors, then reshape into an array of n 4x4 
#matrices which is subsequently further used for the follow-up calculations

set_S_se3=np.array([])
for S in set_S:
    S_se3=np.array([[0, -S[2], S[1], S[3]], [S[2], 0, -S[0], S[4]], [-S[1], S[0], 0, S[5]], [0, 0, 0, 0]])
    set_S_se3=np.append(set_S_se3, S_se3)
    
set_S_se3=np.reshape(set_S_se3, (len(set_S), 4, 4))

#second step:
#building the matrix exponentials of each S using the scipy function *expm* - alternatively the Rodrigues formula can be used
#effectively, these matrix exponentials are the transformation matrices of the screw axes in 'set_S'

set_T=np.array([])

for (S_se3, theta) in zip(set_S_se3, thetalist):
    T=expm(np.dot(S_se3, theta))
    set_T=np.append(set_T, T)

set_T=np.reshape(set_T, (len(set_S), 4, 4))

#final step:
#building the forward kinematics *FK* from the product of all *T*s together with M

FK=np.dot(set_T[0], set_T[1])
i=2
while i < len(set_T):
    FK=np.dot(FK, set_T[i])
    i=i+1

FK=np.dot(FK, M)

#print (set_S_se3)
#print (set_T)
print (FK)

