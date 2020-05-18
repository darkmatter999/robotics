import math
import numpy as np 
from scipy.linalg import expm

'''
This algorithm finds the space jacobian of given screw axes and a list of angles
The reference frame is the space frame ('Space Jacobian')
The goal is to find the jacobian Js of the end-effector frame 
given a particular set of joint coordinates **theta**, i.e. the twist of the end-effector frame 
given a certain configuration of the robot joints.
Unlike the forward kinematics, the screw axes input parameters here depend on their respective **theta** (i.e. are a function 
of **theta**), while in FK all joint angles are set to zero.
'''

set_S_orig=np.array([[0,0,1,0,0,0],[0,0,1,0,-1,0],[0,0,1,0,-2,0]])
#set_S_orig=np.array([[0,0,0,0,0,0],[0,1,1,1,0,0],[1,0,0,0,0,1], [0,0,1,-0.73,0,0], [0,1,2.73,3.73,1,0]])
#set_S_orig=np.array([[0,0,0,0,0,0],[0,1,1,1,0,0],[1,0,0,0,0,1]])
#set_S_orig=np.array([[0,0,0,0,0,0],[0,1,1,1,0,0],[1,0,0,0,0,1], [0,0,1,-0.73,0,0], [0,1,2.73,3.73,1,0], [1,0,0,0,0,1],
#[0,1,2.73,3.73,1,0]])

thetalist=np.array([0, math.pi/4, 0])
#thetalist=np.array([-math.pi/2, math.pi/2, math.pi/3, 1, 0.524])
#thetalist=np.array([-math.pi/2, math.pi/2, math.pi/3, 1, 0.524, math.pi/4, math.pi/2])
#thetalist=np.array([-math.pi/2, math.pi/4, 0.524])

#first step:
#building the 4x4 (R --> se3) matrix [S] (here called S_se3) from each S (screw axis)
#take your R6 vector S (one for each joint / screw axis and re-arrange its first three elements 
#(i.e. the R3 vector **w**) into the 
#rows of the to-be-created skew-symmetric matrix as follows:
#R1 --> 0, -x3, x2 *** R2 --> x3, 0, -x1 *** R3 --> -x2, x1, 0
#then add the linear component as the first three rows of the last column
#finally add a row of all zeros

#iteratively build a flattened array from the *w* (angular velocity) and *v* (linear velocity) vectors, then reshape into an array of n 4x4 
#matrices which is subsequently further used for the follow-up calculations

set_S_se3=np.array([])
for S in set_S_orig:
    S_se3=np.array([[0, -S[2], S[1], S[3]], [S[2], 0, -S[0], S[4]], [-S[1], S[0], 0, S[5]], [0, 0, 0, 0]])
    set_S_se3=np.append(set_S_se3, S_se3)
    
set_S_se3=np.reshape(set_S_se3, (len(set_S_orig), 4, 4))

#second step:
#building the matrix exponentials of each S using the scipy function *expm* - alternatively the Rodrigues formula 
#can be used.
#effectively, these matrix exponentials are the transformation matrices of the screw axes in 'set_S'

set_T=np.array([])

for (S_se3, theta) in zip(set_S_se3, thetalist):
    T=expm(np.dot(S_se3, theta))
    set_T=np.append(set_T, T)

set_T=np.reshape(set_T, (len(set_S_orig), 4, 4))

#final step:
#building the jacobian column for column by first constructing the adjoint matrix (6x6 representation of 4x4 transformation matrix/
#matrix exponential) **S-1** for each **S**, subsequently generating the jacobian matrix by allocating S1 as the first column (see above)
#and then iteratively adding the other columns.
#For the jacobian the first screw axis (S1) itself is the first column of the jacobian and can thus be appended to the jacobian before
#the loop.

#initializing the jacobian J with the first (fixed) column S1
J=np.array([set_S_orig[0]])

set_p=np.array([])
set_pR=np.array([])
set_R=np.array([])
set_Ad_T=np.array([])

for T in set_T:
    R=np.array([[T[0][0], T[0][1], T[0][2]], [T[1][0], T[1][1], T[1][2]], [T[2][0], T[2][1], T[2][2]]])
    set_R=np.append(set_R, R)

set_R=np.reshape(set_R, (len(set_S_orig), 3, 3))

for T in set_T:
    p=np.array([[0, -T[2][3], T[1][3]], [T[2][3], 0, -T[0][3]], [-T[1][3], T[0][3], 0]])
    set_p=np.append(set_p, p)

set_p=np.reshape(set_p, (len(set_S_orig), 3, 3))

for (p, R) in zip(set_p, set_R):
    pR=np.dot(p, R)
    set_pR=np.append(set_pR, pR)

set_pR=np.reshape(set_pR, (len(set_S_orig), 3, 3))

for (T, pR) in zip(set_T, set_pR):
    Ad_T=np.array([[T[0][0], T[0][1], T[0][2], 0, 0, 0], [T[1][0], T[1][1], T[1][2], 0, 0, 0], [T[2][0], T[2][1], T[2][2], 0, 0, 0],
    [pR[0][0], pR[0][1], pR[0][2], T[0][0], T[0][1], T[0][2]], [pR[1][0], pR[1][1], pR[1][2], T[1][0], T[1][1], T[1][2]],
    [pR[2][0], pR[2][1], pR[2][2], T[2][0], T[2][1], T[2][2]]])
    set_Ad_T=np.append(set_Ad_T, Ad_T)

set_Ad_T=np.reshape(set_Ad_T, (len(set_S_orig), 6, 6))

for (S, Ad_T) in zip(set_S_orig[1:], set_Ad_T):
    Ji=np.dot(Ad_T, S)
    J=np.append(J, Ji)

J=np.reshape(J, (len(set_S_orig), 6, ))
J=np.transpose(J)

#print (set_S_se3)
#print (set_T[1][1][3])
#print (np.transpose(J))
print (J)
#print (set_Ad_T[1])

