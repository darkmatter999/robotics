import numpy as np

def IKinBodyIterates(Blist, M, T, thetalist0, eomg, ev):
    """Computes inverse kinematics in the body frame for an open chain robot. 
    It furthermore outputs the configuration of each Newton-Raphson iteration
    before the result and the result once the desired end-effector configuration has been found.
    In addition, a .csv file containing the joint vectors of each Newton-Raphson iteration is created.

    :param Blist: The joint screw axes in the end-effector frame when the
                  manipulator is at the home position, in the format of a
                  matrix with axes as the columns
    :param M: The home configuration of the end-effector
    :param T: The desired end-effector configuration Tsd
    :param thetalist0: An initial guess of joint angles that are close to
                       satisfying Tsd
    :param eomg: A small positive tolerance on the end-effector orientation
                 error. The returned joint angles must give an end-effector
                 orientation error less than eomg
    :param ev: A small positive tolerance on the end-effector linear position
               error. The returned joint angles must give an end-effector
               position error less than ev
    :return thetalist: Joint angles that achieve T within the specified
                       tolerances,
    :return success: A logical value where TRUE means that the function found
                     a solution and FALSE means that it ran through the set
                     number of maximum iterations without finding a solution
                     within the tolerances eomg and ev.
    Uses an iterative Newton-Raphson root-finding method.
    The maximum number of iterations before the algorithm is terminated has
    been hardcoded in as a variable called maxiterations. It is set to 20 at
    the start of the function, but can be changed if needed.
    
    Example Input:
    
    Blist = np.array([[0, 0, -1, 2, 0,   0],
                          [0, 0,  0, 0, 1,   0],
                          [0, 0,  1, 0, 0, 0.1]]).T
    M = np.array([[-1, 0,  0, 0],
                      [ 0, 1,  0, 6],
                      [ 0, 0, -1, 2],
                      [ 0, 0,  0, 1]])
    T = np.array([[0, 1,  0,     -5],
                      [1, 0,  0,      4],
                      [0, 0, -1, 1.6858],
                      [0, 0,  0,      1]])
    thetalist0 = np.array([1.5, 2.5, 3])
    eomg = 0.01
    ev = 0.001
    
    Output:
        (np.array([1.57073819, 2.999667, 3.14153913]), True)
    """

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#Test parameters

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


#Book example 6.1 (Page 231)
Blist2 = np.array([[0, 0, 1, 0, 2, 0], [0, 0, 1, 0, 1, 0]]).T
M2 = np.array([[1, 0, 0, 2], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
T2 = np.array([[-0.5, -0.866, 0, 0.366], [0.866, -0.5, 0, 1.366], [0, 0, 1, 0], [0, 0, 0, 1]])
thetalist2 = np.array([0, 0.523599])
eomg2 = 0.001
ev2 = 0.0001

#Universal Robots UR5 6R robot arm example
Blist_UR5 = \
np.array([[0, 1, 0, 0.191, 0, 0.817],  \
[0, 0, 1, 0.095, -0.817, 0], \
[0, 0, 1, 0.095, -0.392, 0], \
[0, 0, 1, 0.095, 0, 0], \
[0, -1, 0, -0.082, 0, 0], \
[0, 0, 1, 0, 0, 0]]).T

M_UR5 = np.array([[-1, 0, 0, 0.817], [0, 0, 1, 0.191], [0, 1, 0, -0.006], [0, 0, 0, 1]])
T_UR5 = np.array([[0, 1, 0, -0.5], [0, 0, -1, 0.1], [-1, 0, 0, 0.1], [0, 0, 0, 1]])
thetalist_UR5 = np.array([-3.417,  -1,  2, -0.9, -0.1, -1.6])
eomg_UR5 = 0.001
ev_UR5 = 0.0001

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    thetalist = np.around(np.array(thetalist0).copy(), decimals=3)

    #output the initial guess of theta angles for each joint
    print ("IK of Universal Robots UR5 6R robot arm with initial joint vector guess:"'\n')
    print (str(thetalist) + '\n')

    #initialize empty array for joint vector matrix to be subsequently output in a .csv file
    joint_vector = np.array([])
    
    i = 0
    maxiterations = 20
    Vb = se3ToVec(MatrixLog6(np.dot(TransInv(FKinBody(M, Blist, \
                                                      thetalist)), T)))
    err = np.linalg.norm([Vb[0], Vb[1], Vb[2]]) > eomg \
          or np.linalg.norm([Vb[3], Vb[4], Vb[5]]) > ev

    while err and i < maxiterations:
        #calculate the current Tsb to be output at each iteration
        Tsb = np.around(FKinBody(M, Blist, thetalist), decimals=3)

        #output the parameters of each current configuration at each iteration
        print ("Iteration " + str(i) + ":"'\n')
        print ("joint vector: " + '\n' + str(thetalist) + '\n')
        print ("SE3(3) end-effector config: " + '\n' + str(Tsb) + '\n')
        print ("Vb: " + '\n' + str(Vb) + '\n')
        print ("angular error magnitude omega_b: " + '\n' + str(np.around(np.linalg.norm([Vb[0], Vb[1], Vb[2]]), decimals=3)) + '\n')
        print ("linear error magnitude v_b: " + '\n' + str(np.around(np.linalg.norm([Vb[3], Vb[4], Vb[5]]), decimals=3)) + '\n')

        #append the joint vector of the latest iteration to the joint vector matrix
        joint_vector = np.append(joint_vector, thetalist)

        #calculate the new thetalist
        thetalist = np.around(thetalist \
                    + np.dot(np.linalg.pinv(JacobianBody(Blist, \
                                                         thetalist)), Vb), decimals=3)
        i = i + 1

        #update Vb
        Vb \
        = np.around(se3ToVec(MatrixLog6(np.dot(TransInv(FKinBody(M, Blist, \
                                                       thetalist)), T))), decimals=3)

        #check if Vb at the current iteration passes the error test, if so, exit loop
        err = np.linalg.norm([Vb[0], Vb[1], Vb[2]]) > eomg \
              or np.linalg.norm([Vb[3], Vb[4], Vb[5]]) > ev

    #output the resulting inverse kinematics once the loop has finalized and found the result
    print ("Result IK:"'\n')
    print ("joint vector: " + '\n' + str(thetalist) + '\n')
    print ("SE3(3) end-effector config: " + '\n' + str(Tsb) + '\n')
    print ("Vb: " + '\n' + str(Vb) + '\n')
    print ("angular error magnitude omega_b: " + '\n' + str(np.around(np.linalg.norm([Vb[0], Vb[1], Vb[2]]), decimals=3)) + '\n')
    print ("linear error magnitude v_b: " + '\n' + str(np.around(np.linalg.norm([Vb[3], Vb[4], Vb[5]]), decimals=3)) + '\n')

    #append the resulting IK to the joint vector matrix
    joint_vector = np.append(joint_vector, thetalist)

    #reshape the joint vector list into an (iterations + result x number of joints) matrix
    joint_vector= np.around(np.reshape(joint_vector, (i+1, len(thetalist))), decimals=3)

    #save the joint vector matrix to a .csv file
    np.savetxt("iterates.csv", joint_vector, delimiter=",")

#call IKinBodyIterates with the Universal Robots UR5 parameters given above
IKinBodyIterates(Blist_UR5, M_UR5, T_UR5, thetalist_UR5, eomg_UR5, ev_UR5)