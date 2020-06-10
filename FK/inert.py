import math
import numpy as np
import modern_robotics as mr

pi = math.pi

print("\n------ Question 1 ------")

density = 5600
rad_cyl = 0.02
len_cyl = 0.2
rad_sph = 0.1

mass_cyl = density * pi * len_cyl * rad_cyl**2
print("\nMass of Cylinder: ", mass_cyl, sep='')
Ixx_cyl = mass_cyl * (3 * rad_cyl**2 + len_cyl**2) / 12
Iyy_cyl = Ixx_cyl
Izz_cyl = mass_cyl * rad_cyl**2 / 2
RIMatrix_cyl = np.diag([Ixx_cyl, Iyy_cyl, Izz_cyl])
print("\nRIMatrix_cyl:\n", np.array2string(RIMatrix_cyl, separator=','), sep='')

mass_sph = density * 4/3 * pi * rad_sph**3
print("\nMass of Sphere: ", mass_sph, sep='')
Ixx_sph = mass_sph * 2/5 * rad_sph**2
Iyy_sph = Ixx_sph
Izz_sph = Ixx_sph
RIMatrix_sph = np.diag([Ixx_sph, Iyy_sph, Izz_sph])
print("\nRIMatrix_sph:\n", np.array2string(RIMatrix_sph, separator=','), sep='')

'''exploit Steiner's theorem'''
q1 = np.array([[0], [0], [rad_sph + len_cyl/2]])
RIMatrix_sph1 = RIMatrix_sph + mass_sph * (np.dot(q1.T, q1) * np.eye(3) - np.dot(q1, q1.T))
q2 = np.array([[0], [0], [-rad_sph - len_cyl/2]])
RIMatrix_sph2 = RIMatrix_sph + mass_sph * (np.dot(q2.T, q2) * np.eye(3) - np.dot(q2, q2.T))
print("\nRIMatrix_sph1:\n", np.array2string(RIMatrix_sph1, separator=','), sep='')
print("\nRIMatrix_sph2:\n", np.array2string(RIMatrix_sph2, separator=','), sep='')

RIMatrix_q1 = RIMatrix_cyl + RIMatrix_sph1 + RIMatrix_sph2
RIMatrix_q1_off = np.around(RIMatrix_q1, decimals=2)
print("\nQuestion 1:\n", np.array2string(RIMatrix_q1_off, separator=','), sep='')