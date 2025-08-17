import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from numpy import inf, nan
import numpy.pi as pi
from scipy.integrate import simps
from math import erf
from Gauss_integrate import Smat_gauss, Tmat_gauss, Amat_gauss, Qmat_gauss, Rmat_gauss

e = 1.602*10**(-19)
hbar = 6.626*10 ** (-34) * 1 / (2 * np.pi)
m_e = 9.109 * 10 ** (-31)
m_p = 1.673 * 10 ** (-27)
epsilon_0 = 8.854 * 10 ** (-12)
a_0 = (4 * np.pi * epsilon_0 * hbar ** 2) / (m_e * e ** 2)


R = np.array([0, 0.8 * 10 ** (-10)])  # initialise positions of ions
Z = np.array([1, 2])  # initialise ion charges
N = 2  # number of electrons


basis = np.zeros([2, 3])  # 3 entries per wavefunction - C, alpha, R
basis[0, :] = np.array([0.3696, 0.4166, R[0]])
basis[1, :] = np.array([0.5881, 0.7739, R[1]])


Rmatrix = Rmat_gauss(basis[:, 2], basis[:, 1])
S = Smat_gauss(basis[:, 2], basis[:, 1], basis[:, 0], N)
T = Tmat_gauss(basis[:, 2], basis[:, 1], basis[:, 0], N, S)
A = Amat_gauss(R, Rmatrix, basis[:, 1], N, S)
Q = Qmat_gauss(Rmatrix, basis[:, 1], basis[:, 0], N, S)


