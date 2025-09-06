import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from numpy import inf, nan
from numpy import pi
from scipy.integrate import simps
from math import erf
from Gauss_integrate import Smat_gauss, Tmat_gauss, Amat_gauss, Qmat_gauss, Rmat_gauss
from GuassHF import Hartree_Fock
from constants import a_0

R = np.array([0, 0.8 * 10 ** (-10)])  # initialise positions of ions
R /= a_0  # put into units of Bohr radii
Z = np.array([1, 2])  # initialise ion charges
N = 2  # number of electrons


basis = np.zeros([2, 3])  # 3 entries per wavefunction - C, alpha, R
basis[0, :] = np.array([0.3696, 0.4166, R[0]])
basis[1, :] = np.array([0.5881, 0.7739, R[1]])

HFsolver = Hartree_Fock(Z, R, N, molecule = True)
HFsolver.get_staq(basis)
HFsolver.solve(verbose = True, thresh = 0.0000001)


