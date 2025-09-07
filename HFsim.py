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

R = np.array([0, 0.74 * 10 ** (-10)])  # initialise positions of ions
R /= a_0  # put into units of Bohr radii
Z = np.array([1, 1])  # initialise ion charges
N = 2  # number of electrons


basis = np.zeros([2, 3, 3])  # 3 entries per wavefunction - C, alpha, R
basis[0, 0, :] = np.array([0.0835, 0.1689, R[0]])
basis[0, 1, :] = np.array([0.2678, 0.6239, R[0]])
basis[0, 2, :] = np.array([0.2769, 3.4253, R[0]])

'''
basis[1, 0, :] = np.array([0.15432897, 6.36242139, R[1]])
basis[1, 1, :] = np.array([0.53532814, 1.15892300, R[1]])
basis[1, 2, :] = np.array([0.44463454, 0.31364979, R[1]])
'''
basis[1, 0, :] = np.array([0.0835, 0.1689, R[1]])
basis[1, 1, :] = np.array([0.2678, 0.6239, R[1]])
basis[1, 2, :] = np.array([0.2769, 3.4253, R[1]])

HFsolver = Hartree_Fock(Z, R, N, molecule = True)
HFsolver.get_staq(basis)
HFsolver.solve(verbose = True, thresh = 0.0000001)
print("done!")
Es = np.zeros(100)
for n in range(20, 120):
    print(n)
    R = np.array([0, n/100 * 10 ** (-10)])  # initialise positions of ions
    R /= a_0  # put into units of Bohr radii
    Z = np.array([1, 1])  # initialise ion charges
    N = 2  # number of electrons
    basis = np.zeros([2, 3, 3])  # 3 entries per wavefunction - C, alpha, R
    basis[0, 0, :] = np.array([0.0835, 0.1689, R[0]])
    basis[0, 1, :] = np.array([0.2678, 0.6239, R[0]])
    basis[0, 2, :] = np.array([0.2769, 3.4253, R[0]])

    basis[1, 0, :] = np.array([0.0835, 0.1689, R[1]])
    basis[1, 1, :] = np.array([0.2678, 0.6239, R[1]])
    basis[1, 2, :] = np.array([0.2769, 3.4253, R[1]])
    HFsolver = Hartree_Fock(Z, R, N, molecule = True)
    HFsolver.get_staq(basis)
    Es[n-20] = HFsolver.solve(verbose = False, thresh = 0.0000001)

plt.plot(np.linspace(20,120,100), Es)
plt.show()

