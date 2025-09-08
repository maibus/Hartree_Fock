import numpy as np
import matplotlib.pyplot as plt
from Gauss_integrate import Smat_gauss, Tmat_gauss, Amat_gauss, Qmat_gauss, Rmat_gauss
from GuassHF import Hartree_Fock
from constants import a_0
from optimise import sweep1D


def energy(r):
    R = np.array([0, r])  # initialise positions of ions
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
    E = HFsolver.solve(verbose = True, thresh = 0.0000001)
    print("done!")
    return E

energy(0.74 * 10 ** (-10))  # known H-2 bond length

r_min, Emin = sweep1D(energy, [0.2 * 10 ** (-10), 1 * 10 ** (-10)], num = 100, visualise = True)
print(r_min, Emin)

