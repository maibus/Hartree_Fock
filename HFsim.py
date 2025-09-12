import numpy as np
import matplotlib.pyplot as plt
from Gauss_integrate import Smat_gauss, Tmat_gauss, Amat_gauss, Qmat_gauss, Rmat_gauss
from GuassHF import Hartree_Fock
from constants import a_0
from optimise import sweep1D, MH1D
from Gauss_fit import getGTO


def energy(r):
    R = np.array([0, r])  # initialise positions of ions
    R /= a_0  # put into units of Bohr radii
    Z = np.array([1, 1])  # initialise ion charges
    N = 2  # number of electrons
    nG = 6  # number of Gaussians used to approximate orbital

    basis = np.zeros([2, nG, 3])  # 3 entries per wavefunction - C, alpha, R

    Cs, alphas = getGTO(1, nG)
    basis[0, :, 0] = Cs
    basis[0, :, 1] = alphas
    basis[0, :, 2] = R[0]

    Cs, alphas = getGTO(2, nG)
    basis[1, :, 0] = Cs
    basis[1, :, 1] = alphas
    basis[1, :, 2] = R[1]


    HFsolver = Hartree_Fock(Z, R, N, molecule = False)
    HFsolver.get_staq(basis)
    E = HFsolver.solve(verbose = True, thresh = 0.0000001)
    #print("done!")
    return E

print(energy(0.0))  # known H-2 bond length

#r_min, Emin = sweep1D(energy, [0.2 * 10 ** (-10), 1 * 10 ** (-10)], num = 100, visualise = True)
#print(r_min, Emin * 27.2114, "-eV")


