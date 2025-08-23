import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from numpy import inf, nan
from numpy import pi
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
R /= a_0  # put into units of Bohr radii
Z = np.array([1, 2])  # initialise ion charges
N = 2  # number of electrons


basis = np.zeros([2, 3])  # 3 entries per wavefunction - C, alpha, R
basis[0, :] = np.array([0.3696, 0.4166, R[0]])
basis[1, :] = np.array([0.5881, 0.7739, R[1]])


Rmatrix = Rmat_gauss(basis[:, 2], basis[:, 1], N)
S = Smat_gauss(basis[:, 2], basis[:, 1], basis[:, 0], N)
T = Tmat_gauss(basis[:, 2], basis[:, 1], basis[:, 0], N, S)
A = Amat_gauss(R, Rmatrix, basis[:, 1], N, S, Z)
Q = Qmat_gauss(Rmatrix, basis[:, 1], basis[:, 0], N, S)
print(Q)
print("hi")

def getP(C):
    P_out = np.empty_like(C)
    for m in range(np.shape(C)[0]):
        for n in range(np.shape(C)[0]):
            P_out[m, n] = 2 * np.sum(C[m, :] * np.conjugate(C[n, :]))
    return P_out
    

def matmul4d(A, B, N):
    output = np.zeros([N, N])
    for m in range(N):
        for n in range(N):
            output[m, n] = np.sum(np.sum(A * B[m, n, :, :]))
    return output


def solver(N, S, T, A, Q):
    C = np.random.random([N, N])
    P = getP(C)
    diff = 1
    thresh = 0.0001

    Sval, Svec = np.linalg.eigh(S)  # from Adam Baskerville
    Sval_inverseroot = np.diag(Sval**(-0.5))
    X =  np.dot(Svec, np.dot(Sval_inverseroot, np.transpose(Svec)))

    while diff > thresh:
        h = T + A
        J = matmul4d(P, np.transpose(Q, (0, 1, 3, 2)), N)
        K = 0.5 * matmul4d(P, np.transpose(Q, (0, 3, 1, 2)), N)

        H = h + J + K
        print(A)

        H_prime = np.matmul(X.T, np.matmul(H, X))
        epsilon, C_prime = np.linalg.eigh(H_prime)

        C = np.matmul(X, C_prime)
        print(C)
        P_old = P
        P = getP(C)
        diff = np.sum(np.sum((P_old - P) ** 2))
        print(diff)
    print(np.sum(epsilon))

solver(N, S, T, A, Q)
        

    
    
    

