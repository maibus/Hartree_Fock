import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from numpy import inf, nan
from numpy import pi
from scipy.integrate import simps
from math import erf
from scipy import special


e = 1.602*10**(-19)
hbar = 6.626*10 ** (-34) * 1 / (2 * np.pi)


def boys_0(x):
    return np.sqrt(pi) / (2 * np.sqrt(x)) * special.erf(1 * np.sqrt(x))


def Smat_gauss(R, alpha, C, N):
    out_matrix = np.zeros([N, N])
    for n in range(N):
        for m in range(N):
            prefactor = (pi / (alpha[n] + alpha[m]))** (3/2)
            exponential = -alpha[n] * alpha[m] / (alpha[n] + alpha[m]) * (R[n] - R[m]) ** 2
            out_matrix[n, m] = prefactor * np.exp(exponential) * C[n] * C[m]
    return out_matrix


def Tmat_gauss(R, alpha, C, N, Smat):
    out_matrix = np.zeros([N, N])
    for n in range(N):
        for m in range(N):
            X = alpha[n] * alpha[m] / (alpha[n] + alpha[m])  # just a repeated part
            prefactor = 0.5 * X * Smat[n, m]
            main = (6 - 4 * X * (R[m] - R[n]) ** 2)
            out_matrix[n, m] = prefactor * main# * C[n] * C[m]
    return out_matrix
            

def Amat_gauss(R, Rmat, alpha, N, Smat, Z):
    out_matrix = np.zeros([N, N])
    for a in range(np.size(Z)):
        for n in range(N):
            for m in range(N):
                R_a = R[a] + 10 ** (-10)  # to overcome /0 errors, temporary fix
                prefactor = -Smat[n, m] / np.abs(Rmat[n, m] - R_a)
                main = np.sqrt(alpha[n] + alpha[m]) * np.abs(Rmat[n, m] - R_a)
                out_matrix[n, m] += prefactor * special.erf(main) * Z[a]
    return out_matrix


def Qmat_gauss(Rmat, alpha, C, N, Smat):
    out_matrix = np.zeros([N, N, N, N])
    for m in range(N):
        for n in range(N):
            for o in range(N):
                for p in range(N):
                    dx = 10 ** (-10)  # for /0 prevention
                    prefactor = Smat[m, o] * Smat[n, p] / np.abs(Rmat[m, o] - Rmat[n, p] + dx)
                    p1 = alpha[m] + alpha[o]
                    p2 = alpha[n] + alpha[p]
                    main = p1 * p2 / (p1 + p2) * np.abs(Rmat[m, o] - Rmat[n, p])
                    out_matrix[m, n, o, p] = prefactor * special.erf(main) * C[m] * C[n] * C[o] * C[p]
    return out_matrix


def Rmat_gauss(R, alpha, N):
    out_matrix = np.zeros([N, N])
    for m in range(N):
        for n in range(N):
            out_matrix[m, n] = (alpha[m] * R[m] + alpha[n] * R[n]) / (alpha[m] + alpha[n])
    return out_matrix



    
    

            
    
