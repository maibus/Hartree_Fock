import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from numpy import inf, nan
import numpy.pi as pi
from scipy.integrate import simps
from math import erf

e = 1.602*10**(-19)
hbar = 6.626*10 ** (-34) * 1 / (2 * np.pi)


def Smat_gauss(R, alpha, N):
    out_matrix = np.zeros([N, N])
    for n in range(N):
        for m in range(N):
            prefactor = pi / (alpha[n] + alpha[m]) ** (3/2)
            exponential = -alpha[n] * alpha[m] / (alpha[n] + alpha[m]) * (R[n] - R[m]) ** 2
            out_matrix[n, m] = prefactor * np.exp(exponential)
    return out_matrix


def Tmat_gauss(R, alpha, N, Smat):
    out_matrix = np.zeros([N, N])
    for n in range(N):
        for m in range(N):
            X = alpha[n] * alpha[m] / (alpha[n] + alpha[m])  # just a repeated part
            prefactor = 0.5 * X * Smat[n, m]
            main = (6 - 4 * X * (R[m] - R[n]) ** 3)
            out_matrix[n, m] = prefactor * main
    return out_matrix
            

def Amat_gauss(R, Rmat, alpha, N, Smat):
    out_matrix = np.zeros([N, N])
    for n in range(N):
        for m in range(N):
            prefactor = -Smat[n, m] / np.abs(R[n, m] - R)
            main = np.sqrt(alpha[n] + alpha[m]) * np.abs(R[n, m] - R)
            out_matrix[n, m] = prefactor * erf(main)
    return out_matrix


def Qmat_gauss(Rmat, alpha, N, Smat):
    out_matrix = np.zeros([N, N, N, N])
    for m in range(N):
        for n in range(N):
            for o in range(N):
                for p in range(N):
                    prefactor = Smat[m, o] * Smat[n, p] / np.abs(Rmat[m, o] - Rmat[n, p])
                    p1 = alpha[m] + alpha[o]
                    p2 = alpha[n] + alpha[p]
                    main = p1 * p2 / (p1 + p2) * np.abs(R[m, o] - R[m, p])
                    out_matrix[m, n, o, p] = prefactor * erf(main)
    return out_matrix


def Rmat_gauss(R, alpha):
    out_matrix = np.zeros([N, N])
    for m in range(N):
        for n in range(N):
            out_matrix[m, n] = (alpha[m] * R[m] + alpha[n] * R[n]) / (alpha[m] + alpha[n])
    return out_matrix



    
    

            
    
