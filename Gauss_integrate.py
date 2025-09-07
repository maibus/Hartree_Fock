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


def S_element(C1, alpha1, R1, C2, alpha2, R2):
    prefactor = (pi / (alpha1 + alpha2))** (3/2)
    exponential = -alpha1 * alpha2 / (alpha1 + alpha2) * (R1 - R2) ** 2
    return prefactor * np.exp(exponential) * C1 * C2

def T_element(alpha1, R1, alpha2, R2, S):
    X = alpha1 * alpha2 / (alpha1 + alpha2)  # just a repeated part
    prefactor = 0.5 * X * S
    main = (6 - 4 * X * (R2 - R1) ** 2)
    return prefactor * main# * C[n] * C[m]

def A_element(alpha1, R1, alpha2, R2, mu, S, Z, R_a):
    R_a += 10 ** (-10)  # to overcome /0 errors, temporary fix
    prefactor = -S / np.abs(mu - R_a)
    main = np.sqrt(alpha1 + alpha2) * np.abs(mu - R_a)
    return prefactor * special.erf(main) * Z

def Q_element(C1, C2, C3, C4, alpha1, alpha2, alpha3, alpha4, R1, R2, R3, R4):
    gamma_1 = alpha1 + alpha2
    mu_1 = (alpha1 * R1 + alpha2 * R2) / gamma_1
    C_1 = C1 * C2 * np.exp(-alpha1*alpha2 / gamma_1 * (R2 - R1) ** 2)
    gamma_2 = alpha3 + alpha4
    mu_2 = (alpha3 * R3 + alpha4 * R4) / gamma_2
    C_2 = C3 * C4 * np.exp(-alpha3 * alpha4 / gamma_2 * (R3 - R4) ** 2)
    alpha_tot = (gamma_1 * gamma_2) / (gamma_1 + gamma_2)  #
    R_12 = (gamma_1 * mu_1 + gamma_2 * mu_2) / (gamma_1 + gamma_2)  #
    return C_1 * C_2 * 2 * pi ** (5/2) / (gamma_1 * gamma_2 * np.sqrt(gamma_1 + gamma_2)) * boys_0(alpha_tot * (mu_1 - mu_2) ** 2 + 10 ** (-10))  #
   


def Smat_gauss(basis, N, M):
    out_matrix = np.zeros([N, N, M, M])
    
    for n in range(N):
        for m in range(N):
            for i in range(M):
                for j in range(M):
                    out_matrix[n, m, i, j] = S_element(basis[n, i, 0], basis[n, i, 1], basis[n, i, 2], basis[m, j, 0], basis[m, j, 1], basis[m, j, 2])
    return out_matrix


def Tmat_gauss(basis, N, M, Smat):
    out_matrix = np.zeros([N, N])
    for n in range(N):
        for m in range(N):
            for i in range(M):
                for j in range(M):
                    out_matrix[n, m] += T_element(basis[n, i, 1], basis[n, i, 2], basis[m, j, 1], basis[m, j, 2], Smat[n, m, i, j])
    return out_matrix
            

def Amat_gauss(basis, R_a, Rmat, N, M, Smat, Z):
    out_matrix = np.zeros([N, N])
    for a in range(np.size(Z)):
        for n in range(N):
            for m in range(N):
                for i in range(M):
                    for j in range(M):
                        out_matrix[n, m] += A_element(basis[n, i, 1], basis[n, i, 2], basis[m, j, 1], basis[m, j, 2], Rmat[n, m, i, j], Smat[n, m, i, j], Z[a], R_a[a])
    return out_matrix


def Qmat_gauss(basis, N, M):
    out_matrix = np.zeros([N, N, N, N])
    for m in range(N):
        for n in range(N):
            for o in range(N):
                for p in range(N):
                    for i in range(M):
                        for j in range(M):
                            for k in range(M):
                                for l in range(M):
                                    out_matrix[m, n, o, p] += Q_element(basis[m, i, 0], basis[n, j, 0], basis[o, k, 0], basis[p, l, 0],
                                                                       basis[m, i, 1], basis[n, j, 1], basis[o, k, 1], basis[p, l, 1],
                                                                       basis[m, i, 2], basis[n, j, 2], basis[o, k, 2], basis[p, l, 2])
                    
    return out_matrix


def Rmat_gauss(basis, N, M):
    out_matrix = np.zeros([N, N, M, M])
    for m in range(N):
        for n in range(N):
            for i in range(M):
                for j in range(M):
                    out_matrix[m, n, i, j] = (basis[m, i, 1] * basis[m, i, 2] + basis[n, j, 1] * basis[n, j, 2]) / (basis[m, i, 1] + basis[n, j, 1])
    return out_matrix



    
    

            
    
