import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from numpy import inf, nan
from numpy import pi
from scipy.integrate import simps
from math import erf
from scipy import special


def laplacian(r, f, dr):
    output = r ** (-2) * np.gradient(r ** 2 * np.gradient(f, dr), dr)
    output[0] = 0  # temporary fix
    return output


def getS(basis, r, dr):
    N = np.shape(basis)[0]
    output = np.zeros([N, N])
    for m in range(N):
        for n in range(N):
            f1 = basis[m, :]
            f2 = basis[n, :]
            integrand = f1 * f2
            plt.plot(integrand)
            plt.show()
            J = 4 * np.pi * r ** 2
            output[m, n] = simps(J * integrand, r)
    return output


def getT(basis, r, dr):
    N = np.shape(basis)[0]
    output = np.zeros([N, N])
    for m in range(N):
        for n in range(N):
            f1 = basis[m, :]
            f2 = basis[n, :]            
            integrand = f1 * laplacian(r, f2, dr)
            J = 4 * np.pi * r ** 2
            output[m, n] = simps(J * integrand, r)
    return output


def getA(R, Z, basis, r, dr):
    N = np.shape(basis)[0]
    output = np.zeros([N, N])
    for a in range(np.size(Z)):
        for m in range(N):
            for n in range(N):
                f1 = basis[m, :]
                f2 = basis[n, :]
                R_index = int(R[a] / dr)
                rho = f1 * f2
                inner = (1 / r) * np.sum((rho * r ** 2)[:R_index] * dr)
                outer = np.sum((rho * r)[R_index:] * dr)
                integral = 4 * np.pi * (inner + outer)
                output[m, n] += Z * integral
    return output


def getQ(basis, r, dr):
    N = np.shape(basis)[0]
    output = np.zeros([N, N])
    for m in range(N):
        for n in range(N):
            for o in range(N):
                for p in range(N):
                    rho1 = f1 * f2
                    rho2 = f3 * f4
                    inner = (1 / r) * np.cumsum(rho1 * r ** 2) * dr
                    outer = np.sum(rho1 * r) * dr - np.cumsum(rho1 * r) * dr
                    integrand = 4 * np.pi * r ** 2 * rho2 * 4 * np.pi * (inner + outer)
                    output[m, n, o, p] = simps(integrand, r)
    return output


