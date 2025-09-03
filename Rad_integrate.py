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


def getS(f1, f2, r, dr):
    integrand = f1 * f2
    J = 4 * np.pi * r ** 2
    return simps(J * integrand, r)


def getT(f1, f2, r, dr):
    integrand = f1 * laplacian(r, f2, dr)
    J = 4 * np.pi * r ** 2
    return simps(J * integrand, r)


def getA(R, Z, f1, f2, r, dr):
    R_index = int(R / dr)
    rho = f1 * f2
    inner = (1 / r) * np.sum((rho * r ** 2)[:R_index] * dr)
    outer = np.sum((rho * r)[R_index:] * dr)
    integral = 4 * np.pi * (inner + outer)
    return integral

def getQ(f1, f2, f3, f4, r, dr):
    rho1 = f1 * f2
    rho2 = f3 * f4
    inner = (1 / r) * np.cumsum(rho1 * r ** 2) * dr
    outer = np.sum(rho1 * r) * dr - np.cumsum(rho1 * r) * dr
    integrand = 4 * np.pi * r ** 2 * rho2 * 4 * np.pi * (inner + outer)
    return simps(integrand, r)
