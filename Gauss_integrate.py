import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from numpy import inf, nan
import numpy.pi as pi
from scipy.integrate import simps

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


            
    
