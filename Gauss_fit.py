import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from numpy.linalg import lstsq

def STO(r):
    return 0.5641895835477563 * np.exp(-1 * r)


def G(a, r):
    C = (2 * a/np.pi) ** (3/4)
    return C * np.exp(-a * r ** 2)


def getCs(alphas, r, target, weight):
    F = np.column_stack([G(a, r) for a in alphas]) * weight[:, None]
    Y = target(r) * weight
    c = lstsq(F, Y.ravel(), rcond = None)[0]
    return c


def error(alphas, r, weight, target):
    c = getCs(alphas, r, target, weight)
    approx = np.dot(np.column_stack([G(a, r) for a in alphas]), c)
    return np.sum((target(r) - approx) ** 2 * weight)


def getGTO(zeta, N):   
    r = np.linspace(0, 50, 500)
    weight = r ** 2

    init_alphas = np.zeros(N)
    init_alphas.fill(4)
    init_alphas = 0.1 * init_alphas ** np.arange(N)
    
    errorfunc = lambda alphas: error(alphas, r, weight, STO)
    alphas = minimize(errorfunc, init_alphas, method='L-BFGS-B')
    alphas = np.array(alphas['x'])
    Cs = getCs(alphas, r, STO, weight)
    return Cs * (2 * alphas/np.pi) ** (3/4) * 1.24 ** (3/2), alphas * zeta **2,

'''
for N in range(1, 7):
    if N != 2:
        C, alpha = getGTO(1.24, N)
        print(C)
        print(alpha)
        r = np.linspace(0, 5, 500)
        plt.plot(r, 0.779036114974963 * np.exp(-1.24 * r), c='black')
        F = np.zeros(500)
        for n in range(N):
            F += C[n] * np.exp(-alpha[n] * r ** 2)

        plt.plot(r, F, label='%s -G' % N)
plt.legend()
plt.show()
'''
