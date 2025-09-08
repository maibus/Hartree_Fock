import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def STO(r):
    return 0.7790 * np.exp(-1.24 * r)

def fitfunc(r, a1, c1):
    return c1 * np.exp(-a1 * r ** 2)# + c2 * np.exp(-a2 * r ** 2) + c3 * np.exp(-a3 * r ** 2)

r = np.linspace(1, 10, 500)
guess = [0.4166, 0.3696]

popt, pcov = curve_fit(fitfunc, r, STO(r), p0=guess)
print(popt)

plt.plot(r, STO(r))
plt.plot(r, fitfunc(r, *popt))
plt.plot(r, fitfunc(r, 0.4166, 0.3696))
plt.show()


