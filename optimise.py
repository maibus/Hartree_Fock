import numpy as np
import matplotlib.pyplot as plt
from random import choice, random

def sweep1D(energy, limits, num, visualise = False):
    r = np.linspace(limits[0], limits[1], num)
    energies = np.zeros(num)
    for n in range(num):
        energies[n] = energy(r[n])

    if visualise:
        plt.scatter(r, energies, s=2)
        plt.show()

    return r[np.argmin(energies)], np.min(energies)

def MH1D(energy, step, shrinkage, r_0, num):
    r = r_0
    E_0 = energy(r)
    for n in range(num):
        step = choice([-1, 1]) * step * (shrinkage) ** n
        E_n = energy(r + step)
        exp = np.exp(100 * (E_n - E_0))
        print(r, exp, step, E_n, E_0)
        if exp > 1:
            r += step
        else:
            move = int(random() < exp)
            r += step
            E_0 = E_0 * (1 - move) + E_n * move
            
    return r, E_0
        
        
        
        
