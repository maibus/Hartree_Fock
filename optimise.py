import numpy as np
import matplotlib.pyplot as plt

def sweep1D(energy, limits, num, visualise = False):
    r = np.linspace(limits[0], limits[1], num)
    energies = np.zeros(num)
    for n in range(num):
        energies[n] = energy(r[n])

    if visualise:
        plt.scatter(r, energies, s=2)
        plt.show()

    return r[np.argmin(energies)], np.min(energies)
        
        
