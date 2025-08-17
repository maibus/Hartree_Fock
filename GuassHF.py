import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from numpy import inf, nan
import numpy.pi as pi
from scipy.integrate import simps
from math import erf

e = 1.602*10**(-19)
hbar = 6.626*10 ** (-34) * 1 / (2 * np.pi)
m_e = 9.109 * 10 ** (-31)
m_p = 1.673 * 10 ** (-27)
epsilon_0 = 8.854 * 10 ** (-12)
a_0 = (4 * np.pi * epsilon_0 * hbar ** 2) / (m_e * e ** 2)

