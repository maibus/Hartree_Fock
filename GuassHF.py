import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from numpy import inf, nan
from numpy import pi
from scipy.integrate import simps
from math import erf
from Gauss_integrate import Smat_gauss, Tmat_gauss, Amat_gauss, Qmat_gauss, Rmat_gauss
from Rad_integrate import getT


class Hartree_Fock:
    def __init__(self, Z, R, N, molecule):
        '''
        Z is charge in units of e
        R is nuclear positions in units of a_0
        N is the number of electrons
        '''
        self.Z = Z
        self.R = R
        self.N = N
        self.molecule = molecule
        self.e = 1.602*10**(-19)
        self.hbar = 6.626*10 ** (-34) * 1 / (2 * np.pi)
        self.m_e = 9.109 * 10 ** (-31)
        self.m_p = 1.673 * 10 ** (-27)
        self.epsilon_0 = 8.854 * 10 ** (-12)
        self.a_0 = (4 * np.pi * self.epsilon_0 * self.hbar ** 2) / (self.m_e * self.e ** 2)
        #self.R /= self.a_0  # puts R in units of a_0

    def get_staq(self, basis):
        self.M = np.shape(basis)[1]
        self.Rmatrix = Rmat_gauss(basis, self.N, self.M)
        self.Stens = Smat_gauss(basis, self.N, self.M)  # 4D tensor
        self.S = np.sum(np.sum(self.Stens, axis=2), axis=2)
        self.T = Tmat_gauss(basis, self.N, self.M, self.Stens)
        self.A = Amat_gauss(basis, self.R, self.Rmatrix, self.N, self.M, self.Stens, self.Z)
        self.Q = Qmat_gauss(basis, self.N, self.M)

    def internuclear(self, R, Z):
        E = 0
        N = np.size(Z)
        nums = np.arange(N)
        for n in range(N):
            E += Z[n] * np.sum(Z * (1 - (nums == n)) / np.abs(R[n] - R + 10**(-10)))
        return E*0.5

    def getP(self, C):
        P_out = np.empty_like(C)
        for a in range(1):  # how do we make this general
            for m in range(np.shape(C)[0]):
                for n in range(np.shape(C)[1]):
                    P_out[m, n] = 2 * C[m, a] * np.conjugate(C[n, a])  # probably remove conjugate
        return P_out    

    def matmul4d(self, A, B, N):
        output = np.zeros([N, N])
        for m in range(N):
            for n in range(N):
                output[m, n] = np.sum(A * B[m, n, :, :])
        return output

    def plotwavefuncs(self, basis):
        r = np.linspace(0, 5, 500)
        for m in range(self.N):
            F = np.zeros(500)
            for n in range(self.N):
                for o in range(self.M):
                    F += self.C[n, m] * basis[n, o, 0] * np.exp(-basis[n, o, 1] * (r - basis[n, o, 2]) ** 2)
            plt.plot(r, F**2 * r**2, label = "orbital " + str(m + 1))
        plt.legend()
        plt.xlabel("r/a_0")
        plt.show()

    def solve(self, verbose, thresh):
        self.C = np.zeros([self.N, self.N])
        P = self.getP(self.C)
        diff = 1
        Enuc = 0
        if self.molecule:
            Enuc = self.internuclear(self.R, self.Z)

        Sval, Svec = np.linalg.eigh(self.S)  # from Adam Baskerville
        Sval_inverseroot = np.diag(Sval**(-0.5))
        X =  np.dot(Svec, np.dot(Sval_inverseroot, np.transpose(Svec)))

        while diff > thresh:
            h = self.T + self.A
            J = self.matmul4d(P, np.transpose(self.Q, (0, 1, 3, 2)), self.N)
            K = -0.5 * self.matmul4d(P, np.transpose(self.Q, (0,2,3,1)), self.N)

            H = h + J + K

            H_prime = np.matmul(X.T, np.matmul(H, X))
            epsilon, C_prime = np.linalg.eigh(H_prime)

            self.C = np.matmul(X, C_prime)
            P_old = P
            P = self.getP(self.C)
            diff = np.sum(np.sum((P_old - P) ** 2))
            energy = 0.5 * np.sum(np.dot(P, h + H))
            if verbose:
                print(np.sum(P * self.S))
                print(0.5 * np.sum(P * (h + H)) + Enuc, "E")
                print(epsilon, "levels")
        return 0.5 * np.sum(P * (h + H)) + Enuc


