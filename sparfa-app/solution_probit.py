# solution.h
# solution.cpp

import numpy as np
import random
from scipy.stats import norm
# import DataRetriever # TBD

class Solution:

    def __init__(self, learner_responses, q, n, k):
        self.Y = learner_responses
        self.Q = q
        self.N = n
        self.K = k  # K = number of abstract underlying concepts + 1
        self.W = np.random.normal(size=self.Q * self.K)  # The intrinsic difficulty vector is incorporated as an additional column of W
        self.C = np.random.normal(size=(self.K - 1) * self.N)  # C is augmented with an all-ones row accordingly
        for j in range(0, self.N):
            self.C = np.append(self.C, 1)

    def getW(self):
        return self.W

    def getC(self):
        return self.C

    def get_intrinsic_difficulty(self):
        mu = np.zeros(self.Q)
        for i in range(0, self.Q):
            mu[i] = self.W[i * (self.K + 1) + self.K]
        return mu

    def objectiveFunction(self, lbd, mu, gamma):
        sumLog = 0
        sumWl1 = 0
        sumWl2 = 0
        sumC = 0

        for i in range(0, self.Q):
            for j in range(0, self.N):
                if self.Y[i][j] != -1:  # The sum is subject to all the observed graded learner response data.
                    WiCj = 0
                    for k in range(0, self.K):
                        WiCj += self.W[i * self.K + k] * self.C[j + k * self.N]
                    # print(f"WiCj is to: {WiCj}, and norm.cdf(WiCj) is {norm.cdf(WiCj)}.")
                    # sumLog += self.Y[i][j] * np.log(norm.cdf(WiCj)) + (
                    #             1 - self.Y[i][j]) * np.log(1 - norm.cdf(WiCj))
                    # print(f"np.log(norm.cdf(WiCj)) is: {np.log(norm.cdf(WiCj))} and np.log(1-norm.cdf(WiCj)) is {np.log(1 - norm.cdf(WiCj))}.")
                    # print(f"norm.logcdf(WiCj) is: {norm.logcdf(WiCj)}.")
                    # print(f"norm.logcdf(-WiCj) is: {norm.logcdf(-WiCj)}.")
                    sumLog += self.Y[i][j] * norm.logcdf(WiCj) + (
                            1 - self.Y[i][j]) * norm.logcdf(-WiCj)


            for k in range(0, self.K):  # l1 norm of w_i and the square of l2 norm of w_i
                sumWl1 += -self.W[i * self.K + k] if self.W[i * self.K + k] < 0 else self.W[i * self.K + k]
                sumWl2 += self.W[i * self.K + k] * self.W[i * self.K + k]

        for i in range(0, self.K):
            for j in range(0, self.N):
                sumC += self.C[i * self.N + j] * self.C[i * self.N + j]

        return -sumLog + lbd * sumWl1 + mu * sumWl2 + gamma * sumC

    def objectiveFunctionW(self, lbd, mu):
        sumLog = 0
        sumWl1 = 0
        sumWl2 = 0

        for i in range(0, self.Q):
            for j in range(0, self.N):
                if self.Y[i][j] != -1:
                    WiCj = 0
                    for k in range(0, self.K):
                        WiCj += self.W[i * self.K + k] * self.C[j + k * self.N]
                    sumLog += self.Y[i][j] * np.log(norm.cdf(WiCj)) + (
                                1 - self.Y[i][j]) * np.log(1 - norm.cdf(WiCj))

            for k in range(0, self.K):
                sumWl1 += -self.W[i * self.K + k] if self.W[i * self.K + k] < 0 else self.W[i * self.K + k]
                sumWl2 += self.W[i * self.K + k] * self.W[i * self.K + k]

        return -sumLog + lbd * sumWl1 + mu * sumWl2

    def objectiveFunctionC(self, gamma):
        sumLog = 0
        sumC = 0

        for i in range(0, self.Q):
            for j in range(0, self.N):
                if self.Y[i][j] != -1:
                    WiCj = 0
                    for k in range(0, self.K):
                        WiCj += self.W[i * self.K + k] * self.C[j + k * self.N]
                    sumLog += self.Y[i][j] * np.log(norm.cdf(WiCj)) + (
                                1 - self.Y[i][j]) * np.log(1 - norm.cdf(WiCj))

        for i in range(0, self.K):
            for j in range(0, self.N):
                sumC += self.C[i * self.N + j] * self.C[i * self.N + j]
        return -sumLog + gamma * sumC

    def updateWij(self, index_i, index_k, lbd, mu, stepSize):
        yipi = np.zeros(self.N)  # y_index_i - p_pro^index_i is an N * 1 column vector
        Diyipi = np.zeros(self.N)
        term = 0

        for j in range(0, self.N):
            WiCj = 0  # inner product of w_index_i and c_j
            for k in range(0, self.K):
                WiCj += self.W[index_i * self.K + k] * self.C[j + k * self.N]
            # (y^index_i - p_pro^index_i)_j
            if self.Y[index_i][j] != -1:
                yipi[j] = self.Y[index_i][j] - norm.cdf(WiCj)
            else:  # if the response is unobserved, we impute the missing observation with randomly generated 0 and 1
                yipi[j] = random.randint(0, 1) - norm.cdf(WiCj)
            Diyipi[j] = norm.pdf(WiCj) / (0.01 + norm.cdf(WiCj) * (1 - norm.cdf(WiCj))) * yipi[j]  # divide by zero error without adding 0.01

            # For logistic regression case, delF is Equation (9) on page 1969
            # For probit regression, delF is Equation (5) on page 1968, and the mathematical deduction is as follows:
            # -CD(y^i - p_pro^i): by the associative property of matrix multiplication, we can compute D(y^i - p_pro^i) first to get a N * 1 column vector, and then compute its product with C
            # an N * N diagonal matrix times an N * 1 column vector is equal to a N * 1 column vector with the j-th element equals the j-th element on the diagonal times the j-th element in the column vector
            term += self.C[index_k * self.N + j] * Diyipi[j]  # the k-th element is equal to the k-th row of C times the column vector yipi

        delF = -term + mu * self.W[index_i * self.K + index_k]  # delF_(index_k)
        gradient_step = self.W[index_i * self.K + index_k] - stepSize * delF
        if index_k == self.K - 1:  # intrinsic difficulty vector (The last column incorporated inside W) doesn't need to comply with Non-negativity assumption (A3)
            self.W[index_i * self.K + index_k] = max(abs(gradient_step) - lbd * stepSize, 0) * np.sign(gradient_step)
        else:
            self.W[index_i * self.K + index_k] = max(gradient_step - lbd * stepSize, 0)

        return self.W[index_i * self.K + index_k]

    def updateCij(self, index_k, index_j, gamma, stepSize):
        yjpj = np.zeros(self.Q)
        Djyjpj = np.zeros(self.Q)
        term = 0

        for i in range(0, self.Q):
            WiCj = 0
            for k in range(0, self.K):
                WiCj += self.W[i * self.K + k] * self.C[index_j + k * self.N]

            if self.Y[i][index_j] != -1:
                yjpj[i] = self.Y[i][index_j] - norm.cdf(WiCj)
            else:
                yjpj[i] = random.randint(0, 1) - norm.cdf(WiCj)
            Djyjpj[i] = norm.pdf(WiCj) / (0.01 + norm.cdf(WiCj) * (1 - norm.cdf(WiCj))) * yjpj[i]  # divide by zero error without adding 0.01

            term += self.W[i * self.K + index_k] * Djyjpj[i]

        delF = -term  # + gamma * self.C[index_k * self.N + index_j]  # delF_(index_k)
        if index_k == self.K - 1:
            self.C[index_k * self.N + index_j] = 1
        else:
            self.C[index_k * self.N + index_j] = (1 / (1 + gamma * stepSize)) * (self.C[index_k * self.N + index_j] - stepSize * delF)

        return self.C[index_k * self.N + index_j]
