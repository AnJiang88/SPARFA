# Learning analytics and content analytics:
# 1. Estimate correct answer likelihood for a learner.
# 2. Estimate concept knowledge for a learner

import numpy as np
from scipy.stats import norm


class Prediction:
    def __init__(self, learner_responses, w, c, q, n, k):
        self.Y = learner_responses
        self.W = w
        self.C = c
        self.Q = q
        self.N = n
        self.K = k

    def get_Y(self, question_number, student_number):
        index_i = question_number
        index_j = student_number
        return self.Y[index_i][index_j]

    def likelihood(self, question_number, student_number):  # probability p(Y_i,j = 1 | w_i, c_j) = Phi(w_i^T * c_j)
        index_i = question_number
        index_j = student_number
        inner_wicj = 0
        for k in range(0, self.K):
            inner_wicj += self.W[index_i * self.K + k] * self.C[index_j + k * self.N]
        return norm.cdf(inner_wicj)

    def concepts(self, question_number):
        index_i = question_number
        underlying_concepts = []
        # the indices of the non-zero entries of w_(index_i) correspond to related underlying concepts
        for k in range(0, self.K - 1):
            if self.W[index_i * self.K + k] != 0:
                underlying_concepts.append(k + 1)
        return underlying_concepts

    def knowledge(self, student_number):  # put a learner's concept knowledge in a dictionary
        index_j = student_number
        knowledge_dict = {}
        for k in range(0, self.K - 1):
            knowledge_dict[k + 1] = round(self.C[index_j + k * self.N], 2)
        return knowledge_dict

    def get_intrinsic_difficulty(self):
        mu = np.zeros(self.Q)
        for i in range(0, self.Q):
            mu[i] = self.W[i * self.K + self.K - 1]
        return mu


















