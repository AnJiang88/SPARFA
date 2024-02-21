# Learning analytics and content analytics:
# 1. Estimate correct answer likelihood for a learner.
# 2. Estimate concept knowledge for a learner

import numpy as np
from scipy.stats import norm


class Inference:
    def __init__(self, learner_responses, w, c, mu, q, n, k):
        self.Y = learner_responses
        self.W = w
        self.C = c
        self.mu = mu
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
        inner_wicj = np.matmul(self.W, self.C)[index_i][index_j]
        return norm.cdf(inner_wicj)

    def concepts(self, question_number):
        index_i = question_number
        underlying_concepts = []
        # the indices of the non-zero entries of w_(index_i) correspond to related underlying concepts
        for k in range(self.K):
            if self.W[index_i * self.K + k] != 0:
                underlying_concepts.append(k + 1)
        return underlying_concepts

    def knowledge(self, student_number):  # put a learner's concept knowledge in a dictionary
        index_j = student_number
        knowledge_dict = {}
        for k in range(self.K):
            knowledge_dict[k + 1] = round(self.C[index_j + k * self.N], 2)
        return knowledge_dict

    def get_intrinsic_difficulty(self):
        return self.mu


















