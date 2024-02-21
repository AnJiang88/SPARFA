# dataPreparation.h
# dataPreparation.cpp

import numpy as np
import random
from datetime import datetime
random.seed(datetime.now().timestamp())


class DataPreparation:

	dataThreshold = 0.5
	obsThreshold = 0.5

	def __init__(self, q, n, k):
		self.Q = q
		self.N = n
		self.K = k
		self.W = np.zeros(self.Q * self.K)
		self.C = np.zeros(self.K * self.N)
		self.Y = np.zeros([self.Q, self.N])
		self.observation = np.zeros([self.Q, self.N])

	def initialization(self):
		# for W:
		for i in range(0, self.Q):
			for j in range(0, self.K):
				randt = random.random()
				if randt < self.dataThreshold:
					self.W[i * self.K+j] = 5 * random.random()
		# for C:
		for i in range(0, self.K):
			for j in range(0, self.N):
				self.C[i * self.N + j] = 2 * random.random() - 1

		# for main data:
		for i in range(0, self.Q):
			for j in range(0, self.N):
				sum_wc = 0
				for k in range(0, self.K):
					sum_wc += self.W[i * self.K + k] * self.C[j + k * +self.N]
				if sum_wc > 0:
					self.Y[i][j] = 1
					self.observation[i][j] = 1
					if random.random() < self.obsThreshold / 5:
						self.observation[i][j] = -1
				else:
					self.Y[i][j] = 0
					self.observation[i][j] = 0
					if random.random() < self.obsThreshold:
						self.observation[i][j] = -1

	def GetMainData(self):
		return self.Y

	def GetObservation(self):
		return self.observation
