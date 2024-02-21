import math
import numpy as np
from scipy.stats import invwishart
from scipy.stats import multivariate_normal
from scipy.stats import gamma
from scipy.stats import beta
from scipy.stats import expon
from scipy.stats import norm


# Sampling methodology - MCMC steps
class Sampler:

    def __init__(self, learner_responses, q, n, k, alph, bet, e, f, h, V_0, mu_0, v_mu):
        self.Y = learner_responses  # Matrix Y, W and C are all in 2D array form
        self.Q = q
        self.N = n
        self.K = k
        self.alph = alph
        self.bet = bet
        self.e = e
        self.f = f
        self.h = h
        self.V_0 = V_0
        self.mu_0 = mu_0
        self.v_mu = v_mu
        # Initiating the parameters by sampling from their prior distributions
        # prior of W
        self.lamd = gamma.rvs(a=self.alph, loc=0, scale=1 / self.bet, size=self.K, random_state=None)
        self.r = beta.rvs(a=self.e, b=self.f, loc=0, scale=1, size=self.K, random_state=None)
        self.W = np.zeros(shape=(self.Q, self.K))
        self.R = np.zeros(shape=(self.Q, self.K))  # inclusion statistics for W
        for i in range(self.Q):  # Initialization of W has setback!
            self.W[i] = self.r * expon.rvs(loc=0, scale=1 / self.lamd, size=self.K, random_state=None) + (1 - self.r)  # * signal.unit_impulse(self.K)  # How does Dirac delta function work here as a pdf??
        # prior of C
        self.V = invwishart.rvs(df=self.h, scale=self.V_0*np.identity(self.K))
        C_T = np.zeros(shape=(self.N, self.K))
        for i in range(C_T.shape[0]):
            C_T[i] = multivariate_normal.rvs(mean=None, cov=self.V, size=1, random_state=None)
        self.C = C_T.T  # or np.transpose(C_T)  # column vector c_j follows multivariate normal distribution
        # prior of mu
        self.mu = np.random.normal(loc=self.mu_0, scale=np.sqrt(self.v_mu), size=(1, self.Q))  # define mu as a row vector for simple access
        # prior of Z = WC + M
        self.Z = np.matmul(self.W, self.C) + np.matmul(self.mu.T, np.ones(shape=(1, self.N)))

    def get_W(self):
        return self.W

    def get_R(self):
        return self.R

    def get_C(self):
        return self.C

    def get_mu(self):
        return self.mu

# 1. For all (i, j) in Omega_obs, draw Z_i,j ~ N((WC)_i,j + mu_i, 1),
# truncating above 0 if Y_i,j = 1, and truncating below 0 if Y_i,j = 0.
    def sample_Z(self):
        for i in range(self.Q):
            for j in range(self.N):
                if self.Y[i][j] != -1:
                    mean = np.matmul(self.W, self.C)[i][j] + self.mu[0][i]
                    self.Z[i][j] = min(np.random.normal(mean, 1), 0) if self.Y[i][j] == 1 else max(np.random.normal(mean, 1), 0)

# 2. For all i = 1,...,Q, draw mu_i ~ N(m_i, v) with v = (v_mu^-1 + n')^-1,
# m_i = mu_0 + vSum_{j:(i,j) in Omega_obs} (Z_i,j - (WC)_i,j),
# and n' the number of learners responding to question i, where mu_0 and v_mu are hyperparameters.
    def sample_Mu(self):
        for i in range(self.Q):
            sum_term = 0
            for j in range(self.N):
                if self.Y[i][j] != -1:
                    sum_term += self.Z[i][j] - np.matmul(self.W, self.C)[i][j]
            number_of_responding = np.count_nonzero(self.Y[i] != -1)
            v = 1 / ((1 / self.v_mu) + number_of_responding)
            m = self.mu_0 + v * sum_term
            self.mu[0][i] = np.random.normal(m, np.sqrt(v))

# 3. For all j = 1,...,N, draw c_j ~ N(m_j, M_j) with M_j = (V^-1 + W^TW)^-1, and m_j = M_jW^T(z_j - mu).
# The notation ~ denotes the restriction of the vector or matrix to the set of rows i: (i, j) in Omega_obs. What does this notation mean?!
    def sample_C(self):
        for j in range(self.N):
            M = np.linalg.inv(np.linalg.inv(self.V) + np.matmul(self.W.T, self.W))  # K by K matrix
            m = np.matmul(np.matmul(M, self.W.T), (self.Z[:, j].reshape(-1, 1) - self.mu.T))  # K by 1 vector
            m = np.squeeze(m)
            self.C[:, j] = multivariate_normal.rvs(mean=m, cov=M, size=1, random_state=None)

# 4. Draw V ~ IW(V_0 + CC^T, N + h).
    def sample_V(self):
        self.V = invwishart.rvs(df=self.N + self.h, scale=self.V_0 * np.identity(self.K) + np.matmul(self.C, self.C.T))

# 5. For all i = 1,...,Q and k = 1,...,K,
# draw W_i,k ~ R_i,k * N^r(M_i,k, S_i,k) + (1 - R_i,k)delta_0,
# where R_i,k, M_i,k and S_i,k are as stated in Theorem 3.
    @staticmethod
    def rectified_normal_pdf(x, m, s, lamd):
        exponent = -lamd ** 2 * s / 2 - (x - m) ** 2 / (2 * s) - math.log(math.sqrt(2 * math.pi * s)) - math.log(max(norm.cdf((m - lamd * s)/math.sqrt(s), loc=0, scale=1), 5e-324))
        res = max(np.exp(exponent), 5e-324)
        print(f"rectified_normal_pdf value is {res}")
        return res

    @staticmethod
    def exponential_pdf(x, lamd):
        res = lamd * math.exp(- lamd * x)
        return res

    def sample_W(self):
        for i in range(self.Q):
            for k in range(self.K):
                S, M = 0, 0
                for j in range(self.N):
                    if self.Y[i][j] != -1:
                        S += self.C[k][j] ** 2
                        M += (self.Z[i][j] - self.mu[0][i] - (np.matmul(self.W, self.C)[i][j] - self.W[i][k] * self.C[k][j])) * self.C[k][j]
                S = 1 / S
                M = M * S
                print(f"M is {M}")
                print(f"S is {S}")
                print(f"lamda_k is {self.lamd[k]}")
                print(f"r_k is {self.r[k]}")
                print(f"expo_pdf is {Sampler.exponential_pdf(0, self.lamd[k])}")
                R_term = Sampler.rectified_normal_pdf(0, M, S, self.lamd[k]) * (1 - self.r[k]) / Sampler.exponential_pdf(0, self.lamd[k])
                self.R[i][k] = R_term / (R_term + self.r[k])
                print(f"R_ik is {self.R[i][k]}")
                # Rewrite the rectified normal distribution pdf in terms of normal distribution pdf,
                # then we can get its sample as follows.
                rectified_rvs = math.exp(-self.lamd[k] ** 2 * S / 2) / norm.cdf((M - self.lamd[k] * S) / math.sqrt(S)) * norm.rvs(loc=M, scale=math.sqrt(S), size=1, random_state=None)
                self.W[i][k] = rectified_rvs
                # self.W[i][k] = self.R[i][k] * rectified_rvs + (1 - self.R[i][k])  # how to use Dirac Delta function??

    # 6. For all k = 1,...,K, let b_k define the number of active (i.e., non-zero) entries of w_k. Draw lambda_k ~ Ga(alpha + b_k, beta + Sum_{i=1,...,Q}W_i,k).
    def sample_Lamda(self):  # the problem of W initialization impacts the value of b and term. Must figure out the usage of dirac delta function.
        for k in range(self.K):
            b = np.count_nonzero(self.W.T[k] != -1)  #
            term = 0
            print(f"Vector w_{k} is:")
            for i in range(self.Q):
                term += self.W[i][k]
                print(self.W[i][k])
            print(f"alpha is {self.alph}")
            print(f"b_k is {b}")
            print(f"1st parameter of Gamma dist is {self.alph + b}")
            print(f"beta is {self.bet}")
            print(f"sum of W_i,k is {term}")
            print(f"2nd parameter of Gamma dist is {1 / (self.bet + term)}")
            self.lamd[k] = gamma.rvs(a=self.alph + b, loc=0, scale=1 / (self.bet + term), random_state=None)  # two parameters in Gamma distribution have to be positive, i.e., a, scale > 0.

    # 7. For all k=1,...,K, draw r_k ~ Beta(e + b_k, f + Q - b_k), with b_k defined as in Step 6.
    def sample_r(self):
        for k in range(self.K):
            b = np.count_nonzero(self.W.T[k] != -1)
            self.r[k] = beta.rvs(a=self.e + b, b=self.f + self.Q - b, loc=0, scale=1, random_state=None)










'''
def gibbs_sampler(initial_point, num_samples, mean, cov):
    point = np.array(initial_point)
    samples = np.empty([num_samples + 1, 2])  # sampled points
    samples[0] = point
    tmp_points = np.empty([num_samples, 2])  # inbetween points

    for i in range(num_samples):
        # Sample from p(x_0|x_1)
        point = conditional_sampler(0, point, mean, cov)
        tmp_points[i] = point
        # Sample from p(x_1|x_0)
        point = conditional_sampler(1, point, mean, cov)
        samples[i + 1] = point

    return samples, tmp_points


def conditional_sampler(sampling_index, current_x, mean, cov):
    conditioned_index = 1 - sampling_index
    # The above line works because we only have 2 variables, x_0 & x_1
    a = cov[sampling_index, sampling_index]
    b = cov[sampling_index, conditioned_index]
    c = cov[conditioned_index, conditioned_index]

    mu = mean[sampling_index] + (b * (current_x[conditioned_index] - mean[conditioned_index])) / c

    sigma = np.sqrt(a - (b ** 2) / c)
    new_x = np.copy(current_x)
    new_x[sampling_index] = np.random.randn() * sigma + mu
    return new_x
'''