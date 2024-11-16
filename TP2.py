import random
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import multivariate_normal
from scipy.special import gamma


##########################################################Exercise 1##########################################################
def invertion_method(Px, X):
    U = random.uniform(0, 1)
    Fx = 0
    for i in range(len(Px)):
        Fx += Px[i]
        if U < Fx:
            return X[i]
    return X[-1]

def generate_iid_sequence(Px, X, N):
    sequence = [invertion_method(Px, X) for _ in range(N)]
    return sequence

def plot_distributions(Px, X, N):
    sequence = generate_iid_sequence(Px, X, N)
        
    # Empirical distrib
    hist, bins = np.histogram(sequence, bins=len(X), density=True)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
        
    # Theoretical distrib
    theoretical_dist = Px
    plt.figure(figsize=(10, 6))
    plt.bar(bin_centers, hist, width=0.1, label='Empirical Distribution', alpha=0.6, color='b')
    plt.plot(X, theoretical_dist, 'ro-', label='Theoretical Distribution')
    plt.xlabel('X')
    plt.ylabel('Probability')
    plt.legend()
    plt.title(f'Empirical vs Theoretical Distribution for N={N}')
    plt.savefig("TP2_results/distribution.png")
    plt.show()

##########################################################Exercise 2##########################################################
def sample_from_gmm(alpha, mu, sigma, n):
    m = len(alpha) 
    samples = []
    for _ in range(n):
        # Class Zi
        zi = np.random.choice(range(m), p=alpha)
        # Gaussian corresponding to Zi
        xi = np.random.multivariate_normal(mu[zi], sigma[zi])
        samples.append(xi)
    return np.array(samples)

def plot_gmm_samples(samples, mu, sigma):
    plt.figure(figsize=(10, 6))
    plt.scatter(samples[:, 0], samples[:, 1], label='Sampled Data', alpha=0.6, color='b')
    plt.scatter(mu[:, 0], mu[:, 1], label='Means', color='r', marker='x')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.title('Samples of Gaussian Mixture Model')
    plt.savefig("TP2_results/sample_gmm.png")
    plt.show()

class GMMEM:
    def __init__(self, X, m):
        n, d = X.shape
        self.n = n
        self.d = d
        self.X = X
        self.m = m
        self.alpha = np.ones(m) / m
        self.mu = X[np.random.choice(n, m, replace=False)]
        self.sigma = [np.eye(d) for _ in range(m)]
        self.gamma = np.zeros((n, m))
        self.log_likelihoods = []
        self.max_distances_to_alpha = []
        self.max_distances_to_mu = []
        self.max_distances_to_sigma = []

    def e_step(self):
        n, m = self.n, self.m
        gamma = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                gamma[i, j] = alpha[j] * self.multivariate_gaussian(i, j)
            gamma[i, :] /= np.sum(gamma[i, :])
        self.gamma = gamma

    def m_step(self):
        n, d = self.n, self.d
        m = self.m
        X = self.X
        gamma = self.gamma
        alpha = np.sum(gamma, axis=0) / n
        mu = np.dot(gamma.T, X) / np.sum(gamma, axis=0)[:, None]
        sigma = []
        for j in range(m):
            diff = X - mu[j]
            sigma_j = np.dot(gamma[:, j] * diff.T, diff) / np.sum(gamma[:, j])
            sigma.append(sigma_j)
        self.alpha = alpha
        self.mu = mu
        self.sigma = sigma

    def multivariate_gaussian(self, i, j):
        d = self.d
        x = self.X[i]
        mu = self.mu[j]
        x_mu = x - mu
        sigma = self.sigma[j]
        inv_sigma = np.linalg.inv(sigma)
        return np.exp(-0.5 * np.dot(np.dot(x_mu.T, inv_sigma), x_mu)) / np.sqrt((2 * np.pi) ** d * np.linalg.det(sigma))

    def log_likelihood(self):
        n = self.n
        alpha = self.alpha
        ll = 0
        for i in range(n):
            temp = 0
            for j in range(len(alpha)):
                temp += alpha[j] * self.multivariate_gaussian(i, j)
            ll += np.log(temp)
        self.log_likelihoods.append(ll)

    def em_algorithm(self, max_iter=1000, tol=1e-4, true_sigma=None, true_alpha=None, true_mu=None):
        for _ in range(max_iter):
            self.e_step()
            self.m_step()
            if true_sigma is not None:
                self.max_distances_to_sigma.append(max([np.linalg.norm(self.sigma[i] - true_sigma[i]) for i in range(len(true_sigma))]))
            if true_alpha is not None:
                self.max_distances_to_alpha.append(np.linalg.norm(self.alpha - true_alpha))
            if true_mu is not None:
                self.max_distances_to_mu.append(max([np.linalg.norm((self.mu)[i] - true_mu[i]) for i in range(len(true_mu))]))
            self.log_likelihood()
            if len(self.log_likelihoods) > 1 and abs(self.log_likelihoods[-1] - self.log_likelihoods[-2]) < tol:
                break

def calculate_bic(gmm, X):
    n, d = X.shape
    m = gmm.m
    ll = gmm.log_likelihoods[-1]
    num_params = m * (d + d * (d + 1) / 2 + 1) - 1
    bic = -2 * ll + num_params * np.log(n)
    return bic

def plot_contour(gmm, data):
    x = np.linspace(min(data[:, 0]), max(data[:, 0]), 100)
    y = np.linspace(min(data[:, 1]), max(data[:, 1]), 100)
    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T
    
    Z = np.zeros((100, 100))
    for j in range(gmm.m):
        mu = gmm.mu[j]
        sigma = gmm.sigma[j]
        rv = multivariate_normal(mu, sigma)
        Z += gmm.alpha[j] * rv.pdf(XX).reshape(100, 100)
    
    plt.contour(X, Y, Z, levels=10, cmap='viridis')

##########################################################Exercise 3A##########################################################
# result = gamma(0.975)
# print(result)

def p(x):
    return x**(1.65-1) * np.exp(-x**2 / 2) * (x >= 0)

def q(x):
    return 2 / np.sqrt(2 * np.pi * 1.5) * np.exp(-((0.8 - x)**2) / (2 * 1.5))

def f(x):
    return 2 * np.sin(np.pi / 1.5 * x) * (x >= 0)

def sample_q(size):
    rng = np.random.default_rng()
    return rng.normal(loc=0.8, scale=np.sqrt(1.5), size=size)

def importance_sampling(n_samples):
    samples = sample_q(n_samples)
    samples = samples[samples >= 0]
    weights = p(samples) / q(samples)
    return samples, weights

def compute_average_var(n_samples):
    samples, weights = importance_sampling(n_samples)
    weighted_f = f(samples) * weights
    average = np.sum(weighted_f) / np.sum(weights)
    var = np.sum(weights * (f(samples) - average)**2) / np.sum(weights)
    return average, var


##########################################################Main##########################################################

if __name__ == "__main__":
    # Ex1
    # Px = [0.1, 0.3, 0.4, 0.2]
    # X = [1, 2, 3, 4]
    # N = 10000
    # plot_distributions(Px, X, N)

    # # Ex2
    # # Parameters of the GMM
    # alpha = [0.3, 0.4, 0.3] 
    # mu = np.array([[0, 0], [3, 3], [0, 3]])  
    # sigma = [np.eye(2), np.eye(2), np.eye(2)] 
    # # Samples
    # n = 10000
    # samples = sample_from_gmm(alpha, mu, sigma, n)
    # plot_gmm_samples(samples, mu, sigma)

    # # EM Algorithm
    # show = True
    # gmm = GMMEM(samples, 3)
    # if show:
    #     gmm.em_algorithm(true_sigma=sigma, true_alpha=alpha, true_mu=mu)
    # else:
    #     gmm.em_algorithm()
    # # plot log likelihood
    # plt.figure(figsize=(10, 6))
    # plt.plot(gmm.log_likelihoods)
    # plt.xlabel('Iterations')
    # plt.ylabel('Log Likelihood')
    # plt.title('Log Likelihood of EM Algorithm')
    # plt.savefig("TP2_results/log_likelihood.png")
    # plt.show()
    # if show:
    #     k = range(len(gmm.max_distances_to_mu))
    #     # plot all normalised max distances
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(k, gmm.max_distances_to_alpha/max(gmm.max_distances_to_alpha), label='Alpha')
    #     plt.plot(k, gmm.max_distances_to_mu/max(gmm.max_distances_to_mu), label='Mu')
    #     plt.plot(k, gmm.max_distances_to_sigma/max(gmm.max_distances_to_sigma), label='Sigma')
    #     plt.xlabel('Iterations')
    #     plt.ylabel('Normalised Max Distance')
    #     plt.legend()
    #     plt.title('Normalised Max Distance to True Parameters')
    #     plt.savefig("TP2_results/max_distances.png")
    #     plt.show()
    # print("Alpha")
    # print(gmm.alpha)
    # print("Mu")
    # print(gmm.mu)
    # if show:
    #     print("Max distance to sigma :")
    #     print(gmm.max_distances_to_sigma[-1])

    # # Analyze Crude Birth/Death Rate Data
    # # Data
    # data_file_path = 'TP2_results/crude_birth_death_rate.csv' 
    # data = pd.read_csv(data_file_path)

    # # Preprocess the data
    # data = data[['Crude Birth Rate', 'Crude Death Rate']].dropna()

    # # Plot the scatter graph
    # plt.figure(figsize=(10, 6))
    # plt.scatter(data['Crude Birth Rate'], data['Crude Death Rate'], alpha=0.6)
    # plt.xlabel('Crude Birth Rate')
    # plt.ylabel('Crude Death Rate')
    # plt.title('Scatter Plot of Crude Birth/Death Rate')
    # plt.savefig("TP2_results/scatter_plot.png")
    # plt.show()

    # bics = []
    # models = []
    # for m in range(1, 6):
    #     gmm = GMMEM(data, m)
    #     gmm.em_algorithm()
    #     bic = calculate_bic(gmm, data)
    #     bics.append(bic)
    #     models.append(gmm)
    #     print(f'Model with {m} components: BIC = {bic}')

    # # Plot the best GMM model (compared by BIC)
    # best_model = models[np.argmin(bics)]
    # plot_contour(best_model, data)
    # plt.savefig("TP2_results/gmm_contour.png")
    # plt.show()

    # Ex3A 
    # Importance Sampling
    # Number of samples
    n_samples = 10000

    # Compute the average
    average, var = compute_average_var(n_samples)
    print(f"Average of f(x) over the density p(x) using importance sampling: {average}")
    print(f"Variance of f(x) over the density p(x) using importance sampling: {var}")

    samples, weights = importance_sampling(n_samples)
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.hist(samples, bins=50, density=True, alpha=0.6, color='b', label='Weighted Samples')
    x = np.linspace(0, np.max(samples), 1000)
    plt.plot(x, p(x), 'r-', label='Target Distribution p(x)')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.legend()
    plt.title('Importance Sampling')
    plt.savefig("TP2_results/importance_sampling.png")
    plt.show()

    # Compute average and variance for different number of samples and plot the results
    n_samples = [100, 1000, 10000, 100000]
    averages = []
    variances = []
    for n in n_samples:
        average, var = compute_average_var(n)
        averages.append(average)
        variances.append(var)
    plt.figure(figsize=(10, 6))
    plt.plot(n_samples, averages, 'ro-', label='Average')
    plt.plot(n_samples, variances, 'bo-', label='Variance')
    plt.xlabel('Number of Samples')
    plt.ylabel('Value')
    plt.legend()
    plt.title('Average and Variance of f(x) over p(x) using Importance Sampling')
    plt.savefig("TP2_results/average_variance.png")
    plt.show()