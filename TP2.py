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
        zi = np.random.choice(range(m), p=alpha/np.sum(alpha))
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
                gamma[i, j] = self.alpha[j] * self.multivariate_gaussian(i, j)
            gamma[i, :] /= np.sum(gamma[i, :])
        self.gamma = gamma

    def m_step(self, weighted=None):
        n, d = self.n, self.d
        m = self.m
        X = self.X
        gamma = self.gamma
        if weighted is None or len(weighted) != n:
            weighted = np.ones(n)*1/n
        for j in range(n):
            gamma[j, :] *= weighted[j]
        self.alpha = np.sum(gamma, axis=0)
        self.alpha /= np.sum(self.alpha)
        mu = np.dot(gamma.T, X) / np.sum(gamma, axis=0)[:, None]
        sigma = []
        for j in range(m):
            diff = X - mu[j]
            sigma_j = np.dot(gamma[:, j] * diff.T, diff) / np.sum(gamma[:, j])
            sigma.append(sigma_j)
        self.mu = mu
        self.sigma = sigma

    def multivariate_gaussian(self, i, j):
        d = self.d
        x = self.X[i]
        mu = self.mu[j]
        x_mu = x - mu
        sigma = self.sigma[j]
        inv_sigma = np.linalg.inv(sigma)
        final = np.exp(-0.5 * np.dot(np.dot(x_mu.T, inv_sigma), x_mu)) / np.sqrt((2 * np.pi) ** d * np.linalg.det(sigma))
        return max(final, 1e-20)

    def log_likelihood(self):
        n = self.n
        ll = 0
        for i in range(n):
            temp = 0
            for j in range(len(self.alpha)):
                temp += self.alpha[j] * self.multivariate_gaussian(i, j)
            ll += np.log(temp)
        self.log_likelihoods.append(ll)

    def em_algorithm(self, max_iter=1000, tol=1e-4, true_sigma=None, true_alpha=None, true_mu=None, weighted=None, verbosity=False):
        for _ in range(max_iter):
            self.e_step()
            self.m_step(weighted=weighted)
            if true_sigma is not None:
                self.max_distances_to_sigma.append(max([np.linalg.norm(self.sigma[i] - true_sigma[i]) for i in range(len(true_sigma))]))
            if true_alpha is not None:
                self.max_distances_to_alpha.append(np.linalg.norm(self.alpha - true_alpha))
            if true_mu is not None:
                self.max_distances_to_mu.append(max([np.linalg.norm((self.mu)[i] - true_mu[i]) for i in range(len(true_mu))]))
            self.log_likelihood()
            if verbosity:
                print(f'Iteration {_+1}, Log Likelihood: {self.log_likelihoods[-1]}')
            if len(self.log_likelihoods) > 1 and abs(self.log_likelihoods[-1] - self.log_likelihoods[-2]) < tol:
                break

def calculate_bic(gmm, X):
    n, d = X.shape
    m = gmm.m
    ll = gmm.log_likelihoods[-1]
    num_params = m * (d + d * (d + 1) / 2 + 1) - 1
    bic = -2 * ll + num_params * np.log(n)
    return bic

def plot_contour(gmm, data, dim1=0, dim2=1):
    x = np.linspace(min(data[:, dim1]), max(data[:, dim1]), 100)
    y = np.linspace(min(data[:, dim2]), max(data[:, dim2]), 100)
    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T

    # scatter plot of data
    plt.scatter(data[:, dim1], data[:, dim2], label='Data', alpha=0.6, color='b')
    
    Z = np.zeros((100, 100))
    for j in range(gmm.m):
        mu = gmm.mu[j][[dim1, dim2]]
        sigma = gmm.sigma[j][np.ix_([dim1, dim2], [dim1, dim2])]
        rv = multivariate_normal(mu, sigma)
        Z += gmm.alpha[j] * rv.pdf(XX).reshape(100, 100)
    
    plt.contour(X, Y, Z, levels=10, cmap='viridis')
    plt.xlabel(f'Dimension {dim1}')
    plt.ylabel(f'Dimension {dim2}')
    plt.title('GMM Contour Plot')
    plt.legend()
    plt.show()
##########################################################Exercise 3A##########################################################
# result = gamma(0.975)
# print(result)

def p_1(x):
    return x**(1.65-1) * np.exp(-x**2 / 2) * (x >= 0)

def q_1(x, mu=0.8, sigma=1.5):
    return 2 / np.sqrt(2 * np.pi * sigma) * np.exp(-((mu - x)**2) / (2 * sigma))

def f(x):
    return 2 * np.sin(np.pi / 1.5 * x) * (x >= 0)

def sample_q_1(size, mu=0.8, sigma=1.5):
    rng = np.random.default_rng()
    samples = rng.normal(loc=mu, scale=np.sqrt(sigma), size=size)
    return samples[samples >= 0]

def importance_sampling(n_samples, mu = None, sigma = None, p=p_1, q=q_1, sample_q=sample_q_1):
    if mu is None:
        samples = sample_q(n_samples)
        weights = p(samples) / q(samples)
    else:
        if sigma is None:
            sigma = 1.5
        samples = sample_q(n_samples, mu=mu, sigma=sigma)
        weights = p(samples) / q(samples, mu=mu, sigma=sigma)
    return samples, weights

def compute_average_var(n_samples, mu=None, sigma=None):
    if mu is None:
        samples, weights = importance_sampling(n_samples)
        weighted_f = f(samples) * weights
        average = np.sum(weighted_f) / np.sum(weights)
        var = np.sum(weights * (f(samples) - average)**2) / np.sum(weights)
    else:
        if sigma is None:
            sigma = 1.5
        samples, weights = importance_sampling(n_samples, mu=mu, sigma=sigma)
        weighted_f = f(samples) * weights
        average = np.sum(weighted_f) / np.sum(weights)
        var = np.sum(weights * (f(samples) - average)**2) / np.sum(weights)
    return average, var

##########################################################Exercise 3C##########################################################
def multivariate_gaussian(x, mu, sigma):
        d = x.shape[1]
        x_mu = x - mu
        inv_sigma = np.linalg.inv(sigma)
        final = [max(np.exp(-0.5 * np.dot(np.dot(x_mu[i,:].T, inv_sigma), x_mu[i,:])) / np.sqrt((2 * np.pi) ** d * np.linalg.det(sigma)), 1e-9) for i in range(x.shape[0])]
        return np.array(final)

def v(X, sigma_1=1, b=0.4):
    d = X.shape[1]
    Sigma = np.ones(d)
    Sigma[0] = sigma_1
    Sigma = np.diag(Sigma)
    x = X.copy()
    x[1] = x[1] + b * ( x[0]**2 - sigma_1**2)
    law = multivariate_gaussian(x, np.zeros(d), Sigma)
    return law


def q_GMM(x, alpha, mu, sigma):
    law = 0
    for j in range(len(alpha)):
        law += alpha[j] * multivariate_gaussian(x, mu[j], sigma[j])
    return law

def population_monte_carlo(n_samples, d, M, sigma_1=1, b=0.4, max_iter=20, tol=1e-4, em_max_iter=100):
    # Initialisation
    alpha = np.ones(M) / M
    alpha_old = np.zeros(M)
    mu = np.random.uniform(-5, 5, (M, d))
    sigma = [np.eye(d) for _ in range(M)]
    iteration = 0
    while iteration < max_iter and np.linalg.norm(alpha - alpha_old) > tol:
        # Importance Sampling
        sample_gmm = lambda n: sample_from_gmm(alpha, mu, sigma, n)
        q_2 = lambda x: q_GMM(x, alpha, mu, sigma)
        samples, weights = importance_sampling(n_samples, p=v, q=q_2, sample_q=sample_gmm)

        # Eighted EM Algorithm
        gmm = GMMEM(samples, M)
        gmm.em_algorithm(max_iter=em_max_iter, tol=tol, weighted=weights)
        alpha_old = alpha.copy()

        # Update parameters
        alpha = gmm.alpha/np.sum(gmm.alpha)
        mu = gmm.mu
        sigma = gmm.sigma
        iteration += 1
        print(f'Iteration {iteration}, Alpha: {alpha}')
    return alpha, mu, sigma
        






##########################################################Main##########################################################

if __name__ == "__main__":
    # Ex1
    Px = [0.1, 0.3, 0.4, 0.2]
    X = [1, 2, 3, 4]
    N = 10000
    plot_distributions(Px, X, N)

    # Ex2
    # Parameters of the GMM
    alpha = [0.3, 0.4, 0.3] 
    mu = np.array([[0, 0], [3, 3], [0, 3]])  
    sigma = [np.eye(2), np.eye(2), np.eye(2)] 
    # Samples
    n = 10000
    samples = sample_from_gmm(alpha, mu, sigma, n)
    plot_gmm_samples(samples, mu, sigma)

    # EM Algorithm
    show = True
    gmm = GMMEM(samples, 3)
    if show:
        gmm.em_algorithm(true_sigma=sigma, true_alpha=alpha, true_mu=mu)
    else:
        gmm.em_algorithm()
    # plot log likelihood
    plt.figure(figsize=(10, 6))
    plt.plot(gmm.log_likelihoods)
    plt.xlabel('Iterations')
    plt.ylabel('Log Likelihood')
    plt.title('Log Likelihood of EM Algorithm')
    plt.savefig("TP2_results/log_likelihood.png")
    plt.show()
    if show:
        k = range(len(gmm.max_distances_to_mu))
        # plot all normalised max distances
        plt.figure(figsize=(10, 6))
        plt.plot(k, gmm.max_distances_to_alpha/max(gmm.max_distances_to_alpha), label='Alpha')
        plt.plot(k, gmm.max_distances_to_mu/max(gmm.max_distances_to_mu), label='Mu')
        plt.plot(k, gmm.max_distances_to_sigma/max(gmm.max_distances_to_sigma), label='Sigma')
        plt.xlabel('Iterations')
        plt.ylabel('Normalised Max Distance')
        plt.legend()
        plt.title('Normalised Max Distance to True Parameters')
        plt.savefig("TP2_results/max_distances.png")
        plt.show()
    print("Alpha")
    print(gmm.alpha)
    print("Mu")
    print(gmm.mu)
    if show:
        print("Max distance to sigma :")
        print(gmm.max_distances_to_sigma[-1])

    # Analyze Crude Birth/Death Rate Data
    # Data
    data_file_path = 'TP2_results/WPP2024_Demographic_Indicators_Medium.csv'
    data = pd.read_csv(data_file_path)
    data = data[['CBR', 'CDR']].dropna()
    X = data.values
    print(X.shape)

    # Plot the scatter graph
    plt.figure(figsize=(10, 6))
    plt.scatter(data['CBR'],  data['CDR'], alpha=0.6)
    plt.xlabel('Crude Birth Rate')
    plt.ylabel('Crude Death Rate')
    plt.title('Scatter Plot of Crude Birth/Death Rate')
    plt.savefig("TP2_results/scatter_plot.png")
    plt.show()

    bics = []
    models = []
    for m in range(1, 6):
        gmm = GMMEM(X, m)
        gmm.em_algorithm(verbosity=True)
        bic = calculate_bic(gmm, X)
        bics.append(bic)
        models.append(gmm)
        print(f'Model with {m} components: BIC = {bic}')

    # Plot the best GMM model (compared by BIC)
    best_model = models[np.argmin(bics)]
    plot_contour(best_model, X)
    plt.savefig("TP2_results/gmm_contour.png")
    plt.show()

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
    plt.plot(x, p_1(x), 'r-', label='Target Distribution p(x)')
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

    # Experiment with a different proposal distribution q(x) = N(mu, sigma) where mu = 6 and sigma = 1.5
    n_samples = 10000

    # Compute the average
    average, var = compute_average_var(n_samples, mu=6)
    print(f"Average of f(x) over the density p(x) using importance sampling: {average}")
    print(f"Variance of f(x) over the density p(x) using importance sampling: {var}")

    samples, weights = importance_sampling(n_samples, mu=6)
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.hist(samples, bins=50, density=True, alpha=0.6, color='b', label='Weighted Samples')
    x = np.linspace(0, np.max(samples), 1000)
    plt.plot(x, p(x), 'r-', label='Target Distribution p(x)')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.legend()
    plt.title('Importance Sampling')
    plt.savefig("TP2_results/importance_sampling_2.png")
    plt.show()

    # Ex3C
    # Population Monte Carlo
    n_samples = 1000
    d = 5
    M = 3
    alpha, mu, sigma = population_monte_carlo(n_samples, d, M, max_iter=2, tol=1e-4, em_max_iter=600)
    print("Alpha")
    print(alpha)
    print("Mu")
    print(mu)
    print("Sigma")
    print(sigma)
    # Plot samples of laws GMM
    samples = sample_from_gmm(alpha, mu, sigma, n_samples)
    plt.figure(figsize=(10, 6))
    plt.scatter(samples[:, 0], samples[:, 1], label='Sampled Data', alpha=0.6, color='b')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.title('Samples of Gaussian Mixture Model')
    plt.savefig("TP2_results/sample_gmm_2.png")

    # Plot the true law v(x) with gaussian contours
    samples_new = np.random.multivariate_normal(np.zeros(d), np.eye(d), n_samples)
    samples_new[:,1] += 0.4*(samples_new[:,0]**2 - 1)
    gmm = GMMEM(samples_new, M)
    gmm.alpha = alpha
    gmm.mu = mu
    gmm.sigma = sigma
    plt.figure(figsize=(10, 6))
    plot_contour(gmm, samples_new)
    plt.scatter(samples_new[:, 0], samples_new[:, 1], label='Sampled Data', alpha=0.6, color='b')
    plt.savefig("TP2_results/true_law_with_MC.png")

    # Use EM on the true law
    gmm = GMMEM(samples_new, M)
    gmm.em_algorithm()
    plt.figure(figsize=(10, 6))
    plot_contour(gmm, samples_new)
    plt.scatter(samples_new[:, 0], samples_new[:, 1], label='Sampled Data', alpha=0.6, color='b')
    plt.savefig("TP2_results/true_law_with_EM.png")

