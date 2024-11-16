import random
import math
import numpy as np
import matplotlib.pyplot as plt

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
    n = 1000 
    samples = sample_from_gmm(alpha, mu, sigma, n)
    plot_gmm_samples(samples, mu, sigma)