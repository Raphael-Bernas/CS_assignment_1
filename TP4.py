import numpy as np
import matplotlib.pyplot as plt
import random
import math
import pandas as pd
from scipy.stats import multivariate_normal
from scipy.special import gamma
from scipy.stats import invgamma, norm


###############################################################Exercice 1###############################################################
def sample_P1(z, sigma1):
    x, y = z
    x_new = np.random.normal(x, sigma1)
    return (x_new, y)

def sample_P2(z, sigma2):
    x, y = z
    y_new = np.random.normal(y, sigma2)
    return (x, y_new)

def PI(x, y, a=1):
    return np.exp(-(x/a)**2 - y**2 - (1/4)*((x/a)**2 - y**2)**2)

def Step_MH_P(z, sigma1, sigma2, a=1, kernel_proba=0.5):
    choice = 0
    if np.random.rand() < kernel_proba:
        z_new = sample_P1(z, sigma1)
        choice = 1
    else:
        z_new = sample_P2(z, sigma2)
    alpha = min(PI(z_new[0], z_new[1], a=a) / PI(z[0], z[1], a=a), 1)
    if np.random.rand() < alpha:
        return z_new, alpha, choice, True
    else:
        return z, alpha, choice, False

def MH_P(z, sigma1, sigma2, num_samples, n_acc=100, a=1, kernel_proba=0.5):
    samples = np.zeros((num_samples, 2))
    samples[0] = z
    accept_x = []
    accept_y = []
    alpha_x = []
    alpha_y = []
    sigma1 = sigma1
    sigma2 = sigma2
    current_accept_x = 0
    current_accept_y = 0
    for i in range(1, num_samples):
        samples[i], alpha, choice, accepted = Step_MH_P(samples[i-1], sigma1, sigma2, a=a, kernel_proba=kernel_proba)
        if choice == 1:
            current_accept_x += accepted
            alpha_x.append(alpha)
        else:
            current_accept_y += accepted
            alpha_y.append(alpha)
        if i % n_acc == 0:
            accept_x.append(current_accept_x / n_acc)
            accept_y.append(current_accept_y / n_acc)
            current_accept_x = 0
            current_accept_y = 0
    return samples, accept_x, accept_y, alpha_x, alpha_y

def Adaptative_MH_P(z, sigma, num_samples, batch_size=50, acc_threshold=0.24, Law= lambda z: PI(z[0], z[1], a=10), verbosity=False):
    d = len(z)
    samples = np.zeros((num_samples, 2))
    samples[0] = z
    accepts = np.zeros((num_samples, d))
    acc_rates = np.zeros((num_samples//batch_size, d))
    alpha = np.zeros((num_samples, d))
    sigma = sigma
    sigmas = [sigma]
    for i in range(1, num_samples):
        j = np.random.randint(0, d)
        z_new = np.copy(samples[i-1])
        z_new[j] = np.random.normal(samples[i-1, j], sigma[j])
        current_alpha = min(Law(z_new) / Law(samples[i-1]), 1)
        alpha[i, j] = current_alpha
        if np.random.rand() < current_alpha:
            samples[i] = z_new
            accepts[i, j] = 1
        else:
            samples[i] = samples[i-1]
        if i % batch_size == 0:
            acc_rate = np.mean(accepts[(i-batch_size):i], axis=0)
            acc_rates[i//batch_size] = acc_rate
            acc_mask = acc_rate > acc_threshold
            sigma = sigma + (2*acc_mask - 1)*min(0.01, 1/np.sqrt(i//batch_size))
            sigma = np.maximum(sigma, 0.01)
            sigmas.append(sigma)
            if verbosity:
                print('Batch', i//batch_size, ':', acc_rate)
    return samples, acc_rates, alpha, sigmas

def f_B(z, B=0.03):
    x1, x2, *rest = z
    rest_sum = sum(xi**2 for xi in rest)
    return np.exp(- x1**2 / 200 - 0.5 * (x2 + B * x1**2 - 100 * B)**2 - 0.5 * rest_sum)

###############################################################Exercice 2###############################################################

def Mixture_law(z, mu, sigma, weights):
    d = len(z)
    n = len(mu)
    res = 0
    for i in range(n):
        res += weights[i]*multivariate_normal.pdf(z, mean=mu[i], cov=sigma[i])
    return res

def Parallel_Tempering_MH_P(z, temp, sigma, num_samples, batch_size=50, Law = lambda z: PI(z[0], z[1], a=1), proposal_law = 'Gibbs', verbosity=False):
    K = len(temp)
    d = len(z[0])
    samples = np.zeros((num_samples, K, d))
    samples[0] = z
    accepts = np.zeros((num_samples, K))
    swap_accepts = np.zeros(num_samples)
    acc_rates = np.zeros((num_samples//batch_size, K))
    swap_acc_rates = np.zeros(num_samples//batch_size)
    sigma = sigma
    for i in range(1, num_samples):
        for k in range(K):
            if proposal_law == 'Gibbs':
                Law_k = lambda z: Law(z) ** (1/temp[k])
                current_sample, _, _, _ = Adaptative_MH_P(samples[i-1, k], sigma, 1, Law=Law_k, verbosity=False)
                samples[i, k] = current_sample[0]
            elif proposal_law == 'Gaussian':
                if sigma is np.array:
                    sigma = sigma[0]
                current_sample = np.random.multivariate_normal(mean=samples[i-1, k], cov=sigma**2*temp[k]*(np.eye(d)))
                alpha = min(1, Law(current_sample) ** (1/temp[k]) / Law(samples[i-1, k]) ** (1/temp[k]))
                if np.random.rand() < alpha:
                    samples[i, k] = current_sample
                    accepts[i, k] = 1
                else:
                    samples[i, k] = samples[i-1, k]
        j0 = np.random.randint(0, K-1)
        l = np.random.randint(0, 1)
        i0 = j0 + 2*l - 1
        alpha_swap = min(1, Law(samples[i, i0]) ** (1/temp[j0]) * Law(samples[i, j0]) ** (1/temp[i0]) / (Law(samples[i, i0]) ** (1/temp[i0]) * Law(samples[i, j0]) ** (1/temp[j0])))
        if np.random.rand() < alpha_swap:
            samples[i, [i0, j0]] = samples[i, [j0, i0]]
            swap_accepts[i] = 1
        if i % batch_size == 0:
            acc_rate = np.mean(accepts[(i-batch_size):i], axis=0)
            acc_rates[i//batch_size] = acc_rate
            swap_acc_rate = np.mean(swap_accepts[(i-batch_size):i])
            swap_acc_rates[i//batch_size] = swap_acc_rate
            if verbosity:
                print('Batch', i//batch_size, ':', swap_acc_rate)
    return samples, acc_rates, swap_acc_rates

###############################################################Exercice 3###############################################################

def generate_synthetic_Y(N, k, alpha, beta, gamma, mu):
    # Sample sigma2 and tau2
    sigma2 = invgamma.rvs(a=alpha, scale=beta)
    tau2 = invgamma.rvs(a=gamma, scale=beta)
    # Sample X
    X = norm.rvs(loc=mu, scale=np.sqrt(sigma2), size=N)
    # Generate Y
    Y = []
    for i in range(N):
        group_y = norm.rvs(loc=X[i], scale=np.sqrt(tau2), size=k[i])
        Y.append(group_y.tolist())
    
    true_parameters = {
        "sigma2": sigma2,
        "tau2": tau2,
        "mu": mu,
        "X": X
    }
    return Y, true_parameters

def plot_Y(Y):
    plt.figure(figsize=(10, 6))
    for i, group_y in enumerate(Y):
        x_coords = [i + 1] * len(group_y)
        plt.scatter(x_coords, group_y, label=f'Group {i + 1}')
    plt.xlabel("Group")
    plt.ylabel("Observations")
    plt.title("Synthetic Data: Observations by Group")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

def gibbs_sampler(Y, alpha, beta, gamma, num_samples=1000, burn_in=500, sigma2_init=1.0, tau2_init=1.0, mu_init=0.0, X_init=None):
    N = len(Y)  # Groups
    k = [len(y_i) for y_i in Y]  # Observations in each group (k_i)
    K = sum(k)
    sigma2 = sigma2_init
    tau2 = tau2_init
    mu = mu_init
    if X_init is None:
        X = np.zeros(N)
    else:
        X = X_init
    # samples dictionary
    samples = {
        "sigma2": [],
        "tau2": [],
        "mu": [],
        "X": []
    }
    for iteration in range(num_samples):
        # Update sigma2
        sum_squared_differences = np.sum((X - mu)**2)
        sigma2 = invgamma.rvs(alpha + N/2, scale=beta + sum_squared_differences/2)
        # Update tau2
        sum_squared_residuals = np.sum([
            np.sum((np.array(y_i) - X[i])**2) for i, y_i in enumerate(Y)
        ])
        tau2 = invgamma.rvs(gamma + K/2, scale=beta + sum_squared_residuals/2)
        # Update mu
        mean_mu = np.mean(X)
        var_mu = sigma2/N
        mu = norm.rvs(loc=mean_mu, scale=np.sqrt(var_mu))
        # Update X
        for i in range(N):
            denom = k[i]*sigma2 + tau2
            mean_X = (np.sum(Y[i])*sigma2 + mu*tau2)/denom
            var_X = sigma2*tau2/denom
            X[i] = norm.rvs(loc=mean_X, scale=np.sqrt(var_X))
        if iteration >= burn_in:
            samples["sigma2"].append(sigma2)
            samples["tau2"].append(tau2)
            samples["mu"].append(mu)
            samples["X"].append(X.copy())
    return samples

def block_gibbs_sampler(Y, alpha, beta, gamma, num_samples=1000, burn_in=500, sigma2_init=1.0, tau2_init=1.0, mu_init=0.0, X_init=None):
    N = len(Y)
    k = [len(y_i) for y_i in Y]
    K = sum(k)
    sigma2 = sigma2_init
    tau2 = tau2_init
    mu = mu_init
    if X_init is None:
        X = np.zeros(N)
    else:
        X = X_init
    # samples dictionary
    samples = {
        "sigma2": [],
        "tau2": [],
        "mu": [],
        "X": []
    }
    for iteration in range(num_samples):
        ## Update sigma2
        sum_squared_differences = np.sum((X - mu)**2)
        sigma2 = invgamma.rvs(alpha + N/2, scale=beta + sum_squared_differences/2)
        # Update tau2
        sum_squared_residuals = np.sum([
            np.sum((np.array(y_i) - X[i])**2) for i, y_i in enumerate(Y)
        ])
        tau2 = invgamma.rvs(gamma + K/2, scale=beta + sum_squared_residuals/2)
        
        # Update the block (X, mu) with multivariate normal
        sum_y = np.array([np.sum(y_i) for y_i in Y])
        inv_Sigma = np.diag(k/tau2 + 1/sigma2)
        inv_Sigma = np.vstack((inv_Sigma, -np.ones(N)/sigma2))
        inv_Sigma = np.hstack((inv_Sigma, np.append(-np.ones(N)/sigma2, N/sigma2).reshape(-1, 1)))
        Sigma = np.linalg.inv(inv_Sigma)
        mean = Sigma @ np.append(sum_y/tau2, 0)
        X_mu = np.random.multivariate_normal(mean, Sigma)
        X = X_mu[:-1]
        mu = X_mu[-1]
        
        if iteration >= burn_in:
            samples["sigma2"].append(sigma2)
            samples["tau2"].append(tau2)
            samples["mu"].append(mu)
            samples["X"].append(X.copy())
    return samples

def plot_distances_between_parameters(samples, true_parameters, normalize=True):
    num_samples = len(samples["sigma2"])
    distances_sigma2 = np.zeros(num_samples)
    distances_tau2 = np.zeros(num_samples)
    distances_mu = np.zeros(num_samples)
    distances_X = np.zeros(num_samples)
    
    true_sigma2 = true_parameters["sigma2"]
    true_tau2 = true_parameters["tau2"]
    true_mu = true_parameters["mu"]
    true_X = true_parameters["X"]
    
    for i in range(num_samples):
        sigma2 = samples["sigma2"][i]
        tau2 = samples["tau2"][i]
        mu = samples["mu"][i]
        X = samples["X"][i]
        
        distances_sigma2[i] = np.abs(sigma2 - true_sigma2)
        distances_tau2[i] = np.abs(tau2 - true_tau2)
        distances_mu[i] = np.abs(mu - true_mu)
        distances_X[i] = np.sqrt(np.sum((X - true_X)**2))
    
    # Normalize distances
    if normalize:
        distances_sigma2 /= np.max(distances_sigma2)
        distances_tau2 /= np.max(distances_tau2)
        distances_mu /= np.max(distances_mu)
        distances_X /= np.max(distances_X)
    
    plt.figure(figsize=(10, 6))
    plt.plot(distances_sigma2, label="sigma2")
    plt.plot(distances_tau2, label="tau2")
    plt.plot(distances_mu, label="mu")
    plt.plot(distances_X, label="X")
    plt.xlabel("Iteration")
    plt.ylabel("Normalized Distance")
    plt.title("Normalized Distance Between True and Estimated Parameters")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

def plot_hist(samples, true_parameters):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # Plot histogram for sigma2
    axs[0, 0].hist(samples["sigma2"], bins=30, alpha=0.7, label="sigma2")
    axs[0, 0].axvline(true_parameters["sigma2"], color='r', linestyle='--', label="True sigma2")
    axs[0, 0].set_xlabel("sigma2")
    axs[0, 0].set_ylabel("Frequency")
    axs[0, 0].set_title("Histogram of sigma2")
    axs[0, 0].legend()
    axs[0, 0].grid(True, linestyle='--', alpha=0.7)

    # Plot histogram for tau2
    axs[0, 1].hist(samples["tau2"], bins=30, alpha=0.7, label="tau2")
    axs[0, 1].axvline(true_parameters["tau2"], color='r', linestyle='--', label="True tau2")
    axs[0, 1].set_xlabel("tau2")
    axs[0, 1].set_ylabel("Frequency")
    axs[0, 1].set_title("Histogram of tau2")
    axs[0, 1].legend()
    axs[0, 1].grid(True, linestyle='--', alpha=0.7)

    # Plot histogram for mu
    axs[1, 0].hist(samples["mu"], bins=30, alpha=0.7, label="mu")
    axs[1, 0].axvline(true_parameters["mu"], color='r', linestyle='--', label="True mu")
    axs[1, 0].set_xlabel("mu")
    axs[1, 0].set_ylabel("Frequency")
    axs[1, 0].set_title("Histogram of mu")
    axs[1, 0].legend()
    axs[1, 0].grid(True, linestyle='--', alpha=0.7)

    # Plot histogram for X_i (taking the first element of X as an example)
    axs[1, 1].hist([x[0] for x in samples["X"]], bins=30, alpha=0.7, label="X[0]")
    axs[1, 1].axvline(true_parameters["X"][0], color='r', linestyle='--', label="True X[0]")
    axs[1, 1].set_xlabel("X[0]")
    axs[1, 1].set_ylabel("Frequency")
    axs[1, 1].set_title("Histogram of X[0]")
    axs[1, 1].legend()
    axs[1, 1].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    np.random.seed(42)
    ex = 3 # set the exercice number you want to run
    #Ex1
    if ex == 1:
        z=(0,0)
        sigma1 = 3.0
        sigma2 = 3.0
        a=10

        num_samples = 100000
        samples, accept_x, accept_y, _, _ = MH_P(z, sigma1, sigma2, num_samples, a=a)
        plot_samples = np.copy(samples)
        # plot samples
        plt.plot(samples[:,0], samples[:,1], 'o')
        # add the contour of the density
        x = samples[:, 0]
        y = samples[:, 1]
        X, Y = np.meshgrid(np.linspace(min(x), max(x), 100), np.linspace(min(y), max(y), 100))
        Z = np.zeros(X.shape)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = PI(X[i, j], Y[i, j], a=a)
        plt.contour(X, Y, Z, 20)
        plt.show()
        # plot acceptance rates
        plt.plot(accept_x, 'r')
        plt.plot(accept_y, 'b')
        plt.legend(['acceptance rate for x', 'acceptance rate for y'])
        plt.show()
        ## plot autocorrelation of the samples
        # pd.plotting.autocorrelation_plot(samples[:,0])
        # pd.plotting.autocorrelation_plot(samples[:,1])
        # plt.legend(['autocorrelation for x', 'autocorrelation for y'])
        # plt.show()
        # Test to optimize the algorithm
        test_other_method = False
        if test_other_method:
            # plot for other kernel probabilities
            kernel_proba = 0.2
            samples, accept_x, accept_y, _, _ = MH_P(z, sigma1, sigma2, num_samples, a=a, kernel_proba=kernel_proba)
            plt.plot(samples[:,0], samples[:,1], 'o')
            x = samples[:, 0]
            y = samples[:, 1]
            X, Y = np.meshgrid(np.linspace(min(x), max(x), 100), np.linspace(min(y), max(y), 100))
            Z = np.zeros(X.shape)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    Z[i, j] = PI(X[i, j], Y[i, j], a=a)
            plt.contour(X, Y, Z, 20)
            plt.show()
            plt.plot(accept_x, 'r')
            plt.plot(accept_y, 'b')
            plt.legend(['acceptance rate for x', 'acceptance rate for y'])
            plt.show()
            # plot for other parameters
            sigma1 = .5
            sigma2 = 3.0
            samples, accept_x, accept_y, _, _ = MH_P(z, sigma1, sigma2, num_samples, a=a)
            plt.plot(samples[:,0], samples[:,1], 'o')
            x = samples[:, 0]
            y = samples[:, 1]
            X, Y = np.meshgrid(np.linspace(min(x), max(x), 100), np.linspace(min(y), max(y), 100))
            Z = np.zeros(X.shape)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    Z[i, j] = PI(X[i, j], Y[i, j], a=a)
            plt.contour(X, Y, Z, 20)
            plt.show()
            plt.plot(accept_x, 'r')
            plt.plot(accept_y, 'b')
            plt.legend(['acceptance rate for x', 'acceptance rate for y'])
            plt.show()
        # Test the adaptative method
        z = (0, 0)
        a=10
        sigma = np.array([3.0, 3.0])
        num_samples = 100000
        # To define the PI law : Law = lambda z: PI(z[0], z[1], a=1)
        test_banana = True
        if test_banana:
            Law = lambda z: f_B(z, B=0.03)
        else:
            Law = lambda z: PI(z[0], z[1], a=a)
        samples, acc_rates, alpha, sigmas = Adaptative_MH_P(z, sigma, num_samples, batch_size=50, Law=Law)
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))

        # plot samples
        axs[0, 0].plot(samples[:,0], samples[:,1], 'o')
        axs[0, 0].set_title('Samples')

        # add the contour of the density
        x = samples[:, 0]
        y = samples[:, 1]
        if test_banana:
            plot_x = x
            plot_y = y
        else:
            plot_x = plot_samples[:,0]
            plot_y = plot_samples[:,1]
        X, Y = np.meshgrid(np.linspace(min(plot_x), max(plot_x), 100), np.linspace(min(plot_y), max(plot_y), 100))
        Z = np.zeros(X.shape)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = Law([X[i, j], Y[i, j]])
        axs[0, 0].contour(X, Y, Z, 20)

        # plot acceptance rates
        axs[0, 1].plot(acc_rates[:,0], 'r')
        axs[0, 1].plot(acc_rates[:,1], 'b')
        axs[0, 1].legend(['acceptance rate for x', 'acceptance rate for y'])
        axs[0, 1].set_title('Acceptance Rates')

        # plot sigmas
        axs[1, 0].plot(sigmas)
        axs[1, 0].legend(['sigma for x', 'sigma for y'])
        axs[1, 0].set_title('Sigmas')

        # plot autocorrelation of the samples
        pd.plotting.autocorrelation_plot(samples[:,0], ax=axs[1, 1])
        pd.plotting.autocorrelation_plot(samples[:,1], ax=axs[1, 1])
        axs[1, 1].legend(['autocorrelation for x', 'autocorrelation for y'])
        axs[1, 1].set_title('Autocorrelation')

        plt.tight_layout()
        plt.show()
    #Ex2
    if ex == 2:
        mu = [
            (2.18, 5.76),
            (8.67, 9.59),
            (4.24, 8.48),
            (8.41, 1.68),
            (3.93, 8.82),
            (3.25, 3.47),
            (1.70, 0.50),
            (4.59, 5.60),
            (6.91, 5.81),
            (6.87, 5.40),
            (5.41, 2.65),
            (2.70, 7.88),
            (4.98, 3.70),
            (1.14, 2.39),
            (8.33, 9.50),
            (4.93, 1.50),
            (1.83, 0.09),
            (2.26, 0.31),
            (5.54, 6.86),
            (1.69, 8.11),
        ]
        weights = [0.05]*20
        sigma_law = [0.1*np.eye(2) for _ in range(20)]
        z = (0, 0)
        Law = lambda z: Mixture_law(z, mu, sigma_law, weights)
        num_samples = 1000
        sigma = np.array([0.1, 0.1]) # To see the all the gaussian clusters attained : sigma = np.array([5, 5])
        samples, acc_rates, alpha, sigmas = Adaptative_MH_P(z, sigma, num_samples, batch_size=50, Law=Law, verbosity=True)
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        # plot samples
        axs[0, 0].plot(samples[:,0], samples[:,1], 'o')
        axs[0, 0].set_title('Samples')
        # add the contour of the density
        x = samples[:, 0]
        y = samples[:, 1]
        # X, Y = np.meshgrid(np.linspace(min(x), max(x), 100), np.linspace(min(y), max(y), 100))
        X, Y = np.meshgrid(np.linspace(-1, 10, 100), np.linspace(-1, 10, 100))
        Z = np.zeros(X.shape)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = Law([X[i, j], Y[i, j]])
        axs[0, 0].contour(X, Y, Z, 20)
        # plot acceptance rates
        axs[0, 1].plot(acc_rates[:,0], 'r')
        axs[0, 1].plot(acc_rates[:,1], 'b')
        axs[0, 1].legend(['acceptance rate for x', 'acceptance rate for y'])
        axs[0, 1].set_title('Acceptance Rates')
        # plot sigmas
        axs[1, 0].plot(sigmas)
        axs[1, 0].legend(['sigma for x', 'sigma for y'])
        axs[1, 0].set_title('Sigmas')
        # plot autocorrelation of the samples
        pd.plotting.autocorrelation_plot(samples[:,0], ax=axs[1, 1])
        pd.plotting.autocorrelation_plot(samples[:,1], ax=axs[1, 1])
        axs[1, 1].legend(['autocorrelation for x', 'autocorrelation for y'])
        axs[1, 1].set_title('Autocorrelation')
        plt.tight_layout()
        plt.show()

        # Parallel Tempering
        z = np.array([[0, 0] for _ in range(5)])
        temp = [60, 21.6, 7.7, 2.8, 1]
        sigma = 0.25
        # sigma = np.array([1., 1.])
        num_samples = 10000
        # Law = lambda z: PI(z[0], z[1], a=10)
        # Law = lambda z: f_B(z, B=0.03)
        Law = lambda z: Mixture_law(z, mu, sigma_law, weights)
        samples, acc_rates, swap_acc_rates = Parallel_Tempering_MH_P(z, temp, sigma, num_samples, batch_size=50, Law=Law, proposal_law='Gaussian', verbosity=True)
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        # plot samples
        axs[0, 0].plot(samples[:,-1,0], samples[:,-1,1], 'o')
        axs[0, 0].set_title('Samples')
        # add the contour of the density
        x = samples[:, 0, 0]
        y = samples[:, 0, 1]
        X, Y = np.meshgrid(np.linspace(min(x), max(x), 100), np.linspace(min(y), max(y), 100))
        Z = np.zeros(X.shape)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = Law([X[i, j], Y[i, j]])
        axs[0, 0].contour(X, Y, Z, 20)
        # plot acceptance rates
        for k in range(len(temp)):
            axs[0, 1].plot(acc_rates[:,k])
        axs[0, 1].set_title('Acceptance Rates')
        # plot autocorrelation of the samples
        pd.plotting.autocorrelation_plot(samples[:,0,0], ax=axs[1, 0])
        pd.plotting.autocorrelation_plot(samples[:,0,1], ax=axs[1, 0])
        axs[1, 0].legend(['autocorrelation for x', 'autocorrelation for y'])
        axs[1, 0].set_title('Autocorrelation')
        # plot swap acceptance rates
        axs[1, 1].plot(swap_acc_rates)
        axs[1, 1].set_title('Swap Acceptance Rates')
        plt.tight_layout()
        plt.show()
    #Ex3
    if ex == 3:
        N = 20
        k = np.random.randint(5, 10, N)
        alpha = 2
        beta = 1
        gamma = 2
        mu = 0
        Y, true_parameters = generate_synthetic_Y(N, k, alpha, beta, gamma, mu)
        plot_Y(Y)
        samples = gibbs_sampler(Y, alpha, beta, gamma, num_samples=1000, burn_in=0)
        plot_distances_between_parameters(samples, true_parameters, normalize=False)
        plot_hist(samples, true_parameters)
        samples = block_gibbs_sampler(Y, alpha, beta, gamma, num_samples=1000, burn_in=0)
        plot_distances_between_parameters(samples, true_parameters, normalize=False)
        plot_hist(samples, true_parameters)


