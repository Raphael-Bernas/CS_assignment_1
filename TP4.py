import numpy as np
import matplotlib.pyplot as plt
import random
import math
import pandas as pd
from scipy.stats import multivariate_normal
from scipy.special import gamma


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

if __name__ == "__main__":
    np.random.seed(42)
    ex = 1
    #Ex1
    if ex == 1:
        z=(0,0)
        sigma1 = 3.0
        sigma2 = 3.0
        a=10

        num_samples = 100000
        samples, accept_x, accept_y, _, _ = MH_P(z, sigma1, sigma2, num_samples, a=a)
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
        test_other_method = True
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

    
    