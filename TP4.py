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

def Step_MH_P(z, sigma1, sigma2, a=1):
    choice = 0
    if np.random.rand() < 0.5:
        z_new = sample_P1(z, sigma1)
        choice = 1
    else:
        z_new = sample_P2(z, sigma2)
    alpha = min(PI(z_new[0], z_new[1], a=a) / PI(z[0], z[1], a=a), 1)
    if np.random.rand() < alpha:
        return z_new, alpha, choice
    else:
        return z, alpha, choice

def MH_P(z, sigma1, sigma2, num_samples, a=1):
    samples = np.zeros((num_samples, 2))
    samples[0] = z
    accept_x = []
    accept_y = []
    for i in range(1, num_samples):
        samples[i], alpha, choice = Step_MH_P(samples[i-1], sigma1, sigma2, a=a)
        if choice == 1:   
            accept_x.append(alpha)
        else:
            accept_y.append(alpha)
    return samples, accept_x, accept_y

if __name__ == "__main__":
    np.random.seed(42)
    #Ex1
    z=(0,0)
    sigma1 = 3.0
    sigma2 = 3.0
    a=10

    num_samples = 100000
    samples, accept_x, accept_y = MH_P(z, sigma1, sigma2, num_samples, a=a)
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
    
    