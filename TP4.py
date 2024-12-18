import numpy as np
import matplotlib.pyplot as plt

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

def Step_MH_P(z, sigma1, sigma2):
    choice = 0
    if np.random.rand() < 0.5:
        z_new = sample_P1(z, sigma1)
        choice = 1
    else:
        z_new = sample_P2(z, sigma2)
    alpha = min(PI(z_new[0], z_new[1]) / PI(z[0], z[1]), 1)
    if np.random.rand() < alpha:
        return z_new, alpha, choice
    else:
        return z, alpha, choice

def MH_P(z, sigma1, sigma2, num_samples):
    samples = np.zeros((num_samples, 2))
    samples[0] = z
    accept_x = []
    accept_y = []
    for i in range(1, num_samples):
        samples[i], alpha, choice = Step_MH_P(samples[i-1], sigma1, sigma2)
        if choice == 1:   
            accept_x.append(alpha)
        else:
            accept_y.append(alpha)
    return samples, accept_x, accept_y

if __name__ == "__main__":
    np.random.seed(42)
    #Ex1
    z=(0,0)
    sigma1 = 1.0
    sigma2 = 1.0

    num_samples = 100000
    samples, accept_x, accept_y = MH_P(z, sigma1, sigma2, num_samples)
    # plot samples
    plt.plot(samples[:,0], samples[:,1], 'o')
    plt.show()
    # plot acceptance rates
    plt.plot(accept_x, 'r')
    plt.plot(accept_y, 'b')
    plt.legend(['acceptance rate for x', 'acceptance rate for y'])
    plt.show()
    
    