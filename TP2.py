import random
import math
import numpy as np
import matplotlib.pyplot as plt

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

if __name__ == "__main__":
    Px = [0.1, 0.3, 0.4, 0.2]
    X = [1, 2, 3, 4]
    N = 10000
    plot_distributions(Px, X, N)