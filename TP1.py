import random
import math
import numpy as np
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
set_seed = 1234
#random.seed(set_seed)

def marsaglia_bray():
    V1 = 1
    V2 = 1
    while V1**2 + V2**2 > 1:
        U1 = random.uniform(0, 1)
        U2 = random.uniform(0, 1)
        V1 = 2*U1-1
        V2 = 2*U2-1
        

    S = -2*math.log(V1**2 + V2**2)
    X = S*V1/math.sqrt(V1**2 + V2**2)
    Y = S*V2/math.sqrt(V1**2 + V2**2)
    
    return X, Y

def line(a, x):
    return a*x 

def Jn(w, X, y):
        # Calculates the cost J_n(w) as Mean Squared Error for simplicity.
        return 0.5 * np.mean((y - X.dot(w)) ** 2)
    

def stochastic_gradient_descent(N_max, w0, eta, epsilon, X, y, N_batch = 1, return_j=False, learning_rate_decay=1., sign = False):
    w = w0.copy()
    w = w / np.linalg.norm(w)
    k = 1  
    n = X.shape[0]  # Number of samples
    # normalize the data
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    w = w.reshape(X.shape[1],1)
    # Initial cost calculation
    J_prev = Jn(w, X, y)
    if return_j:
        J = [J_prev]
    stop_crit = N_max
    eta_curr = eta
    
    while stop_crit > epsilon and k < N_max :
        S = 0
        for i in range(N_batch):
            # Randomly select an index I from uniform distribution U([1, n])
            I = random.randint(0, n-1)
            # Update the gradient
            if sign:
                S += -np.sign(X[I]*(y[I] - X[I].dot(w)))
            else:
                #print((y[I] - X[I].dot(w)))
                S += 2*(X[I].dot(w)-y[I])*X[I]
        S /= N_batch
        S = S.reshape(w.shape)
        # Update weights
        w = w - eta_curr*S
        J_curr = Jn(w, X, y)
        print(J_curr)
        stop_crit = abs(J_curr - J_prev)/J_prev+1    
        # Update for next iteration
        J_prev = J_curr
        if return_j:
            J.append(J_curr)
        k += 1
        eta_curr = eta_curr * learning_rate_decay
        # #print(k)
        # #print(w[0]/w[1])
    if return_j:
        print(J[0],J[-1])
        return w, J
    else:
        return w

def test_cancer_data():
    breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 
    # data (as pandas dataframes) 
    X = breast_cancer_wisconsin_diagnostic.data.features 
    y = breast_cancer_wisconsin_diagnostic.data.targets

    # Ensure that y is a binary classification
    y = y.map(lambda x: 1 if x == 'M' else -1)
    X_reduced = X.values#[:,:2]
    # # Center the data
    # mean = [(1/X_reduced.shape[0])*np.sum(X_reduced[:,0]),(1/X_reduced.shape[0])*np.sum(X_reduced[:,1])]
    # print(mean)
    # X_reduced = np.array([X_reduced[i,:] - mean for i in range(X_reduced.shape[0])])

    eta = 0.001
    w0 = np.ones(X_reduced.shape[1])
    N_batch = 1
    epsilon = 10e-10
    N_max = 10**4
    w_hat, J_val = stochastic_gradient_descent(N_max, w0, eta, epsilon, X_reduced, y.values, N_batch = N_batch, return_j=True, sign=False, learning_rate_decay=.999)
    
    print("Found w :")
    print(-w_hat[0]/w_hat[1])
    h_hat = lambda x: line(-w_hat[0]/w_hat[1], x)
    
    # plot the samples with color and the line
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y.values)
    plt.plot([-10, 20], [h_hat(-10), h_hat(20)])
    plt.savefig("TP1_results/cancer.png")
    plt.show()
    # plot the cost function
    plt.plot(J_val)
    plt.savefig("TP1_results/cost.png")
    plt.show()


def main(noised = False):
    N = 150
    a = 1.3
    h = lambda x: line(a, x)
    N_max = 10**5
    # create a list of samples of size N
    samples_x = []
    samples_y = []
    for i in range(N):
        X, Y = marsaglia_bray()
        if i < N/2:
            samples_x.append(X*math.sqrt(5)-5)
            samples_y.append(Y*math.sqrt(5))
        else :
            samples_x.append(X*math.sqrt(5)+5)
            samples_y.append(Y*math.sqrt(5))
    samples_label = []
    # divide the samples with the line
    for i in range(N):
        if samples_y[i] > h(samples_x[i]):
            samples_label.append(1)
        else:
            samples_label.append(-1)
    X = np.array([[samples_x[i],samples_y[i]] for i in range(N)])
    y = np.array(samples_label)
    eta = 0.01
    w0 = (1,1)
    N_batch = N//10
    epsilon = 10e-10
    if not noised:
        w_hat = stochastic_gradient_descent(N_max, w0, eta, epsilon, X, y)
        print("Found w :")
        print(-w_hat[0]/w_hat[1])
        h_hat = lambda x: line(-w_hat[0]/w_hat[1], x)
        # plot the samples with color and the line
        plt.scatter(samples_x, samples_y, c=samples_label)
        plt.plot([-10, 15], [h(-10), h(15)])
        # add title with the line coeffictients
        plt.title(f"Line: y = {a}x")
        # save the figure
        plt.savefig("TP1_results/samples.png")
        # plot the estimated line 
        plt.plot([-10, 15], [h_hat(-10), h_hat(15)])
        plt.savefig("TP1_results/estimated.png")
        plt.show()
    else:
        # noising the samples
        noise = 0.1
        samples_x_noised = np.zeros(N)
        samples_y_noised = np.zeros(N)
        for i in range(N):
            x1,x2 = marsaglia_bray()
            samples_x_noised[i] += noise*x1
            samples_y_noised[i] += noise*x2
        X_noised = np.array([[samples_x[i]+samples_x_noised[i],samples_y[i]+samples_y_noised[i]] for i in range(N)])
        w_hat_noised = stochastic_gradient_descent(N_max, w0, eta, epsilon, X_noised, y)
        print("Found w noised:")
        print(-w_hat_noised[0]/w_hat_noised[1])
        h_hat_noised = lambda x: line(-w_hat_noised[0]/w_hat_noised[1], x)
        plt.scatter(samples_x, samples_y, c=samples_label)
        plt.plot([-10, 15], [h(-10), h(15)])
        # add title with the line coeffictients
        plt.title(f"Line: y = {a}x")
        # save the figure
        plt.savefig("TP1_results/samples.png")
        # plot the noised line
        plt.plot([-10, 15], [h_hat_noised(-10), h_hat_noised(15)])
        plt.savefig("TP1_results/noised.png")
        plt.show()
    
if __name__ == "__main__":
    generate = False
    if generate:
        main(noised=True)
    else:
        test_cancer_data()
