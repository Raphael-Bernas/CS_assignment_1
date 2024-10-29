import random
import math
import matplotlib.pyplot as plt
set_seed = 1234
random.seed(set_seed)

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

def line(a, b, x):
    return a*x + b

def main():
    N = 150
    a = 1.3
    b = -0.9
    h = lambda x: line(a, b, x)
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
    sample_label = []
    # divide the samples with the line
    for i in range(N):
        if samples_y[i] > h(samples_x[i]):
            sample_label.append(1)
        else:
            sample_label.append(-1)
    # plot the samples with color and the line
    plt.scatter(samples_x, samples_y, c=sample_label)
    plt.plot([-10, 15], [h(-10), h(15)])
    # add title with the line coeffictients
    plt.title(f"Line: y = {a}x + {b}")
    # save the figure
    plt.savefig("samples.png")
    plt.show()
    
if __name__ == "__main__":
    main()
    
