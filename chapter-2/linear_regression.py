import numpy as np
import os

from plot import plot_data, x_edge

__dirname = os.path.dirname(__file__)

# load data
X, Y = np.loadtxt(os.path.join(__dirname, "./pizza.txt"), skiprows=1, unpack=True)

def predict(X, w, b):
    """
    model: y = x * weight + bias
    linear equation = y = mx + b
    """
    return X * w + b

def loss(X, Y, w, b):
    """
    error = prediction - real
    mean squared error: average(error ^ 2)
    """
    return np.average((predict(X, w, b) - Y) ** 2)

def train(X, Y, iterations, learning_rate):
    """
    adjust weights every iteration by the learning rate
    """
    w = b = 0

    for i in range(iterations):
        current_loss = loss(X, Y, w, b)
        print("iteration: %4d => loss: %.6f" % (i, current_loss))
        
        # adjust weights
        if loss(X, Y, w + learning_rate, b) < current_loss:
            w += learning_rate
        elif loss(X, Y, w - learning_rate, b) < current_loss:
            w -= learning_rate

        # adjust bias
        elif loss(X, Y, w, b + learning_rate) < current_loss:
            b += learning_rate
        elif loss(X, Y, w, b - learning_rate) < current_loss:
            b -= learning_rate

        # line converged
        else:
            return w, b

    print("could not converge in %4d iterations" % iterations)

(weight, bias) = train(X, Y, iterations=10000, learning_rate=0.01)
print("weight: %.3f, bias: %.3f" % (weight, bias))
print("prediction: x=%d => y=%.2f" % (20, predict(20, weight, bias)))

plot_data(X, Y, predict, weight, bias)