import numpy as np
import os

from plot import plot_data, x_edge

__dirname = os.path.dirname(__file__)

# load data
X, Y = np.loadtxt(os.path.join(__dirname, "./pizza.txt"), skiprows=1, unpack=True)

def predict(X, w):
    """
    model: y = x * w
    """
    return X * w

def loss(X, Y, w):
    """
    error = prediction - real
    mean squared error: average(error ^ 2)
    """
    return np.average((predict(X, w) - Y) ** 2)

def train(X, Y, iterations, learning_rate):
    """
    adjust weights every iteration by the learning rate
    """
    w = 0
    for i in range(iterations):
        current_loss = loss(X, Y, w)
        print("iteration: %4d => loss: %.6f" % (i, current_loss))
        
        if loss(X, Y, w + learning_rate) < current_loss:
            w += learning_rate
        elif loss(X, Y, w - learning_rate) < current_loss:
            w -= learning_rate
        else:
            return w

    print("could not converge in %4d iterations" % iterations)

weight = train(X, Y, iterations=10000, learning_rate=0.01)
print("weight: %.3f" % weight)
print("prediction: x=%d => y=%.2f" % (20, predict(20, weight)))

plot_data(X, Y, predict(x_edge, weight))