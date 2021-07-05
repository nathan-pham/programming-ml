import numpy as np
import os

__dirname = os.path.dirname(__file__)

X, Y = np.loadtxt(os.path.join(__dirname, "../chapter-2/pizza.txt"), skiprows=1, unpack=True)

"""
pitfalls of gradient descent,  the loss function should be:
- convex (few features, one global maximum)
- differentiable (also continuous)

mean squared error is best for gradient descent 
- absolute error has a non-differentiable cusp around 0
- squaring errors means steeper slopes & faster, smooth improvements
"""

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

def gradient(X, Y, w, b):
    """
    derivative of original loss function
    adding parameters (bias) requires partial derivatives
    """
    w_gradient = 2 * np.average(X * (predict(X, w, b) - Y))
    b_gradient = 2 * np.average(predict(X, w, b) - Y)
    return (w_gradient, b_gradient)

def train(X, Y, iterations, learning_rate):
    w = b = 0

    for i in range(iterations):
        current_loss = loss(X, Y, w, b)
        print("iteration: %4d => loss: %.6f" % (i, current_loss))
        
        w_gradient, b_gradient = tuple(learning_rate * g for g in gradient(X, Y, w, b))
        w -= w_gradient
        b -= b_gradient

    return w, b

(weight, bias) = train(X, Y, iterations=20000, learning_rate=0.001)
print("weight: %.3f, bias: %.3f" % (weight, bias))
print("prediction: x=%d => y=%.2f" % (20, predict(20, weight, bias)))
