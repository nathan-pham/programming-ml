import numpy as np
import os

__dirname = os.path.dirname(__file__)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def forward(X, w):
    """
    forward propogation: feeding data through a system
    weighted sum: np.matmul
    """
    return sigmoid(np.matmul(X, w))

def classify(X, w):
    """
    binary classification, round between 0 or 1
    """
    return np.round(forward(X, w))

def loss(X, Y, w):
    """
    mean squared error is uneven when used with sigmoids & gradient descent
    log loss is smoother; works the same: closer to ground truth, less loss
    """
    y_hat = forward(X, w)
    return -np.average((Y * np.log(y_hat)) + ((1 - Y) * np.log(1 - y_hat)))

def gradient(X, Y, w):
    return np.matmul(X.T, (forward(X, w) - Y)) / X.shape[0]

def train(X, Y, iterations, learning_rate):
    w = np.zeros((X.shape[1], 1))

    for i in range(iterations):
        print("iteration: %4d, loss: %.20f" % (i, loss(X, Y, w)))
        w -= gradient(X, Y, w) * learning_rate

    return w

def test(X, Y, w):
    """
    get accuracy of model weights
    number of correct / number of examples
    """
    total_examples = X.shape[0]
    correct_results = np.sum(classify(X, w) == Y)
    success_percent = correct_results / total_examples
    print("\nsuccess: %d/%d (%.2f%%)" % (correct_results, total_examples, success_percent * 100))


x1, x2, x3, y = np.loadtxt(os.path.join(__dirname, "./police.txt"), skiprows=1, unpack=True)
X = np.column_stack((np.ones(x1.size), x1, x2, x3))
Y = y.reshape(-1, 1)

weights = train(X, Y, iterations=10000, learning_rate=0.001)
test(X, Y, weights)