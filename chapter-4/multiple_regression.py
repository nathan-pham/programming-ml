import numpy as np
import os

__dirname = os.path.dirname(__file__)

"""
weighted sum of inputs: y = (x1 * y1) + (x2 * y2) + (x3 * y3) + ... + b
multi-dimensional loss functions require matrix manipulation

matrix multiplication golden rule: number of columns of m1 must match number of rows of m2
[ 1 2 3 ]     [ 7    8 ]     [ 58,   64 ]
[ 4 5 6 ]  x  | 9   10 |  =  [ 139, 154 ]
              [ 11  12 ]

calculating matrix[1][1] = (1, 2, 3) • (7, 8, 11) = (1 * 7) + (2 * 8) + (3 * 11) = 58

matrix transpose: swaps matrix dimensions, flips matrix over diagonal
[ 1 2 3 ] = tranpose => [ 1 4 ]
[ 4 5 6 ]               [ 2 5 ]
                        [ 3 6 ]
"""

x1, x2, x3, y = np.loadtxt(os.path.join(__dirname, "./pizza_3_vars.txt"), skiprows=1, unpack=True)
X = np.column_stack((np.ones(x1.size), x1, x2, x3)) # => column 1 represents bias; x0 * b
Y = y.reshape(-1, 1)

def predict(X, w):
    return np.matmul(X, w)

def loss(X, Y, w):
    return np.average((predict(X, w) - Y) ** 2)

def gradient(X, Y, w):
    return 2 * np.matmul(X.T, (predict(X, w) - Y)) / X.shape[0]

def train(X, Y, iterations, learning_rate):
    w = np.zeros((X.shape[1], 1))

    for i in range(iterations):
        print("iteration: %4d => loss: %.20f" % (i, loss(X, Y, w)))
        w -= gradient(X, Y, w) * learning_rate

    return w

weights = train(X, Y, iterations=10_000, learning_rate=0.001)
print("\nweights: %s" % weights.T)

print("\npredictions:")
for i in range(5):
    print("X[%d] => %.4f (label: %d)" % (i, predict(X[i], weights), Y[i]))
