import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import os

__dirname = os.path.dirname(__file__)

sb.set()

x_edge = y_edge = 50

plt.axis([0, x_edge, 0, y_edge])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel("Reservations", fontsize=15)
plt.ylabel("Pizzas", fontsize=15)

def plot_data(X, Y, prediction=None):
    plt.plot(X, Y, "bo")

    if prediction is not None:
        plt.plot([0, x_edge], [0, prediction], linewidth=1, color="g")
    
    plt.show()

if __name__ == "__main__":
    X, Y = np.loadtxt(os.path.join(__dirname, "./pizza.txt"), skiprows=1, unpack=True)
    plot_data(X, Y)