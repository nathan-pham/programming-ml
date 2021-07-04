import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import os

__dirname = os.path.dirname(__file__)

sb.set()

plt.axis([0, 50, 0, 50])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel("Reservations", fontsize=15)
plt.ylabel("Pizzas", fontsize=15)

X, Y = np.loadtxt(os.path.join(__dirname, "./pizza.txt"), skiprows=1, unpack=True)
plt.plot(X, Y, "bo")
plt.show()