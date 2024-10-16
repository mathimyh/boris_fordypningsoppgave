import matplotlib.pyplot as plt
import numpy as np

def f(x):
    return np.tanh((-x+50)) + np.tanh((x-550)) + 3

def g(x):
    return 1 + 1000 * (np.exp(-(x)**2 / 500e-18 ) + np.exp(-(x-800e-9)**2 / 2000e-18))

def h(x):
    return  1000* (1 / (1 + np.exp(x)) + 1 / (1 + np.exp(-x + 800))) + 1

x = np.linspace(0, 800e-9, 800)

plt.plot(x, g(x))
plt.show()
