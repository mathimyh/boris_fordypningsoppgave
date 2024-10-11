import matplotlib.pyplot as plt
import numpy as np

def f(x):
    return np.tanh((-x+50)) + np.tanh((x-550)) + 3

def g(x):
    return 1 + 10000 * (np.exp(-(x)**2 / 500e-18 ) + np.exp(-(x-600e-9)**2 / 500e-18))


x = np.linspace(0, 600e-9, 600)

plt.plot(x, g(x))
plt.show()