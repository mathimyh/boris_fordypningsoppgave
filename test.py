import matplotlib.pyplot as plt
import numpy as np

def f(x):
    return step(-x+200e-9)*(100*np.tanh((-x) / 50e-9) + 101) + step(x-2300e-9)*(100*np.tanh((x-2500e-9)/ 50e-9) + 101) + step(x-200e-9) - step(x-2300e-9)

def g(x):
    return 1 + 1000 * (np.exp(-(x)**2 / 500e-18 ) + np.exp(-(x-1200e-9)**2 / 2000e-18))

def h(x):
    return  1000* (1 / (1 + np.exp(x)) + 1 / (1 + np.exp(-x + 800))) + 1

x = np.linspace(0, 2500e-9, 5000)

def step(x):
    return 1 * (x > 0)

plt.plot(x, f(x))
plt.show()
