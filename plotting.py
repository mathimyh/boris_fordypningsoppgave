import matplotlib.pyplot as plt
import numpy as np

def plot_something(filename):
    
    f = open(filename, 'r')

    lines = f.readlines()

    x = []

    for i in range(len(lines)):
        x.append(i)

    plt.plot(x, lines)
    plt.show()


def SA_plotting(filename):

    f = open(filename, 'r')

    lines = f.readlines()

    data = lines[12].strip()

    data = data.split('\t')

    ys = []


    # Calculate only one of the three dimensions
    i = 2
    while i < len(data):
        ys.append(float(data[i]))
        i += 3

    xs = []

    for i in range(len(ys)):
        xs.append(i)

    plt.plot(xs, ys)
    plt.show()

    # Maybe the total length of the vector? (Doesn't seem like it)

    ys = []

    for i in range(0, len(data), 3):
        ys.append(np.sqrt(float(data[i])**2 + float(data[i+1])**2 + float(data[i+2])**2))

    xs = []

    for i in range(len(ys)):
        xs.append(i)

    plt.plot(xs, ys)
    plt.show()

SA_plotting('temp/try1.txt')