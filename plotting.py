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


def SA_plotting(filename, plotname, title):

    f = open(filename, 'r')

    lines = f.readlines()

    data = lines[12].strip()

    data = data.split('\t')


    # Calculate only one of the three dimensions
    # 
    ys = []
    i = 0
    while i < len(data):
        ys.append(float(data[i]))
        i += 3

    xs = []

    for i in range(len(ys)):
        xs.append(i)

    plt.plot(xs, ys)
    plt.show()
    # plt.xlabel("Distance from injector (nm)")
    # plt.ylabel("Spin accumulation in x-direction")
    # plt.title(title)

    # plotname = "plots/" + plotname
    # plt.savefig(plotname, dpi=500)

    # Total length of the vector

    # ys = []

    # for i in range(0, len(data), 3):
    #     ys.append(np.sqrt(float(data[i])**2 + float(data[i+1])**2 + float(data[i+2])**2))

    # xs = []

    # for i in range(len(ys)):
    #     xs.append(i)

    # plt.plot(xs, ys)
    # plt.xlabel("Distance from injector (nm)")
    # plt.ylabel("Spin accumulation")
    # plt.title(title)

    # plotname = "plots/" + plotname
    # plt.savefig(plotname, dpi=500)


def plateau_plot(filename, plotname, title):

    f = open(filename, 'r')

    lines = f.readlines()
    lines = lines[13:]

    ts = []
    vals = []

    for line in lines:
        vec = line.split('\t')
        ts.append(float(vec[0])*1e12)
        vals.append(float(vec[1]))

    plt.plot(ts, vals)
    plt.xlabel("Time (ps)")
    plt.ylabel("Magnetization")
    plt.title(title)

    plt.savefig(plotname, dpi=500)


def main():
    # SA_plotting('cache/testy.txt', "afm_transport/x-axis_mxdmdt2.png", "Spin accumulation in AFM (mxdmdt2), V = 140μV")
    plateau_plot("cache/plateau_%data%_%x_val%.txt", "plots/afm_transport/plateau_290.png", "Magnetization at 290 nm")

if __name__ == '__main__':
    main()