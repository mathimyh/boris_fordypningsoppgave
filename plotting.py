import matplotlib.pyplot as plt
import numpy as np
from textwrap import wrap

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

    data = lines[70].strip()

    data = data.split('\t')
    data = data[1:]


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


def plot_plateau(t, V, data, damping, x_vals):

    x_vals_string = 'nm_'.join(str(x_val) for x_val in x_vals)

    filename = 'C:/Users/mathimyh/Documents/Boris Data/Simulations/boris_fordypningsoppgave/cache/plateau/plateau_V' + str(V) + '_damping' + str(damping) + '_' + data[1:-1] + '_' + x_vals_string + 'nm.txt'  
    f = open(filename, 'r')

    lines = f.readlines()
    lines = lines[12:]
    

    indexer = 0

    fig, ax = plt.subplots(2, 3)
    

    for i in range(1, len(lines[0].split('\t')), 3):
        ts = []
        vals = []

        for line in lines:
            vec = line.split('\t')
            ts.append(float(vec[0])*1e12)
            vals.append(float(vec[i]))

        text = str(x_vals[indexer]) + 'nm'
        if indexer == 0:
            ax[0][0].plot(ts, vals)
            ax[0][0].title.set_text(text)
        elif indexer == 1:
            ax[0][1].plot(ts, vals)
            ax[0][1].title.set_text(text)
        elif indexer == 2:
            ax[0][2].plot(ts, vals)
            ax[0][2].title.set_text(text)
        elif indexer == 3:
            ax[1][0].plot(ts, vals)
            ax[1][0].title.set_text(text)
        elif indexer == 4:
            ax[1][1].plot(ts, vals)
            ax[1][1].title.set_text(text)
        elif indexer == 5:
            ax[1][2].plot(ts, vals)
            ax[1][2].title.set_text(text)

        indexer += 1

    plotname = 'C:/Users/mathimyh/Documents/Boris Data/Simulations/boris_fordypningsoppgave/plots/plateau/plateau_V' + str(V) + '_damping' + str(damping) + '_' + data[1:-1] + '_' + x_vals_string + 'nm.png'
    fig.suptitle(' Spin accumulation over time')
    fig.tight_layout()
    fig.savefig(plotname, dpi=600)



def plot_tAvg_SA(t, V, damping, data, x_start, x_stop):

    filename = 'cache/t_avg/tAvg_damping' + str(damping) + '_V' + str(V) + '_' + str(data[1:-1]) + '.txt'
    f = open(filename, 'r')

    lines = f.readlines()
    lines = lines[10:]

    xs = np.linspace(x_start, x_stop, x_stop - x_start)

    vals = []

    for line in lines:
        vec = line.split('\t')
        all_vals = vec[1:]
        i = 0
        temp = []
        while i < len(all_vals):
            temp.append(float(all_vals[i]))
            i += 3
        vals.append(temp)
        

    ys = []

    for i in range(len(vals[0])):
        val = 0
        for j in range(len(vals)):
            val += float(vals[j][i])
        val /= len(vals)
        ys.append(val)


    plt.plot(xs, ys)
    plt.xlabel("x (nm)")
    plt.ylabel("Spin accumulation")
    title = 'tAvg '+ str(data[1:-1])
    plt.title("\n".join(wrap(title, 60)))
    plt.tight_layout()

    plotname = 'plots/t_avg/tAvg_damping' + str(damping) + '_V' + str(V) + '_' + str(data[1:-1]) + '_t' + str(t) + 'ps.png'
    plt.savefig(plotname, dpi=500)   


def plot_tAvg_comparison(positive, negative, title, savename):
    
    posotive = positive.split('/')[-1]
    temps = posotive.split('_')
    temp = temps[2]
    V = float(temp[1:])


    f1 = open(positive, 'r')
    f2 = open(negative, 'r')

    lines1 = f1.readlines()
    lines2 = f2.readlines()
    lines1 = lines1[10:]
    lines2 = lines2[10:]

    xs = np.linspace(0, len(lines1[0]))

    vals1 = []
    vals2 = []

    for i in range(len(lines1)):
        vec1 = lines1[i].split('\t')
        vec2 = lines2[i].split('\t')
        all_vals1 = vec1[1:]
        all_vals2 = vec2[1:]
        j = 0
        temp1 = []
        temp2 = []
        while j < len(all_vals1):
            temp1.append(float(all_vals1[j]))
            temp2.append(float(all_vals2[j]))
            j += 3
        vals1.append(temp1)
        vals2.append(temp2)
        

    ys1 = []
    ys2 = []

    for i in range(len(vals1[0])):
        val1 = 0
        val2 = 0
        for j in range(len(vals1)):
            val1 += float(vals1[j][i])
            val2 += float(vals2[j][i])
        val1 /= len(vals1)
        val2 /= len(vals2)
        ys1.append(val1)
        ys2.append(val2)

    pos_V = 'V = ' + str(V * 1e6) + 'μV'
    neg_V = 'V = ' + str(-V * 1e6) + 'μV'


    plt.plot(xs, ys1, color='r', label=pos_V)
    plt.plot(xs, ys2, color='b', label=neg_V)

    plt.xlabel('Distance from injector (nm)')
    plt.ylabel('μ')
    plt.title(title)

    plt.legend()

    plt.savefig(savename, dpi=600)

def main():
    a = 0
    # SA_plotting('cache/tAvg_damping0.001_V0.145_mxdmdt.txt', "afm_transport/x_axis_mxdmdt_400nm.png", "Spin accumulation in AFM (mxdmdt) at 400 nm, V = -160μV")
    # plateau_plot("cache/plateau_V-0.15_damping0.005_mxdmdt_250nm_350nm_450nm_550nm.txt", "plots/plateau/plateau_250_V-0.2_0.005_mxdmdt.png", "Spin accumulation (mxdmdt) at 250 nm with V = -0.15 μV")
    plot_tAvg_comparison('cache/t_avg/7000long/tAvg_damping0.0002_V0.009_mxdmdt.txt', 'cache/t_avg/7000long/tAvg_damping0.0002_V-0.009_mxdmdt.txt', 'Time averaged spin accumulation in AFM with virtual current', 'plots/t_avg/7000long/tAvg_comparison_mxdmdt.png')


if __name__ == '__main__':
    main()