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


def plot_tAvg_comparison(plots, legends, title, savename):
    
    indexer = 0

    for plot in plots:


        f = open(plot, 'r')

        plat = plot.split('/')[-1]
        temps = plat.split('_')
        temp = temps[2]
        V = float(temp[1:])

        lines = f.readlines()
        lines = lines[10:]

        xs = np.linspace(0, 3490, 3490)

        vals = []

        for i in range(len(lines)):
            vec1 = lines[i].split('\t')
            vec2 = lines[i].split('\t')
            all_vals = vec1[1:]
            j = 0
            temp = []
            while j < len(all_vals):
                temp.append(float(all_vals[j]))
                j += 3
            vals.append(temp)

        ys = []

        for i in range(len(vals[0])):
            val = 0
            for j in range(len(vals)):
                val += float(vals[j][i])
            val /= len(vals)
            ys.append(val)

        ys = [y / ys[0] for y in ys]

        plt.plot(xs, ys, label=legends[indexer])

        indexer += 1

    plt.xlabel('Distance from injector (nm)')
    plt.ylabel('μ')
    plt.title(title)

    plt.legend()

    plt.savefig(savename, dpi=600)


def main():
    a = 0
    # SA_plotting('cache/tAvg_damping0.001_V0.145_mxdmdt.txt', "afm_transport/x_axis_mxdmdt_400nm.png", "Spin accumulation in AFM (mxdmdt) at 400 nm, V = -160μV")
    # plateau_plot("cache/plateau_V-0.15_damping0.005_mxdmdt_250nm_350nm_450nm_550nm.txt", "plots/plateau/plateau_250_V-0.2_0.005_mxdmdt.png", "Spin accumulation (mxdmdt) at 250 nm with V = -0.15 μV")
    f1 = 'cache/t_avg/7000long/tAvg_damping0.0002_V-0.009_mxdmdt.txt'
    f2 = 'cache/t_avg/7000long_10thick/tAvg_damping0.0002_V-0.018_mxdmdt.txt'
    f3 = 'cache/t_avg/7000long_15thick/tAvg_damping0.0002_V-0.027_mxdmdt.txt'

    l1 = '1 layer'
    l2 = '2 layers'
    l3 = '3 layers'

    title = 'Normalized spin accumulation for different thicknesses'

    savename = 'plots/t_avg/7000long_15thick/tAvg_meshthickness_comparison.png'

    plot_tAvg_comparison((f1,f2,f3), (l1,l2,l3), title, savename)

if __name__ == '__main__':
    main()