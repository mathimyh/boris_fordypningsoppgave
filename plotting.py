import matplotlib.pyplot as plt
import numpy as np
from textwrap import wrap
import os
from itertools import chain

plt.rcParams.update({'font.size': 22})

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


def plot_plateau(meshdims, cellsize, t, V, data, damping, x_vals, MEC, ani):

    x_vals_string = 'nm_'.join(str(x_val) for x_val in x_vals)

    mec_folder = ''
    if MEC:
        mec_folder = 'MEC/'

    folder_name = ani + '/plots/' + mec_folder + 'plateau/' + str(meshdims[0]) + 'x' + str(meshdims[1]) + 'x' + str(meshdims[2])
    print(folder_name)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        

    filename = 'C:/Users/mathimyh/Documents/Boris Data/Simulations/boris_fordypningsoppgave/' + ani + '/cache/' + mec_folder + 'plateau/' +  str(meshdims[0]) + 'x' + str(meshdims[1]) + 'x' + str(meshdims[2]) + '/plateau_V'  + str(V) + '_damping' + str(damping) + '_' + data[1:-1] + '_' + x_vals_string + 'nm.txt'  
    f = open(filename, 'r')

    lines = f.readlines()
    lines = lines[12:]
    

    indexer = 0

    fig, ax = plt.subplots(2, 3)

    fig.set_figheight(10)
    fig.set_figwidth(14)
    
    direction = 1
    if ani == 'OOP':
        direction = 3

    for i in range(direction, len(lines[0].split('\t')), 3):
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


    plotname = 'C:/Users/mathimyh/Documents/Boris Data/Simulations/boris_fordypningsoppgave/' + ani + '/plots/' + mec_folder + 'plateau/' + str(meshdims[0]) + 'x' + str(meshdims[1]) + 'x' + str(meshdims[2]) + '/plateau_V' + str(V) + '_damping' + str(damping) + '_' + data[1:-1] + '_' + x_vals_string + 'nm.png'
    fig.suptitle(' Spin accumulation over time')
    fig.tight_layout()
    fig.savefig(plotname, dpi=600)


def plot_tAvg_SA(meshdims, cellsize, t, V, damping, data, x_start, x_stop, MEC, ani):

    mec_folder = ''
    if MEC:
        mec_folder = 'MEC/'

    folder_name = ani + '/plots/' + mec_folder + 't_avg/' + str(meshdims[0]) + 'x' + str(meshdims[1]) + 'x' + str(meshdims[2])
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    filename = ani + '/cache/' + mec_folder + 't_avg/'  + str(meshdims[0]) + 'x' + str(meshdims[1]) + 'x' + str(meshdims[2]) + '/tAvg_damping' + str(damping) + '_V' + str(V) + '_' + str(data[1:-1]) + '.txt'
    f = open(filename, 'r')

    lines = f.readlines()
    lines = lines[10:]

    xs = np.linspace(x_start, x_stop, x_stop - x_start)

    vals = []


    for line in lines:
        # This is the component we look at. In plane means x-component (0) and out-of-plane means z (2)
        direction = 0
        if ani == 'OOP':
            direction = 2
        vec = line.split('\t')
        all_vals = vec[1:]
        temp = []
        while direction < len(all_vals):
            temp.append(float(all_vals[direction]))
            direction += 3
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

    plotname = ani + '/plots/' + mec_folder + 't_avg/' +  str(meshdims[0]) + 'x' + str(meshdims[1]) + 'x' + str(meshdims[2]) + '/tAvg_damping' + str(damping) + '_V' + str(V) + '_' + str(data[1:-1]) + '_t' + str(t) + 'ps.png'
    plt.savefig(plotname, dpi=500)   


def plot_tAvg_SA_2D(meshdims, cellsize, t, V, damping, data, x_start, x_stop, MEC, ani):

    mec_folder = ''
    if MEC:
        mec_folder = 'MEC/'

    folder_name = ani + '/plots/' + mec_folder + 't_avg/' + str(meshdims[0]) + 'x' + str(meshdims[1]) + 'x' + str(meshdims[2])
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    filename = ani + '/cache/' + mec_folder + 't_avg/'  + str(meshdims[0]) + 'x' + str(meshdims[1]) + 'x' + str(meshdims[2]) + '/2D_tAvg_damping' + str(damping) + '_V' + str(V) + '_' + str(data[1:-1]) + '.txt'
    f = open(filename, 'r')

    lines = f.readlines()
    lines = lines[10:]

    # Turn the raw data into a list of numpy arrays. Every entry in the arrays are floats.
    raw_data = []
    for line in lines:
        
        # Make a list of all entries and an empty array to fill only the component we want in
        vec = line.strip().split('\t')[1:]
        temp = np.empty(int(len(vec)/3))
        
         # This is the component we look at. In plane means x-component (0) and out-of-plane means z (2)
        direction = 0
        if ani == 'OOP':
            direction = 2

        # Iterate over all and add only the component we want. Convert to float
        indexer = 0
        while direction < len(vec):
            temp[indexer] = float(vec[direction])
            indexer += 1
            direction += 3

        # Reshape to 2D array and add to the data list
        raw_data.append(temp.reshape(meshdims[2], int(len(temp)/(meshdims[2]))))

    # Now find the time averages for all the data
    tAvg_data = np.zeros_like(raw_data[0]) # Haven't tried this before, but should work, right?

    for k, matrix in enumerate(raw_data):
        for i, row in enumerate(matrix):
            for j, col in enumerate(row):
                tAvg_data[i][j] += col
                if k == len(raw_data)-1:
                    tAvg_data[i][j] /= len(raw_data)

    plt.plot(tAvg_data[50])
    plt.show()

    # plt.imshow((tAvg_data), extent=[0, (x_stop-x_start)/1000, 0, meshdims[2]], aspect='auto', interpolation='bilinear', cmap='inferno')
    # plt.colorbar()
    # plt.tight_layout()

    # plotname = ani + '/plots/' + mec_folder + 't_avg/' +  str(meshdims[0]) + 'x' + str(meshdims[1]) + 'x' + str(meshdims[2]) + '/2D_tAvg_damping' + str(damping) + '_V' + str(V) + '_' + str(data[1:-1]) + '_t' + str(t) + 'ps.png'
    # plt.savefig(plotname, dpi=500) 
    # plt.show()

def plot_tAvg_SA_z(meshdims, cellsize, t, V, damping, data, x_start, x_stop, MEC, ani):

    mec_folder = ''
    if MEC:
        mec_folder = 'MEC/'

    folder_name = ani + '/plots/' + mec_folder + 't_avg/' + str(meshdims[0]) + 'x' + str(meshdims[1]) + 'x' + str(meshdims[2])
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    filename = ani + '/cache/' + mec_folder + 't_avg/'  + str(meshdims[0]) + 'x' + str(meshdims[1]) + 'x' + str(meshdims[2]) + '/tAvg_damping' + str(damping) + '_V' + str(V) + '_' + str(data[1:-1]) + '_zdir.txt'
    f = open(filename, 'r')

    lines = f.readlines()
    lines = lines[10:]

    vals = []

    for line in lines:
        # This is the component we look at. In plane means x-component (0) and out-of-plane means z (2)
        direction = 0
        if ani == 'OOP':
            direction = 2
        vec = line.split('\t')
        all_vals = vec[1:]
        temp = []
        while direction < len(all_vals):
            temp.append(float(all_vals[direction]))
            direction += 3
        vals.append(temp)
        

    ys = []

    for i in range(len(vals[0])):
        val = 0
        for j in range(len(vals)):
            val += float(vals[j][i])
        val /= len(vals)
        ys.append(val)

    xs = np.linspace(0, len(ys), len(ys))

    plt.plot(xs, ys)
    plt.xlabel("x (nm)")
    plt.ylabel("Spin accumulation")
    title = 'tAvg '+ str(data[1:-1])
    plt.title("\n".join(wrap(title, 60)))
    plt.tight_layout()

    plotname = ani + '/plots/' + mec_folder + 't_avg/' +  str(meshdims[0]) + 'x' + str(meshdims[1]) + 'x' + str(meshdims[2]) + '/tAvg_zdir_damping_' + str(damping) + '_V' + str(V) + '_' + str(data[1:-1]) + '_t' + str(t) + 'ps.png'
    plt.savefig(plotname, dpi=500)

def plot_tAvg_comparison(plots, legends, savename, ani):
    
    indexer = 0
    plt.figure(figsize=(14,10))


    colors = ['darkgreen', 'green', 'forestgreen', 'mediumseagreen', 'limegreen', 'red', 'blue', 'yellow', 'cyan']

    for plot in plots:

        f = open(plot, 'r')

        plat = plot.split('/')[-1]
        temps = plat.split('_')
        temp = temps[2]
        V = float(temp[1:])

        lines = f.readlines()
        lines = lines[10:]


        vals = []

        for i in range(len(lines)):
            vec1 = lines[i].split('\t')
            all_vals = vec1[1:]
            ani_int = 0
            if ani == 'OOP':
                ani_int = 2
            temp = []
            while ani_int < len(all_vals):
                temp.append(float(all_vals[ani_int]))
                ani_int += 3
            vals.append(temp)

        ys = []

        for i in range(len(vals[0])):
            val = 0
            for j in range(len(vals)):
                val += float(vals[j][i])
            val /= len(vals)
            ys.append(val)


        # if len(ys) > 1980:
        #     ys = ys[len(ys)-1980:]

        ys = [(y) / ys[0]  for y in ys]
        xs = np.linspace(0, len(ys), len(ys))

        plt.plot(xs, ys, label=legends[indexer])

        indexer += 1

    plt.xlabel('Distance from injector (nm)')
    plt.ylabel('μ')

    plt.legend()

    plt.savefig(savename, dpi=600)


def plot_magnon_dispersion(meshdims, damping, MEC, ani, dir):
    
    mec_folder = ''
    if MEC:
        mec_folder = 'MEC/'

    time_step = 0.1e-12

    output_file = 'C:/Users/mathimyh/documents/boris data/simulations/boris_fordypningsoppgave/' + ani + '/cache/' + mec_folder + 'dispersions/' + str(meshdims[0]) + 'x' + str(meshdims[1]) + 'x' + str(meshdims[2]) +  '/dir' + dir + '_dispersion.txt'

    pos_time = np.loadtxt(output_file)

    fourier_data = np.fft.fftshift(np.abs(np.fft.fft2(pos_time)))

    freq_len = len(fourier_data)
    k_len = len(fourier_data[0])
    freq = np.fft.fftfreq(freq_len, time_step)
    kvector = np.fft.fftfreq(k_len, 5e-9)

    k_max = 2*np.pi*kvector[int(0.5 * len(kvector))]*5e-9
    f_min = np.abs(freq[0])
    f_max = np.abs(freq[int(0.5 * len(freq))])/1e12 # to make it THz
    f_points = int(0.5 * freq_len)

    result = [fourier_data[i] for i in range(int(0.5 *freq_len),freq_len)]
    
    fig1,ax1 = plt.subplots()

    ax1.imshow(result, origin='lower', interpolation='bilinear', extent = [-k_max, k_max,f_min, f_max], aspect ="auto", clim=(0, 1200))

    ax1.set_xlabel('qa')
    ax1.set_ylabel('f (THz)')
    # ax1.set_ylim(0, 0.1)

    plt.tight_layout()

    folder_name = ani + '/plots/' + mec_folder + 'dispersions/' + str(meshdims[0]) + 'x' + str(meshdims[1]) + 'x' + str(meshdims[2])
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    savename = ani + '/plots/' + mec_folder + 'dispersions/' + str(meshdims[0]) + 'x' + str(meshdims[1]) + 'x' + str(meshdims[2]) + '/damping' + str(damping) + 'dir' + dir + '_magnon_dispersion.png' 

    plt.savefig(savename, dpi=600)

    plt.show()


def plot_phonon_dispersion(meshdims, damping, MEC, ani, dir,time_step):
    
    mec_folder = ''
    if MEC:
        mec_folder = 'MEC/'


    output_file = 'C:/Users/mathimyh/documents/boris data/simulations/boris_fordypningsoppgave/' + ani + '/cache/' + mec_folder + 'dispersions/' + str(meshdims[0]) + 'x' + str(meshdims[1]) + 'x' + str(meshdims[2]) +  '/dir' + dir + '_phonon_dispersion.txt'

    pos_time = np.loadtxt(output_file)

    fourier_data = np.fft.fftshift(np.abs(np.fft.fft2(pos_time)))

    freq_len = len(fourier_data)
    k_len = len(fourier_data[0])
    freq = np.fft.fftfreq(freq_len, time_step)
    kvector = np.fft.fftfreq(k_len, 5e-9)

    k_max = 2*np.pi*kvector[int(0.5 * len(kvector))]*5e-9
    f_min = np.abs(freq[0])
    f_max = np.abs(freq[int(0.5 * len(freq))])/1e12 # to make it THz
    f_points = int(0.5 * freq_len)

    result = [fourier_data[i] for i in range(int(0.5 *freq_len),freq_len)]
    
    fig1,ax1 = plt.subplots()

    ax1.imshow(result, origin='lower', interpolation='bilinear', extent = [-k_max, k_max,f_min, f_max], aspect ="auto", clim=(0,1200), vmax=20)

    ax1.set_xlabel(r'$q_x$')
    ax1.set_ylabel('f (THz)')
    # ax1.set_ylim(0, 0.1)

    plt.tight_layout()

    folder_name = ani + '/plots/' + mec_folder + 'dispersions/' + str(meshdims[0]) + 'x' + str(meshdims[1]) + 'x' + str(meshdims[2])
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    savename = ani + '/plots/' + mec_folder + 'dispersions/' + str(meshdims[0]) + 'x' + str(meshdims[1]) + 'x' + str(meshdims[2]) + '/damping' + str(damping) + 'dir' + dir + '_phonon_dispersion.png' 

    plt.savefig(savename, dpi=600)

    plt.show()

def plot_phonon_dispersion_specific(output_file, savename, time_step):


    pos_time = np.loadtxt(output_file)

    fourier_data = np.fft.fftshift(np.abs(np.fft.fft2(pos_time)))

    freq_len = len(fourier_data)
    k_len = len(fourier_data[0])
    freq = np.fft.fftfreq(freq_len, time_step)
    kvector = np.fft.fftfreq(k_len, 5e-9)

    k_max = 2*np.pi*kvector[int(0.5 * len(kvector))]*5e-9
    f_min = np.abs(freq[0])
    f_max = np.abs(freq[int(0.5 * len(freq))])/1e12 # to make it THz
    f_points = int(0.5 * freq_len)

    result = [fourier_data[i] for i in range(int(0.5 *freq_len),freq_len)]
    
    fig1,ax1 = plt.subplots()

    ax1.imshow(result, origin='lower', interpolation='bilinear', extent = [-k_max, k_max,f_min, f_max], aspect ="auto", clim=(0,1000), vmax=20)

    ax1.set_xlabel(r'$q_x$')
    ax1.set_ylabel('f (THz)')
    # ax1.set_ylim(0, 0.1)

    plt.tight_layout()

    plt.savefig(savename, dpi=600)

    plt.show()


def plot_dispersions(plots, savename):

    dim1 = 0
    dim2 = 0

    if len(plots) == 2:
        dim1 = 1
        dim2 = 2
    
    elif len(plots) == 4:
        dim1 = 2
        dim2 = 2

    fig, axs = plt.subplots(dim1, dim2)
    
    fig.set_figheight(10)
    fig.set_figwidth(14)

    annotations = ['a', 'b', 'c', 'd']

    for i, ax in enumerate(list(chain.from_iterable(axs))):

        output_file = plots[i]
        
        time_step = 0.1e-12
        pos_time = np.loadtxt(output_file)

        fourier_data = np.fft.fftshift(np.abs(np.fft.fft2(pos_time)))

        freq_len = len(fourier_data)
        k_len = len(fourier_data[0])
        freq = np.fft.fftfreq(freq_len, time_step)
        kvector = np.fft.fftfreq(k_len, 5e-9)

        k_max = 2*np.pi*kvector[int(0.5 * len(kvector))]*5e-9
        f_min = np.abs(freq[0])
        f_max = np.abs(freq[int(0.5 * len(freq))])/1e12 # to make it THz
        f_points = int(0.5 * freq_len)

        result = [fourier_data[i] for i in range(int(0.5 *freq_len),freq_len)]
        
        clim_max = 1200
        if i == 1:
            clim_max = 800

        ax.imshow(result, origin='lower', interpolation='bilinear', extent = [-k_max, k_max,f_min, f_max], aspect ="auto", clim=(0, clim_max))

        ax.annotate(annotations[i], (0.05, 0.85), xycoords = 'axes fraction', color='white', fontsize=32)

        ax.set_xlabel('qa')
        ax.set_ylabel('f (THz)')
    # ax1.set_ylim(0, 0.1)

    fig.tight_layout()

    plt.savefig(savename, dpi=600)

    plt.show()


def plot_trajectory(meshdims, damping, MEC, ani, dir):
    
    mec_folder = ''
    if MEC:
        mec_folder = 'MEC/'

    output_file1 = 'C:/Users/mathimyh/documents/boris data/simulations/boris_fordypningsoppgave/' + ani + '/cache/' + mec_folder + 'trajectory/' + str(meshdims[0]) + 'x' + str(meshdims[1]) + 'x' + str(meshdims[2]) +  '/' +  dir + '_trajectory_M1.txt'
    output_file2 = 'C:/Users/mathimyh/documents/boris data/simulations/boris_fordypningsoppgave/' + ani + '/cache/' + mec_folder + 'trajectory/' + str(meshdims[0]) + 'x' + str(meshdims[1]) + 'x' + str(meshdims[2]) +  '/' +  dir + '_trajectory_M2.txt'

    f1 = open(output_file1, 'r')

    lines1 = f1.readlines()[:-1]

    ys1 = np.zeros(len(lines1[0].strip().split('\t')))

    for line1 in lines1:
        vals = line1.strip().split('\t')
        for i, val in enumerate(vals):
            ys1[i] += float(val)

    ys1 = [y/len(ys1) for y in ys1]

    f2 = open(output_file2, 'r')

    lines2 = f2.readlines()[:-1]

    ys2 = np.zeros(len(lines2[0].strip().split('\t')))

    for line2 in lines2:
        vals = line2.strip().split('\t')
        for i, val in enumerate(vals):
            ys2[i] += float(val)

    ys2 = [y/len(ys2) for y in ys2]

    ys = []

    for i in range(len(ys1)):
        ys.append(1/2 * (ys1[i] - ys2[i]))

    xs = np.linspace(0, len(ys), len(ys))

    plt.plot(xs, ys)

    folder_name = ani + '/plots/' + mec_folder + 'trajectory/' + str(meshdims[0]) + 'x' + str(meshdims[1]) + 'x' + str(meshdims[2])
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    savename = ani + '/plots/' + mec_folder + 'trajectory/' + str(meshdims[0]) + 'x' + str(meshdims[1]) + 'x' + str(meshdims[2]) + '/damping' + str(damping) + '_' + dir + '_trajectory.png' 
    plt.savefig(savename, dpi=600)
    plt.show()


def plot_neel_T(meshdims, damping, MEC, ani):
    
    plt.figure(figsize=(10,6))

    mec_folder = ''
    if MEC:
        mec_folder = 'MEC/'

    output_file = 'C:/Users/mathimyh/documents/boris data/simulations/boris_fordypningsoppgave/' + ani + '/cache/' + mec_folder + 'neel/' + str(meshdims[0]) + 'x' + str(meshdims[1]) + 'x' + str(meshdims[2]) +  '/neel_T.txt'

    f = open(output_file, 'r')

    lines = f.readlines()[9:-1]

    xs = []
    ys = []

    for line in lines:
        vals = line.strip().split('\t')
        xs.append(float(vals[0])*1e12)
        ys.append(float(vals[1]))

    plt.plot(xs, ys)

    folder_name = ani + '/plots/' + mec_folder + 'neel/' + str(meshdims[0]) + 'x' + str(meshdims[1]) + 'x' + str(meshdims[2])
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    savename = ani + '/plots/' + mec_folder + 'neel/' + str(meshdims[0]) + 'x' + str(meshdims[1]) + 'x' + str(meshdims[2]) + '/damping' + str(damping) + '_' + 'neel_T.png' 
    plt.xlabel(r'Temperature ($K$)')
    plt.ylabel(r'$\mathbf{m}_{z}$')
    plt.tight_layout()
    plt.savefig(savename, dpi=600)
    plt.show()


def main():
    # a = 0
    # # SA_plotting('cache/tAvg_damping0.001_V0.145_mxdmdt.txt', "afm_transport/x_axis_mxdmdt_400nm.png", "Spin accumulation in AFM (mxdmdt) at 400 nm, V = -160μV")
    # # plateau_plot("cache/plateau_V-0.15_damping0.005_mxdmdt_250nm_350nm_450nm_550nm.txt", "plots/plateau/plateau_250_V-0.2_0.005_mxdmdt.png", "Spin accumulation (mxdmdt) at 250 nm with V = -0.15 μV")
    # f2 = 'OOP/cache/MEC/t_avg/5000x50x5/tAvg_damping0.0004_V0.014_mxdmdt.txt'
    # f1 = 'OOP/cache/t_avg/5000x50x5/tAvg_damping0.0004_V0.014_mxdmdt.txt'
    # # f6 = 'IP/cache/t_avg/4000x50x50/tAvg_damping0.0004_V-0.6_mxdmdt.txt'
    # # f8 = 'IP/cache/t_avg/4000x50x75/tAvg_damping0.0004_V-1_mxdmdt.txt'
    # # f9 = 'IP/cache/t_avg/4000x50x100/tAvg_damping0.0004_V-1.3_mxdmdt.txt'
    # # f7 = 'IP/cache/t_avg/4000x50x55/tAvg_damping0.0004_V-0.65_mxdmdt.txt'
    # # f5 = 'IP/cache/t_avg/4000x50x45/tAvg_damping0.0004_V-0.5_mxdmdt.txt'
    # # f4 = 'IP/cache/t_avg/4000x50x40/tAvg_damping0.0004_V-0.35_mxdmdt.txt'
    # # f3 = 'IP/cache/t_avg/4000x50x35/tAvg_damping0.0004_V-0.25_mxdmdt.txt'
    

    # l1 = 'Without MEC'
    # l2 = 'With MEC'
    # # l6 = '10 layers'
    # # l8 = '15 layers'
    # # l9 = '20 layers'
    # # l7 = '11 layers'
    # # l5 = '9 layers'
    # # l4 = '8 layers'
    # # l3 = '7 layers'

    # # # title = 'Normalized spin accumulation with/without MEC'

    # savename = 'OOP/plots/MEC/t_avg/5000x50x5/tAvg_MEC_comparison.png'

    # plot_tAvg_comparison((f1,f2), (l1,l2), savename, 'OOP')

    # plot_dispersion([4000, 50, 5], 4e-4, 1, 'OOP', 'y')

    # FOR DISPERSIONS DOWN HERE

    f1 = 'OOP/cache/dispersions/4000x50x5/dirx_dispersion.txt'
    f2 = 'OOP/cache/MEC/dispersions/4000x50x5/dirx_dispersion.txt'
    f3 = 'OOP/cache/dispersions/4000x50x5/diry_dispersion.txt'
    f4 = 'OOP/cache/MEC/dispersions/4000x50x5/diry_dispersion.txt'


    savename = 'OOP/plots/dispersions/4000x50x5/MEC_comparison_dispersion.png'

    plot_dispersions((f1,f2,f3,f4), savename)

if __name__ == '__main__':
    main()