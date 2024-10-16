import sys
sys.path.insert(0, 'C:/users/mathimyh/documents/boris data/borispythonscripts/')

from NetSocks import NSClient   # type: ignore
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

font = {'size' : 20}
mpl.rc('font', **font)

ns = NSClient()
ns.configure(True)

mesh_step = 5e-9
meshdim =  [1e-6, mesh_step, mesh_step]
cellsize = [mesh_step,mesh_step,mesh_step]

time_step = 0.1e-12
t0 = 10e-12
equiltime = 10e-12
total_time = (2 * t0 ) + equiltime

H0 = 0
He = 500e3

A = 76e-15
Ahom = -460e3
K1 = 21
Ms = 2.1e3

ns.setafmesh('hematite', meshdim)
ns.cellsize(cellsize)
ns.pbc('hematite', 'x', 5)

ns.setangle(90, 0)

ns.setparam('hematite', 'damping_AFM', [0.001, 0.001])
ns.setparam('hematite', 'Ms_AFM', [Ms,Ms])
ns.setparam('hematite', 'A_AFM', [A,A])
ns.setparam('hematite','Ah',[Ahom,Ahom])
ns.setparam('hematite','Anh',[0,0])
ns.setparam('hematite','K1_AFM',[K1,K1])

output_file = 'C:/Users/mathimyh/documents/boris data/simulations/boris_fordypningsoppgave/tests/dispersion.txt'

ns.setode('sLLG', 'RK4')
ns.setdt(1e-15) # time step of the solver, should be <= 1fs
ns.temperature(1)
ns.delmodule('hematite','Zeeman') 

time = 0.0
ns.cuda(1)

ns.editstagestop(0, 'time', time + equiltime)
ns.Run()

while time < total_time:

    ns.editstagestop(0, 'time', time + equiltime + time_step)
    ns.Run()
    ns.dp_getexactprofile([cellsize[0]/2, meshdim[1]/2 + cellsize[1]/2, 0], [meshdim[0] - cellsize[0]/2, meshdim[1]/2 + cellsize[1]/2, 0], mesh_step, 0)
    ns.dp_div(2, Ms)
    ns.dp_saveappendasrow(output_file, 2)
    time += time_step

pos_time = np.loadtxt(output_file)

fourier_data = np.fft.fftshift(np.abs(np.fft.fft2(pos_time)))

freq_len = len(fourier_data)
k_len = len(fourier_data[0])
freq = np.fft.fftfreq(freq_len, time_step)
kvector = np.fft.fftfreq(k_len, mesh_step)

k_max = 2*np.pi*kvector[int(0.5 * len(kvector))]*mesh_step
f_min = np.abs(freq[0])
f_max = np.abs(freq[int(0.5 * len(freq))])/1e12 # to make it THz
f_points = int(0.5 * freq_len)

result = [fourier_data[i] for i in range(int(0.5 *freq_len),freq_len)]

fig1,ax1 = plt.subplots()

ax1.imshow(result, origin='lower', interpolation='bilinear', extent = [-k_max, k_max,f_min, f_max], aspect ="auto")

ax1.set_xlabel('qa')
ax1.set_ylabel('f (THz)')

plt.tight_layout()

plt.savefig('dispersion_test.pdf', dpi=600)

plt.show()