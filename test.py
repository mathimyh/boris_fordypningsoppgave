import sys
import os
sys.path.insert(0, 'C:/users/mathimyh/documents/boris data/borispythonscripts/')

from NetSocks import NSClient   # type: ignore
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ns = NSClient(); ns.configure(True, False)
ns.reset()
ns.clearelectrodes()

meshdims = (1000, 50, 100)
cellsize = 5

t0 = 100 # Time to find steady state. 100 ps is long enough
t = 100 # How long the time average lasts
V = -5 # -5 is a nice value for the system here

Base = np.array([0, 0, 0, meshdims[0], meshdims[1], meshdims[2]]) * 1e-9
ns.setafmesh("base", Base)
ns.cellsize("base", np.array([cellsize, cellsize, cellsize]) * 1e-9)

ns.temperature("0.3K")
ns.addmodule("base", "aniuni")
ns.setode('sLLG', 'RK4')
ns.setdt(1e-15)

# Set parameters
ns.setparam("base", "grel_AFM", (1, 1)) # 1
ns.setparam("base", "damping_AFM", (2e-3, 2e-3)) # 1
ns.setparam("base", "Ms_AFM", 2.1e3) # A/m
ns.setparam("base", "Nxy", (0, 0))
ns.setparam("base", "A_AFM", 76e-15) #J/m
ns.setparam("base", "Ah", -460e3) #J/m^3
ns.setparam("base", "Anh", (0.0, 0.0)) #J/m^3
ns.setparam("base", "J1", 0)
ns.setparam("base", "J2", 0)
ns.setparam("base", "K1_AFM", (21, 21)) #J/m^3
ns.setparam("base", "K2_AFM", 0)
ns.setparam("base", "K3_AFM", 0)
ns.setparam("base", "cHa", 1) # 1
ns.setparam('base', 'ea1', (1,0,0))

# Set spesific params and modules here for torque
ns.addmodule("base", "SOTfield")
ns.addmodule("base", "transport")
ns.setparam("base", "SHA", '1')
ns.setparam("base", "flST", '1')
ns.delmodule("base", "Zeeman")

# Current along y-direction
ns.addelectrode(np.array([(meshdims[0]/2 - 100), 0, (meshdims[2]-cellsize), (meshdims[0]/2 + 100), 0, meshdims[2]])* 1e-9)
ns.addelectrode(np.array([(meshdims[0]/2 - 100), meshdims[1], (meshdims[2]-cellsize), (meshdims[0]/2 + 100), meshdims[1], meshdims[2]]) * 1e-9)
ns.designateground('1')

# Add step function so that torque only acts on region in the injector
width = 40
func = '(step(x-' + str(meshdims[0]/2 - width/2) + 'e-9)-step(x-' + str(meshdims[0]/2 + width/2) + 'e-9)) * (step(z-' + str(meshdims[2]-cellsize) + 'e-9)-step(z-' + str(meshdims[2]) + 'e-9))'
ns.setparamvar('SHA','equation', func)
ns.setparamvar('flST','equation',func)

# Damping at edges to prevent reflections. Also add damping at the bottom. 
ns.setparamvar('damping_AFM', 'abl_tanh', [100/meshdims[0], 100/meshdims[0], 0, 0, 20/meshdims[2], 0, 1, 1e3, 200]) 

# Find steady state stage first
ns.setstage('V')
ns.editstagevalue('0', str(0.001*V))
ns.editstagestop(0, 'time', t0 * 1e-12)

# Find time average stage
ns.addstage('V')
ns.editstagevalue('1', str(0.001*V))
ns.editstagestop(1, 'time', t * 1e-12)
ns.editdatasave(1, 'time', t * 1e-12 /200)

ns.setdata('time')
# Save mxdmdt z-direction directly beneath the injector. Average over y-direction
for i in range(int(meshdims[2]/cellsize-1)):
    rect = np.array([meshdims[0]/2-cellsize, 0, i*cellsize, meshdims[0]/2+cellsize, meshdims[1], (i+1)*cellsize]) * 1e-9
    ns.adddata('<mxdmdt>', "base", rect)

savename = 'mxdmdt_z_direction.txt'
ns.savedatafile(savename)

ns.cuda(1)
ns.Run()

f = open(savename, 'r')

lines = f.readlines()
lines = lines[10:]
vals = []

for line in lines: 
    direction = 1 # This is the component we look at (x in this case)
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

xs = np.linspace(0, meshdims[2], len(ys))

plt.plot(xs, ys)
plt.xlabel(r"z (m)")
plt.ylabel(r'$\mu_x$')
plt.tight_layout()
plt.savefig('mxdmdt_z_dir.png', dpi=600)
plt.show()