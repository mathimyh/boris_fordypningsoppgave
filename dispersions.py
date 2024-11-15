import sys
import os
sys.path.insert(0, 'C:/users/mathimyh/documents/boris data/borispythonscripts/')

from NetSocks import NSClient   # type: ignore
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import plotting

def dispersion_relation(meshdims, t, damping, x_start, x_stop, MEC, ani, dir):

    dir1 = 0

    if dir == 'x':
        dir1 = 1
    elif dir == 'y':
        dir1 = 2
    elif dir == 'z':
        dir1 = 3
    else:
        print('Choose direction')
        exit()

    mec_folder = ''
    if MEC:
        mec_folder = 'MEC/'

    folder_name = ani + '/cache/' + mec_folder + 'dispersions/' + str(meshdims[0]) + 'x' + str(meshdims[1]) + 'x' + str(meshdims[2])
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    time_step = 0.1e-12
    total_time = (2 * t)*1e-12

    Ms = 2.1e3

    # sim_name = 'C:/Users/mathimyh/documents/boris data/simulations/boris_fordypningsoppgave/sims/V' + str(V) + '_damping' + str(damping) + '_steady_state.bsm'
    sim_name = 'C:/Users/mathimyh/Documents/Boris Data/Simulations/boris_fordypningsoppgave/' + ani + '/sims/' + mec_folder + str(meshdims[0]) + 'x' + str(meshdims[1]) + 'x' + str(meshdims[2]) + '/ground_state.bsm'

    ns = NSClient(); ns.configure(True, False)
    
    ns.loadsim(sim_name)
    ns.reset()

    time = 0.0
    ns.cuda(1)

    output_file = 'C:/Users/mathimyh/documents/boris data/simulations/boris_fordypningsoppgave/' + ani + '/cache/' + mec_folder + 'dispersions/' + str(meshdims[0]) + 'x' + str(meshdims[1]) + 'x' + str(meshdims[2]) +  'dispersion.txt'
    ns.dp_newfile(output_file)

    # Add this here to find the ground state. MEC is still bugged in Boris so cant save simulations beforehand
    if MEC:
        ns.editstagestop(0, 'time', 1000) # Very long so I can manually stop when it is found
        ns.Run()
        ns.setstage('relax')

    while time < total_time:
        # ns.setstage('V')
        # ns.editstagevalue('0', str(0.001*V))
        ns.editstagestop(0, 'time', time + time_step)
        ns.Run()
        ns.dp_getexactprofile([x_start * 1e-9 + 5e-9/2, 50e-9/2 + 5e-9/2, 0], [x_stop * 1e-9 - 5e-9/2, 50e-9/2 + 5e-9/2, 0], 5e-9, 0)
        ns.dp_div(dir1, Ms)
        ns.dp_saveappendasrow(output_file, dir1)
        time += time_step

    plotting.plot_dispersion(meshdims, damping, MEC, ani, dir)
