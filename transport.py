import sys
import os
sys.path.insert(0, 'C:/users/mathimyh/documents/boris data/borispythonscripts/')

from NetSocks import NSClient   # type: ignore
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import plotting

# Initializes an AFM mesh with its parameters and a relax stage. Saves the ground state after the simuation is over
def Init(meshdims, cellsize, t0, MEC, ani):

    ns = NSClient(); ns.configure(True, False)
    ns.reset()
    

    # Add the base layer
    Base = np.array([0, 0, 0, meshdims[0], meshdims[1], meshdims[2]]) * 1e-9
    ns.setafmesh("base", Base)
    ns.cellsize("base", np.array([cellsize, cellsize, cellsize]) * 1e-9)

    # Add the modules
    ns.addmodule("base", "aniuni")

    # Set temperature
    ns.temperature("0.3K")

    # Set parameters
    ns.setparam("base", "grel_AFM", (1, 1))  
    ns.setparam("base", "damping_AFM", (2e-4, 2e-4))
    ns.setparam("base", "Ms_AFM", 2.1e3)
    ns.setparam("base", "Nxy", (0, 0))
    ns.setparam("base", "A_AFM", 76e-15)
    ns.setparam("base", "Ah", -460e3)
    ns.setparam("base", "Anh", (0.0, 0.0))
    ns.setparam("base", "J1", 0)
    ns.setparam("base", "J2", 0)
    ns.setparam("base", "K1_AFM", (21, 21))
    ns.setparam("base", "K2_AFM", 0)
    ns.setparam("base", "K3_AFM", 0)
    ns.setparam("base", "cHa", 1)
    # ns.setparam("base", "D_AFM", (0, 250e-6))
    if ani == 'OOP':
        ns.setparam("base", "ea1", (0,0,1))
    elif ani == 'IP':
        ns.setparam('base', 'ea1', (1,0,0))
    else:
        print("Choose anisotropy direction")
        return

    # Add the magnetoelastic coupling if this is desired
    Mec_folder = ''
    if MEC:
        ns.addmodule('base', 'melastic') # Add the new module
        ns.surfacefix('-z') # Fix one face
        ns.seteldt(1e-14) # I will do the timestep of the magnetisation
        ns.setparam('base', 'cC', (36.3e10, 17e10, 8.86e10)) # N/m^2       A. Yu. Lebedev et al (1989)
        ns.setparam('base', 'density', 5250) #kg/m^3       found this on google
        ns.setparam('base', 'MEc', (-3.44e6, 7.5e6)) #J/m^3     G. Wedler et al (1999)
        ns.setparam('base', 'mdamping', 1e15)
        Mec_folder = 'MEC/'                 

    # Set the first relax stage, this finds the ground state
    ns.setstage('Relax')
    ns.editstagestop(0, 'time', t0 * 1e-12)

    ns.setode('sLLG', 'RK4')
    ns.setdt(1e-15)
    # ns.random()

    folder_name = ani + '/sims/' + Mec_folder + str(meshdims[0]) + 'x' + str(meshdims[1]) + 'x' + str(meshdims[2])
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    ns.cuda(1)
    ns.Run()


    savename = 'C:/Users/mathimyh/Documents/boris data/simulations/boris_fordypningsoppgave/' + ani + '/sims/' + Mec_folder + str(meshdims[0]) + 'x' + str(meshdims[1]) + 'x' + str(meshdims[2]) + '/ground_state.bsm'
    ns.savesim(savename)

# Sets up a simulation with a virtual current
def virtual_current(meshdims, cellsize, t, V, damping, sim_name, MEC, ani):

    ns = NSClient(); ns.configure(True, False)
    ns.reset()
    
    ns.loadsim(sim_name)
    ns.clearelectrodes()
    ns.reset()

    ns.setstage('V')
    ns.editstagevalue('0', str(0.001*V))
    ns.editstagestop(0, 'time', t * 1e-12)


    # Set spesific params and modules here for torque
    ns.addmodule("base", "SOTfield")
    ns.addmodule("base", "transport")
    ns.setparam("base", "SHA", '1')
    ns.setparam("base", "flST", '1')
    ns.setparam("base", "damping_AFM", (damping, damping))
    ns.delmodule("base", "Zeeman")
    
    # Add the electrodes
    if ani == 'IP':
        ns.addelectrode(np.array([(meshdims[0]/2 - 100), 0, (meshdims[2]-cellsize), (meshdims[0]/2 + 100), 0, meshdims[2]])* 1e-9)
        ns.addelectrode(np.array([(meshdims[0]/2 - 100), meshdims[1], (meshdims[2]-cellsize), (meshdims[0]/2 + 100), meshdims[1], meshdims[2]]) * 1e-9)
        ns.designateground('1')

    elif ani == 'OOP':
        ns.addelectrode(np.array([(meshdims[0]/2 - 100), 0, 0, (meshdims[0]/2+100), meshdims[1], 0]) * 1e-9)
        ns.addelectrode(np.array([(meshdims[0]/2 - 100), 0, meshdims[2], (meshdims[0]/2+100), meshdims[1], meshdims[2]]) * 1e-9)
        ns.designateground('1')
    
    else:
        print('Which anisotropy?')
        exit(1)
    
    # Add step function so that torque only acts on region in the injector
    func = 'step(x-' + str(meshdims[0]/2 - 20) + 'e-9)-step(x-' + str(meshdims[0]/2 + 20) + 'e-9)'
    ns.setparamvar('SHA','equation', func)
    ns.setparamvar('flST','equation',func)

    # Add damping function so it increases at the edges
    # ns.setparamvar('damping_AFM', 'equation', '1 + 10000 * (exp(-(x)^2 / 1000e-18) + exp(-(x-1200e-9)^2 / 1000e-18))')
    damp_func = 'step(-x+200e-9)*(100*tanh((-x) / 50e-9) + 101) + step(x-' + str(meshdims[0]-200) + 'e-9)*(100*tanh((x-' + str(meshdims[0]) + 'e-9)/ 50e-9) + 101) + step(x-200e-9) - step(x-' + str(meshdims[0]-200) + 'e-9)'
    ns.setparamvar('damping_AFM', 'equation', damp_func)

    # Maybe try periodic boundary conditions for the large one, instead of damping equation?
    # ns.pbc('base', 'x')

    ns.cuda(1)
    # ns.selectcudadevice([1,0])

    return ns

# Runs the simulation and saves the spin accumulation. NOT time averaged
def runSimulation(t, V, data, x_start, x_stop):

    ns = virtual_current(t, V)
    ns.editdatasave(0, 'time', t * 1e-12)

    ns.cuda(1)

    first = np.array([x_start, 0, 0, x_start+1, 20, 8]) * 1e-9
    ns.setdata(data, "base", first)
    for i in range(x_stop - x_start -1):
        temp = np.array([x_start + 1 + 1*i, 0, 0, x_start + 2 + 1*i, 20, 8]) * 1e-9
        ns.adddata(data, "base", temp)

    # Saving 
    if data == '<mxdmdt>':
            savedata = 'mxdmdt'
    elif data == '<mxdmdt2>':
        savedata = 'mxdmdt2'

    savename = 'C:/Users/mathimyh/Documents/Boris Data/Simulations/boris_fordypningsoppgave/cache/V' + str(V) + '_' + savedata + '_' + str(x_start) + '_' + str(x_stop) + '.txt'
    ns.savedatafile(savename)

    ns.Run()

# A function that runs the virtual current from a simulation for a given time and saves the simulation after
def save_steadystate(meshdims, cellsize, t, V, damping, MEC):
    
    mec_folder = ''
    if MEC:
        mec_folder = 'MEC/'

    loadname = 'C:/Users/mathimyh/Documents/Boris Data/Simulations/boris_fordypningsoppgave/sims/' + mec_folder + str(meshdims[0]) + 'x' + str(meshdims[1]) + 'x' + str(meshdims[2]) + '/ground_state.bsm'
    print(loadname)
    ns = virtual_current(meshdims, cellsize, t, V, damping, loadname, MEC, ani)
    ns.iterupdate(200)

    folder_name = 'sims/' + mec_folder + str(meshdims[0]) + 'x' + str(meshdims[1]) + 'x' + str(meshdims[2])
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    ns.Run()

    # Remove module because of bug
    if MEC:
        ns.delmodule('base', 'melastic')

    savename = 'C:/Users/mathimyh/Documents/Boris Data/simulations/boris_fordypningsoppgave/sims/' + mec_folder + str(meshdims[0]) + 'x' + str(meshdims[1]) + 'x' + str(meshdims[2]) + '/V' + str(V) + '_damping' + str(damping) + '_steady_state.bsm'
    ns.savesim(savename)

# Function for finding the plateau. Saves data from one point along the x-axis.
def find_plateau(meshdims, cellsize, t, V, data, damping, x_vals, MEC, ani):

    mec_folder = ''
    if MEC:
        mec_folder = 'MEC/'

    loadname = 'C:/Users/mathimyh/Documents/Boris Data/Simulations/boris_fordypningsoppgave/' + ani + '/sims/' + mec_folder + str(meshdims[0]) + 'x' + str(meshdims[1]) + 'x' + str(meshdims[2]) + '/ground_state.bsm'
    
    ns = virtual_current(meshdims, cellsize, t, V, damping, loadname, MEC, ani)
    ns.iterupdate(200)

    folder_name = ani + '/cache/' + mec_folder + 'plateau/' + str(meshdims[0]) + 'x' + str(meshdims[1]) + 'x' + str(meshdims[2])
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


    # Save a profile to find the plateau
    if x_vals != False:
        
        ns.editdatasave(0, 'time', 5e-12)
        ns.setdata('time')

        for x_val in x_vals:
            temp = np.array([x_val, 0, 0, x_val + 10, 50, 5]) * 1e-9
            ns.adddata(data, "base", temp)
    
        x_vals_string = 'nm_'.join(str(x_val) for x_val in x_vals)
        
        savename = 'C:/Users/mathimyh/Documents/Boris Data/Simulations/boris_fordypningsoppgave/' + ani + '/cache/' +  mec_folder + 'plateau/' + str(meshdims[0]) + 'x' + str(meshdims[1]) + 'x' + str(meshdims[2]) + '/plateau_V' + str(V) + '_damping' + str(damping) + '_' + data[1:-1] + '_' + x_vals_string + 'nm.txt'
        
        ns.savedatafile(savename)

    ns.Run()

    if x_vals != False:
        plotting.plot_plateau(meshdims, cellsize, t, V, data, damping, x_vals, MEC, ani)

# Load a simulation in steady state, run the simulation and save the SA along with the time
def time_avg_SA(meshdims, cellsize, t, V, damping, data, x_start, x_stop, MEC, ani):

    savedata = data[1:-1]
    mec_folder = ''
    if MEC:
        mec_folder = 'MEC/'

    folder_name = ani + '/cache/' + mec_folder + 't_avg/' + str(meshdims[0]) + 'x' + str(meshdims[1]) + 'x' + str(meshdims[2])
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


    sim_name = 'C:/Users/mathimyh/documents/boris data/simulations/boris_fordypningsoppgave/' + ani + '/sims/' + mec_folder + str(meshdims[0]) + 'x' + str(meshdims[1]) + 'x' + str(meshdims[2]) + '/V' + str(V) + '_damping' + str(damping) + '_steady_state.bsm'
    
    ns = NSClient(); ns.configure(True, False)
    ns.reset()
    
    ns.loadsim(sim_name)
    ns.reset()

    # Voltage stage
    ns.setstage('V')

    ns.editstagevalue('0', str(0.001*V))
    
    ns.editstagestop(0, 'time', t * 1e-12)

    ns.editdatasave(0, 'time', t * 1e-12 /200)

    ns.setdata('time')
    for i in range(x_stop - x_start):
        temp = np.array([x_start + 1*i, 0, 0, x_start + 1 + 1*i, meshdims[1], meshdims[2]]) * 1e-9
        ns.adddata(data, "base", temp)


    savename = 'C:/Users/mathimyh/documents/boris data/simulations/boris_fordypningsoppgave/' + ani + '/cache/' + mec_folder + 't_avg/' + str(meshdims[0]) + 'x' + str(meshdims[1]) + 'x' + str(meshdims[2]) + '/tAvg_damping' + str(damping) + '_V' + str(V) + '_' + savedata  + '.txt'

    ns.savedatafile(savename)

    ns.cuda(1)
    # ns.selectcudadevice([0,1])

    ns.Run()

    plotting.plot_tAvg_SA(meshdims, cellsize, t, V, damping, data, x_start, x_stop, MEC, ani)
    
# Get a profile of the magnetization
def profile_from_sim(t, V, damping, sim_name, x_start, x_stop):

    ns = NSClient(); ns.configure(True)
    ns.reset()
    
    ns.loadsim(sim_name)
    ns.reset()

    # Voltage stage
    ns.setstage('V')

    ns.editstagevalue('0', str(0.001*V))
    
    ns.editstagestop(0, 'time', t * 1e-12)

    ns.setdata("commbuf")
    ns.adddata("time")

    start = str(x_start) + 'e-9, 10e-9, 0'
    end = str(x_stop) + 'e-9, 10e-9, 0'

    savedt = 1e-12

    for i in range(0, 6):
        ns.editdatasave(i, "time", savedt)

    ns.dp_getexactprofile(start = start, end = end, step = '4e-9', dp_index = '0', bufferCommand = True)
    ns.dp_save("C:/Users/mathimyh/Documents/Boris data/Simulations/boris_fordypningsoppgave/cache/profile_test.txt", dp_indexes = 1, bufferCommand = True)

    ns.cuda(1)

    ns.Run()