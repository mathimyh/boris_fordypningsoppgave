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
    ns.setparam("base", "A_AFM", 76e-15) #J/m
    ns.setparam("base", "Ah", -460e3) #J/m^3
    ns.setparam("base", "Anh", (0.0, 0.0)) #J/m^3
    ns.setparam("base", "J1", 0)
    ns.setparam("base", "J2", 0)
    ns.setparam("base", "K1_AFM", (21, 21)) #J/m^3
    ns.setparam("base", "K2_AFM", 0)
    ns.setparam("base", "K3_AFM", 0)
    ns.setparam("base", "cHa", 1)
    # ns.setparam("base", "D_AFM", (0, 250e-6))
    if ani == 'OOP':
        ns.setparam("base", "ea1", (0,0,1))
        ns.setangle(0,90)
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
        ns.seteldt(1e-15) # I will do the timestep of the magnetisation
        ns.setparam('base', 'cC', (36.3e10, 17e10, 8.86e10)) # N/m^2       A. Yu. Lebedev et al (1989)
        ns.setparam('base', 'density', 5250) #kg/m^3       found this on google
        ns.setparam('base', 'MEc', (-3.44e6, 7.5e6)) #J/m^3  (Original B2 = 7.5e6)   G. Wedler et al (1999)
        ns.setparam('base', 'mdamping', 1e15) # I found 1e15 to be nice until now
        # ns.setparamvar('base', 'mdamping', 'abl_tanh', [200e-9/meshdims[0], 200e-9/meshdims[0], 0, 0, 0, 0, 1, 1e4, 200]) # Use one similar to tut 33 in Boris
        Mec_folder = 'MEC/'                 

    # Set the first relax stage, this finds the ground state
    ns.setstage('Relax')
    ns.editstagestop(0, 'time', t0 * 1e-12)

    ns.setode('sLLG', 'RK4')
    ns.setdt(1e-15)
    # ns.random()

    z_damping = 0
    if meshdims[2] == 110:
        z_damping = 80

    ns.setparamvar('damping_AFM', 'abl_tanh', [z_damping/meshdims[0], z_damping/meshdims[0], 0, 0, z_damping/meshdims[0], 0, 1, 1e2, z_damping]) 

    folder_name = ani + '/sims/' + Mec_folder + str(meshdims[0]) + 'x' + str(meshdims[1]) + 'x' + str(meshdims[2])
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


    # Test with periodic boundary conditions for dispersions:
    # ns.pbc('base', 'x', 10)

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
    
    # If it is OOP ani, then we need to change the spin current direction
    if ani == 'OOP':
        ns.setparam('base', 'STp', (1,0,0)) # x-dir spin current and y-dir electrode gives z-dir torque

    # Add the electrodes

    # Current along y-direction
    ns.addelectrode(np.array([(meshdims[0]/2 - 100), 0, (meshdims[2]-cellsize), (meshdims[0]/2 + 100), 0, meshdims[2]])* 1e-9)
    ns.addelectrode(np.array([(meshdims[0]/2 - 100), meshdims[1], (meshdims[2]-cellsize), (meshdims[0]/2 + 100), meshdims[1], meshdims[2]]) * 1e-9)
    ns.designateground('1')

    # # This is current along x-direction
    # # elif ani == 'OOP':
    # ns.addelectrode(np.array([0, 0, 0, 0, meshdims[1], meshdims[2]]) * 1e-9)
    # ns.addelectrode(np.array([meshdims[0], 0, 0, meshdims[0], meshdims[1], meshdims[2]]) * 1e-9)
    # ns.designateground('1')

    # # Electrode along z-direction
    # ns.addelectrode(np.array([meshdims[0]/2 - 100, 0, 0, meshdims[0]/2 + 100, meshdims[1], 0]) * 1e-9)
    # ns.addelectrode(np.array([meshdims[0]/2 - 100, 0, meshdims[2], meshdims[0]/2 + 100, meshdims[1], meshdims[2]]) * 1e-9)
    # ns.designateground('1')
    
    # else:
    #     print('Which anisotropy?')
    #     exit(1)
    
    # Add step function so that torque only acts on region in the injector
    width = 40
    func = '(step(x-' + str(meshdims[0]/2 - width/2) + 'e-9)-step(x-' + str(meshdims[0]/2 + width/2) + 'e-9)) * (step(z-' + str(meshdims[2]-cellsize) + 'e-9)-step(z-' + str(meshdims[2]) + 'e-9))'
    ns.setparamvar('SHA','equation', func)
    ns.setparamvar('flST','equation',func)

    # Use the built-in generator for damping at the edges. 
    # Also for the 2 deepest layers for the 150 thick
    z_damping = 0
    if meshdims[2] == 25:
        z_damping = 40

    ns.setparamvar('damping_AFM', 'abl_tanh', [300/meshdims[0], 300/meshdims[0], 0, 0, 0, 0, 1, 1e2, z_damping]) 

    # # Maybe try periodic boundary conditions for the large one, instead of damping equation?
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
def save_steadystate(meshdims, cellsize, t, V, damping, MEC, ani):
    
    mec_folder = ''
    if MEC:
        mec_folder = 'MEC/'

    loadname = 'C:/Users/mathimyh/Documents/Boris Data/Simulations/boris_fordypningsoppgave/' + ani + '/sims/' + mec_folder + str(meshdims[0]) + 'x' + str(meshdims[1]) + 'x' + str(meshdims[2]) + '/ground_state.bsm'
    print(loadname)
    ns = virtual_current(meshdims, cellsize, t, V, damping, loadname, MEC, ani)
    ns.iterupdate(200)

    folder_name = ani + '/sims/' + mec_folder + str(meshdims[0]) + 'x' + str(meshdims[1]) + 'x' + str(meshdims[2])
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    ns.Run()

    savename = 'C:/Users/mathimyh/Documents/Boris Data/simulations/boris_fordypningsoppgave/' + ani + '/sims/' + mec_folder + str(meshdims[0]) + 'x' + str(meshdims[1]) + 'x' + str(meshdims[2]) + '/V' + str(V) + '_damping' + str(damping) + '_steady_state.bsm'
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
            temp = np.array([x_val, 0, 0, x_val + 10, meshdims[1], meshdims[2]]) * 1e-9
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
    
    # Loading the sim. All the parameters and parameters variation is still there so don't need to add back
    ns.loadsim(sim_name)
    ns.reset()

    # Voltage stage
    ns.setstage('V')

    ns.editstagevalue('0', str(0.001*V))
    
    ns.editstagestop(0, 'time', t * 1e-12)

    ns.editdatasave(0, 'time', t * 1e-12 /200)

    ns.setdata('time')
    for i in range(int((x_stop - x_start)/cellsize)):
        temp = np.array([x_start + (1*i*cellsize), 0, meshdims[2], x_start + (1 + i)*cellsize, meshdims[1], meshdims[2]]) * 1e-9 # Only measure at the top
        ns.adddata(data, "base", temp)

    savename = 'C:/Users/mathimyh/documents/boris data/simulations/boris_fordypningsoppgave/' + ani + '/cache/' + mec_folder + 't_avg/' + str(meshdims[0]) + 'x' + str(meshdims[1]) + 'x' + str(meshdims[2]) + '/tAvg_damping' + str(damping) + '_V' + str(V) + '_' + savedata  + '.txt'

    ns.savedatafile(savename)

    ns.cuda(1)
    # ns.selectcudadevice([0,1])

    ns.Run()

    plotting.plot_tAvg_SA(meshdims, cellsize, t, V, damping, data, x_start, x_stop, MEC, ani)
 
# Save 2D magnetization
def time_avg_SA_2D(meshdims, cellsize, t, V, damping, data, x_start, x_stop, MEC, ani):
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


    # To not have an enormous amount of data, x-direction will only sample every 10th cellsize. 
    # z-direction will sample every nm.
    # At least for now, I will see if the resolution is fine or not. 
    for j in range(meshdims[2]):
        for i in range(int((x_stop - x_start)/cellsize*0.1)): 
            temp = np.array([x_start + i*cellsize*10, 0, j, x_start + (i+1)*cellsize*10, meshdims[1], j]) * 1e-9 # Average over y direction
            ns.adddata(data, "base", temp)


    savename = 'C:/Users/mathimyh/documents/boris data/simulations/boris_fordypningsoppgave/' + ani + '/cache/' + mec_folder + 't_avg/' + str(meshdims[0]) + 'x' + str(meshdims[1]) + 'x' + str(meshdims[2]) + '/2D_tAvg_damping' + str(damping) + '_V' + str(V) + '_' + savedata  + '.txt'

    ns.savedatafile(savename)

    ns.cuda(1)
    # ns.selectcudadevice([0,1])

    ns.Run()

    plotting.plot_tAvg_SA_2D(meshdims, cellsize, t, V, damping, data, x_start, x_stop, MEC, ani)

def time_avg_SA_2D_y(meshdims, cellsize, t, V, damping, data, x_start, x_stop, MEC, ani):
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


    # To not have an enormous amount of data, x-direction will only sample every 10th cellsize. 
    # z-direction will sample every nm.
    # At least for now, I will see if the resolution is fine or not. 
    for j in range(meshdims[1]):
        for i in range(int((x_stop - x_start)/cellsize*0.1)): 
            temp = np.array([x_start + i*cellsize*10, j, meshdims[2], x_start + (i+1)*cellsize*10, j, meshdims[2]]) * 1e-9 # Average over y direction
            ns.adddata(data, "base", temp)


    savename = 'C:/Users/mathimyh/documents/boris data/simulations/boris_fordypningsoppgave/' + ani + '/cache/' + mec_folder + 't_avg/' + str(meshdims[0]) + 'x' + str(meshdims[1]) + 'x' + str(meshdims[2]) + '/ydir_2D_tAvg_damping' + str(damping) + '_V' + str(V) + '_' + savedata  + '.txt'

    ns.savedatafile(savename)

    ns.cuda(1)
    # ns.selectcudadevice([0,1])

    ns.Run()

    plotting.plot_tAvg_SA_2D_y(meshdims, cellsize, t, V, damping, data, x_start, x_stop, MEC, ani)

def time_avg_SA_z(meshdims, cellsize, t, V, damping, data, x_start, x_stop, MEC, ani):
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
    
    for p in range(meshdims[2]):
        temp = np.array([meshdims[0]/2 - cellsize, 0, meshdims[2]-p, meshdims[0]/2 + cellsize, meshdims[1], meshdims[2]-p]) * 1e-9
        ns.adddata(data, "base", temp)


    savename = 'C:/Users/mathimyh/documents/boris data/simulations/boris_fordypningsoppgave/' + ani + '/cache/' + mec_folder + 't_avg/' + str(meshdims[0]) + 'x' + str(meshdims[1]) + 'x' + str(meshdims[2]) + '/tAvg_damping' + str(damping) + '_V' + str(V) + '_' + savedata  + '_zdir.txt'

    ns.savedatafile(savename)

    ns.cuda(1)
    # ns.selectcudadevice([0,1])

    ns.Run()

    plotting.plot_tAvg_SA_z(meshdims, cellsize, t, V, damping, data, x_start, x_stop, MEC, ani)

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