import sys
sys.path.insert(0, 'C:/users/mathimyh/documents/boris data/borispythonscripts/')

from NetSocks import NSClient   # type: ignore
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from plotting import tAvg_SA_plotting


font = {'size' : 20}
mpl.rc('font', **font)

# Initializes an AFM mesh with its parameters and a relax stage. Saves the ground state after the simuation is over
def Init(t0):

    ns = NSClient(); ns.configure(True, False)
    ns.reset()
    
    # Initialize the mesh
    Lx = 800
    Ly = 20
    Lz = 8
    
    # Add the base layer
    Base = np.array([0, 0, 0, Lx, Ly, Lz]) * 1e-9
    ns.setafmesh("base", Base)
    ns.cellsize("base", np.array([4, 4, 4]) * 1e-9)

    # Add the modules
    ns.addmodule("base", "aniuni")

    # Set temperature
    ns.temperature("0.3K")

    # Set parameters
    ns.setparam("base", "grel_AFM", (1, 1))
    ns.setparam("base", "damping_AFM", (0.002, 0.002))
    ns.setparam("base", "Ms_AFM", 2.1e3)
    ns.setparam("base", "Nxy", (0, 0))
    ns.setparam("base", "A_AFM", 1e-12)
    ns.setparam("base", "Ah", -200e3)
    ns.setparam("base", "Anh", (0.0, 0.0))
    ns.setparam("base", "J1", 0)
    ns.setparam("base", "J2", 0)
    ns.setparam("base", "K1_AFM", (10e3, 10e3))
    ns.setparam("base", "K2_AFM", 0)
    ns.setparam("base", "K3_AFM", 0)
    ns.setparam("base", "cHa", 1)
    ns.setparam("base", "D_AFM", (0, 250e-6))
    ns.setparam("base", "ea1", (1,0,0))

    # Set the first relax stage, this finds the ground state
    ns.setstage('Relax')
    ns.editstagestop(0, 'time', t0 * 1e-12)

    ns.setode('sLLG', 'RK4')
    ns.setdt(1e-15)
    ns.random()

    ns.Run()

    ns.savesim('C:/Users/mathimyh/Documents/boris data/simulations/boris_fordypningsoppgave/sims/ground_state.bsm')

# Sets up a simulation with a virtual current
def virtual_current(t, V, damping, sim_name):

    ns = NSClient(); ns.configure(True, False)
    ns.reset()
    
    ns.loadsim(sim_name)
    ns.reset()

    # Voltage stage
    ns.setstage('V')

    ns.editstagevalue('0', str(0.001*V))
    
    ns.editstagestop(0, 'time', t * 1e-12)


    # Set spesific params and modules here for torque
    ns.addmodule("base", "SOTfield")
    ns.addmodule("base", "transport")
    ns.setparam("base", "SHA", '1')
    ns.setparam("base", "flST", '1')
    ns.setparam("base", "damping_AFM", (damping, damping))
    
    # Add the electrodes
    ns.addelectrode('0,0,0,600e-9,0,8e-9')
    ns.addelectrode('0,20e-9,0,600e-9,20e-9,8e-9')
    ns.designateground('1')
    
    # Add step function so that torque only acts on region in the injector
    ns.setparamvar('SHA','equation','step(x-150e-9)-step(x-170e-9)')
    ns.setparamvar('flST','equation','step(x-150e-9)-step(x-170e-9)')

    # Add damping function so it increases at the edges
    ns.setparamvar('damping_AFM', 'equation', '1 + 1000 * (exp(-(x)^2 / 500e-18) + exp(-(x-800e-9)^2 / 500e-18))')

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
    elif data == 'mxdmdt2':
        savedata = 'mxdmdt2'

    savename = 'C:/Users/mathimyh/Documents/Boris Data/Simulations/boris_fordypningsoppgave/cache/V' + str(V) + '_' + savedata + '_' + str(x_start) + '_' + str(x_stop) + '.txt'
    ns.savedatafile(savename)

    ns.Run()

# A function that runs the virtual current from a simulation for a given time and saves the simulation after
def run_and_save(t, V, damping, loadname, savename):
    
    ns = virtual_current(t, V, damping, loadname)

    ns.cuda(1)
    ns.iterupdate(2000)

    ns.Run()

    ns.savesim(savename)


# Function for finding the plateau. Saves data from one point along the x-axis.
def find_plateau(t, V, data, damping, x_vals=False):

    ns = virtual_current(t, V, damping, 'C:/Users/mathimyh/Documents/Boris Data/Simulations/boris_fordypningsoppgave/sims/ground_state.bsm')

    ns.cuda(1)
    ns.iterupdate(2000)

    # Save a profile to find the plateau
    if x_vals != False:

        if data == '<mxdmdt>':
            savedata = 'mxdmdt'
        elif data == 'mxdmdt2':
            savedata = 'mxdmdt2'
        
        ns.editdatasave(0, 'time', 5e-12)
        ns.setdata('time')

        for x_val in x_vals:
            temp = np.array([x_val, 0, 0, x_val + 10, 20, 8]) * 1e-9
            ns.adddata(data, "base", temp)
    
        x_vals_string = 'nm_'.join(str(x_val) for x_val in x_vals)
        
        savename = 'C:/Users/mathimyh/Documents/Boris Data/Simulations/boris_fordypningsoppgave/cache/plateau_V' + str(V) + '_damping' + str(damping) + '_' + savedata + '_' + x_vals_string + 'nm.txt'
        
        ns.savedatafile(savename)

    ns.Run()

# Load a simulation in steady state, run the simulation and save the SA along with the time
def time_avg_SA(t, V, damping, data, x_start, x_stop):

    if data == '<mxdmdt>':
        savedata = 'mxdmdt'
    elif data == '<mxdmdt2>':
        savedata = 'mxdmdt2'

    sim_name = 'C:/Users/mathimyh/documents/boris data/simulations/boris_fordypningsoppgave/sims/V' + str(V) + '_damping' + str(damping) + '_steady_state.bsm'
    
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
        temp = np.array([x_start + 1*i, 0, 0, x_start + 1 + 1*i, 20, 8]) * 1e-9
        ns.adddata(data, "base", temp)

    savename = 'C:/Users/mathimyh/documents/boris data/simulations/boris_fordypningsoppgave/cache/tAvg_damping' + str(damping) + '_V' + str(V) + '_' + savedata  + '.txt'

    ns.savedatafile(savename)

    ns.cuda(1)

    ns.Run()

def main():
    
    # The parameters one often changes 
    t0 = 20
    t = 400
    V = -0.15
    data = '<mxdmdt>'
    damping = 0.001

    # runSimulation(t, V, data, negative=True)
    # find_plateau(t, V, data, damping, x_vals=[250,350,450,550])
    # Init(t0)
    savename = 'C:/Users/mathimyh/Documents/Boris Data/Simulations/boris_fordypningsoppgave/sims/V'+ str(V) + '_damping' + str(damping) + '_steady_state.bsm'
    run_and_save(t, V, damping, loadname="C:/Users/mathimyh/Documents/Boris Data/Simulations/boris_fordypningsoppgave/sims/ground_state.bsm", savename=savename)
    # time_avg_SA(t, V, damping, data, 170, 500)
    # tAvg_SA_plotting(t, V, damping, data, 170, 500)



if __name__ == '__main__':
    main()