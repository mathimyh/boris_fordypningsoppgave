import sys
sys.path.insert(0, 'C:/users/mathimyh/documents/boris data/borispythonscripts/')

from NetSocks import NSClient  
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt



font = {'size' : 20}
mpl.rc('font', **font)

# Initializes an AFM mesh with its parameters and a relax stage. Saves the ground state after the simuation is over
def Init(t0):
    
    ns = NSClient(); ns.configure(True, False)
    ns.reset()
    
    # Initialize the 3 meshes. Idk if it does anything nice to do it using classes
    Lx = 1000
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
    
    

    
    ns.setstage('Relax')
    ns.editstagestop(0, 'time', t0 * 1e-12)

    ns.setode('sLLG', 'RK4')
    ns.setdt(1e-15)
    ns.random()

    ns.Run()

    ns.savesim('ground_state.bsm')

# Sets up a simulation with a virtual current
def virtual_current(t, V, negative):

    ns = NSClient(); ns.configure(True, False)
    ns.reset()
    
    ns.loadsim('ground_state.bsm')
    ns.reset()

    # Voltage stage
    ns.setstage('V')
    neg = 1
    if negative:
        neg = -1

    ns.editstagevalue('0', str(neg*0.001*V))
    
    ns.editstagestop(0, 'time', t * 1e-12)


    # Set spesific params and modules here for torque
    ns.addmodule("base", "SOTfield")
    ns.addmodule("base", "transport")
    ns.setparam("base", "SHA", '1')
    ns.setparam("base", "flST", '1')
    
    # Add the electrodes
    ns.addelectrode('0,0,0,500e-9,0,8e-9')
    ns.addelectrode('0,20e-9,0,500e-9,20e-9,8e-9')
    ns.designateground('1')
    
    # Add step function so that torque only acts on region in the injector
    ns.setparamvar('SHA','equation','step(x-225e-9)-step(x-275e-9)')
    ns.setparamvar('flST','equation','step(x-225e-9)-step(x-275e-9)')

    # Add damping function so it increases at the edges
    ns.setparamvar('damping_AFM', 'equation', 'tanh(-x-100e-9) + tanh(x-1000e-9) + 3')

    return ns

# Runs the simulation and saves the spin accumulation. NOT time averaged
def runSimulation(t, V, data, negative):

    ns = virtual_current(t, V, negative)
    ns.editdatasave(0, 'time', t * 1e-12)

    ns.cuda(1)

    first = np.array([275, 0, 0, 276, 20, 8]) * 1e-9
    ns.setdata(data, "base", first)
    for i in range(500):
        temp = np.array([275 + 1*i, 0, 0, 276 + 1*i, 20, 8]) * 1e-9
        ns.adddata(data, "base", temp)


    # Saving 
    if negative:
        if data == '<mxdmdt>':
            ns.savedatafile('C:/Users/mathimyh/Documents/Boris Data/Simulations/testing/cache/neg_V_<mxdmdt>.txt')
        elif data == '<mxdmdt2>':
            ns.savedatafile('C:/Users/mathimyh/Documents/Boris Data/Simulations/testing/cache/neg_V_<mxdmdt2>.txt')
    else:
        if data == '<mxdmdt>':
            ns.savedatafile('C:/Users/mathimyh/Documents/Boris Data/Simulations/testing/cache/<mxdmdt>.txt')
        elif data == '<mxdmdt2>':
            ns.savedatafile('C:/Users/mathimyh/Documents/Boris Data/Simulations/testing/cache/<mxdmdt2>.txt')
    

    ns.Run()

def find_plateau(t, V, data, negative, x_val):

    ns = virtual_current(t, V, negative)


    # Only save if I want to
    if x_val != False:

        ns.editdatasave(0, 'time', 5e-12)
        
        temp = np.array([x_val, 0, 0, x_val + 10, 20, 8]) * 1e-9
        ns.savedatafile("C:/Users/mathimyh/Documents/Boris Data/Simulations/testing/cache/plateau_%data%_%x_val%.txt")

        ns.setdata('time')

    ns.adddata(data, "base", temp)

    ns.editstagestop(0, 'time', 200e-12)

    ns.editstagestop(0, 'time', 200e-12)

    ns.Run()

    # After running this it takes around 200ps for the magnetization to stabilize...
    # Save the sim so its easy to start simulations from this

    if negative:
        ns.savesim("neg_V_steady_state.bsm")
    else:
        ns.savesim("steady_state.bsm")

def main():
    
    t0 = 20
    t = 25
    
    # 0.085 gives a good signal without flipping the magnetization
    V = 0.17
    data = '<mxdmdt>'

    runSimulation(t, V, data, negative=True)
    # find_plateau(t, V, data, negative=True, x_val=False)
    # Init(t0)


if __name__ == '__main__':
    main()