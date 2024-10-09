import sys
sys.path.insert(0, 'C:/users/mathimyh/documents/boris data/borispythonscripts/')

from NetSocks import NSClient   , customize_plots # type: ignore
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt



font = {'size' : 20}
mpl.rc('font', **font)


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

    # Add two meshes with really high Gilbert damping to avoid reflections
    Damper1 = np.array([-50, 0, 0, 0, Ly, Lz]) * 1e-9
    ns.addafmesh("damper1", Damper1)
    Damper2 = np.array([Lx, 0, 0, Lx + 50, Ly, Lz]) * 1e-9
    ns.addafmesh("damper2", Damper2)
    ns.cellsize("damper1", np.array([4, 4, 4]) * 1e-9)
    ns.cellsize("damper2", np.array([4, 4, 4]) * 1e-9)

    # Add the modules
    ns.addmodule("base", "aniuni")
    ns.addmodule("damper1", "aniuni")
    ns.addmodule("damper2", "aniuni")


    # Set temperature
    ns.temperature("0.3K")
    ns.temperature("damper1", "0.3K")
    ns.temperature("damper2", "0.3K")

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

    # Set gilbert damping really high for the edges
    ns.setparam("damper1", "grel_AFM", (1, 1))
    ns.setparam("damper1", "damping_AFM", (1e10, 1e10))
    ns.setparam("damper1", "Ms_AFM", 2.1e3)
    ns.setparam("damper1", "Nxy", (0, 0))
    ns.setparam("damper1", "A_AFM", 1e-12)
    ns.setparam("damper1", "Ah", -200e3)
    ns.setparam("damper1", "Anh", (0.0, 0.0))
    ns.setparam("damper1", "J1", 0)
    ns.setparam("damper1", "J2", 0)
    ns.setparam("damper1", "K1_AFM", (10e3, 10e3))
    ns.setparam("damper1", "K2_AFM", 0)
    ns.setparam("damper1", "K3_AFM", 0)
    ns.setparam("damper1", "cHa", 1)
    ns.setparam("damper1", "D_AFM", (0, 250e-6))
    ns.setparam("damper1", "ea1", (0,0,1))
    
    ns.setparam("damper2", "grel_AFM", (1, 1))
    ns.setparam("damper2", "damping_AFM", (1000, 1000))
    ns.setparam("damper2", "Ms_AFM", 2.1e3)
    ns.setparam("damper2", "Nxy", (0, 0))
    ns.setparam("damper2", "A_AFM", 1e-12)
    ns.setparam("damper2", "Ah", -200e3)
    ns.setparam("damper2", "Anh", (0.0, 0.0))
    ns.setparam("damper2", "J1", 0)
    ns.setparam("damper2", "J2", 0)
    ns.setparam("damper2", "K1_AFM", (10e3, 10e3))
    ns.setparam("damper2", "K2_AFM", 0)
    ns.setparam("damper2", "K3_AFM", 0)
    ns.setparam("damper2", "cHa", 1)
    ns.setparam("damper2", "D_AFM", (0, 250e-6))
    ns.setparam("damper2", "ea1", (0,0,1))
    

    
    ns.setstage('Relax')
    ns.editstagestop(0, 'time', t0 * 1e-12)

    ns.setode('sLLG', 'RK4')
    ns.setdt(1e-15)
    ns.random()
    ns.random("damper1")
    ns.random("damper2")

    ns.Run()

    ns.savesim('ground_state.bsm')


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
    ns.addmodule("damper1", "SOTfield")
    ns.addmodule("damper1", "transport")
    ns.addmodule("damper2", "SOTfield")
    ns.addmodule("damper2", "transport")
    ns.setparam("base", "SHA", '1')
    ns.setparam("base", "flST", '1')
    
    # Add the electrodes
    ns.addelectrode('225e-9,0,0,275e-9,0,8e-9')
    ns.addelectrode('225e-9,20e-9,0,275e-9,20e-9,8e-9')
    ns.designateground('1')

    return ns


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
        ns.savedatafile("C:/Users/mathimyh/Documents/Boris Data/Simulations/testing/cache/neg_V_%data%.txt")
    else:
        ns.savedatafile("C:/Users/mathimyh/Documents/Boris Data/Simulations/testing/cache/%data%.txt")

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

    ns.Run()

    # After running this it takes around 200ps for the magnetization to stabilize...
    # Save the sim so its easy to start simulations from this

    if negative:
        ns.savesim("neg_V_steady_state.bsm")
    else:
        ns.savesim("steady_state.bsm")

def main():
    
    t0 = 15
    t = 500
    # 0.085 gives a good signal without flipping the magnetization
    V = 0.085
    data = "<mxdmdt>"

    # runSimulation(t, V, data, negative=True)
    find_plateau(t, V, data, negative=True, x_val=False)
    # Init(t0)


if __name__ == '__main__':
    main()