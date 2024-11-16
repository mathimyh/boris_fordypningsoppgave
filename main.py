import plotting
import transport
import dispersions

def main():
    
    # Parameters 
    Lx = 4000
    Ly = 50
    Lz = 100
    cellsize = 5
    meshdims = (Lx, Ly, Lz)

    t = 1000
    V = -1.3
    data = '<mxdmdt>'
    damping = 4e-4
    MEC = 0
    ani = 'IP'
    x_vals = [2020, 2300, 2600, 3000, 3500, 4000]

    transport.save_steadystate(meshdims, cellsize, 200, V, damping, MEC, dir)
    # transport.Init(meshdims, cellsize, 5000, MEC, ani)
    # transport.find_plateau(meshdims, cellsize, t, V, data, damping, False, MEC, ani)
    transport.time_avg_SA(meshdims, cellsize, t, V, damping, data, 2020, 4000, MEC, ani)
    # dispersions.dispersion_relation(meshdims, t, damping, 2020, 4000, MEC, ani, 'y')

if __name__ == '__main__':
    main()