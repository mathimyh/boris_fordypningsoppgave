import plotting
import transport
import dispersions

def main():
    
    # Dimensions
    Lx = 4000
    Ly = 50
    Lz = 5
    cellsize = 5
    meshdims = (Lx, Ly, Lz)

    # Parameters
    t = 10
    V = -0.03
    data = '<mxdmdt>'
    damping = 4e-4
    MEC = 1
    ani = 'OOP'
    x_vals = [2020, 2300, 2600, 3000, 3500, 4000]

    transport.Init(meshdims, cellsize, 1000, MEC, ani)
    # transport.save_steadystate(meshdims, cellsize, 200, V, damping, MEC, ani)
    # transport.find_plateau(meshdims, cellsize, t, V, data, damping, x_vals, MEC, ani)
    # transport.time_avg_SA(meshdims, cellsize, t, V, damping, data, 2020, 4000, MEC, ani)
    # dispersions.magnon_dispersion_relation(meshdims, cellsize, t, damping, 0, 4000, MEC, ani, 'x')
    # plotting.plot_dispersion(meshdims, damping, MEC, ani, 'x')
    # plotting.plot_tAvg_SA(meshdims, cellsize, t, V, damping, data, 2020, 4000, MEC, ani)
    # dispersions.trajectory(meshdims, t, damping, 0, 4000, MEC, ani, 'y')
    # dispersions.phonon_dispersion_relation(meshdims, cellsize, t, damping, 200, 3800, MEC, ani, 'y')
    # dispersions.neel_T(meshdims, t, damping, MEC, ani)
    # plotting.plot_neel_T(meshdims, damping, MEC, ani)

if __name__ == '__main__':
    main()