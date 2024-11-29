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
    t = 1000
    V = 0.014
    data = '<mxdmdt>'
    damping = 4e-4
    MEC = 0
    ani = 'IP'
    x_vals = [2020, 2300, 2600, 3000, 3500, 4000]

    # Vs = [-0.012, -0.03, -0.05, -0.11, -0.25, -0.35, -0.5, -0.6, -0.65, -1, -1.3]
    # zs = [5, 10, 15, 25, 35, 40, 45, 50, 55, 75, 100]

    # dims = []
    # for z in zs:
    #     dims.append((4000,50,z))

    # for i in range(len(dims)):
        # transport.time_avg_SA(dims[i], cellsize, t, Vs[i], damping, data, 2020, 4000, MEC, ani)

    # transport.Init(meshdims, cellsize, 1000, MEC, ani)
    # transport.save_steadystate(meshdims, cellsize, 200, V, damping, MEC, ani)
    # transport.find_plateau(meshdims, cellsize, t, V, data, damping, False, MEC, ani)
    # transport.time_avg_SA(meshdims, cellsize, t, V, damping, data, 2020, 4000, MEC, ani)
    # dispersions.magnon_dispersion_relation(meshdims, cellsize, t, damping, 0, 4000, MEC, ani, 'x')
    # plotting.plot_dispersion(meshdims, damping, MEC, ani, 'x')
    # plotting.plot_tAvg_SA_z(meshdims, cellsize, t, V, damping, data, 0, 4000, MEC, ani)
    # dispersions.trajectory(meshdims, t, damping, 0, 4000, MEC, ani, 'y')
    # dispersions.phonon_dispersion_relation(meshdims, cellsize, t, damping, 0, 4000, MEC, ani, 'x')
    # dispersions.neel_T(meshdims, t, damping, MEC, ani)
    # plotting.plot_neel_T(meshdims, damping, MEC, ani)
    # plotting.plot_phonon_dispersion(meshdims, damping, MEC, ani, 'x', 0.1e-12)
    # outfile = 'OOP/cache/MEC/dispersions/working_phonon/4000x50x5/dirx_phonon_dispersion.txt'
    # savename = 'OOP/plots/MEC/dispersions/working_phonon/4000x50x5\damping0.0004dirx_phonon_dispersion.png'
    # plotting.plot_phonon_dispersion_specific(output_file=outfile, savename=savename, time_step=0.1e-12)

if __name__ == '__main__':
    main()