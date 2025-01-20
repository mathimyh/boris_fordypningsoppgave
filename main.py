import plotting
import transport
import dispersions

def main():
    
    # Dimensions
    Lx = 4000
    Ly = 50
    Lz = 25
    cellsize = 5
    meshdims = (Lx, Ly, Lz)

    # Parameters
    t = 50
    V = -0.4
    data = '<mxdmdt>'
    damping = 4e-4
    MEC = 0
    ani = 'IP'
    # x_vals = [2020, 2300, 2600, 3000, 3500, 4000]
    x_vals = [500, 600, 700, 800, 900, 1000]

    # Vs = [-0.012, -0.03, -0.05, -0.11, -0.25, -0.35, -0.5, -0.6, -0.65, -1, -1.3]
    # zs = [5, 10, 15, 25, 35, 40, 45, 50, 55, 75, 100]

    # dims = []
    # for z in zs:
    #     dims.append((4000,50,z))

    # for i in range(len(dims)):
        # transport.time_avg_SA(dims[i], cellsize, t, Vs[i], damping, data, 2020, 4000, MEC, ani)

    
    # transport.Init(meshdims, cellsize, 1, MEC, ani)
    # transport.save_steadystate(meshdims, cellsize, 200, V, damping, MEC, ani)
    # transport.find_plateau(meshdims, cellsize, t, V, data, damping, False, MEC, ani)
    # transport.time_avg_SA_2D_y(meshdims, cellsize, t, V, damping, data, 0, 4000, MEC, ani)
    # dispersions.magnon_dispersion_relation(meshdims, cellsize, t, V, damping, 0, 4000, MEC, ani, dir = 'y', axis = 'x')
    # plotting.plot_dispersion(meshdims, damping, MEC, ani, 'x')
    # plotting.plot_tAvg_SA_2D_y(meshdims, cellsize, t, V, damping, data, 1000, 4000, MEC, ani)
    # dispersions.trajectory(meshdims, t, damping, 0, 4000, MEC, ani, 'y')
    # dispersions.phonon_dispersion_relation(meshdims, cellsize, t, damping, 0, 4000, MEC, ani, 'x')
    # dispersions.neel_T(meshdims, t, damping, MEC, ani)
    # plotting.plot_neel_T(meshdims, damping, MEC, ani)
    # plotting.plot_phonon_dispersion(meshdims, damping, MEC, ani, 'x', 0.1e-12)
    # outfile = 'OOP/cache/MEC/dispersions/working_phonon/4000x50x5/dirx_phonon_dispersion.txt'
    # savename = 'OOP/plots/MEC/dispersions/working_phonon/4000x50x5\damping0.0004dirx_phonon_dispersion.png'
    # plotting.plot_phonon_dispersion_specific(output_file=outfile, savename=savename, time_step=0.1e-12)
    # plotting.plot_magnon_dispersion(meshdims, damping, MEC, ani, 'y', 'x')

    # test_meshes = [(4000, 50, 5), (1500, 50, 50), (1000, 50, 100), (1000, 50, 150)]
    # test_voltages = [-0.0115, -0.85, -2.9, -8.5]


    # for i in range(4):
    #     transport.Init(test_meshes[i], cellsize, 50, MEC, ani)
    #     transport.save_steadystate(test_meshes[i], cellsize, 200, test_voltages[i], damping, MEC, ani)
    #     transport.time_avg_SA(test_meshes[i], cellsize, t, test_voltages[i], damping, data, 0, test_meshes[i][0], MEC, ani)
    #     transport.time_avg_SA_z(test_meshes[i], cellsize, t, test_voltages[i], damping, data, 0, test_meshes[i][0], MEC, ani)
    #     transport.time_avg_SA_2D(test_meshes[i], cellsize, t, test_voltages[i], damping, data, 0, test_meshes[i][0], MEC, ani)
if __name__ == '__main__':
    main()