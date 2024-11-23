# A function that calculates the steady state for a defined equations of motion,
# importing parameter files where possible to save time.
# The steady state is then saved as a pickle file

import os 
import numpy as np
from scipy.integrate import solve_ivp
import time
import pickle

from Equations_DoS import diff_eqs, initial_state_all_down # Defines the equations of motion and initial state
from Pickling_parameters import pickle_this # Function that calculates necessary tensors, if the files do not exist


def SteadyState(P, om0, g_scaled, N, Nph, DoS, data_fld=None):

    ##### Parameters to tune #####

    # Photon dispersion
    k = np.arange(Nph)
    Nm = 1e8        # Number of emitters
    E_rho_2D = 5e5  # eV, Energy scale related to density
    om_k = om0 + E_rho_2D/Nm*k**2   #eV, Photon dispersion

    kappa = 1e-4    # eV, photon loss rate


    # Integration times
    t0 = 0
    t1 = 50000
    dt = 10000
    tend = 500000
    Nt = 100

    # convergence parameter
    eps = 1e-6

    # Folders
    if data_fld==None:
        data_fld = f'data/Nph{Nph}/'
    param_fld = 'param-pickles/'

    if not os.path.exists(data_fld):
        print(f'creating data folder: {data_fld}')
        os.mkdir(data_fld)

    ##### loading parameters and coefficients ####

    f_lam = (param_fld + f'lam_pickle_N{N}.pkl')
    f_Ham = (param_fld + f'Ham-coeffs-N{N}-g{g_scaled:.1f}-Nm1e{np.log10(Nm):.0f}.pkl')
    f_dis = (param_fld + f'dissipative-coeffs-N{N}.pkl')

    try:
        with open(f_lam, 'rb') as handle:
            lx, lz, lam, f, g, zeta = pickle.load(handle)
        with open(f_Ham,'rb') as handle:
            A, Bp, fmB, ZpB = pickle.load(handle)
        with open(f_dis, 'rb') as handle:
            phi0, phi_pump, xi0_z, xi_z_pump, xi0_p, xi_p_pump, Gam_down = pickle.load(handle)
    except FileNotFoundError:
        print('Need to calculate coefficients')
        if not os.path.exists(param_fld):
            os.mkdir(param_fld)

        pickle_this(N, g_scaled, Nm)

        with open(f_lam, 'rb') as handle:
            lx, lz, lam, f, g, zeta = pickle.load(handle)
        with open(f_Ham,'rb') as handle:
            A, Bp, fmB, ZpB = pickle.load(handle)
        with open(f_dis, 'rb') as handle:
            phi0, phi_pump, xi0_z, xi_z_pump, xi0_p, xi_p_pump, Gam_down = pickle.load(handle)

    Gam_up = P*Gam_down
    phi = phi0 + Gam_up * phi_pump
    xi_p = xi0_p + Gam_up * xi_p_pump
    xi_z = xi0_z + Gam_up * xi_z_pump


    #The initial value
    initial = initial_state_all_down(lx, lz, lam, N, Nph)

    # ODE to solve
    def diff(t, state): return diff_eqs(t, state, N, Nph, Nm, lx, lz, om_k, kappa, phi, xi_z, xi_p, fmB, ZpB, Bp, DoS )


    #### Solving ODE ####
    t = t1
    Tvec = np.linspace(t0, t, Nt)
    start_time = time.time()
    r = solve_ivp(diff, (t0, t), initial , t_eval=Tvec, rtol=1e-6)
    test = np.abs((r.y[:Nph-1, -2]-r.y[:Nph-1, -1])/(r.t[-2]-r.t[-1])/(r.y[:Nph-1, -1])).max()
    i = 0
    while (r.success) & (t < tend) & (test > eps):
        Tvec = np.linspace(t, t+dt, Nt)
        r = solve_ivp(diff, (t, t + dt), r.y[:, -1], t_eval=Tvec, rtol=1e-6)
        t += dt
        i += 1
        test = np.abs((r.y[:Nph-1, -2]-r.y[:Nph-1, -1])/(r.t[-2]-r.t[-1])/(r.y[:Nph-1, -1])).max()
    if not r.success:
        print(f'Integration failed at step: {i}')
    elif t >= tend:
        print(f"Integration didn't converge before reaching tend: {test:.1e}")
        print(f'Time: {time.time() - start_time}')
    else:
        print('Success!')
        print(f'Timesteps: {i} \t Time elapsed: {(time.time() - start_time):.1f}')


    fname = data_fld + f'data_g{g_scaled:.2f}_om0_{om0:.3f}_P{P:.3f}_N{N}.pkl'
    with open(fname, 'wb') as handle:
        pickle.dump([r.y[:, -1], om_k, Nm, test], handle)
