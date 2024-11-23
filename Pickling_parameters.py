# Function that defines parameters related to the molecules
# and calculates the tensors called in the equations of motion.

import numpy as np
import pickle
import os

from GGM import GGM_matr #Function that gives us the GGMs and other related parameters
from Coeffs import Ham_coef, rates, pump_terms, equat_coeff # Functions for calculating coefficients 


def pickle_this(N, g_scaled, Nm):

    # Parameters we are keeping constant
    eps = 1.0       # excit freq
    S = 0.1         # exciton-phonon coupling
    om_v = .2       # phon freq
    Gam_down = 1e-4 # excit decay
    Gam_z = 0.03    # exciton dephasing
    T = 0.025       # phon bath temp
    gam_phon = 0.02 # phonon-bath coupling
    Gam_up = 0.     # Pump is calculated seperately 

    # Hamiltonian constants
    g_light_mat = g_scaled / np.sqrt(Nm)   # light-mat coupling

    all_files = os.listdir('./param-pickles') # For checking if parameter file aleady exists

    pkl_files = []
    for item in all_files:
        if item[-4:] == '.pkl':
            pkl_files.append(item)

    del(all_files)

    fname = ('param-pickles/lam_pickle_N%i.pkl' % N)
    if any([pkl == fname for pkl in pkl_files]):
        with open(fname, 'rb') as handle:
            lx, lz, lam, f, g, zeta = pickle.load(handle)
        print('lam-file exists')
    else:
        lx, lz, lam, f, g, zeta = GGM_matr(N)
        with open(fname, 'wb') as handle:
            pickle.dump([lx, lz, lam, f, g, zeta], handle)


    fname1 = ('param-pickles/Ham-coeffs-N%i-g%.1f-Nm1e%i.pkl' % (N,g_scaled,np.log10(Nm)))
    fname2 = ('param-pickles/dissipative-coeffs-N%i.pkl' % N)

    if any([pkl==fname1 for pkl in pkl_files]) and any([pkl==fname2 for pkl in pkl_files]):
        print('Hamiltonian and dissipative files exist')
    else:
        A, B = Ham_coef(N, eps, om_v, S, g_light_mat, lam)
        Bp = 2*B[:lx]

        gam = rates(lam, Gam_up, Gam_down, Gam_z, gam_phon, T, om_v, S)

        # Total xi_p will be xi0_pi + Gam_up * xi_pump (phi similar)
        phi0, xi0_z, xi0_p, fmB, ZpB = equat_coeff(gam, f, g, zeta, A, B, N)
        phi_pump, xi_p_pump, xi_z_pump = pump_terms(lam, N, f, zeta)

        with open(fname1, 'wb') as handle:
            pickle.dump([A, Bp, fmB, ZpB],handle)

        with open(fname2, 'wb') as handle:
            pickle.dump([phi0, phi_pump, xi0_z, xi_z_pump, xi0_p, xi_p_pump, Gam_down],handle)


if __name__ == '__main__':
    N = 4
    g_scaled = 0.4
    Nm = 1e8

    pickle_this(N, g_scaled, Nm)
