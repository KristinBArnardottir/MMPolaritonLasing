# Functions that calculate the coefficients used in 'Equations_DoS.py'

import numpy as np
from Funct import tenz_dot_to_matr, dag, scal_matr_prod


#Defining the Hamiltonian coefficients A_i and B_i
def Ham_coef(N, eps, om_v, S, g_light_mat, lam):

    n_lam = lam.shape[0]
    # Pauli matrices
    sig_x = np.array([[0,1],[1,0]], dtype = complex)
    sig_y = np.array([[0, -1j],[1j, 0]])
    sig_z = np.array([[1, 0],[0, -1]], dtype = complex)
    sig_minus = 0.5 * (sig_x - 1j * sig_y)

    # phonon annihilation operator b
    b = np.diag( np.sqrt( np.arange(1,N) ), 1 )

    # phonon displacement operator
    x = dag(b) + b

    # phonon number operator matrix nb = b^{\dagger} b
    nb = np.dot(dag(b), b)

    # A anb B Hamiltonian matrices
    A_ham = (
            0.5 * eps * tenz_dot_to_matr( sig_z, np.identity(N) ) +

            om_v * (
                    tenz_dot_to_matr( np.identity(2), nb ) +
                    np.sqrt(S) * tenz_dot_to_matr(sig_z, x )
                    )
            )

    B_ham = (
            g_light_mat * tenz_dot_to_matr( sig_minus, np.identity(N) )
            )

    # Expansion coefficients for matrices A_ham and B_ham - A_i and B_i
    A = np.empty( [n_lam], dtype = np.complex128 )
    B = np.empty( [n_lam], dtype = np.complex128 )

    for i in range(n_lam):
        A[i] = 0.5 * scal_matr_prod(A_ham, lam[i])
        B[i] = 0.5 * scal_matr_prod(B_ham, lam[i])

    return A, B



#Defining the rate coefficients gam^\mu_i 
def rates(lam, Gam_up, Gam_down, Gam_z, gam_phon, T, om_v, S):

    n_lam = lam.shape[0]
    N = lam.shape[1]//2

    sig_z = np.array([[1, 0],[0, -1]], dtype = complex)
    sig_plus = np.array([[0,1],[0,0]], dtype = complex)
    b = np.diag( np.sqrt( np.arange(1,N) ), 1 )

    # jump operators in original basis O_i
    sig_plus_jump = tenz_dot_to_matr( sig_plus, np.identity(N) )
    sig_minus_jump = dag(sig_plus_jump)
    sig_z_jump = tenz_dot_to_matr( sig_z, np.identity(N) )
    b_dag_jump = tenz_dot_to_matr( np.identity(2), dag(b) ) - np.sqrt(S) * sig_z_jump
    b_jump = dag(b_dag_jump)

    O = np.array([sig_plus_jump, sig_minus_jump, b_dag_jump, b_jump, sig_z_jump])

    n_mean_phon = 1.0 / (np.exp(om_v / T) - 1)  # mean number delocalized phonons
    phon_up = gam_phon * n_mean_phon            # phonon gamma up
    phon_down = gam_phon * (n_mean_phon + 1)    # phonon gamma down

    # Original rates \Gamma_i
    old_rates = np.array([Gam_up, Gam_down, phon_up, phon_down, Gam_z])

    # new rates \gamma_i^{\mu}
    mu_num = O.shape[0]                         # number of old jump operators
    gam_mu_i = np.empty( [mu_num, n_lam], dtype = complex )

    for mu in range(mu_num):
        for i in range(n_lam):
            c = 0.5 * scal_matr_prod(O[mu], lam[i])
            gam_mu_i[mu, i] = np.sqrt(old_rates[mu]) * c

    return gam_mu_i

# Coefficients that scale with pump strength
def pump_terms(lam, N, f, zeta):
        # Defining the jump operator associated with the pump
        sig_plus = np.array([[0,1],[0,0]], dtype = complex)
        sig_plus_jump = tenz_dot_to_matr( sig_plus, np.identity(N) )

        lx = N*N
        lxy = 2*lx
        gam0 = np.empty(lam.shape[0], dtype = complex)
        for i in range(lam.shape[0]):
            gam0[i] = 0.5 * scal_matr_prod(sig_plus_jump, lam[i])

        phi_pump = 2j/N * np.einsum('ijk, j, k -> i', f, gam0, gam0.conj())[lxy:]

        xi_pump = 1j*(np.einsum('j, k, ijkp ->ip',
                        gam0, gam0.conj(),
                        (  np.einsum('ijl, klp -> ijkp', f, zeta)
                        + np.einsum('kil, ljp -> ijkp', f, zeta)
                        ) ))
        xi_p_pump = xi_pump[:lx,:lx] + 1j*xi_pump[lx:lxy,:lx]
        xi_z_pump = xi_pump[lxy:, lxy:]

        return phi_pump, xi_p_pump, xi_z_pump



# Calculating the equation coefficients xi, zeta, phi
# and the pre-contracted tensors fBzx zB
def equat_coeff(gam, f, g, zeta, A, B, N):

    lx = N*N
    lxy = 2*lx

    gam_gam = np.matmul(gam.T, gam.conj())

    # calculating xi (two indices)
    xi = (1j * ( np.einsum('jk, ijkp ->ip',
                          gam_gam,
                          (  np.einsum('ijl, klp -> ijkp', f, zeta)
                           + np.einsum('kil, ljp -> ijkp', f, zeta)
                          ) )
                )
             + 2 * np.einsum('ijp, j -> ip', f, A)
        )
    xi_z = xi[lxy:, lxy:]
    xi_p = xi[:lx,:lx] + 1j*xi[lx:lxy,:lx]

    # calculating phi (one index: z)
    phi = ( ( 2j/N ) *
              np.einsum( 'ijp, jp -> i', f[ lxy:, :, :] , gam_gam)
            )

    # Calculating fBzx (Two indices: z x)
    fBxz = np.einsum( 'ijk, j -> ik', f[ :lxy, :, lxy:], B.conj() )
    fmB = 0.5*(fBxz[:lx,:] - 1j*fBxz[lx:lxy,:])

    # Calculating zB (Two indices: x z)
    zB = np.einsum('ijk, j -> ik', zeta[ :lxy, : , lxy: ], B )
    ZpB = 0.5*(zB[:lx, :] + 1j* zB[lx:lxy,:])


    return phi, xi_z, xi_p, fmB, ZpB
