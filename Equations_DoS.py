# This file defines the differential equations based on the second order cumulant expansion
# in the basis of photon operators, and a basis of modified GGM, such that we only keep
# matrices proportional to sigma^+ and not sigma^-

import numpy as np


def diff_eqs(t, state, N, Nph, Nm, lx, lz, om_k, kappa, phi, xi_z, xi_p, fmB, ZpB, Bp, DoS):

#####Unpacking 'state' #####

    # Photon number operator
    ata = state[0:Nph]
    loc = Nph

    # Single lambda_z
    ell = state[loc: loc + lz]
    loc += lz

    # Photon - lambda_+
    # Rank 2 tensor: (i,k) - (matter, photon)
    Cp = np.reshape( state[ loc:loc + Nph*lx ], [ lx, Nph ])
    loc += lx*Nph

    # Fourier component for lambda_+ lambda_- (on different molecules)
    # Rank 3 tensor (i,j,k) (matter, matter, reciprocal for position) - Hermitian in i,j
    Dpm = np.reshape( state[ loc : loc + lx*lx*Nph], [ lx, lx, Nph ] )

    del(loc)
###########################

#####Building equations#####

    # Derived parameters for simplicity:
    # (subscripts x and z mean summation only happens in lam_x/z subspace)

    Cp_sum = np.matmul(Cp, DoS)

    # Photon number < a_dag a > (Vector - len Nph)
    ata_dot = ( - kappa * ata
                - 2.*Nm * np.matmul(Cp.T, Bp).imag
                )

    # single lambda < lam_z > (vector - len lz)
    ell_dot = ( np.matmul(xi_z, ell) + phi
    # - from switching indices
                - 8. * np.matmul( fmB.T, Cp_sum ).real
                )

    # Photon - matter < a_k lam_x > (matrix - dim (lx * Nph))
    # [:,None] makes sure that matter deg is always first index (photon is second)
    # Note: fBzx includes B.conj()
    Cp_dot = ( - ( 1.j * om_k + 0.5*kappa ) * Cp + np.matmul(xi_p, Cp)
                 + 2. * np.matmul( fmB.conj(), ell)[:, None] * ata
                 - 1.j * ( np.matmul(ZpB, ell) + 0.5*Bp/N)[:, None]
                 #- 1j * Nm * np.einsum(' ijk, j -> ik ', Dpm, Bx)
                 - 1.j * Nm * np.matmul(Bp, Dpm)
                 )

    # Fourier comp of matter - matter: odd lambdas < lam_+ lam_- >
    D_first_part = ( np.matmul(xi_p.conj(), Dpm)      # np.einsum(' ip, pjk -> ijk', xi_p, Dpm).T.conj()
                     #+ 2*np.einsum('jp, p, ik -> ijk', fmB, ell, Cp_a)
                     + 2.*np.transpose(np.matmul(fmB, ell)*(Cp - Cp_sum[:,None]/Nm)[:, :, None], (0, 2, 1))
                     )

    D_dot = D_first_part + np.transpose(D_first_part, (1, 0, 2)).conj()
############################

#####Re-Packing of results#####

    x_dot = np.concatenate((
                ata_dot,
                ell_dot,
                Cp_dot.flatten(),
                D_dot.flatten()
                          ))
###############################

    return x_dot


# Defining the initial state

from Funct import scal_matr_prod, tenz_dot_to_matr


def initial_state_all_down(lx, lz, lam, N, Nph):
    # Gives the state of all molecules in ground state, no photons
    init = np.empty([Nph + lz + lx*Nph + lx*lx*Nph], dtype = complex)

    # The ground state ket
    psi = np.zeros([2*N, 1], dtype = complex)
    psi[N] = 1.

    # The density matrix
    rho = tenz_dot_to_matr(psi.T, psi)

    # Photon number <a^+ a>
    init[ : Nph] = 0.
    loc = Nph

    # <lambda^z> = tr[ rho . lambda^z ]
    for i in range(lz):
        init[ loc + i] = scal_matr_prod( rho, lam[2*lx + i]).real
    loc += lz

    # <a lambda^x>
    init[ loc : loc + Nph*lx] = 0.
    loc += Nph*lx

    # <lambda_i^x lambda_j^x> = tr[ rho . lambda_i^x ] * tr[ rho . lambda_j^x ] = 0
    init[ loc : loc + lx*lx*Nph] = 0.
    loc += lx*lx*Nph

    return init
