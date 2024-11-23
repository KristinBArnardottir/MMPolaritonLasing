# Code for defining GGM and related objects

import numpy as np
from Funct import scal_matr_prod

def GGM_matr(N):    

    if N == 1:
        lx = 1
        lz = 1
        lam = np.concatenate((
            [np.array([[0,1],[1,0]])], #sigma_x
            [np.array([[0,-1j],[1j,0]])], #sigma_y
            [np.array([[1,0],[0,-1]])] #sigma_z ] 
            ))
    else:
        Nl = 2 * N #Dim of lambda

        # defining lambda_x and lambda_y matrices
        lambda_x = []
        lambda_y = []
        
        for i in range(N):
            for j in range(N,Nl):
                ar = np.zeros(shape=(Nl, Nl))
                ar[j,i] = 1
                # symmetric
                lambda_x.append(ar+ar.T)
                # anti-symmetric
                lambda_y.append(1j*ar -1j*ar.T)


        # defining lambda_z matrices
        lam_st = []
        lam_sb = []
        lam_at = []
        lam_ab = []
        
        #Non-diagonal
        for i in range(N-1):
            for j in range(i+1,N):
                #Top part
                ar = np.zeros(shape=(Nl, Nl))
                ar[j,i] = 1
                lam_st.append(ar + ar.T)
                lam_at.append(1j*ar - 1j*ar.T)
                #Bottom part
                ar = np.zeros(shape=(Nl, Nl))
                ar[j+N,i+N] = 1
                lam_sb.append(ar + ar.T)
                lam_ab.append(1j*ar - 1j*ar.T)

        # diagonal
        lam_d = []

        ar = np.zeros(shape = (2*N, 2*N))
        for i in range(1,2*N):
            ar[0:i,0:i] = np.identity(i)
            ar[i,i] = -i
            lam_d.append(ar * np.sqrt(2./(i*(i+1))))

        lambda_z = np.concatenate((lam_st,lam_sb, lam_at, lam_ab, lam_d)) # array of matrises lambda_z


        lx = len(lambda_x)
        lz = len(lambda_z)        


        lam = np.concatenate((lambda_x, lambda_y, lambda_z), axis=0) # joint array of both lambda_x and lambda_z matrices
        
    n_lam = lam.shape[0]
    
    f = np.empty( [n_lam, n_lam, n_lam], dtype = complex )
    g = np.empty( [n_lam, n_lam, n_lam], dtype = complex )


    for i in range(n_lam):
        for j in range(n_lam):
            for k in range(n_lam):
                com = np.dot(lam[i], lam[j]) - np.dot(lam[j], lam[i]) 
                anticom = np.dot(lam[i], lam[j]) + np.dot(lam[j], lam[i]) 
                f[i,j,k] = - 0.25 * 1j * scal_matr_prod(com, lam[k])
                g[i,j,k] = 0.25 * scal_matr_prod(anticom, lam[k])
    
    zeta = 1j * f + g
    
    return lx, lz, lam, f, g, zeta

