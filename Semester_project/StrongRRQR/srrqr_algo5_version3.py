import numpy as np
from scipy.linalg import qr, solve_triangular
from givens import givens
from householder import *


def srrqr_algo5(M, f=2, tol=1e-3, k_stop=None, type_of_algo='tol'):

    # Dimension of the given matrix
    m, n = M.shape
    # Initialization
    k = 0
    R = np.asarray(M.copy())
    P = np.eye(n)

    # Compute the first gamma
    gamma = np.linalg.norm(R, axis=0)
    gamma = np.reshape(gamma, (len(gamma), 1))

    print(gamma.shape)
    # outer loop
    while True:
        jmax = np.argmax(gamma)
        if type_of_algo == 'tol':
            if k == n - 1:
                print("k is equal to n-1: stopping the algorithm")
                break
            gamma_max = gamma[jmax]
            if gamma_max < tol:
                print("Gamma max is smaller than the tolerance: stopping the algorithm")
                break
        elif type_of_algo == 'rank':
            if k == k_stop:
                print("k is equal to k_stop: stopping the algorithm")
                break

        # Exchange columns k and k+j_max
        P[:, [k, k + jmax]] = P[:, [k + jmax, k]]
        R[:, [k, k + jmax]] = R[:, [k + jmax, k]]
        gamma[0], gamma[jmax] = gamma[jmax], gamma[0]
        if k >= 1:
            AB[:, [0, jmax]] = AB[:, [jmax, 0]]

        R[k:, k:] = householder_reflection_first_column2(R[k:, k:])
        ga = R[k, k]
        c_T = R[k, k + 1:]
        if k == 0:
            AB = np.atleast_2d(c_T / ga)
            omega = np.atleast_2d([ga])
            omega = np.reshape(omega, (len(omega), 1))

        else:
            u, U = AB[:, 0], AB[:, 1:]
            AB = np.vstack([U - np.outer(u, c_T) / ga, c_T / ga])
            print(omega.shape)
            print(u.shape)
            print(ga.shape)
            tmp = 1 / (omega[:, 0] ** 2) + (u ** 2) / (ga ** 2)
            print(tmp.shape)
            omega = np.append(np.sqrt(1 / tmp), ga).reshape(-1, 1)
            print(omega)
        print(k)
        # Update gamma
        gamma[:-1, 0] = np.sqrt(gamma[1:, 0] ** 2 - c_T ** 2)
        gamma = gamma[:-1]
        print("here")
        
        # inner loop
        while True:
            print(gamma.shape)
            print(omega.shape)
            AB_abs = np.abs(AB)
            ratio_matrix = (1 / omega * np.transpose(gamma))  # np.outer(gamma, 1/omega).T
            print(ratio_matrix.shape)
            print(AB_abs.shape)
            max_matrix = np.maximum(AB_abs, ratio_matrix)
            rho_hat = np.max(max_matrix)

            if rho_hat <= f:
                break
            print("la")
            max_idx = np.argmax(max_matrix)
            i, j = np.unravel_index(max_idx, max_matrix.shape)
            P[:, [i, k + j]] = P[:, [k + j, i]]
            # If j > 0 exchange columns k+1 and k+j
            # Else interchanging the i and k th columns
            if j > 0:
                AB[:, [0, j]] = AB[:, [j, 0]]
                gamma[[0, j]] = gamma[[j, 0]]
                R[:, [k, k + j]] = R[:, [k + j, k]]
                
            if i < k:
                ind = list(range(i+1, k)) + [i]
                #P[:, i:k + 1] = np.concatenate((P[:, i + 1:k + 1], P[:, i:i + 1]))
                R[:, i:k] = R[:, ind]
                omega[i:k] = omega[ind]
                AB[i:k, :] = AB[ind, :]

                for ii in range(i, k-1):
                    G = givens(R[ii, ii], R[ii + 1, ii])
                    if G[0, :] @ np.array([R[ii, ii], R[ii + 1, ii]]) < 0:
                        G = -G  # guarantee R(ii,ii) > 0
                    R[ii:ii + 2, :] = G @ R[ii:ii + 2, :]
                if R[k-1, k-1] < 0:
                    R[k-1, :] = -R[k-1, :]

            # Third step : zeroing out the below-diag of k+1 th columns
            ## Zhipeng's correction
            ## column index should be k+1 not k
            if k < m - 1:
                for ii in range(k + 1, m):
                    G = givens(R[k-1, k-1], R[ii, k-1])
                    if G[0, :] @ np.array([R[k, k], R[ii, k]]) < 0:
                        G = -G  # guarantee R(k+1,k+1) > 0
                    R[[k, ii], :] = G @ R[[k, ii], :]


            # Get the different values needed for the update
            print("here4")
            ga = R[k-1, k-1]
            mu = R[k-1, k] / ga
            if k-1 < m:
                nu = R[k, k] / ga
            else:
                nu = 0
            rho = np.sqrt(mu ** 2 + nu ** 2)
            ga_bar = ga * rho
            b1 = R[:k-1, k-1]
            b2 = R[:k-1, k]
            c1T = R[k, k + 1:]
            if k + 1 > m:
                c2T = np.zeros_like(c1T)
            else:
                c2T = R[k, k + 1:]
            c1T_bar = (mu * c1T + nu * c2T) / rho
            c2T_bar = (nu * c1T - mu * c2T) / rho
            c1T_bar = np.reshape(c1T_bar, (1, -1))
            c2T_bar = np.reshape(c2T_bar, (1, -1))

            # Modify R
            R[:k-1, k-1] = b1
            R[:k-1, k] = b2
            R[k-1, k-1] = ga_bar
            R[k-1, k] = ga * mu / rho
            R[k, k] = ga * nu / rho
            R[k-1, k + 1:] = c1T_bar
            R[k, k + 1:] = c2T_bar

            # Update AB
            u = np.linalg.solve(R[:k-1, :k-1], b1)
            u = np.reshape(u, (-1, 1))
            u1 = AB[:k-1, 0]
            u1 = np.reshape(u1, (-1, 1))
            AB[:k-1, 0] = ((nu ** 2) * (u - mu * u1) / (rho ** 2))[:, 0]
            AB[k-1, 0] = mu / rho ** 2
            AB[k-1, 1:] = (c1T_bar / ga_bar)[0, :]
            AB[:k-1, 1:] = AB[:k-1, 1:] + (nu * u*c2T_bar - u1*c1T_bar) / ga_bar

            # Update gamma
            gamma[0] = ga * nu / rho
            gamma[1:] = np.sqrt(gamma[1:] ** 2 + (c2T_bar.T ** 2) - (c2T ** 2))
            print("gamma inside loop", np.isnan(gamma).sum())

            # Update omega
            u_bar = u1 + mu * u
            omega[k-1] = ga_bar
            omega[:k-1] = np.reciprocal(np.sqrt(omega[:k-1] ** (-2) + u_bar ** 2 / (ga_bar ** 2) - u ** 2 / (ga ** 2)))
            print("omega inside loop", np.isnan(omega).sum())
            ## Zhipeng's check
            A = R[:k+1, :k+1]
            A_inv = np.linalg.inv(A)
            omega_ex = 1/np.linalg.norm(A_inv,axis = 1)
            print("---------------- error of omega", np.linalg.norm(omega - omega_ex))
            C = R[k+1:, k+1:]
            gamma_ex = np.linalg.norm(C, axis = 0)
            print("---------------- error of gamma", np.linalg.norm(gamma - gamma_ex))
        k += 1
        print("now k is",k,"\n")
    return R, P, k

    ##################################################################################################################