from scipy.linalg import solve_triangular
from givens import givens
from householder import householder_reflection_first_column2
from implementation import *


def srrqr_algo5(M, f=2, tol=1e-3, k_stop=None, type_of_algo='tol'):

    # Dimension of the given matrix
    m, n = M.shape
    # Initialization
    tau_k = 0
    k = 0
    R = np.asarray(M.copy())
    P = np.eye(n)

    # Compute the first gamma
    gamma = np.linalg.norm(R, axis=0)

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
        print(gamma[0])
        R[k:, k:] = householder_reflection_first_column2(R[k:, k:])
        print("R after householder\n", np.round(R, 2))
        ga = R[k, k]
        c_T = R[k, k + 1:]
        if k == 0:
            AB = np.atleast_2d(c_T / ga)
            omega = np.array(ga)
        else:
            u, U = AB[:, 0], AB[:, 1:]
            AB = np.vstack([U - np.outer(u, c_T) / ga, c_T / ga])
            tmp = 1 / (omega ** 2) + (u ** 2) / (ga ** 2)
            omega = np.append(np.sqrt(1 / tmp), ga)

        # Update gamma
        gamma[:-1] = np.sqrt(gamma[1:] ** 2 - c_T ** 2)
        gamma = gamma[:-1]
        
        # inner loop
        while True:
            AB_abs = np.abs(AB)
            ratio_matrix = np.outer(gamma, 1/omega).T
            max_matrix = np.maximum(AB_abs, ratio_matrix)
            rho_hat = np.max(max_matrix)

            if rho_hat <= f:
                break
            max_idx = np.argmax(max_matrix)
            i, j = np.unravel_index(max_idx, max_matrix.shape)
            P[:, [i, k + j]] = P[:, [k + j, i]]
            tau_k += 1

            print("i, j", i, j)

            # If j > 0 exchange columns k+1 and k+j
            if j > 0:
                AB[:, [0, j]] = AB[:, [j, 0]]
                gamma[[0, j]] = gamma[[j, 0]]
                R[:, [k + 1, k + j + 1]] = R[:, [k + j + 1, k + 1]]

            # Else interchanging the i and k th columns
            if i < k:
                print("k", k)
                print("R before shifts\n", np.round(R, 2))
                R[:, i:k + 1] = np.concatenate((R[:, i + 1:k + 1], R[:, i:i + 1]), axis=1)
                omega[i:k + 1] = np.concatenate((omega[i + 1:k + 1], omega[i:i + 1]))
                AB[i:k + 1, :] = np.concatenate((AB[i + 1:k + 1, :], AB[i:i + 1, :]), axis=0)
                print("R after shifts\n", np.round(R, 2))

                for ii in range(i, k):
                    G = givens(R[ii, ii], R[ii + 1, ii])
                    if G[0, :] @ np.array([R[ii, ii], R[ii + 1, ii]]) < 0:
                        G = -G  # guarantee R(ii,ii) > 0
                    R[ii:ii + 2, :] = G @ R[ii:ii + 2, :]
                if R[k, k] < 0:
                    R[k, :] = -R[k, :]

                print("R after givens\n", np.round(R, 2))

            # Third step : zeroing out the below-diag of k+1 th columns
            print("R before zeroing\n", np.round(R, 2))
            if k < n - 1:
                for ii in range(k + 2, m):
                    G = givens(R[k + 1, k + 1], R[ii, k + 1])
                    if G[0, :] @ np.array([R[k + 1, k + 1], R[ii, k + 1]]) < 0:
                        G = -G  # guarantee R(k+1,k+1) > 0
                    R[[k + 1, ii], :] = G @ R[[k + 1, ii], :]
            print("R after zeroing\n", np.round(R, 2))

            # Get the different values needed for the update
            ga = R[k, k]
            mu = R[k, k + 1] / ga
            if k < n:
                nu = R[k + 1, k + 1] / ga
            else:
                nu = 0
            rho = np.sqrt(mu ** 2 + nu ** 2)
            ga_bar = ga * rho
            b1 = R[:k, k]
            b2 = R[:k, k + 1]
            c1T = R[k, k + 2:]
            if k + 1 > m:
                c2T = np.zeros_like(c1T)
            else:
                c2T = R[k + 1, k + 2:]
            c1T_bar = (mu * c1T + nu * c2T) / rho
            c2T_bar = (nu * c1T - mu * c2T) / rho
            U = AB[:k, 1:]

            print("ga\n", ga)
            print("mu\n", mu)
            print("nu\n", nu)
            print("rho\n", rho)
            print("b1\n", b1)
            print("b2\n", b2)
            print("c1T\n", c1T)
            print("c2T\n", c2T)
            print("c1T_bar\n", c1T_bar)
            print("c2T_bar\n", c2T_bar)
            print("U\n", U)

            # Modify R
            R_tmp = R.copy()
            R_tmp[:k, k + 1] = b1
            R_tmp[:k, k] = b2
            R_tmp[k, k] = ga_bar
            R_tmp[k, k + 1] = ga * mu / rho
            R_tmp[k + 1, k + 1] = ga * nu / rho
            R_tmp[k, k + 2:] = c1T_bar
            R_tmp[k + 1, k + 2:] = c2T_bar
            R = R_tmp

            # Update AB
            u = solve_triangular(R[:k, :k], b1, lower=False)
            u1 = AB[:k, 0].copy()
            AB[:k, 0] = (nu ** 2 * u - mu * u1) / rho ** 2
            AB[k, 0] = mu / rho ** 2
            AB[k, 1:] = c1T_bar / ga_bar
            AB[:k, 1:] = U + (nu * np.outer(u, c2T_bar) - np.outer(u1, c1T_bar)) / ga_bar

            # Update gamma
            gamma[0] = ga * nu / rho
            gamma[1:] = np.sqrt(gamma[1:] ** 2 + (c2T_bar ** 2) - (c2T ** 2))
            print("gamma inside loop", np.isnan(gamma).sum())

            # Update omega
            u_bar = u1 + mu * u
            omega[k] = ga_bar
            omega[:k] = (omega[:k] ** (-2) + u_bar ** 2 / (ga_bar ** 2) - u ** 2 / (ga ** 2)) ** (-1 / 2)
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
    return R, P, k, tau_k

    ##################################################################################################################