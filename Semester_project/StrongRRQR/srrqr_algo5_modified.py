import numpy as np
from scipy.linalg import qr, solve_triangular
from givens import givens
from householder import *

def srrqr_algo5(M, f, tol):
    # Check constant bound f
    if f <= 1:
        print('parameter f given is less or equal than 1. Automatically set f = 2')
        f = 2

    # Dimension of the given matrix
    m, n = M.shape

    # Initialization
    k = 0
    R = M.copy()
    P = np.eye(n)
    # ---------------------------------
    # Compute gamma
    gamma = np.linalg.norm(R, axis=0)
    while True:
        print("First while loop")
        print("----------------------------------------------------------------------------------")
        print("k = ", k, " ----------------------------------------------------------------------")
        #Zhipeng's modification
        if k == n-1:
            return R, P, k
        # Compute jmax and gamma_max
        jmax = np.argmax(gamma)
        gamma_max = gamma[jmax]
        if gamma_max < tol:
            print("Gamma max is smaller than the tolerance: stopping the algorithm")
            break
        #print("jmax: ", jmax, "--> gamma_max: ", gamma_max, "\n")

        # Exchange columns
        #print("Exchanging the columns of P, R, gamma and AB")
        P[[k, k + jmax]] = P[[k + jmax, k]]
        #print("R: ", R.shape)
        #print(np.round(R, 3), "\n")
        R[:, [k, k + jmax]] = R[:, [k + jmax, k]]

        #print("gamma before: ", gamma, "\n")
        gamma[0], gamma[jmax] = gamma[jmax], gamma[0]
        #print("gamma after: ", gamma, "\n")
        if k == 1:
            #print("AB before: ", AB, "\n")
            AB[0], AB[jmax] = AB[jmax], AB[0]
            #print("AB after: ", AB, "\n")
        elif k > 1:
            #print("AB before: ", AB, "\n")
            AB[:, [0, jmax]] = AB[:, [jmax, 0]]
            #print("AB after: ", AB, "\n")
        #print("Interchanging the columns of R")
        #print("R: ", R.shape)
        #print(np.round(R, 3), "\n")
        #print("P: ", P.shape)
        #print(P, "\n")

        if k == 0:
            #print("R before householder: ", R.shape)
            #print(np.round(R, 3), "\n")
            R = householder_first_column(R)
            #print("R after householder: ", R.shape)
            #print(np.round(R, 3), "\n")
            ga = R[0, 0]
            c_T = R[0, 1:]
            #print("ga: ")
            #print(np.round(ga, 3), "\n")
            #print("c_T: ", c_T.shape)
            #print(np.round(c_T, 3), "\n")

            # Compute AB
            AB = c_T / ga
            #print("AB: ", AB.shape)
            #print(np.round(AB, 3), "\n")
            A = R[:k + 1, :k + 1]
            B_ = R[:k + 1, k + 1:]
            #print("A: ", A.shape)
            #print(np.round(A, 3), "\n")
            #print("B_: ", B_.shape)
            #print(np.round(B_, 3), "\n")
            AB_ = np.linalg.inv(A) @ B_
            #print("AB with linalg inv: ", AB_.shape)
            #print(np.round(AB_, 3), "\n")

            omega = ga
            #print("omega: ", omega)
            #print(omega, "\n")

            for i in range(n - (k + 1)):
                #print("gamma i+1:", gamma[i + 1])
                #print("v_j:", c_T[i])
                gamma[i] = np.sqrt(gamma[i + 1] ** 2 - c_T[i] ** 2)
                #print("gamma:", gamma[i])

            #gamma[:-k-1] = np.sqrt(gamma[1:-k]**2 - c_T**2)
            #print("gamma: ", gamma.shape)
            #print(gamma, "\n")
            #print("R: ", R.shape)
            #print(np.round(R, 3), "\n")
            gamma_norm = np.linalg.norm(R[k + 1:, k + 1:], axis=0)
            #print("gamma with norm: ", gamma_norm.shape)
            #print(gamma_norm)
            gamma = np.delete(gamma, -1)
            #print("gamma: ", gamma.shape)
            #print(gamma, "\n")

        else:
            #print("Updating the values when k > 0")
            #print("R: ", R.shape)
            #print(np.round(R, 3), "\n")
            R[k:, k:] = householder_first_column(R[k:, k:])
            print("R after householder: ", R.shape)
            print(np.round(R, 3), "\n")

            # Updating values
            b = R[:k, k]
            #print("b \n", b)
            B = R[:k, k + 1:]
            #print("B\n", B)
            ga = R[k, k]
            #print("ga\n", ga)
            c_T = R[k, k + 1:]
            #print("c_T\n", c_T)
            #print("k", k)
            #print("AB: ", AB.shape)
            #print(np.round(AB, 3), "\n")
            if k == 1:
                #print("ici et k = ", k)
                u = AB[0]
                #print("u\n", u)
                U = AB[1:]
                #print("U\n", U)
            else:
                u = AB[:k, 0]
                #print("u\n", u)
                U = AB[:k, 1:]
                #print("U\n", U)

            AB = np.vstack([U - np.outer(u, c_T) / ga, c_T / ga])
            #print("AB: ", AB.shape)
            #print(np.round(AB, 3), "\n")

            A = R[:k + 1, :k + 1]
            B_ = R[:k + 1, k + 1:]
            #print("A: ", A.shape)
            #print(np.round(A, 3), "\n")
            #print("B_: ", B_.shape)
            #print(np.round(B_, 3), "\n")
            AB = np.linalg.inv(A) @ B_
            #print("AB with solve triangular: ", AB.shape)
            #print(np.round(AB, 3), "\n")
            # Why they don't have exactly the same values ???

            if k == 1:
                omega = np.array(omega)
                tmp = 1 / omega ** 2 + u ** 2 / ga ** 2
                omega = np.sqrt(1 / tmp)
                omega = np.append(omega, ga)
                #print("omega: \n", omega)
                #print(omega.shape)
            else:
                #print("updating omega")
                #Zhipeng's modification
                tmp = 1 / omega**2 + u**2 / ga**2
                omega = np.sqrt(1/tmp)
                omega = np.append(omega,ga)
                
                # for i in range(k):
                #     tmp = 1 / omega[i] ** 2 + (u[i] ** 2) / (ga ** 2)
                #     omega[i] = np.sqrt(1 / tmp)
                # omega = np.append(omega, ga)
                #print("omega: \n", omega)
                #print(omega.shape)
            #Zhipeng's modification
            #print("computed updated omega:\n",omega)
            A = R[:k + 1, :k + 1]
            #print("now A is: \n",np.round(A,3))
            #print("inverse of A is: \n",np.round(np.linalg.inv(A),3))
            omega_ex = np.linalg.norm(np.linalg.inv(A), axis=1) ** (-1)
            #print("updated omega should be:\n",omega_ex)

            for i in range(n - (k + 1)):
                #print("gamma i+1:", gamma[i + 1])
                #print("v_j:", c_T[i])
                gamma[i] = np.sqrt(gamma[i + 1] ** 2 - c_T[i] ** 2)
                #print("gamma:", gamma[i])
            #print("gamma: ", gamma.shape)
            #print(gamma, "\n")
            gamma = np.delete(gamma, -1)
            #print("gamma: ", gamma.shape)
            #print(gamma, "\n")

            gamma = np.linalg.norm(R[k + 1:, k + 1:], axis=0)
            #print("gamma with norm: ", gamma.shape)
            #print(gamma, "\n")
            # tmp = solve_triangular(R[:k+1, :k+1], np.eye(k), lower=False)
            #print("omega with norm: ", omega.shape)
            #print(omega, "\n")
        A = R[:k + 1, :k + 1]
        print(f"A at {k} is: \n", np.round(A,3))
        print("R:")
        print(np.round(R, 3), "\n")
        if k == n - 1:
            k += 1
            break
        while True:
            # Compute rho_hat
            print("Second while loop")
            print("------------------------------------------")
            AB_abs = np.abs(AB)
            # print("AB_abs: ", AB_abs.shape)
            # print(np.round(AB_abs, 3), "\n")
            if k > 0:
                tmp = omega[:, np.newaxis]
            else:
                tmp = omega
            ratio_matrix = gamma / tmp
            # print("ratio_matrix: ", ratio_matrix.shape)
            # print(np.round(ratio_matrix, 3), "\n")
            max_matrix = np.maximum(AB_abs, ratio_matrix)
            # print("max_matrix: ", max_matrix.shape)
            # print(np.round(max_matrix, 3), "\n")
            rho_hat = np.max(max_matrix)
            # print("rho_hat: ", rho_hat, "\n")

            if rho_hat <= f:
                break
            print("We have to do some interchange.\n")
            # Get indices i and j
            max_idx = np.argmax(max_matrix)
            i, j = np.unravel_index(max_idx, max_matrix.shape)
            print("i: ", i)
            print("j: ", j)

            # If j > 0 exchange columns k+1 and k+j
            if j > 0:
                # print("Interchanging columns (k+1) and (k+j)")
                # Just for checking that everything works fine
                # print("before exchange")
                # print(np.round(AB, 3), "\n")
                # print(np.round(gamma, 3), "\n")
                # print(np.round(R, 3), "\n")
                # print(np.round(P, 3), "\n")
                AB[:, [0, j]] = AB[:, [j, 0]]
                gamma[[0, j]] = gamma[[j, 0]]
                R[:, [k, k + j]] = R[:, [k + j, k]]
                P[[k, k + j]] = P[[k + j, k]]
                # Just for checking that everything works fine
                # print("after exchange")
                # print(np.round(AB, 3), "\n")
                # print(np.round(gamma, 3), "\n")
                # print(np.round(R, 3), "\n")
                # print(np.round(P, 3), "\n")

            # Second step : interchanging the i and k th columns
            if i < k:
                # print("Interchanging columns i and k")
                # print("before echange")
                # print("AB:")
                # print(np.round(AB, 3), "\n")
                # print("gamma:")
                # print(np.round(gamma, 3), "\n")
                # print("R:")
                # print(np.round(R, 3), "\n")
                # print("P:")
                # print(np.round(P, 3), "\n")
                print("section 4.2: before:")
                print("R:")
                print(np.round(R, 3), "\n")
                print("AB:")
                print(np.round(AB, 3), "\n")
                print("A:\n")
                print(np.round(A, 3))
                #P = np.concatenate((P[:, :i], P[:, i + 1:k + 1], P[:, i:i + 1], P[:, k + 1:]), axis=1)
                #omega = np.concatenate((omega[:i], omega[i + 1:k + 1], [omega[i]], omega[k + 1:]))
                #R = np.concatenate((R[:, :i], R[:, i + 1:k + 1], R[:, i:i + 1], R[:, k + 1:]), axis=1)
                #print("test", AB[:, i + 1:k + 1])
                #AB = np.concatenate((AB[:, :i], AB[:, i + 1:k + 1], AB[:, i:i + 1], AB[:, k + 1:]), axis=0)
                #p[i:k + 1] = np.concatenate((p[i + 1:k + 1], p[i:i + 1]))
                #R[:, i:k + 1] = np.concatenate((R[:, i + 1:k + 1], R[:, i:i + 1]), axis=1)
                #omega[i:k + 1] = np.concatenate((omega[i + 1:k + 1], omega[i:i + 1]))
                #AB[i:k + 1, :] = np.concatenate((AB[i + 1:k + 1, :], AB[i:i + 1, :]), axis=0)
                P[i:k + 1] = np.concatenate((P[i + 1:k + 1], P[i:i + 1]))
                R[:, i:k + 1] = np.concatenate((R[:, i + 1:k + 1], R[:, i:i + 1]), axis=1)
                omega[i:k + 1] = np.concatenate((omega[i + 1:k + 1], omega[i:i + 1]))
                AB[i:k + 1, :] = np.concatenate((AB[i + 1:k + 1, :], AB[i:i + 1, :]), axis=0)
                print("section 4.2: after:")
                print("R:")
                print(np.round(R, 3), "\n")
                print("AB:")
                print(np.round(AB, 3), "\n")
                # print("After echange")
                # print("AB:")
                # print(np.round(AB, 3), "\n")
                # print("gamma:")
                # print(np.round(gamma, 3), "\n")
                # print("R:")
                # print(np.round(R, 3), "\n")
                # print("P:")
                # print(np.round(P, 3), "\n")
                # # givens rotation for the triangulation of R(1:k, 1:k)
                # print("Using givens rotation")
                # print("R before: ")
                # print(np.round(R, 3))

                for ii in range(i, k):
                    G = givens(R[ii, ii], R[ii + 1, ii])
                    #print("Applying givens rotation :")
                    if G[0, :] @ np.array([R[ii, ii], R[ii + 1, ii]]) < 0:
                        G = -G  # guarantee R(ii,ii) > 0
                    R[ii:ii + 2, :] = G @ R[ii:ii + 2, :]
                    #print(np.round(R, 3))
                    #Q[:, ii:ii + 2] = Q[:, ii:ii + 2] @ G.T
                if R[k, k] < 0:
                    R[k, :] = -R[k, :]
                    #Q[:, k] = -Q[:, k]
                # print("R after: ")
                # print(np.round(R, 3))

                A = R[:k + 1, :k + 1]
                B_ = R[:k + 1, k + 1:]
                print("section 4.2: after zeroing out:")
                print("R:")
                print(np.round(R, 3), "\n")
                print("AB:")
                print(np.round(AB, 3), "\n")
                print("A: ", A.shape)
                print(np.round(A, 3), "\n")
                #print("B_: ", B_.shape)
                #print(np.round(B_, 3), "\n")
                # AB_ = np.linalg.inv(A) @ B_
                # print("AB_ with solve: ", AB_.shape)
                # print(np.round(AB_, 3), "\n")

            # Third step : zeroing out the below-diag of k+1 th columns
            if k < n-1:
                #print("Zeroing out the below-diag of k+1 th columns")
                # print("R before: ")
                # print(np.round(R, 3))
                for ii in range(k + 1, m):
                    G = givens(R[k, k], R[ii, k])
                    if G[0, :] @ np.array([R[k, k], R[ii, k]]) < 0:
                        G = -G  # guarantee R(k+1,k+1) > 0
                    R[[k, ii], :] = G @ R[[k, ii], :]
                # print("R after: ")
                # print(np.round(R, 3))
                #print("R: ", R.shape)
                #print(np.round(R, 3), "\n")

            # Get the different values needed for the update
            print("Before we modify R we have:\n")
            print("R: ", R.shape)
            print(np.round(R[:8, :], 3), "\n")
            print("A: ", R[:k + 1, :k + 1].shape)
            print(np.round(R[:k + 1, :k + 1], 3), "\n")
            print("B: ", R[:k + 1, k + 1:].shape)
            print(np.round(R[:k + 1, k + 1:], 3), "\n")
            print("C: ", R[k + 1:, k + 1:].shape)
            print(np.round(R[k + 1:, k + 1:], 3), "\n")
            AB_ = np.linalg.inv(R[:k + 1, :k + 1]) @ R[:k + 1, k + 1:]
            print("AB with inv: ", AB.shape)
            print(np.round(AB, 3), "\n")
            print("AB with formulas: ", AB.shape)
            print(np.round(AB, 3), "\n")

            P[[k, k + 1]] = P[[k + 1, k]]
            print("so we have:\n")
            ga = R[k, k]
            print("gamma = ", ga)
            mu = R[k, k + 1] / ga
            print("mu", R[k, k + 1], "/", ga, "=", mu)
            if k < n:
                nu = R[k + 1, k + 1] / ga
                print("nu", R[k + 1, k + 1], "/", ga, "=", nu)
            else:
                nu = 0
            rho = np.sqrt(mu ** 2 + nu ** 2)
            print("rho = ", rho)
            ga_bar = ga * rho
            print("ga_bar = ", ga_bar)
            b1 = R[:k, k]
            print("b1 = ", b1)
            b2 = R[:k, k + 1]
            print("b2 = ", b2)
            c1T = R[k, k + 2:]
            print("c1T = ", c1T)
            if k + 1 > m:
                c2T = np.zeros_like(c1T)
            else:
                c2T = R[k + 1, k + 2:]
                print("c2 = T", c2T)

            c1T_bar = (mu * c1T + nu * c2T) / rho
            print("c1T_bar = ", c1T_bar)
            c2T_bar = (nu * c1T - mu * c2T) / rho
            print("c2T_bar = ", c2T_bar)
            U = AB[:k, 1:]
            print("U:\n")
            print(U)
            
            #Zhipeng's modification
            #print("We interchange",k+1,"th and ",k+2,"th column")
            #print("before modifing, R is: ", R.shape)
            #print(np.round(R, 3), "\n")
            A = R[:k + 1, :k + 1]
            #print("A is: ", np.round(A,3))
            #print("inverse of A: ", np.round(np.linalg.inv(A),3))
            #print("in storage omega is: ",omega)
            omega_ex = np.linalg.norm(np.linalg.inv(A), axis=1) ** (-1)
            #print("omega should be: ",omega_ex)
            
            # Modify R
            R_tmp = R.copy()
            # A = R[:k + 1, :k + 1]
            # omega_ex = np.linalg.norm(np.linalg.inv(A), axis=1) ** (-1)
            R_tmp[:k, k + 1] = b1
            R_tmp[:k, k] = b2
            R_tmp[k, k] = ga_bar
            R_tmp[k, k + 1] = ga * mu / rho
            R_tmp[k + 1, k + 1] = ga * nu / rho
            R_tmp[k, k + 2:] = c1T_bar
            R_tmp[k + 1, k + 2:] = c2T_bar
            #print("R tmp : ", R_tmp.shape)
            #print(np.round(R_tmp, 3), "\n")
            R = R_tmp
            #Zhipeng's modification
            print("after modifying, R is: ", R.shape)
            print(np.round(R[:10, :], 3), "\n")
            A = R[:k + 1, :k + 1]
            #print("A_bar is: \n", np.round(A,3))
            #print("inverse of A_bar: \n", np.round(np.linalg.inv(A),3))
            
            # update AB
            print("A_k-1:\n", np.round(R[:k, :k], 3))
            u = solve_triangular(R[:k, :k], b1, lower=False)
            u_bbar = solve_triangular(R[:k, :k], b2, lower=False)
            print("u", u)
            #U = AB[:k, 1:]
            #print("U:\n")
            #print(U)
            #print("mu should be:", mu)
            #print(AB[-1, 0])
            print("u_bar should be:", u_bbar)
            #print("-u/ga:", -u/ga)
            #print("1/ga:", 1/ga)
            u1 = AB[:k, 0].copy()
            print("u1", u1)

            # Update AB and check if it is equal to A^-1 * B using numpy
            A = R[:k + 1, :k + 1]
            B_ = R[:k + 1, k + 1:]
            AB_ = np.linalg.inv(A) @ B_
            # print("AB_ with solve: ", AB_.shape)
            # print(np.round(AB_, 3), "\n")
            # print("AB before update\n", np.round(AB_, 3))
            #print("AB after update\n", np.round(AB, 3))
            AB[:k, 0] = (nu ** 2 * u - mu * u1) / rho ** 2
            #print("AB after update\n", np.round(AB, 3))
            AB[k, 0] = mu / rho ** 2
            #print("AB after update\n", np.round(AB, 3))
            AB[k, 1:] = c1T_bar / ga_bar
            #print("AB after update\n", np.round(AB, 3))
            # print("AB[:k - 1, 1:] shape", AB[:k - 1, 1:].shape)
            #print("nu shape", nu.shape)
            #print("u shape", u.shape)
            #print("c2T_bar shape", c2T_bar.shape)
            #print("u1 shape", u1.shape)
            #print("c1T_bar shape", c1T_bar.shape)
            #print("nu * u * c2T_bar shape", (nu * np.outer(u, c2T_bar)).shape)
            #print("u1 * c1T_bar shape", (u1 * c1T_bar).shape)
            #U = AB[:k, 1:]
            #print("U:\n")
            #print(U)
            print("u", u.shape, "\n", u, "\n")
            print("u1", u1.shape, "\n", u1, "\n")
            print("c2T_bar", c2T_bar.shape, "\n", c2T_bar, "\n")
            print("c1T_bar", c1T_bar.shape, "\n", c1T_bar, "\n")
            AB[:k, 1:] = U + (nu * np.outer(u, c2T_bar) - np.outer(u1, c1T_bar)) / ga_bar
            print("AB after update\n", np.round(AB, 3))
            A = R[:k + 1, :k + 1]
            B_ = R[:k + 1, k + 1:]
            #print("A: ", A.shape)
            #print(np.round(A, 3), "\n")
            #print("B_: ", B_.shape)
            #print(np.round(B_, 3), "\n")
            AB_ = np.linalg.inv(A) @ B_
            print("AB with solve: ", AB_.shape)
            print(np.round(AB_, 3), "\n")

            # Update gamma
            # print("gamma before : ", gamma.shape)
            # print(np.round(gamma, 3), "\n")
            gamma[0] = ga * nu / rho
            gamma[1:] = np.sqrt(gamma[1:] ** 2 + (c2T_bar ** 2) - (c2T ** 2))
            # print("gamma after: ", gamma.shape)
            # print(np.round(gamma, 3), "\n")
            # gamma_norm = np.linalg.norm(R[k + 1:, k + 1:], axis=0)
            # print("gamma with norm: ", gamma_norm.shape)
            # print(gamma_norm)

            # Update omega
            u_bar = u1 + mu * u
            # omega_ = np.linalg.norm(np.linalg.inv(A), axis=1) ** (-1)
            # print("omega with norm: ", omega_.shape)
            # print(omega, "\n")
            print("Now k is: ",k,"\n")
            print("before modification, omega is: ",omega)
            omega[k] = ga_bar
            omega[:k] = (omega[:k] ** (-2) + u_bar ** 2 / (ga_bar ** 2) - u ** 2 / (ga ** 2)) ** (-1 / 2)
            # print("omega", omega)
            # print("omega[:k]", omega[:k])
            # print("omega[:k]**2", omega[:k] ** 2)
            # print("u_bar ** 2 / (ga_bar ** 2)", (u_bar ** 2) / (ga_bar ** 2))
            # print("u ** 2 / (ga ** 2)", (u ** 2) / (ga ** 2))
            # print("u", u)
            # print("ga", ga)
            # print("sans sqrt", ((omega[:k] ** 2) + (u_bar ** 2) / (ga_bar ** 2) - (u ** 2) / (ga ** 2)))
            print("computed modified omega: ", omega.shape, "\n")
            print(omega, "\n")
            A = R[:k + 1, :k + 1]
            omega_ex = np.linalg.norm(np.linalg.inv(A), axis=1) ** (-1)
            print("modified omega should be: ", omega.shape, "\n")
            print(omega_ex, "\n")
            v1 = -u/ga
            v2 = -u_bar/ga_bar
            print("the last column of inv(A) with formulas:\n",v1)
            print("the last column of inv(A_bar) with formulas:\n",v2)
            print("u_bar is:\n",u_bar)
            print("u_bar computed by another way:\n",u_bbar)
        k += 1
    # Only return the truncated version of the strong RRQR decomposition
    return R, P, k

    ##################################################################################################################
