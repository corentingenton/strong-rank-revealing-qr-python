import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hadamard, svd, qr
from scipy.sparse.linalg import svds
from rrqr_k import rrqr_k

# Function to generate a Kahan matrix equivalent
import numpy as np


def kahan_matrix(n, theta):
    """
    Generate an n x n Kahan matrix with parameter theta.

    Args:
    n (int): Size of the matrix.
    theta (float): The angle parameter for cos(θ) and sin(θ).

    Returns:
    np.ndarray: The n x n Kahan matrix.
    """
    c = np.cos(theta)
    s = np.sin(theta)

    # Initialize an n x n matrix of zeros
    K = np.zeros((n, n))

    # Fill the first row
    K[0, 0] = 1
    K[0, 1] = 1

    # Fill the rest of the matrix
    for i in range(1, n):
        K[i, i] = c
        if i < n - 1:
            K[i, i + 1] = s

    return K

#    if ( I = J )
#      A(I,I) =  sin(ALPHA)^(I)
#    elseif ( I < J )
#      A(I,J) = - sin(ALPHA)^(I) * cos(ALPHA)
#    else
#      A(I,J) = 0

# Generate Kahan matrix
m, n = 8192, 400
M = kahan(n)
M = np.vstack([M, np.zeros((m-n, n))])  # Add zeros
k, f = 399, 2

# SRHT Sketching
d = 5 * n
D = np.diag(np.random.choice([-1, 1], size=m))
H = hadamard(m) / np.sqrt(m)
sampled_rows = np.random.choice(m, d, replace=True)
HDM = H @ D @ M
M_sk = np.sqrt(m/d) * HDM[sampled_rows, :]

# RRQR on sketched matrix
Q_sk, R_sk, P, nb = rrqr_k(M_sk, k, f)
Q, R = np.linalg.qr(M @ P)

# Singular values
R11 = R[:k, :k]
sig_R11 = svd(R11, compute_uv=False)
sig_M = svd(M, compute_uv=False)
ratio1 = sig_M[:k] / sig_R11

R11_sk = R_sk[:k, :k]
sig_R11_sk = svd(R11_sk, compute_uv=False)
sig_M_sk = svd(M_sk, compute_uv=False)
ratio2 = sig_M_sk[:k] / sig_R11_sk
ratio3 = sig_M[:k] / sig_R11_sk

# QR with column pivoting (QRCP)
Q_cp, R_cp, P_cp = qr(M, pivoting=True)
R11_cp = R_cp[:k, :k]
ratio_cp = sig_M[:k] / svd(R11_cp, compute_uv=False)

# Randomized SVD (RSVD)
_, S_rsvd, _ = svds(M, k=2*n)
sig_M_rsvd = np.diag(S_rsvd)
nb_rsvd = min(k, len(sig_M_rsvd))
ratio_rsvd = sig_M[:nb_rsvd] / sig_M_rsvd[:nb_rsvd]

# Strong rank-revealing QR
sig_L = svd(R[:k, :], compute_uv=False)
ratio_strong = sig_M[:k] / sig_L[:k]

# Plot singular values of M and M_sk
plt.figure(1)
plt.semilogy(range(1, n+1), sig_M, '-or', markevery=range(0, n, n//100), label=r'$\sigma(M)$')
plt.semilogy(range(1, n+1), sig_M_sk, '-xb', markevery=range(0, n, n//100), label=r'$\sigma(M^{sk})$')
plt.legend()
plt.xlabel('Index i')
plt.ylabel('Singular values')
plt.title('Singular values of M and M_sk')
plt.show()

# Plot the ratios between the leading singular values
plt.figure(2)
plt.plot(range(1, k+1), ratio1, '-+b', label='Rand SRRQR')
plt.plot(range(1, k+1), ratio_cp, '-or', label='QRCP')
plt.plot(range(1, nb_rsvd+1), ratio_rsvd, '-*g', label='Rand SVD')
plt.plot(range(1, k+1), ratio_strong, '-s', label='Rand QDP')
plt.legend()
plt.xlabel('Index i')
plt.ylabel(r'$\sigma_i(M)/\sigma_i(R_{11})$')
plt.title('Ratio of singular values')
plt.show()

# QLP Factorization
Rk = R[:k, :]
Q2, L = qr(Rk.T)
L = L.T
L_val = np.abs(np.diag(L)[:k])
ratio_svd1 = L_val / sig_M[:k]

Rk_cp = R_cp[:k, :]
Q2_cp, L_cp = qr(Rk_cp.T)
L_cp = L_cp.T
L_val_cp = np.abs(np.diag(L_cp)[:k])
ratio_svd2 = L_val_cp / sig_M[:k]

# Plot L-values
plt.figure(3)
plt.semilogy(range(1, k+1), sig_M[:k], '--', linewidth=2, label=r'$\sigma_{i}(M)$')
plt.semilogy(range(1, k+1), sig_L[:k], '--', linewidth=2, label=r'$\sigma_{i}(R_k)$')
plt.semilogy(range(1, k+1), L_val, '-*r', label=r'$L_{ii}$ by RSRRQR')
plt.semilogy(range(1, k+1), L_val_cp, '-og', label=r'$L_{ii}$ by QRCP')
plt.legend()
plt.xlabel('Index i')
plt.title('QLP Factorization')
plt.show()

# Approximation
f_eps = f * np.sqrt((1 + 0.25) / (1 - 0.25))
low_ranks = range(k // 3, k)
errors = []
Q2 = Q2.T @ P
for ell in low_ranks:
    L_approx = np.linalg.svd(L, full_matrices=False)[0][:, :ell]
    R_approx = np.linalg.svd(Rk, full_matrices=False)[0][:, :ell]
    M_approx = Q[:, :k] @ L_approx @ Q2
    errors.append(np.linalg.norm(M - M_approx))

bounds = sig_M[low_ranks] + sig_M[k] * np.sqrt(1 + f_eps**2 * k * (n - k))

plt.figure(4)
plt.semilogy(low_ranks, errors, linewidth=2.0, label='Errors')
plt.semilogy(low_ranks, bounds, '--+', linewidth=3.0, label='Theoretical bounds')
plt.legend()
plt.xlabel(r'Low rank $\ell$')
plt.ylabel(r'$\|\hat{M}_{\ell} - M\|_2$')
plt.title('Low-rank Approximation')
plt.show()
