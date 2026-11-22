from SRRQR_Algo6 import givens

"""
if k == 0
if np.max(gamma) < tol:
    break
jmax = np.argmax(gamma)
print(jmax)
#k = k+1
#gamma = np.linalg.norm(R[k:, k:], axis=0) if k < R.shape[0] else np.zeros(R.shape[1] - k)
gamma_max = gamma[jmax]
print(gamma_max)

P = np.eye(R.shape[1])
R[:, [k, k + jmax]] = R[:, [k + jmax, k]]
print(np.round(R, 3))
P[[k, k + jmax]] = P[[k + jmax, k]]
P_ = P @ P

R[k:, k:] = householder_first_column(R[k:, k:])

print(np.round(R, 3))
# if R[k, k] < 0:    ---> What to do if R[k, k] is negative ??
#    R[k, k] = np.abs(if R[k, k])

if k == 0:
    AB = R[0, 0]*R[0, 1:]
    print("AB", AB.shape)
    gamma = np.linalg.norm(R[1:, 1:], axis=0)
    print("gamma", gamma.shape)
    #tmp = solve_triangular(R[:k, :k], np.eye(k), lower=False)
    omega = np.array(R[0, 0])  # np.linalg.norm(tmp, axis=1) ** (-1)
    print("omega", omega.shape)

else:
"""