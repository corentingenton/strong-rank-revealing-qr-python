import numpy as np

def partialQR(M, k):
    # Split M into two parts
    M1 = M[:, :k]
    M2 = M[:, k:]

    # Perform QR decomposition on M1
    Q, Rk = np.linalg.qr(M1, mode='complete')

    # Compute Rc
    Rc = np.dot(Q.T, M2)

    # Construct R
    R = np.hstack((Rk, Rc))

    # Split Rc into B and C
    B = Rc[:k, :]
    C = Rc[k:, :]

    # Extract A from Rk
    A = Rk[:k, :]

    return Q, R, A, B, C
