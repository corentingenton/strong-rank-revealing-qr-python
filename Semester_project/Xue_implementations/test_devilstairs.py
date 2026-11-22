import numpy as np
from scipy.linalg import orth

# Clear and initialize variables
m = 2048
n = 100
s = np.zeros(n)
Length = 10
Nst = n // Length

# Populate the vector 's' as per the given logic
for i in range(1, Nst + 1):
    s[Length * (i - 1):Length * i] = -0.6 * (i - 1)

# Handle the remaining elements
s[Length * Nst:] = -0.6 * (Nst - 1)

# Take 10^s element-wise
s = 10 ** s

# Create matrix M
M = orth(np.random.randn(n, n)) @ np.diag(s) @ orth(np.random.randn(n, n))

# Add rows of zeros to make the matrix size (m x n)
M = np.vstack([M, np.zeros((m - n, n))])

# Additional variables k and f
k = 50
f = 1

# Print the final matrix and variables (optional)
print("Matrix M:\n", M)
print("k:", k)
print("f:", f)
