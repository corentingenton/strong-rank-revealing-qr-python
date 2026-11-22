# Step 3: Import the compiled module
import lapack_wrapper


# Step 4: Demonstrate usage
def example_dgeqp3rk():
    # Example parameters
    m, n, nrhs = 5, 4, 2

    # Create input matrices
    A = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0, 16.0],
        [17.0, 18.0, 19.0, 20.0]
    ], dtype=np.float64)

    B = np.array([
        [21.0, 22.0],
        [23.0, 24.0],
        [25.0, 26.0],
        [27.0, 28.0],
        [29.0, 30.0]
    ], dtype=np.float64)

    # Prepare pivot array (initialize to zeros)
    jpvt = np.zeros(n, dtype=np.int32)

    # Determine optimal workspace size
    # First, call with lwork = -1 to query optimal workspace
    work = np.zeros(1, dtype=np.float64)
    info = np.zeros(1, dtype=np.int32)
    tau = np.zeros(min(m, n), dtype=np.float64)

    lapack_wrapper.py_dgeqp3rk(
        m, n, nrhs,
        A, m,  # A and its leading dimension
        jpvt,
        tau,
        B, nrhs,  # B and its leading dimension
        work, -1,  # Query optimal workspace size
        info
    )

    # Extract optimal workspace size
    lwork = int(work[0])
    work = np.zeros(lwork, dtype=np.float64)

    # Call the routine with optimal workspace
    lapack_wrapper.py_dgeqp3rk(
        m, n, nrhs,
        A, m,
        jpvt,
        tau,
        B, nrhs,
        work, lwork,
        info
    )

    # Check for successful execution
    if info[0] != 0:
        raise RuntimeError(f"LAPACK routine failed with info = {info[0]}")

    # Return results
    return {
        'A': A,  # Modified matrix A
        'B': B,  # Modified right-hand side
        'jpvt': jpvt,  # Pivot array
        'tau': tau  # Reflectors
    }


# Run the example
results = example_dgeqp3rk()
print("Pivots:", results['jpvt'])
print("Modified A:\n", results['A'])
print("Modified B:\n", results['B'])
print("Reflectors (tau):", results['tau'])