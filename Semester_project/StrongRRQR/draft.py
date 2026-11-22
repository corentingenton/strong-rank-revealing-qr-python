
def householder_reflection(x):
    # Ensure x is a numpy array
    x = np.asarray(x)

    # Create a unit vector in the first dimension
    e = np.zeros_like(x)
    e[0] = 1

    # Compute the norm of the input vector
    x_norm = np.linalg.norm(x)

    # Determine the sign for the reflection
    sign = -1 if x[0] < 0 else 1

    # Compute the reflection vector
    # Use x + sign * ||x|| * e to create a vector pointing to the reflection
    v = x + sign * x_norm * e

    # Normalize the reflection vector
    v = v / np.linalg.norm(v)

    # Compute the Householder reflection matrix
    # H = I - 2 * (v * v^T)
    R = np.eye(len(x)) - 2 * np.outer(v, v)

    # Apply the reflection 14 the original vector
    return R @ x  # np.round(R @ x, 14)



"""
def householder_first_column(A):

    m, n = A.shape
    x = A[:m, 0].copy()
    x_norm = np.linalg.norm(x)

    e1 = np.zeros_like(x)
    e1[0] = x_norm

    v = x - e1
    v_norm = np.linalg.norm(v)

    if v_norm == 0:
        # No transformation needed
        return A.copy()

    # Normalize v
    v = v / v_norm

    # Compute Householder matrix H = I - 2vv^T
    H = np.eye(m) - 2 * np.outer(v, v.T)
    transformed_A = H @ A

    return transformed_A


def householder_reflection_first_column_optimized(A):
    A = A.copy().astype(float)  # Ensure float operations
    m, n = A.shape

    # Work with the first column
    x = A[:, 0]

    # Compute norm and sign to avoid instability
    x_norm = np.linalg.norm(x)
    if x_norm == 0:
        return A

    sign = np.sign(x[0]) if x[0] != 0 else 1
    x[0] += sign * x_norm  # In-place modification to form `u`
    x /= np.linalg.norm(x)  # Normalize `x`

    # Apply the transformation in-place
    w = x @ A  # Vector-matrix product
    A -= 2.0 * np.outer(x, w)  # Update A in-place

    return A



def householder_version_3(A):
    x = A[:, 0].copy()
    u = x.copy()
    nu = np.linalg.norm(u)

    if nu == 0:
        u = np.zeros_like(x)
    else:
        u = x / nu
        if u[0] >= 0:
            u[0] += 1
            nu = -nu
        else:
            u[0] -= 1

        u = u / np.linalg.norm(u)

    H = np.eye(x.shape[0]) - 2 * np.outer(u, u)

    A_sub = A[:, 1:]
    for col in range(A_sub.shape[1]):
        y = A_sub[:, col].copy()
        tau = np.dot(u.T, y)
        tmp = y - tau * u
        A_sub[:, col] = tmp

    # Update the original matrix A
    A[:, 1:] = A_sub
    A[:, 0] = np.dot(H, x)  # Apply H to the first column

    return A


def householder(A):
    n, m = A.shape
    R = np.copy(A)
    y = R[:, 0]
    e1 = np.zeros(n)
    e1[0] = 1
    z = np.sign(y[0]) * np.linalg.norm(y) * e1 + y
    z = z / np.linalg.norm(z)

    R = R - 2 * np.outer(z, z) @ R
    return R
"""


def householder_reflection_first_column(A):

    A = A.copy().astype(float)

    m = A.shape[0]
    x = A[:, 0].copy()

    x_norm = np.linalg.norm(x)

    if x_norm == 0:
        return A

    sign = np.sign(x[0]) if x[0] != 0 else 1

    u = x.copy()
    u[0] += sign * np.linalg.norm(x)

    u_norm = np.linalg.norm(u)
    u /= u_norm

    A -= 2.0 * np.outer(u, np.dot(u, A))

    return A



def householder_first_column(A):
    m = A.shape[0]
    x = A[1:, 0].copy()
    if np.all(x == 0):
        return A

    alpha = -np.sign(A[0, 0]) * np.linalg.norm(A[:, 0])
    v = A[:, 0].copy()
    v[0] = v[0] - alpha
    v = v / np.linalg.norm(v)

    A = A - 2 * np.outer(v, (v.T @ A))
    A[0, 0] = abs(alpha)
    A[1:, 0] = 0

    return A


def householder_first_column2(A):
    """
    Applies a Householder transformation to zero out the elements below
    the diagonal in the first column of A while ensuring the diagonal element is positive.

    Parameters:
        A (numpy.ndarray): The input matrix.

    Returns:
        numpy.ndarray: The matrix after applying the Householder transformation.
    """
    m, n = A.shape
    x = A[:, 0]
    norm_x = np.linalg.norm(x)

    # Ensure the diagonal element is positive
    sign = 1 if x[0] >= 0 else -1
    rho = sign * norm_x

    # Householder vector
    v = x.copy()
    v[0] += rho
    v /= np.linalg.norm(v)

    # Apply the transformation
    A = A - 2 * np.outer(v, v @ A)
    return A



"""
        # Add an arrow for the Sketching time with text
        ax.annotate(f'{part1:.3f}', xy=(1, part1 / 2), xytext=(0.3, 0.05),
                    textcoords='axes fraction', ha='center', va='center',
                    fontsize=12, color='black', fontweight='bold',
                    arrowprops=dict(arrowstyle="->", color='black', lw=1))
"""

"""
# Plot randomized times as a stacked bar
rand_parts = randomized_times_parts[i]
rand_bottoms = np.cumsum([0] + rand_parts[:-1])
randomized_bars = []
for part, bottom, color in zip(rand_parts, rand_bottoms, part_colors):
    randomized_bars.append(ax.bar(1, part, bottom=bottom, color=color))

# Add in-bar labels for randomized parts
for bar, part, bottom in zip(randomized_bars, rand_parts, rand_bottoms):
    ax.text(1, bottom + part / 2, f'{part:.2f}', ha='center', va='center', fontsize=10, color='black')

# Plot mixed-precision times as a stacked bar
mixed_parts = mixed_precision_times_parts[i]
mixed_bottoms = np.cumsum([0] + mixed_parts[:-1])
mixed_bars = []
for part, bottom, color in zip(mixed_parts, mixed_bottoms, part_colors):
    mixed_bars.append(ax.bar(2, part, bottom=bottom, color=color))

# Add in-bar labels for mixed-precision parts
for bar, part, bottom in zip(mixed_bars, mixed_parts, mixed_bottoms):
    ax.text(2, bottom + part / 2, f'{part:.2f}', ha='center', va='center', fontsize=10, color='black')
"""


def plot_runtime_comparison_with_mixed_parts2(time_devils, time_devils_rand_parts, time_devils_mixed_parts,
                                              time_stewart, time_stewart_rand_parts, time_stewart_mixed_parts,
                                              time_kahan, time_kahan_rand_parts, time_kahan_mixed_parts):
    """
    Plots runtime comparison with regular, randomized, and mixed-precision times. Randomized and mixed-precision times
    are split into three parts (Sketching, Algo, QR) and displayed as stacked bars.

    Args:
        time_devils: Regular times for Devil's stairs.
        time_devils_rand_parts: A list of tuples (Sketching, Algo, QR) for randomized Devil's stairs.
        time_devils_mixed_parts: A list of tuples (Sketching, Algo, QR) for mixed-precision Devil's stairs.
        time_stewart: Regular times for Stewart matrix.
        time_stewart_rand_parts: A list of tuples (Sketching, Algo, QR) for randomized Stewart matrix.
        time_stewart_mixed_parts: A list of tuples (Sketching, Algo, QR) for mixed-precision Stewart matrix.
        time_kahan: Regular times for Kahan matrix.
        time_kahan_rand_parts: A list of tuples (Sketching, Algo, QR) for randomized Kahan matrix.
        time_kahan_mixed_parts: A list of tuples (Sketching, Algo, QR) for mixed-precision Kahan matrix.
    """
    matrix_names = ["Devil's stairs", "Stewart", "Kahan"]

    # Helper function to compute means for each part
    def mean_parts(trials):
        parts = list(zip(*trials))  # Transpose to separate parts
        return [np.mean(part) for part in parts]

    # Calculate means for each category
    regular_times = [np.mean(time_devils), np.mean(time_stewart), np.mean(time_kahan)]

    randomized_times_parts = [
        mean_parts(time_devils_rand_parts),
        mean_parts(time_stewart_rand_parts),
        mean_parts(time_kahan_rand_parts),
    ]
    mixed_precision_times_parts = [
        mean_parts(time_devils_mixed_parts),
        mean_parts(time_stewart_mixed_parts),
        mean_parts(time_kahan_mixed_parts),
    ]

    # Create figure with more horizontal space
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), layout='constrained')

    # Colors for the parts
    part_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    part_labels = ['Sketching', 'Algo', 'QR']

    for i, (matrix_name, ax) in enumerate(zip(matrix_names, axes)):
        # Plot regular time as a single bar
        rects1 = ax.bar(0, regular_times[i], color='#1f77b4', alpha=0.7, label='Regular')

        # Plot randomized times as a stacked bar
        if i == 0:  # Devil's stairs
            part1_rand, part2_rand, part3_rand = randomized_times_parts[i]
            bottom1_rand = 0
            ax.bar(1, part1_rand, color=part_colors[0], label='Sketching')
            ax.bar(1, part2_rand, bottom=part1_rand, color=part_colors[1], label='Algo')
            ax.bar(1, part3_rand, bottom=part1_rand + part2_rand, color=part_colors[2], label='QR')

        if i == 1:  # Stewart
            part1_rand, part2_rand, part3_rand = randomized_times_parts[i]
            bottom1_rand = 0
            ax.bar(1, part1_rand, color=part_colors[0], label='Sketching')
            ax.bar(1, part2_rand, bottom=part1_rand, color=part_colors[1], label='Algo')
            ax.bar(1, part3_rand, bottom=part1_rand + part2_rand, color=part_colors[2], label='QR')

        if i == 2:  # Kahan
            part1_mixed, part2_mixed, part3_mixed = mixed_precision_times_parts[i]
            bottom1_mixed = 0
            ax.bar(2, part1_mixed, color=part_colors[0], label='Sketching')
            ax.bar(2, part2_mixed, bottom=part1_mixed, color=part_colors[1], label='Algo')
            ax.bar(2, part3_mixed, bottom=part1_mixed + part2_mixed, color=part_colors[2], label='QR')

        # Add label to regular bar
        ax.bar_label(rects1, fmt='%.2f', fontsize=10)

        # Add in-bar labels for the other parts
        if i < 2:  # Randomized
            ax.text(1, part1_rand + part2_rand / 2, f'{part2_rand:.3f}', ha='center', va='center', fontsize=12,
                    color='black', fontweight='bold')
            ax.text(1, part1_rand + part2_rand + part3_rand / 2, f'{part3_rand:.3f}', ha='center', va='center',
                    fontsize=12, color='black', fontweight='bold')

        if i == 2:  # Mixed Precision for Kahan
            ax.text(2, part1_mixed + part2_mixed / 2, f'{part2_mixed:.3f}', ha='center', va='center', fontsize=12,
                    color='black', fontweight='bold')
            ax.text(2, part1_mixed + part2_mixed + part3_mixed / 2, f'{part3_mixed:.3f}', ha='center', va='center',
                    fontsize=12, color='black', fontweight='bold')

        # Set labels and titles
        ax.set_title(matrix_name, fontsize=14)
        ax.set_ylabel('Time (s)', fontsize=14)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['Regular', 'Randomized', 'Mixed Precision'], fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_xlim(-0.5, 2.5)  # Adjust x-axis limits to create more space

    # Add a legend closer to the plots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(labels), fontsize=10, bbox_to_anchor=(0.5, -.05),
               fancybox=True)

    plt.suptitle('Time Comparison for Different Matrices', fontsize=16)

    # Adjust layout to make room for legend
    plt.tight_layout()

    return plt




def plot_runtime_comparison_with_mixed(time_devils, time_devils_rand, time_devils_mixed,
                                       time_stewart, time_stewart_rand, time_stewart_mixed,
                                       time_kahan, time_kahan_rand, time_kahan_mixed):
    matrix = ("Devil's stairs", "Stewart", "Kahan")

    time_means = {
        'regular': [np.mean(time_devils), np.mean(time_stewart), np.mean(time_kahan)],
        'randomized': [np.mean(time_devils_rand), np.mean(time_stewart_rand), np.mean(time_kahan_rand)],
        'mixed_precision': [np.mean(time_devils_mixed), np.mean(time_stewart_mixed), np.mean(time_kahan_mixed)],
    }

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5), layout='constrained')
    axes = [ax1, ax2, ax3]

    for i, matrix_name in enumerate(matrix):
        ax = axes[i]
        regular_time = time_means['regular'][i]
        randomized_time = time_means['randomized'][i]
        mixed_precision_time = time_means['mixed_precision'][i]

        rects1 = ax.bar(0, regular_time, label='Regular')
        rects2 = ax.bar(1, randomized_time, label='Randomized')
        rects3 = ax.bar(2, mixed_precision_time, label='Mixed Precision')

        ax.bar_label(rects1, fmt='%.3f', fontsize=14)
        ax.bar_label(rects2, fmt='%.3f', fontsize=14)
        ax.bar_label(rects3, fmt='%.3f', fontsize=14)

        ax.set_title(matrix_name, fontsize=16)
        ax.set_ylabel('Time (s)', fontsize=16)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['Regular', 'Randomized', 'Mixed Precision'], fontsize=14)  # Set tick labels and font size
        #ax.legend(loc='lower left')
        ax.tick_params(axis='both', which='major', labelsize=14)  # length=8, width=1.5, )  # Major ticks
        # ax.set_yscale('log')

    plt.suptitle('Time Comparison for Different Matrices', fontsize=16)
    return plt