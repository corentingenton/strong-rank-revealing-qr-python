import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.linalg import hadamard

plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'text.latex.preamble': r'\usepackage{amsmath}',
})

plt.rcParams['axes.labelsize'] = 24
plt.rcParams['xtick.labelsize'] = 22
plt.rcParams['ytick.labelsize'] = 22
plt.rcParams['legend.fontsize'] = 20


def compute_ratios(A, C, M, k):
    S_A = scipy.linalg.svd(A, compute_uv=False)
    S_C = scipy.linalg.svd(C, compute_uv=False)
    S_M = scipy.linalg.svd(M, compute_uv=False)

    ratio_MA = S_M[:k] / S_A
    ratio_CM = S_C / S_M[k:]
    return ratio_MA, ratio_CM


def plot_ratios(first_ratios, second_ratios, name='Kahan', mixed=False, MA=True):
    fig, ax = plt.subplots(figsize=(8, 6))
    if mixed:
        default_green = plt.rcParams['axes.prop_cycle'].by_key()['color'][2]
        default_orange = plt.rcParams['axes.prop_cycle'].by_key()['color'][1]
        ax.plot(first_ratios, marker='x', markersize=10, markerfacecolor='None',
                linewidth=2.5, color=default_orange, label='Randomised')

        ax.plot(second_ratios, linewidth=2.2, marker='o', markersize=6, alpha=0.85,
                markerfacecolor='None', color=default_green, label='Mixed Precision')
    else:
        ax.plot(first_ratios, marker='x', markersize=10, markerfacecolor='None',
                linewidth=2.5, label='Regular')
        ax.plot(second_ratios, linewidth=2.2, marker='o', markersize=6, alpha=0.85,
                markerfacecolor='None', label='Randomised')

    if MA:
        ax.set_ylabel(r'$\sigma_i(M)/\sigma_i(A)$')
        ax.set_xlabel(r'Index $i$')
    else:
        ax.set_ylabel(r'$\sigma_j(C)/\sigma_{k+j}(M)$')
        ax.set_xlabel(r'Index $j$')

    if name == 'Kahan':
        ax.set_yscale('log')
    ax.legend(loc='best')
    ax.grid()
    plt.tight_layout()

    return fig


def plot_singular_values(M, sketched=False):
    if sketched:
        M_sk = sketching(M)
        S_M = scipy.linalg.svd(M_sk, compute_uv=False)
    else:
        S_M = scipy.linalg.svd(M, compute_uv=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(S_M, linewidth=2.5, label='Singular Values')
    ax.set_ylabel(r'$\sigma_i(M)$')
    ax.set_yscale('log')
    ax.set_xlabel(r'Index $i$')
    ax.grid()

    return fig


def sketching(M, mixed=False):
    np.random.seed(42)
    m, n = M.shape
    d = 5 * n

    dtype = np.float32 if mixed else np.float64

    D = np.diag(np.random.choice([-1, 1], size=m).astype(dtype))
    H = hadamard(m).astype(dtype) / np.sqrt(m)
    sampled_rows = np.random.choice(m, size=d, replace=True)

    HDM = np.dot(H, np.dot(D, M.astype(dtype)))

    M_sk = np.sqrt(m / d) * HDM[sampled_rows, :]
    return M_sk.astype(np.float64)


def plot_runtime_comparison(time_devils, time_devils_rand, time_stewart,
                            time_stewart_rand, time_kahan, time_kahan_rand, scipy_comparison=False):
    matrix = ("Devil's stairs", "Stewart", "Kahan")

    time_means = {

        'regular': [np.mean(time_devils), np.mean(time_stewart), np.mean(time_kahan)],
        'randomized': [np.mean(time_devils_rand), np.mean(time_stewart_rand), np.mean(time_kahan_rand)]
    }

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5), layout='constrained')
    axes = [ax1, ax2, ax3]

    for i, matrix_name in enumerate(matrix):
        ax = axes[i]
        regular_time = time_means['regular'][i]
        randomized_time = time_means['randomized'][i]

        rects1 = ax.bar(0, regular_time, label='Regular')
        rects2 = ax.bar(1, randomized_time, label='Randomized')

        # Increase the font size of the bar labels
        ax.bar_label(rects1, fmt='%.3f', fontsize=14)  # Larger font size for the labels
        ax.bar_label(rects2, fmt='%.3f', fontsize=14)  # Larger font size for the labels

        ax.set_title(matrix_name, fontsize=16)
        ax.set_ylabel('Time (s)', fontsize=16)
        ax.set_xticks([0, 1])  # Set tick positions
        ax.set_xlim(-0.5, 1.5)
        if scipy_comparison:
            ax.set_xticklabels(['Regular', 'Scipy QR'], fontsize=16)
        else:
            ax.set_xticklabels(['Regular', 'Randomized'], fontsize=16)

        ax.tick_params(axis='both', which='major', labelsize=14)

    return fig


def plot_runtime_comparison_by_parts(time_devils, time_devils_rand_parts,
                             time_stewart, time_stewart_rand_parts,
                             time_kahan, time_kahan_rand_parts):

    matrix_names = ["Devil's stairs", "Stewart", "Kahan"]

    # Calculate means for regular times
    regular_times = [np.mean(time_devils), np.mean(time_stewart), np.mean(time_kahan)]

    # Calculate means for randomized time parts
    randomized_times_parts = [
        [np.mean(time_devils_rand_parts['Sketch']), np.mean(time_devils_rand_parts['Algo']),
         np.mean(time_devils_rand_parts['QR'])],
        [np.mean(time_stewart_rand_parts['Sketch']), np.mean(time_stewart_rand_parts['Algo']),
         np.mean(time_stewart_rand_parts['QR'])],
        [np.mean(time_kahan_rand_parts['Sketch']), np.mean(time_kahan_rand_parts['Algo']),
         np.mean(time_kahan_rand_parts['QR'])]
    ]

    fig, axes = plt.subplots(1, 3, figsize=(10, 5.5))

    part_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for i, (matrix_name, ax) in enumerate(zip(matrix_names, axes)):
        # Plot regular time as a single bar
        rects1 = ax.bar(0, regular_times[i], color='#d62728', alpha=0.8, label='Regular')

        # Plot randomized times as a stacked bar
        part1, part2, part3 = randomized_times_parts[i]
        bottom1 = part1
        bottom2 = part1 + part2
        total_time = part1 + part2 + part3

        rects2 = ax.bar(1, part1, color=part_colors[0], label='Sketching')
        rects3 = ax.bar(1, part2, bottom=bottom1, color=part_colors[1], label='Algo')
        rects4 = ax.bar(1, part3, bottom=bottom2, color=part_colors[2], label='QR')

        # Add in-bar labels for the other parts
        ax.text(1, part1 / 2, f'{part1:.3f}', ha='center', va='center', fontsize=14, color='black', fontweight='bold')
        ax.text(1, bottom1 + part2 / 2, f'{part2:.3f}', ha='center', va='center', fontsize=14, color='black',
                fontweight='bold')
        ax.text(1, bottom2 + part3 / 2, f'{part3:.3f}', ha='center', va='center', fontsize=14, color='black',
                fontweight='bold')

        ax.bar_label(rects1, fmt='%.3f', fontsize=14)
        ax.bar_label(rects4, fmt='%.3f', fontsize=14)

        ax.set_title(matrix_name, fontsize=16)
        ax.set_ylabel('Time (s)', fontsize=16)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Regular', 'Randomized'], fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_xlim(-0.5, 1.5)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(labels), fontsize=14, bbox_to_anchor=(0.5, -.05),
               fancybox=True)

    plt.tight_layout()

    return fig


def compute_theoretical_bound(k, n, f, eps, sigma_l_plus_1):
    f_tilda = np.sqrt((1 + eps) / (1 - eps)) * np.sqrt(2) * f
    bound = sigma_l_plus_1 * (1 + np.sqrt(1 + (f_tilda ** 2) * k * (n - k)))
    return bound


def plot_low_rank_approximation(svd_error, error, array_of_l, bounds):

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(array_of_l, error, marker='x', markersize=12, label='Rand QDP')
    ax.plot(array_of_l, svd_error, linewidth=2.5, label='SVD')
    ax.plot(array_of_l, bounds, linewidth=2.5, label='Theoretical bound')

    ax.set_xlabel(r'Low rank $l$')
    ax.set_ylabel(r'$\| \hat{M}_l - M \|_2$')
    ax.set_yscale('log')
    ax.legend(loc='best')
    ax.grid()
    fig.tight_layout()

    return fig


def truncated_svd(M, rank_l):
    U, D_l, V_T = scipy.sparse.linalg.svds(M, k=rank_l)
    D_l = D_l[::-1]
    U = U[:, ::-1]
    V_T = V_T[::-1, :]
    return U, D_l, V_T


def plot_runtime_comparison_with_mixed_parts(time_devils_rand_parts, time_devils_mixed_parts,
                                             time_stewart_rand_parts, time_stewart_mixed_parts,
                                             time_kahan_rand_parts, time_kahan_mixed_parts):

    matrix_names = ["Devil's stairs", "Stewart", "Kahan"]

    # Calculate means for each category
    randomized_times_parts = [
        [np.mean(time_devils_rand_parts['Sketch']), np.mean(time_devils_rand_parts['Algo']),
         np.mean(time_devils_rand_parts['QR'])],
        [np.mean(time_stewart_rand_parts['Sketch']), np.mean(time_stewart_rand_parts['Algo']),
         np.mean(time_stewart_rand_parts['QR'])],
        [np.mean(time_kahan_rand_parts['Sketch']), np.mean(time_kahan_rand_parts['Algo']),
         np.mean(time_kahan_rand_parts['QR'])]
    ]
    mixed_precision_times_parts = [
        [np.mean(time_devils_mixed_parts['Sketch']), np.mean(time_devils_mixed_parts['Algo']),
         np.mean(time_devils_mixed_parts['QR'])],
        [np.mean(time_stewart_mixed_parts['Sketch']), np.mean(time_stewart_mixed_parts['Algo']),
         np.mean(time_stewart_mixed_parts['QR'])],
        [np.mean(time_kahan_mixed_parts['Sketch']), np.mean(time_kahan_mixed_parts['Algo']),
         np.mean(time_kahan_mixed_parts['QR'])]
    ]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5.5))
    axes = [ax1, ax2, ax3]

    part_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for i, (matrix_name, ax) in enumerate(zip(matrix_names, axes)):

        # Plot randomized times as a stacked bar
        part1_rand, part2_rand, part3_rand = randomized_times_parts[i]
        bottom1_rand = part1_rand
        bottom2_rand = part1_rand + part2_rand
        total_time_rand = part1_rand + part2_rand + part3_rand

        ax.bar(0, part1_rand, color=part_colors[0], label='Sketching')
        ax.bar(0, part2_rand, bottom=bottom1_rand, color=part_colors[1], label='Algo')
        rects4 = ax.bar(0, part3_rand, bottom=bottom2_rand, color=part_colors[2], label='QR')

        part1_mixed, part2_mixed, part3_mixed = mixed_precision_times_parts[i]
        bottom1_mixed = part1_mixed
        bottom2_mixed = part1_mixed + part2_mixed
        total_time_mixed = part1_mixed + part2_mixed + part3_mixed

        ax.bar(1, part1_mixed, color=part_colors[0])
        ax.bar(1, part2_mixed, bottom=bottom1_mixed, color=part_colors[1])
        rects7 = ax.bar(1, part3_mixed, bottom=bottom2_mixed, color=part_colors[2])

        ax.text(0, part1_rand / 2, f'{part1_rand:.3f}', ha='center', va='center', fontsize=14,
                color='black',
                fontweight='bold')
        ax.text(0, bottom1_rand + part2_rand / 2, f'{part2_rand:.3f}', ha='center', va='center', fontsize=14,
                color='black',
                fontweight='bold')
        ax.text(0, bottom2_rand + part3_rand / 2, f'{part3_rand:.3f}', ha='center', va='center', fontsize=14,
                color='black',
                fontweight='bold')
        ax.text(1, part1_mixed / 2, f'{part1_mixed:.3f}', ha='center', va='center', fontsize=14,
                color='black',
                fontweight='bold')
        ax.text(1, bottom1_mixed + part2_mixed / 2, f'{part2_mixed:.3f}', ha='center', va='center', fontsize=14,
                color='black',
                fontweight='bold')
        ax.text(1, bottom2_mixed + part3_mixed / 2, f'{part3_mixed:.3f}', ha='center', va='center', fontsize=14,
                color='black',
                fontweight='bold')



        ax.bar_label(rects4, fmt='%.3f', fontsize=14)
        ax.bar_label(rects7, fmt='%.3f', fontsize=14)

        # Set labels and titles
        ax.set_title(matrix_name, fontsize=16)
        ax.set_ylabel('Time (s)', fontsize=16)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Randomized', 'Mixed Prec.'], fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_xlim(-0.5, 1.5)

    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(labels), fontsize=14, bbox_to_anchor=(0.5, -.05),
               fancybox=True)

    plt.tight_layout()
    return fig
