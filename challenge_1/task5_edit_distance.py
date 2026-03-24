"""
Task 5 — K-means with edit distance between images.
Uses K-medoids with edit distance on binary row sequences.
Run on a small subset because edit distance computation is O(N^2 * d).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
import os, time

PLOTS = 'plots'
os.makedirs(PLOTS, exist_ok=True)

# ============================================================
# 1. Load MNIST (small subset for edit distance)
# ============================================================
print("Loading MNIST...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X_all = mnist.data.astype(np.float64) / 255.0
y_all = mnist.target.astype(int)

# Use only 500 images for the edit distance experiment (O(N^2) distance)
N_EDIT = 500
X_small = X_all[:N_EDIT]
y_small = y_all[:N_EDIT]

print(f"  Using {N_EDIT} images for edit distance clustering.")

# ============================================================
# 2. Convert image to symbolic sequence
# ============================================================

def image_to_sequence(image, threshold=0.3):
    """
    Convert a flattened 28×28 image to a binary sequence.
    Each pixel becomes '1' (ink) or '0' (background) based on threshold.
    Read row by row.

    Returns: list of ints (0 or 1), length 784
    """
    return (image > threshold).astype(np.int8).tolist()


def image_to_rle_sequence(image, threshold=0.3):
    """
    Run-length encoding of binary image (row by row).
    Encode as alternating run lengths starting from 0-pixels.
    This produces a much shorter sequence.

    Returns: list of ints (run lengths)
    """
    binary = (image > threshold).astype(np.int8)
    rle = []
    current = binary[0]
    count = 0
    for b in binary:
        if b == current:
            count += 1
        else:
            rle.append(count)
            count = 1
            current = b
    rle.append(count)
    return rle


# ============================================================
# 3. Edit distance (Levenshtein) between sequences
# ============================================================

def edit_distance(seq1, seq2):
    """
    Standard Levenshtein edit distance between two sequences.
    Uses dynamic programming.
    Cost: insert=1, delete=1, substitute=1.
    """
    m, n = len(seq1), len(seq2)
    # Use 1D DP array for memory efficiency
    dp = np.arange(n + 1, dtype=np.float32)
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if seq1[i - 1] == seq2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1.0 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return float(dp[n])


# ============================================================
# 4. Precompute sequences for all images (RLE for speed)
# ============================================================
print("Converting images to RLE sequences...")
sequences = [image_to_rle_sequence(X_small[i]) for i in range(N_EDIT)]
seq_lengths = [len(s) for s in sequences]
print(f"  Sequence lengths: mean={np.mean(seq_lengths):.1f}, "
      f"min={min(seq_lengths)}, max={max(seq_lengths)}")

# ============================================================
# 5. Compute pairwise edit distance matrix (small subset)
# ============================================================
# For demo, compute full pairwise matrix on N_EDIT images.
print("Computing pairwise edit distances (this may take a minute)...")
t0 = time.time()
D_edit = np.zeros((N_EDIT, N_EDIT), dtype=np.float32)
for i in range(N_EDIT):
    if i % 50 == 0:
        print(f"  Row {i}/{N_EDIT}...", flush=True)
    for j in range(i + 1, N_EDIT):
        d = edit_distance(sequences[i], sequences[j])
        D_edit[i, j] = d
        D_edit[j, i] = d
t_edit = time.time() - t0
print(f"  Done in {t_edit:.1f}s")

# ============================================================
# 6. K-medoids implementation
# ============================================================

def kmedoids(D, K, max_iter=50, n_init=3, random_state=42):
    """
    K-medoids clustering given a precomputed distance matrix D.

    Initialization: random medoids.
    Assignment: each point to nearest medoid.
    Update: find point minimizing total distance to all others in cluster.

    Parameters
    ----------
    D    : ndarray, shape (N, N)  — pairwise distances
    K    : int
    max_iter : int
    n_init   : int
    random_state : int

    Returns
    -------
    best_labels   : ndarray, shape (N,)
    best_medoids  : ndarray, shape (K,)  — indices of medoids
    best_obj      : float  — total distance to medoids
    best_history  : list
    """
    N = D.shape[0]
    rng = np.random.default_rng(random_state)
    best_obj = np.inf
    best_labels = None
    best_medoids = None
    best_history = None

    for run in range(n_init):
        # Initialize: random medoids
        medoids = rng.choice(N, K, replace=False)

        history = []
        for it in range(max_iter):
            # Assignment: argmin distance to medoids
            D_med = D[:, medoids]        # (N, K)
            labels = np.argmin(D_med, axis=1)

            # Objective: sum of distances to assigned medoids
            obj = sum(D[i, medoids[labels[i]]] for i in range(N))
            history.append(obj)

            # Update: new medoid = minimizer of within-cluster sum of distances
            new_medoids = np.empty(K, dtype=int)
            for k in range(K):
                mask = np.where(labels == k)[0]
                if len(mask) == 0:
                    new_medoids[k] = medoids[k]
                    continue
                # Sum of distances within cluster
                D_sub = D[np.ix_(mask, mask)]
                sums = D_sub.sum(axis=1)
                new_medoids[k] = mask[np.argmin(sums)]

            if np.all(new_medoids == medoids):
                break
            medoids = new_medoids

        if obj < best_obj:
            best_obj = obj
            best_labels = labels.copy()
            best_medoids = medoids.copy()
            best_history = history

    return best_labels, best_medoids, best_obj, best_history


# ============================================================
# 7. Run K-medoids with edit distance
# ============================================================
K = 5   # use 5 clusters on 500 images
print(f"\n[Task 5] K-medoids with edit distance (K={K}, N={N_EDIT})...")
t0 = time.time()
labels_med, medoids, obj_med, hist_med = kmedoids(D_edit, K, max_iter=50,
                                                    n_init=3, random_state=42)
t_kmed = time.time() - t0
pur_med = 0.0
from kmeans_core import cluster_purity, normalized_mutual_information
pur_med = cluster_purity(labels_med, y_small)
nmi_med = normalized_mutual_information(labels_med, y_small)
print(f"  K-medoids: obj={obj_med:.2f}, purity={pur_med:.4f}, "
      f"NMI={nmi_med:.4f}, t={t_kmed:.1f}s")

# ============================================================
# 8. Compare with Euclidean K-means on same 500 images
# ============================================================
from kmeans_core import kmeans as kmeans_eucl
print(f"\n[Task 5] Euclidean K-means on same {N_EDIT} images (K={K})...")
labels_eu, C_eu, obj_eu, _ = kmeans_eucl(
    X_small, K, max_iter=100, n_init=3, init='kmeans++', random_state=42
)
pur_eu = cluster_purity(labels_eu, y_small)
nmi_eu = normalized_mutual_information(labels_eu, y_small)
print(f"  K-means:   obj={obj_eu:.4f}, purity={pur_eu:.4f}, NMI={nmi_eu:.4f}")

# ============================================================
# 9. Visualize medoid images
# ============================================================
fig, axes = plt.subplots(1, K, figsize=(3 * K, 3.5))
for k, ax in enumerate(axes):
    med_idx = medoids[k]
    n_k = (labels_med == k).sum()
    ax.imshow(X_small[med_idx].reshape(28, 28), cmap='gray', vmin=0, vmax=1)
    ax.set_title(f'Medoid {k}\n(n={n_k}, digit {y_small[med_idx]})', fontsize=9)
    ax.axis('off')
plt.suptitle(f'K-medoids with Edit Distance (K={K}, N={N_EDIT})', fontsize=12)
plt.tight_layout()
plt.savefig(f'{PLOTS}/task5_medoids.png', dpi=120)
plt.close()
print("\nSaved task5_medoids.png")

# ============================================================
# 10. Convergence of K-medoids
# ============================================================
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(hist_med, 'o-', color='purple')
ax.set_xlabel('Iteration')
ax.set_ylabel('Total distance to medoids')
ax.set_title(f'K-medoids convergence (K={K}, edit distance)')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{PLOTS}/task5_kmedoids_convergence.png', dpi=120)
plt.close()
print("Saved task5_kmedoids_convergence.png")

# ============================================================
# 11. Comparison table: edit distance vs Euclidean
# ============================================================
print("\n=== Task 5 comparison ===")
print(f"{'Method':20s} {'Purity':>8} {'NMI':>8} {'Time(s)':>8}")
print(f"{'K-medoids (edit)':20s} {pur_med:>8.4f} {nmi_med:>8.4f} {t_kmed:>8.1f}")
print(f"{'K-means (Eucl.)':20s} {pur_eu:>8.4f} {nmi_eu:>8.4f} {'<1':>8}")

# --- Bar chart comparison ---
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, vals, title in zip(
    axes,
    [[pur_med, pur_eu], [nmi_med, nmi_eu]],
    ['Cluster Purity', 'NMI']
):
    ax.bar(['K-medoids\n(edit dist)', 'K-means\n(Euclidean)'],
           vals, color=['purple', 'steelblue'], edgecolor='black', alpha=0.85)
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)
plt.suptitle(f'Edit distance K-medoids vs Euclidean K-means (K={K}, N={N_EDIT})',
             fontsize=11)
plt.tight_layout()
plt.savefig(f'{PLOTS}/task5_comparison.png', dpi=120)
plt.close()
print("Saved task5_comparison.png")

# ============================================================
# 12. Visualize binary and RLE representations of a few images
# ============================================================
fig, axes = plt.subplots(3, 4, figsize=(12, 9))
for row_idx, img_idx in enumerate([0, 10, 20]):
    ax = axes[row_idx, 0]
    ax.imshow(X_small[img_idx].reshape(28, 28), cmap='gray', vmin=0, vmax=1)
    ax.set_title(f'Original (digit {y_small[img_idx]})', fontsize=8)
    ax.axis('off')

    ax = axes[row_idx, 1]
    binary = (X_small[img_idx] > 0.3).reshape(28, 28)
    ax.imshow(binary, cmap='gray', vmin=0, vmax=1)
    ax.set_title('Binary (threshold 0.3)', fontsize=8)
    ax.axis('off')

    ax = axes[row_idx, 2]
    rle = image_to_rle_sequence(X_small[img_idx])
    ax.bar(range(len(rle)), rle, color='steelblue', width=0.8)
    ax.set_title(f'RLE sequence (len={len(rle)})', fontsize=8)
    ax.set_xlabel('Position'); ax.set_ylabel('Run length')

    ax = axes[row_idx, 3]
    seq_full = image_to_sequence(X_small[img_idx])
    ax.imshow(np.array(seq_full).reshape(28, 28), cmap='gray', vmin=0, vmax=1)
    ax.set_title('Binary (flat) = pixel seq', fontsize=8)
    ax.axis('off')

plt.suptitle('Image representations used in edit distance', fontsize=12)
plt.tight_layout()
plt.savefig(f'{PLOTS}/task5_representations.png', dpi=120)
plt.close()
print("Saved task5_representations.png")

print("\nTask 5 complete.")
