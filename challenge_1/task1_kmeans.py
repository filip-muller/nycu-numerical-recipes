"""
Task 1 — Basic K-means on MNIST
Runs experiments, saves all plots to plots/
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.datasets import fetch_openml
import os, time

from kmeans_core import (
    kmeans, assign_clusters, compute_objective,
    cluster_purity, normalized_mutual_information,
    silhouette_score_fast, cluster_entropy
)

PLOTS = 'plots'
os.makedirs(PLOTS, exist_ok=True)

# ============================================================
# 1. Load MNIST
# ============================================================
print("Loading MNIST...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X_raw = mnist.data.astype(np.float64)   # (70000, 784)
y_all = mnist.target.astype(int)        # (70000,)

# Normalize to [0, 1]
X_all = X_raw / 255.0

# Use ~60 000 training samples (MNIST standard split is 60k train / 10k test)
X = X_all[:60000]
y = y_all[:60000]

print(f"  Dataset shape: {X.shape}, labels: {np.unique(y)}")

# ============================================================
# 2. Convergence plot — objective vs iteration for K=10
# ============================================================
print("\n[Task 1] Objective vs iteration for K=10 (random vs kmeans++)")

K_demo = 10

labels_rand, C_rand, obj_rand, hist_rand = kmeans(
    X, K_demo, max_iter=100, n_init=1, init='random', random_state=0
)
labels_pp, C_pp, obj_pp, hist_pp = kmeans(
    X, K_demo, max_iter=100, n_init=1, init='kmeans++', random_state=0
)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(hist_rand, label=f'Random init (final={obj_rand:.2e})', marker='o', markersize=3)
ax.plot(hist_pp, label=f'K-means++ init (final={obj_pp:.2e})', marker='s', markersize=3)
ax.set_xlabel('Iteration')
ax.set_ylabel('Within-cluster SSE')
ax.set_title('K-means convergence (K=10, MNIST)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{PLOTS}/task1_convergence.png', dpi=120)
plt.close()
print("  Saved task1_convergence.png")

# ============================================================
# 3. Run K-means for multiple K values
# ============================================================
K_values = [5, 10, 15, 20, 30]
results = {}

for K in K_values:
    print(f"\n  K={K} ...", end=' ', flush=True)
    t0 = time.time()
    labels, centroids, obj, hist = kmeans(
        X, K, max_iter=150, n_init=3, init='kmeans++', random_state=42
    )
    dt = time.time() - t0
    purity = cluster_purity(labels, y)
    nmi = normalized_mutual_information(labels, y)
    results[K] = dict(labels=labels, centroids=centroids, obj=obj,
                      hist=hist, purity=purity, nmi=nmi, time=dt)
    print(f"obj={obj:.3e}, purity={purity:.3f}, NMI={nmi:.3f}, t={dt:.1f}s")

# ============================================================
# 4. Visualize centroids for K=10
# ============================================================
print("\n[Task 1] Visualizing centroids for K=10")
C10 = results[10]['centroids']

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for k, ax in enumerate(axes.flat):
    ax.imshow(C10[k].reshape(28, 28), cmap='gray', vmin=0, vmax=1)
    ax.set_title(f'Centroid {k}')
    ax.axis('off')
plt.suptitle('K-means Centroids (K=10, pixel domain)', fontsize=13)
plt.tight_layout()
plt.savefig(f'{PLOTS}/task1_centroids_k10.png', dpi=120)
plt.close()
print("  Saved task1_centroids_k10.png")

# ============================================================
# 5. Show nearest images to each centroid (K=10)
# ============================================================
print("[Task 1] Nearest images to centroids for K=10")
labels10 = results[10]['labels']
n_show = 5   # images per cluster

fig, axes = plt.subplots(10, n_show + 1, figsize=(2 * (n_show + 1), 21))
D_full = np.sum((X - C10[labels10]) ** 2, axis=1)  # distance to own centroid

for k in range(10):
    mask = np.where(labels10 == k)[0]
    # Sort by distance to centroid k
    dists_k = np.sum((X[mask] - C10[k]) ** 2, axis=1)
    sorted_idx = mask[np.argsort(dists_k)]
    nearest = sorted_idx[:n_show]

    # First column: centroid
    axes[k, 0].imshow(C10[k].reshape(28, 28), cmap='gray', vmin=0, vmax=1)
    axes[k, 0].set_title('centroid', fontsize=7)
    axes[k, 0].axis('off')
    # Remaining: nearest images
    for j, idx in enumerate(nearest):
        axes[k, j + 1].imshow(X[idx].reshape(28, 28), cmap='gray', vmin=0, vmax=1)
        axes[k, j + 1].set_title(f'digit {y[idx]}', fontsize=7)
        axes[k, j + 1].axis('off')

plt.suptitle('Centroids and nearest images (K=10)', fontsize=11)
plt.tight_layout()
plt.savefig(f'{PLOTS}/task1_nearest_images_k10.png', dpi=100)
plt.close()
print("  Saved task1_nearest_images_k10.png")

# ============================================================
# 6. Sensitivity to initialization: multiple runs, K=10
# ============================================================
print("\n[Task 1] Sensitivity to initialization")
n_runs = 8
objs_random = []
objs_pp = []

for run in range(n_runs):
    _, _, o_r, _ = kmeans(X, 10, max_iter=100, n_init=1,
                          init='random', random_state=run * 100)
    _, _, o_p, _ = kmeans(X, 10, max_iter=100, n_init=1,
                          init='kmeans++', random_state=run * 100)
    objs_random.append(o_r)
    objs_pp.append(o_p)

fig, ax = plt.subplots(figsize=(7, 4))
ax.scatter(range(n_runs), objs_random, label='Random init', s=60, zorder=3)
ax.scatter(range(n_runs), objs_pp, label='K-means++ init', marker='s', s=60, zorder=3)
ax.axhline(np.mean(objs_random), color='C0', linestyle='--', alpha=0.5)
ax.axhline(np.mean(objs_pp), color='C1', linestyle='--', alpha=0.5)
ax.set_xlabel('Run index')
ax.set_ylabel('Final objective (SSE)')
ax.set_title('Initialization sensitivity (K=10, single restart each)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{PLOTS}/task1_init_sensitivity.png', dpi=120)
plt.close()
print("  Saved task1_init_sensitivity.png")

# ============================================================
# 7. Centroid gallery for K=5 and K=20
# ============================================================
for K in [5, 20]:
    C = results[K]['centroids']
    cols = min(K, 10)
    rows = (K + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
    axes = np.array(axes).reshape(-1)
    for k in range(K):
        axes[k].imshow(C[k].reshape(28, 28), cmap='gray', vmin=0, vmax=1)
        axes[k].set_title(f'{k}', fontsize=7)
        axes[k].axis('off')
    for k in range(K, len(axes)):
        axes[k].axis('off')
    plt.suptitle(f'Centroids (K={K})', fontsize=11)
    plt.tight_layout()
    plt.savefig(f'{PLOTS}/task1_centroids_k{K}.png', dpi=120)
    plt.close()
    print(f"  Saved task1_centroids_k{K}.png")

# ============================================================
# 8. Summary table
# ============================================================
print("\n=== Summary ===")
print(f"{'K':>4} {'Objective':>14} {'Purity':>8} {'NMI':>8} {'Time(s)':>8}")
for K in K_values:
    r = results[K]
    print(f"{K:>4} {r['obj']:>14.3e} {r['purity']:>8.4f} {r['nmi']:>8.4f} {r['time']:>8.1f}")

# Save results for other tasks
np.save('plots/task1_results_summary.npy', {
    K: {'obj': results[K]['obj'],
        'purity': results[K]['purity'],
        'nmi': results[K]['nmi']}
    for K in K_values
})

print("\nTask 1 complete.")
