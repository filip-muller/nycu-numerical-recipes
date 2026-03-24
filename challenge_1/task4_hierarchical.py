"""
Task 4 — Hierarchical K-means (2-level).
Compares flat K-means vs hierarchical K-means at similar total leaf clusters.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
import os, time

from kmeans_core import (
    kmeans, assign_clusters, compute_objective,
    cluster_purity, normalized_mutual_information, cluster_entropy
)

PLOTS = 'plots'
os.makedirs(PLOTS, exist_ok=True)

# ============================================================
# 1. Load MNIST
# ============================================================
print("Loading MNIST...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X_all = mnist.data.astype(np.float64) / 255.0
y_all = mnist.target.astype(int)

X = X_all[:60000]
y = y_all[:60000]

# ============================================================
# 2. Hierarchical K-means implementation
# ============================================================

def hierarchical_kmeans(X, K1, K2, max_iter=100, n_init=3,
                        init='kmeans++', random_state=42):
    """
    2-level hierarchical K-means.

    Level 1: cluster all data into K1 groups.
    Level 2: within each group, run K-means with K2.

    Returns
    -------
    leaf_labels   : ndarray, shape (N,)  — final leaf cluster index (0 .. K1*K2-1)
    level1_labels : ndarray, shape (N,)  — coarse cluster index
    level1_centroids: ndarray, shape (K1, d)
    level2_centroids: list of K1 arrays, each shape (K2, d)
    leaf_obj      : float  — total SSE in leaf assignment
    hierarchy     : dict mapping coarse cluster → sub-cluster info
    """
    N, d = X.shape

    # --- Level 1 ---
    print(f"  Level 1: K={K1}...", end=' ', flush=True)
    t0 = time.time()
    labels1, C1, obj1, _ = kmeans(
        X, K1, max_iter=max_iter, n_init=n_init,
        init=init, random_state=random_state
    )
    print(f"done ({time.time()-t0:.1f}s), obj={obj1:.3e}")

    # --- Level 2 ---
    leaf_labels = np.empty(N, dtype=int)
    leaf_offset = 0
    level2_centroids = []
    hierarchy = {}
    total_obj = 0.0

    for k in range(K1):
        mask = np.where(labels1 == k)[0]
        X_sub = X[mask]
        n_sub = len(X_sub)
        # Adapt K2 if cluster is too small
        k2_actual = min(K2, n_sub)

        print(f"  Level 2, parent={k} (n={n_sub}): K={k2_actual}...",
              end=' ', flush=True)
        t1 = time.time()

        if k2_actual < 2:
            sub_labels = np.zeros(n_sub, dtype=int)
            C_sub = X_sub.mean(axis=0, keepdims=True)
            obj_sub = compute_objective(X_sub, sub_labels, C_sub)
        else:
            sub_labels, C_sub, obj_sub, _ = kmeans(
                X_sub, k2_actual, max_iter=max_iter, n_init=2,
                init=init, random_state=random_state + k
            )
        print(f"done ({time.time()-t1:.1f}s), obj={obj_sub:.3e}")

        total_obj += obj_sub
        level2_centroids.append(C_sub)

        # Map sub-labels to global leaf index
        leaf_labels[mask] = sub_labels + leaf_offset
        hierarchy[k] = {
            'mask': mask,
            'sub_labels': sub_labels,
            'centroids': C_sub,
            'obj': obj_sub,
        }
        leaf_offset += k2_actual

    return (leaf_labels, labels1, C1, level2_centroids,
            total_obj, hierarchy, leaf_offset)


# ============================================================
# 3. Run flat K-means and hierarchical K-means
# ============================================================
K1, K2 = 5, 4     # hierarchical: 5 × 4 = 20 leaf clusters
K_flat = K1 * K2  # flat: 20 clusters

print(f"\n[Task 4] Flat K-means (K={K_flat})...")
t0 = time.time()
labels_flat, C_flat, obj_flat, hist_flat = kmeans(
    X, K_flat, max_iter=150, n_init=3, init='kmeans++', random_state=42
)
t_flat = time.time() - t0
pur_flat = cluster_purity(labels_flat, y)
nmi_flat = normalized_mutual_information(labels_flat, y)
print(f"  Flat K={K_flat}: obj={obj_flat:.3e}, purity={pur_flat:.4f}, "
      f"NMI={nmi_flat:.4f}, t={t_flat:.1f}s")

print(f"\n[Task 4] Hierarchical K-means (K1={K1}, K2={K2})...")
t0 = time.time()
(labels_hier, labels_l1, C_l1, C_l2,
 obj_hier, hierarchy, n_leaves) = hierarchical_kmeans(
    X, K1, K2, max_iter=150, n_init=3, random_state=42
)
t_hier = time.time() - t0
pur_hier = cluster_purity(labels_hier, y)
nmi_hier = normalized_mutual_information(labels_hier, y)
print(f"\n  Hierarchical K={n_leaves} leaves: obj={obj_hier:.3e}, "
      f"purity={pur_hier:.4f}, NMI={nmi_hier:.4f}, t={t_hier:.1f}s")

# ============================================================
# 4. Visualize level-1 centroids
# ============================================================
fig, axes = plt.subplots(1, K1, figsize=(3 * K1, 3))
for k, ax in enumerate(axes):
    ax.imshow(C_l1[k].reshape(28, 28), cmap='gray', vmin=0, vmax=1)
    n_k = (labels_l1 == k).sum()
    ax.set_title(f'L1 cluster {k}\n(n={n_k})', fontsize=9)
    ax.axis('off')
plt.suptitle('Hierarchical K-means — Level 1 Centroids', fontsize=12)
plt.tight_layout()
plt.savefig(f'{PLOTS}/task4_level1_centroids.png', dpi=120)
plt.close()
print("\nSaved task4_level1_centroids.png")

# ============================================================
# 5. Visualize level-2 centroids for each parent cluster
# ============================================================
fig, axes = plt.subplots(K1, K2, figsize=(2.5 * K2, 2.5 * K1))
for k in range(K1):
    C_sub = hierarchy[k]['centroids']
    for j in range(K2):
        ax = axes[k, j]
        if j < len(C_sub):
            ax.imshow(C_sub[j].reshape(28, 28), cmap='gray', vmin=0, vmax=1)
            ax.set_title(f'L1={k}, L2={j}', fontsize=7)
        ax.axis('off')
plt.suptitle('Hierarchical K-means — Level 2 Centroids', fontsize=12)
plt.tight_layout()
plt.savefig(f'{PLOTS}/task4_level2_centroids.png', dpi=120)
plt.close()
print("Saved task4_level2_centroids.png")

# ============================================================
# 6. Comparison: flat vs hierarchical
# ============================================================
metrics = ['Objective (×1e6)', 'Purity', 'NMI', 'Time (s)']
flat_vals = [obj_flat / 1e6, pur_flat, nmi_flat, t_flat]
hier_vals = [obj_hier / 1e6, pur_hier, nmi_hier, t_hier]

fig, axes = plt.subplots(1, 4, figsize=(14, 4))
for ax, m, fv, hv in zip(axes, metrics, flat_vals, hier_vals):
    ax.bar(['Flat', 'Hier'], [fv, hv],
           color=['steelblue', 'darkorange'], edgecolor='black', alpha=0.85)
    ax.set_title(m)
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate([fv, hv]):
        ax.text(i, v + 0.01 * abs(v), f'{v:.3f}', ha='center', fontsize=9)
plt.suptitle(f'Flat (K={K_flat}) vs Hierarchical ({K1}×{K2}={n_leaves} leaves)',
             fontsize=12)
plt.tight_layout()
plt.savefig(f'{PLOTS}/task4_comparison.png', dpi=120)
plt.close()
print("Saved task4_comparison.png")

# ============================================================
# 7. Digit distribution inside level-1 clusters
# ============================================================
fig, axes = plt.subplots(1, K1, figsize=(3 * K1, 3.5))
for k, ax in enumerate(axes):
    mask = labels_l1 == k
    cnt = np.bincount(y[mask], minlength=10)
    ax.bar(range(10), cnt / cnt.sum(), color='steelblue', alpha=0.8)
    ax.set_xticks(range(10))
    ax.set_xlabel('Digit')
    ax.set_title(f'L1 cluster {k}')
    ax.set_ylabel('Fraction') if k == 0 else None
    ax.grid(axis='y', alpha=0.3)
plt.suptitle('Digit distribution in Level-1 clusters', fontsize=12)
plt.tight_layout()
plt.savefig(f'{PLOTS}/task4_digit_distribution_l1.png', dpi=120)
plt.close()
print("Saved task4_digit_distribution_l1.png")

# ============================================================
# 8. Additional: K1=10, K2=3 (30 leaves) for deeper hierarchy
# ============================================================
print("\n[Task 4] Extended: K1=10, K2=3 (30 leaves)...")
(labels_30, labels_l1_30, C_l1_30, C_l2_30,
 obj_30, hier_30, n_leaves_30) = hierarchical_kmeans(
    X, 10, 3, max_iter=100, n_init=2, random_state=42
)
pur_30 = cluster_purity(labels_30, y)
nmi_30 = normalized_mutual_information(labels_30, y)
labels_flat30, C_flat30, obj_flat30, _ = kmeans(X, 30, max_iter=100, n_init=2,
                                                init='kmeans++', random_state=42)
pur_flat30 = cluster_purity(labels_flat30, y)
nmi_flat30 = normalized_mutual_information(labels_flat30, y)

print(f"\n  Flat K=30:  obj={obj_flat30:.3e}, purity={pur_flat30:.4f}, NMI={nmi_flat30:.4f}")
print(f"  Hier 10×3: obj={obj_30:.3e}, purity={pur_30:.4f}, NMI={nmi_30:.4f}")

print("\nTask 4 complete.")
