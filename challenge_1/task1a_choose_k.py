"""
Task 1a — How to choose K?
Elbow curve, silhouette, purity, stability.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
import os

from kmeans_core import (
    kmeans, assign_clusters, compute_objective,
    cluster_purity, normalized_mutual_information,
    silhouette_score_fast
)

PLOTS = 'plots'
os.makedirs(PLOTS, exist_ok=True)

# ============================================================
# Load MNIST (use a subset for silhouette speed)
# ============================================================
print("Loading MNIST...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X_all = mnist.data.astype(np.float64) / 255.0
y_all = mnist.target.astype(int)

X = X_all[:60000]
y = y_all[:60000]

# Subset for silhouette (expensive O(n^2) per cluster)
sil_size = 3000
rng = np.random.default_rng(0)
sil_idx = rng.choice(len(X), sil_size, replace=False)
X_sil = X[sil_idx]
y_sil = y[sil_idx]

K_values = [2, 5, 8, 10, 12, 15, 20, 25, 30]

objectives   = []
purities     = []
nmis         = []
silhouettes  = []
stab_stds    = []

print("\n[Task 1a] Scanning K values...")
for K in K_values:
    print(f"  K={K}...", end=' ', flush=True)

    # Main run (3 restarts)
    labels, centroids, obj, hist = kmeans(
        X, K, max_iter=150, n_init=3, init='kmeans++', random_state=42
    )
    objectives.append(obj)
    purities.append(cluster_purity(labels, y))
    nmis.append(normalized_mutual_information(labels, y))

    # Silhouette on subset
    labels_sil = assign_clusters(X_sil, centroids)
    sil = silhouette_score_fast(X_sil, labels_sil, sample_size=sil_size)
    silhouettes.append(sil)

    # Stability: run 4 independent restarts, measure std of objectives
    run_objs = []
    for seed in range(4):
        _, _, o, _ = kmeans(X, K, max_iter=100, n_init=1,
                            init='kmeans++', random_state=seed * 7)
        run_objs.append(o)
    stab_stds.append(np.std(run_objs))

    print(f"obj={obj:.3e}, purity={purities[-1]:.3f}, "
          f"NMI={nmis[-1]:.3f}, sil={sil:.3f}, std={stab_stds[-1]:.2e}")

# ============================================================
# Plots
# ============================================================

# --- Elbow curve ---
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(K_values, objectives, 'o-', color='steelblue')
ax.set_xlabel('K (number of clusters)')
ax.set_ylabel('Within-cluster SSE')
ax.set_title('Elbow Curve')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{PLOTS}/task1a_elbow.png', dpi=120)
plt.close()
print("\nSaved task1a_elbow.png")

# --- Silhouette score ---
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(K_values, silhouettes, 's-', color='darkorange')
ax.set_xlabel('K')
ax.set_ylabel('Silhouette score')
ax.set_title('Silhouette Score vs K')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{PLOTS}/task1a_silhouette.png', dpi=120)
plt.close()
print("Saved task1a_silhouette.png")

# --- Purity & NMI ---
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(K_values, purities, 'o-', label='Purity', color='green')
ax.plot(K_values, nmis, 's--', label='NMI', color='purple')
ax.set_xlabel('K')
ax.set_ylabel('Score')
ax.set_title('Cluster Purity and NMI vs K')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{PLOTS}/task1a_purity_nmi.png', dpi=120)
plt.close()
print("Saved task1a_purity_nmi.png")

# --- Stability (std of objective across runs) ---
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(K_values, stab_stds, 'd-', color='crimson')
ax.set_xlabel('K')
ax.set_ylabel('Std of objective (4 runs)')
ax.set_title('Stability vs K')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{PLOTS}/task1a_stability.png', dpi=120)
plt.close()
print("Saved task1a_stability.png")

# --- Combined 4-panel figure ---
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].plot(K_values, objectives, 'o-', color='steelblue')
axes[0, 0].set_title('Elbow Curve (SSE)')
axes[0, 0].set_xlabel('K'); axes[0, 0].set_ylabel('SSE')
axes[0, 0].grid(alpha=0.3)

axes[0, 1].plot(K_values, silhouettes, 's-', color='darkorange')
axes[0, 1].set_title('Silhouette Score')
axes[0, 1].set_xlabel('K'); axes[0, 1].set_ylabel('Score')
axes[0, 1].grid(alpha=0.3)

axes[1, 0].plot(K_values, purities, 'o-', label='Purity', color='green')
axes[1, 0].plot(K_values, nmis, 's--', label='NMI', color='purple')
axes[1, 0].set_title('Purity & NMI (external, uses labels)')
axes[1, 0].set_xlabel('K'); axes[1, 0].set_ylabel('Score')
axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

axes[1, 1].plot(K_values, stab_stds, 'd-', color='crimson')
axes[1, 1].set_title('Stability (std of objective, 4 runs)')
axes[1, 1].set_xlabel('K'); axes[1, 1].set_ylabel('Std(SSE)')
axes[1, 1].grid(alpha=0.3)

plt.suptitle('K Selection Methods — MNIST', fontsize=13)
plt.tight_layout()
plt.savefig(f'{PLOTS}/task1a_combined.png', dpi=120)
plt.close()
print("Saved task1a_combined.png")

print("\nTask 1a complete.")
