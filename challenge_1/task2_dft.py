"""
Task 2 — K-means in the DFT domain with feature weighting.
Task 2a — Optimizing feature weights.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
import os

from kmeans_core import (
    kmeans, assign_clusters, compute_objective,
    compute_weighted_distance_matrix,
    cluster_purity, normalized_mutual_information
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
# 2. DFT feature extraction
# ============================================================

def dft_features(images, mode='magnitude'):
    """
    Compute 2D DFT of each image and return a feature matrix.

    Parameters
    ----------
    images : ndarray, shape (N, 784) — flattened 28x28 images
    mode   : 'magnitude' | 'phase' | 'both'

    Returns
    -------
    features : ndarray, shape (N, d)
    """
    N = len(images)
    imgs = images.reshape(N, 28, 28)
    F = np.fft.fft2(imgs)   # (N, 28, 28) complex

    if mode == 'magnitude':
        # log(1 + |F|) for better dynamic range
        feats = np.log1p(np.abs(F)).reshape(N, -1)
    elif mode == 'phase':
        feats = np.angle(F).reshape(N, -1)
    elif mode == 'both':
        mag = np.log1p(np.abs(F)).reshape(N, -1)
        phase = np.angle(F).reshape(N, -1)
        feats = np.concatenate([mag, phase], axis=1)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return feats.astype(np.float64)


def build_radial_weights(shape=(28, 28), mode='low', sigma=7.0):
    """
    Build frequency-domain weights for a 28×28 DFT grid.

    mode: 'low'    — Gaussian centred on DC (emphasises low frequencies)
          'high'   — 1 - Gaussian (emphasises high frequencies)
          'band'   — band-pass ring
          'flat'   — all ones (no weighting)
    """
    ky = np.fft.fftfreq(shape[0]) * shape[0]   # -14..13
    kx = np.fft.fftfreq(shape[1]) * shape[1]
    KY, KX = np.meshgrid(ky, kx, indexing='ij')
    R = np.sqrt(KY ** 2 + KX ** 2)             # radial frequency

    if mode == 'low':
        w = np.exp(-R ** 2 / (2 * sigma ** 2))
    elif mode == 'high':
        w = 1.0 - np.exp(-R ** 2 / (2 * sigma ** 2))
    elif mode == 'band':
        center = shape[0] / 4
        w = np.exp(-(R - center) ** 2 / (2 * (sigma / 2) ** 2))
    elif mode == 'flat':
        w = np.ones_like(R)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return w.flatten()   # shape (784,)


# ============================================================
# 3. Compute DFT features
# ============================================================
print("Computing DFT features (magnitude, log-scale)...")
X_dft = dft_features(X, mode='magnitude')   # (60000, 784)

# Standardize DFT features (zero mean, unit variance per feature)
mu_dft = X_dft.mean(axis=0)
std_dft = X_dft.std(axis=0) + 1e-8
X_dft_norm = (X_dft - mu_dft) / std_dft

print(f"  DFT feature shape: {X_dft_norm.shape}")

# ============================================================
# 4. Pixel-domain vs DFT-domain K-means (K=10)
# ============================================================
K = 10
print(f"\n[Task 2] K-means in pixel domain (K={K})...")
labels_px, C_px, obj_px, _ = kmeans(
    X, K, max_iter=150, n_init=3, init='kmeans++', random_state=42
)
pur_px = cluster_purity(labels_px, y)
nmi_px = normalized_mutual_information(labels_px, y)
print(f"  Pixel — obj={obj_px:.3e}, purity={pur_px:.4f}, NMI={nmi_px:.4f}")

print(f"[Task 2] K-means in DFT domain (K={K})...")
labels_dft, C_dft, obj_dft, _ = kmeans(
    X_dft_norm, K, max_iter=150, n_init=3, init='kmeans++', random_state=42
)
pur_dft = cluster_purity(labels_dft, y)
nmi_dft = normalized_mutual_information(labels_dft, y)
print(f"  DFT   — obj={obj_dft:.3e}, purity={pur_dft:.4f}, NMI={nmi_dft:.4f}")

# ============================================================
# 5. Feature-weighted DFT K-means
# ============================================================
weight_modes = ['flat', 'low', 'high', 'band']
w_results = {}

for wmode in weight_modes:
    w = build_radial_weights(mode=wmode, sigma=6.0)
    # Weights apply to normalised features
    print(f"[Task 2] Weighted DFT K-means — weights={wmode}...")
    labels_w, C_w, obj_w, _ = kmeans(
        X_dft_norm, K, max_iter=150, n_init=3,
        init='kmeans++', w=w, random_state=42
    )
    pur_w = cluster_purity(labels_w, y)
    nmi_w = normalized_mutual_information(labels_w, y)
    w_results[wmode] = dict(labels=labels_w, centroids=C_w,
                            obj=obj_w, purity=pur_w, nmi=nmi_w, w=w)
    print(f"  {wmode:6s} — obj={obj_w:.3e}, purity={pur_w:.4f}, NMI={nmi_w:.4f}")

# ============================================================
# 6. Visualize weight masks
# ============================================================
fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
for ax, wmode in zip(axes, weight_modes):
    w = build_radial_weights(mode=wmode, sigma=6.0)
    # Shift DC to center for visualization
    w_img = np.fft.fftshift(w.reshape(28, 28))
    im = ax.imshow(w_img, cmap='hot', vmin=0)
    ax.set_title(f'Weight mask: {wmode}')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
plt.suptitle('DFT Frequency Weight Masks', fontsize=12)
plt.tight_layout()
plt.savefig(f'{PLOTS}/task2_weight_masks.png', dpi=120)
plt.close()
print("\nSaved task2_weight_masks.png")

# ============================================================
# 7. Comparison bar chart
# ============================================================
methods = ['Pixel', 'DFT (flat)'] + [f'DFT ({m})' for m in ['low', 'high', 'band']]
purities = [pur_px, w_results['flat']['purity'],
            w_results['low']['purity'],
            w_results['high']['purity'],
            w_results['band']['purity']]
nmis = [nmi_px, w_results['flat']['nmi'],
        w_results['low']['nmi'],
        w_results['high']['nmi'],
        w_results['band']['nmi']]

x = np.arange(len(methods))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.bar(x, purities, color=['steelblue'] + ['darkorange'] * 4)
ax1.set_xticks(x); ax1.set_xticklabels(methods, rotation=20, ha='right', fontsize=9)
ax1.set_ylabel('Cluster Purity'); ax1.set_title('Purity comparison')
ax1.set_ylim(0, 1); ax1.grid(axis='y', alpha=0.3)

ax2.bar(x, nmis, color=['steelblue'] + ['darkorange'] * 4)
ax2.set_xticks(x); ax2.set_xticklabels(methods, rotation=20, ha='right', fontsize=9)
ax2.set_ylabel('NMI'); ax2.set_title('NMI comparison')
ax2.set_ylim(0, 1); ax2.grid(axis='y', alpha=0.3)

plt.suptitle(f'Pixel vs DFT domain K-means (K={K})', fontsize=12)
plt.tight_layout()
plt.savefig(f'{PLOTS}/task2_comparison.png', dpi=120)
plt.close()
print("Saved task2_comparison.png")

# ============================================================
# 8. Visualize DFT centroids back in image space
# ============================================================
# Note: centroids in DFT-normalised space; to visualize we un-normalise
# and invert the log-magnitude DFT.

def invert_logmag_dft(log_mag_row, shape=(28, 28)):
    """Approximate image from log|F|. Uses zero phase → symmetric image."""
    F_mag = np.expm1(log_mag_row.reshape(shape)) * (std_dft.reshape(shape) + 1e-8) \
            + mu_dft.reshape(shape)
    F_mag = np.maximum(F_mag, 0)
    # Reconstruct with zero phase
    img = np.fft.ifft2(F_mag).real
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for k, ax in enumerate(axes.flat):
    img = invert_logmag_dft(C_dft[k])
    ax.imshow(img, cmap='gray', vmin=0, vmax=1)
    ax.set_title(f'DFT centroid {k}')
    ax.axis('off')
plt.suptitle('DFT-domain K-means Centroids (K=10, inverted to image space)', fontsize=11)
plt.tight_layout()
plt.savefig(f'{PLOTS}/task2_dft_centroids.png', dpi=120)
plt.close()
print("Saved task2_dft_centroids.png")

# ============================================================
# Task 2a — Optimizing weights (alternating optimization)
# ============================================================
print("\n[Task 2a] Optimizing feature weights via alternating gradient search")

# Approach: parameterise weights by sigma of Gaussian low-pass filter.
# Scan sigma values (1D grid search), track purity on a validation subset.
# The "held-out" data here is 10 000 images (indices 50k–60k treated as val).

X_tr2a = X_dft_norm[:50000]
y_tr2a = y[:50000]
X_va2a = X_dft_norm[50000:]
y_va2a = y[50000:]

sigmas = [2.0, 4.0, 6.0, 8.0, 11.0, 14.0]
best_sigma = None
best_nmi_va = -1
results_2a = []

for sigma in sigmas:
    w = build_radial_weights(mode='low', sigma=sigma)
    labels_tr_2a, C_2a, _, _ = kmeans(
        X_tr2a, K, max_iter=100, n_init=2,
        init='kmeans++', w=w, random_state=42
    )
    labels_va_2a = assign_clusters(X_va2a, C_2a, w=w)
    purity_va = cluster_purity(labels_va_2a, y_va2a)
    nmi_va = normalized_mutual_information(labels_va_2a, y_va2a)
    obj_va = compute_objective(X_va2a, labels_va_2a, C_2a)
    results_2a.append((sigma, purity_va, nmi_va, obj_va))
    print(f"  sigma={sigma:.1f} → purity_va={purity_va:.4f}, NMI_va={nmi_va:.4f}")
    if nmi_va > best_nmi_va:
        best_nmi_va = nmi_va
        best_sigma = sigma

print(f"\n  Best sigma = {best_sigma} (NMI={best_nmi_va:.4f})")

sigmas_arr, purity_arr, nmi_arr, obj_arr = zip(*results_2a)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
ax1.plot(sigmas_arr, purity_arr, 'o-', color='green')
ax1.axvline(best_sigma, linestyle='--', color='red', label=f'best σ={best_sigma}')
ax1.set_xlabel('σ (Gaussian weight bandwidth)'); ax1.set_ylabel('Purity (validation)')
ax1.set_title('Weight Optimisation: Purity vs σ')
ax1.legend(); ax1.grid(alpha=0.3)

ax2.plot(sigmas_arr, nmi_arr, 's-', color='purple')
ax2.axvline(best_sigma, linestyle='--', color='red', label=f'best σ={best_sigma}')
ax2.set_xlabel('σ'); ax2.set_ylabel('NMI (validation)')
ax2.set_title('Weight Optimisation: NMI vs σ')
ax2.legend(); ax2.grid(alpha=0.3)

plt.suptitle('Task 2a — Grid search over Gaussian frequency weight bandwidth', fontsize=11)
plt.tight_layout()
plt.savefig(f'{PLOTS}/task2a_weight_opt.png', dpi=120)
plt.close()
print("Saved task2a_weight_opt.png")

print("\nTask 2 & 2a complete.")
