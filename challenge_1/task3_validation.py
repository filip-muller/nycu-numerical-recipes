"""
Task 3 — Validating clusters with train/test splits.
Task 3a — Different train/test ratios.
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
    cluster_entropy, train_test_cluster_evaluation
)

PLOTS = 'plots'
os.makedirs(PLOTS, exist_ok=True)

# ============================================================
# 1. Load MNIST + DFT features
# ============================================================
print("Loading MNIST...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X_all = mnist.data.astype(np.float64) / 255.0
y_all = mnist.target.astype(int)

X = X_all[:60000]
y = y_all[:60000]


def dft_features_normalised(images):
    """Return normalised log-magnitude DFT features."""
    N = len(images)
    imgs = images.reshape(N, 28, 28)
    F = np.fft.fft2(imgs)
    feats = np.log1p(np.abs(F)).reshape(N, -1)
    mu = feats.mean(axis=0)
    std = feats.std(axis=0) + 1e-8
    return (feats - mu) / std, mu, std


X_dft, mu_dft, std_dft = dft_features_normalised(X)


def cluster_size_imbalance(labels, K):
    """
    Imbalance metric: max cluster size / mean cluster size.
    1.0 = perfectly balanced; larger = more imbalanced.
    """
    counts = np.bincount(labels, minlength=K).astype(float)
    return counts.max() / (counts.mean() + 1e-10)


# ============================================================
# 2. Task 3 — Main validation (80/20 split)
# ============================================================
K = 10
TRAIN_FRAC = 0.8
N = len(X)
n_train = int(N * TRAIN_FRAC)

rng = np.random.default_rng(0)
perm = rng.permutation(N)
tr_idx = perm[:n_train]
te_idx = perm[n_train:]

X_tr, y_tr = X[tr_idx], y[tr_idx]
X_te, y_te = X[te_idx], y[te_idx]

print(f"\n[Task 3] 80/20 split — train={n_train}, test={N-n_train}")

# --- Pixel domain ---
res_px = train_test_cluster_evaluation(X_tr, X_te, y_tr, y_te,
                                       K=K, n_init=3, random_state=42)
imb_tr_px = cluster_size_imbalance(res_px['train_labels'], K)
imb_te_px = cluster_size_imbalance(res_px['test_labels'], K)

print(f"\nPixel domain (K={K}):")
print(f"  Train obj={res_px['train_obj']:.3e}, test obj={res_px['test_obj']:.3e}")
print(f"  Train purity={res_px['train_purity']:.4f}, test purity={res_px['test_purity']:.4f}")
print(f"  Train NMI={res_px['train_nmi']:.4f}, test NMI={res_px['test_nmi']:.4f}")
print(f"  Train entropy={res_px['train_entropy']:.4f}, test entropy={res_px['test_entropy']:.4f}")
print(f"  Cluster imbalance train={imb_tr_px:.2f}, test={imb_te_px:.2f}")

# --- DFT domain ---
X_dft_tr, X_dft_te = X_dft[tr_idx], X_dft[te_idx]
res_dft = train_test_cluster_evaluation(X_dft_tr, X_dft_te, y_tr, y_te,
                                        K=K, n_init=3, random_state=42)
imb_tr_dft = cluster_size_imbalance(res_dft['train_labels'], K)
imb_te_dft = cluster_size_imbalance(res_dft['test_labels'], K)

print(f"\nDFT domain (K={K}):")
print(f"  Train obj={res_dft['train_obj']:.3e}, test obj={res_dft['test_obj']:.3e}")
print(f"  Train purity={res_dft['train_purity']:.4f}, test purity={res_dft['test_purity']:.4f}")
print(f"  Train NMI={res_dft['train_nmi']:.4f}, test NMI={res_dft['test_nmi']:.4f}")
print(f"  Train entropy={res_dft['train_entropy']:.4f}, test entropy={res_dft['test_entropy']:.4f}")
print(f"  Cluster imbalance train={imb_tr_dft:.2f}, test={imb_te_dft:.2f}")

# --- Comparison figure ---
metrics = ['Purity', 'NMI', 'Entropy', 'Imbalance']
px_tr = [res_px['train_purity'], res_px['train_nmi'],
         res_px['train_entropy'], imb_tr_px]
px_te = [res_px['test_purity'],  res_px['test_nmi'],
         res_px['test_entropy'],  imb_te_px]
dft_tr = [res_dft['train_purity'], res_dft['train_nmi'],
          res_dft['train_entropy'], imb_tr_dft]
dft_te = [res_dft['test_purity'],  res_dft['test_nmi'],
          res_dft['test_entropy'],  imb_te_dft]

x = np.arange(len(metrics))
width = 0.2
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - 1.5*width, px_tr,  width, label='Pixel train',  color='steelblue')
ax.bar(x - 0.5*width, px_te,  width, label='Pixel test',   color='cornflowerblue')
ax.bar(x + 0.5*width, dft_tr, width, label='DFT train',    color='darkorange')
ax.bar(x + 1.5*width, dft_te, width, label='DFT test',     color='moccasin')
ax.set_xticks(x); ax.set_xticklabels(metrics)
ax.set_title(f'Train vs Test metrics — Pixel and DFT domain (K={K}, 80/20 split)')
ax.legend(); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{PLOTS}/task3_train_test_metrics.png', dpi=120)
plt.close()
print("\nSaved task3_train_test_metrics.png")

# ============================================================
# 3. Task 3a — Different train/test ratios
# ============================================================
print("\n[Task 3a] Scanning train/test ratios...")
ratios = [0.5, 0.7, 0.8, 0.9]   # fraction of data used for training

records_px  = {'purity_tr':[], 'purity_te':[], 'nmi_tr':[], 'nmi_te':[],
               'obj_tr':[], 'obj_te':[]}
records_dft = {'purity_tr':[], 'purity_te':[], 'nmi_tr':[], 'nmi_te':[],
               'obj_tr':[], 'obj_te':[]}
ratio_labels = []

for frac in ratios:
    n_tr = int(N * frac)
    rng2 = np.random.default_rng(1)
    perm2 = rng2.permutation(N)
    ti = perm2[:n_tr]; vi = perm2[n_tr:]

    lbl = f"{int(frac*100)}/{100-int(frac*100)}"
    ratio_labels.append(lbl)
    print(f"  Ratio {lbl}...", end=' ', flush=True)

    # Pixel
    rp = train_test_cluster_evaluation(X[ti], X[vi], y[ti], y[vi],
                                       K=K, n_init=2, random_state=42)
    records_px['purity_tr'].append(rp['train_purity'])
    records_px['purity_te'].append(rp['test_purity'])
    records_px['nmi_tr'].append(rp['train_nmi'])
    records_px['nmi_te'].append(rp['test_nmi'])
    records_px['obj_tr'].append(rp['train_obj'])
    records_px['obj_te'].append(rp['test_obj'])

    # DFT
    rd = train_test_cluster_evaluation(X_dft[ti], X_dft[vi], y[ti], y[vi],
                                       K=K, n_init=2, random_state=42)
    records_dft['purity_tr'].append(rd['train_purity'])
    records_dft['purity_te'].append(rd['test_purity'])
    records_dft['nmi_tr'].append(rd['train_nmi'])
    records_dft['nmi_te'].append(rd['test_nmi'])
    records_dft['obj_tr'].append(rd['train_obj'])
    records_dft['obj_te'].append(rd['test_obj'])

    print(f"px purity_te={rp['test_purity']:.4f}, "
          f"dft purity_te={rd['test_purity']:.4f}")

# --- Plots ---
x_pos = np.arange(len(ratios))
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Purity
axes[0].plot(x_pos, records_px['purity_tr'],  'o--', label='Pixel train',  color='steelblue')
axes[0].plot(x_pos, records_px['purity_te'],  'o-',  label='Pixel test',   color='steelblue', alpha=0.5)
axes[0].plot(x_pos, records_dft['purity_tr'], 's--', label='DFT train',    color='darkorange')
axes[0].plot(x_pos, records_dft['purity_te'], 's-',  label='DFT test',     color='darkorange', alpha=0.5)
axes[0].set_xticks(x_pos); axes[0].set_xticklabels(ratio_labels)
axes[0].set_title('Purity vs Train Fraction')
axes[0].set_ylabel('Purity'); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

# NMI
axes[1].plot(x_pos, records_px['nmi_tr'],  'o--', color='steelblue')
axes[1].plot(x_pos, records_px['nmi_te'],  'o-',  color='steelblue', alpha=0.5)
axes[1].plot(x_pos, records_dft['nmi_tr'], 's--', color='darkorange')
axes[1].plot(x_pos, records_dft['nmi_te'], 's-',  color='darkorange', alpha=0.5)
axes[1].set_xticks(x_pos); axes[1].set_xticklabels(ratio_labels)
axes[1].set_title('NMI vs Train Fraction')
axes[1].set_ylabel('NMI'); axes[1].grid(alpha=0.3)
axes[1].legend(['Pixel train','Pixel test','DFT train','DFT test'], fontsize=8)

# Normalised objectives (divide by n_points)
n_tr_list = [int(N * f) for f in ratios]
n_te_list = [N - n for n in n_tr_list]
px_obj_tr_norm  = [o/n for o, n in zip(records_px['obj_tr'],  n_tr_list)]
px_obj_te_norm  = [o/n for o, n in zip(records_px['obj_te'],  n_te_list)]
dft_obj_tr_norm = [o/n for o, n in zip(records_dft['obj_tr'], n_tr_list)]
dft_obj_te_norm = [o/n for o, n in zip(records_dft['obj_te'], n_te_list)]

axes[2].plot(x_pos, px_obj_tr_norm,  'o--', color='steelblue')
axes[2].plot(x_pos, px_obj_te_norm,  'o-',  color='steelblue', alpha=0.5)
axes[2].plot(x_pos, dft_obj_tr_norm, 's--', color='darkorange')
axes[2].plot(x_pos, dft_obj_te_norm, 's-',  color='darkorange', alpha=0.5)
axes[2].set_xticks(x_pos); axes[2].set_xticklabels(ratio_labels)
axes[2].set_title('Objective per point vs Train Fraction')
axes[2].set_ylabel('SSE / N'); axes[2].grid(alpha=0.3)
axes[2].legend(['Pixel train','Pixel test','DFT train','DFT test'], fontsize=8)

plt.suptitle(f'Task 3a — K={K}, varying train/test ratios', fontsize=12)
plt.tight_layout()
plt.savefig(f'{PLOTS}/task3a_ratio_comparison.png', dpi=120)
plt.close()
print("Saved task3a_ratio_comparison.png")

# ============================================================
# 4. Stability across multiple random splits (same ratio 80/20)
# ============================================================
print("\n[Task 3a] Stability across 5 random 80/20 splits...")
stab_records_px  = []
stab_records_dft = []

for seed in range(5):
    rng3 = np.random.default_rng(seed * 13)
    perm3 = rng3.permutation(N)
    n_tr3 = int(N * 0.8)
    ti3 = perm3[:n_tr3]; vi3 = perm3[n_tr3:]

    rp3 = train_test_cluster_evaluation(X[ti3], X[vi3], y[ti3], y[vi3],
                                        K=K, n_init=2, random_state=42)
    rd3 = train_test_cluster_evaluation(X_dft[ti3], X_dft[vi3], y[ti3], y[vi3],
                                        K=K, n_init=2, random_state=42)
    stab_records_px.append(rp3['test_purity'])
    stab_records_dft.append(rd3['test_purity'])
    print(f"  Seed {seed}: px purity_te={rp3['test_purity']:.4f}, "
          f"dft purity_te={rd3['test_purity']:.4f}")

print(f"\nPixel  purity std across splits: {np.std(stab_records_px):.5f}")
print(f"DFT    purity std across splits: {np.std(stab_records_dft):.5f}")

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(stab_records_px, 'o-', label=f'Pixel (std={np.std(stab_records_px):.4f})')
ax.plot(stab_records_dft, 's--', label=f'DFT   (std={np.std(stab_records_dft):.4f})')
ax.set_xlabel('Split seed')
ax.set_ylabel('Test purity')
ax.set_title('Stability across 5 random 80/20 splits (K=10)')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{PLOTS}/task3a_stability.png', dpi=120)
plt.close()
print("Saved task3a_stability.png")

print("\nTask 3 & 3a complete.")
