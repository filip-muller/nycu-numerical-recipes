"""
Core K-means routines for the MNIST clustering challenge.
All functions are implemented from scratch using only numpy.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Distance computation
# ---------------------------------------------------------------------------

def compute_distance_matrix(X, C):
    """
    Compute squared Euclidean distance matrix between data points and centroids.

    Uses the identity ||x - c||^2 = ||x||^2 - 2 x·c + ||c||^2
    to avoid an explicit double for-loop.

    Parameters
    ----------
    X : ndarray, shape (N, d)
    C : ndarray, shape (K, d)

    Returns
    -------
    D : ndarray, shape (N, K)  — squared distances
    """
    # ||x||^2  shape (N,)
    x_sq = np.sum(X ** 2, axis=1, keepdims=True)   # (N, 1)
    # ||c||^2  shape (K,)
    c_sq = np.sum(C ** 2, axis=1, keepdims=True).T  # (1, K)
    # cross term  shape (N, K)
    cross = X @ C.T                                  # (N, K)
    D = x_sq - 2.0 * cross + c_sq
    # Numerical noise can produce tiny negatives; clamp to 0
    np.maximum(D, 0.0, out=D)
    return D


def compute_weighted_distance_matrix(X, C, w):
    """
    Compute weighted squared distance matrix.
    d_w(x, c)^2 = sum_j w_j (x_j - c_j)^2

    Parameters
    ----------
    X : ndarray, shape (N, d)
    C : ndarray, shape (K, d)
    w : ndarray, shape (d,)  — non-negative weights

    Returns
    -------
    D : ndarray, shape (N, K)
    """
    Xw = X * np.sqrt(w)   # (N, d)
    Cw = C * np.sqrt(w)   # (K, d)
    return compute_distance_matrix(Xw, Cw)


# ---------------------------------------------------------------------------
# Assignment step
# ---------------------------------------------------------------------------

def assign_clusters(X, C, w=None):
    """
    Assign each point to its nearest centroid.

    Parameters
    ----------
    X : ndarray, shape (N, d)
    C : ndarray, shape (K, d)
    w : ndarray, shape (d,) or None — if given, use weighted distance

    Returns
    -------
    labels : ndarray, shape (N,)  — cluster index for each point
    """
    if w is None:
        D = compute_distance_matrix(X, C)
    else:
        D = compute_weighted_distance_matrix(X, C, w)
    return np.argmin(D, axis=1)


# ---------------------------------------------------------------------------
# Update step
# ---------------------------------------------------------------------------

def update_centroids(X, labels, K):
    """
    Compute new centroids as the mean of assigned points.

    If a cluster becomes empty, its centroid is kept unchanged
    (handled by the caller who passes in old centroids).

    Parameters
    ----------
    X      : ndarray, shape (N, d)
    labels : ndarray, shape (N,)
    K      : int

    Returns
    -------
    C_new : ndarray, shape (K, d)
    counts: ndarray, shape (K,) — number of points in each cluster
    """
    d = X.shape[1]
    C_new = np.zeros((K, d), dtype=np.float64)
    counts = np.zeros(K, dtype=np.int64)
    for k in range(K):
        mask = labels == k
        counts[k] = mask.sum()
        if counts[k] > 0:
            C_new[k] = X[mask].mean(axis=0)
    return C_new, counts


# ---------------------------------------------------------------------------
# Objective function
# ---------------------------------------------------------------------------

def compute_objective(X, labels, C):
    """
    Compute the K-means within-cluster sum of squared distances.

    Parameters
    ----------
    X      : ndarray, shape (N, d)
    labels : ndarray, shape (N,)
    C      : ndarray, shape (K, d)

    Returns
    -------
    obj : float
    """
    # Vectorized: subtract each point's assigned centroid, then sum squared norms
    diff = X - C[labels]          # (N, d)
    return float(np.sum(diff ** 2))


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def _init_random(X, K, rng):
    """Random initialization: pick K distinct points uniformly."""
    idx = rng.choice(len(X), size=K, replace=False)
    return X[idx].copy()


def _init_kmeanspp(X, K, rng):
    """
    K-means++ initialization.
    First centroid: uniform random.
    Each subsequent centroid: sample proportional to squared distance
    from the nearest already-chosen centroid.
    """
    N = len(X)
    idx0 = rng.integers(0, N)
    centroids = [X[idx0].copy()]

    for _ in range(1, K):
        C = np.array(centroids)
        D = compute_distance_matrix(X, C)           # (N, k_so_far)
        min_dist = D.min(axis=1)                    # (N,)
        probs = min_dist / min_dist.sum()
        next_idx = rng.choice(N, p=probs)
        centroids.append(X[next_idx].copy())

    return np.array(centroids)


# ---------------------------------------------------------------------------
# Main K-means
# ---------------------------------------------------------------------------

def kmeans(X, K, max_iter=300, tol=1e-4, n_init=5, init='kmeans++',
           w=None, random_state=42, verbose=False):
    """
    K-means clustering with multiple restarts.

    Parameters
    ----------
    X            : ndarray, shape (N, d)
    K            : int
    max_iter     : int
    tol          : float  — stop if objective improves by less than tol (relative)
    n_init       : int    — number of random restarts; best result is returned
    init         : 'random' or 'kmeans++'
    w            : ndarray, shape (d,) or None — feature weights
    random_state : int
    verbose      : bool

    Returns
    -------
    best_labels    : ndarray, shape (N,)
    best_centroids : ndarray, shape (K, d)
    best_obj       : float
    best_history   : list of floats — objective per iteration (best run)
    """
    rng = np.random.default_rng(random_state)

    best_obj = np.inf
    best_labels = None
    best_centroids = None
    best_history = None

    for run in range(n_init):
        # --- Initialization ---
        if init == 'kmeans++':
            C = _init_kmeanspp(X, K, rng)
        else:
            C = _init_random(X, K, rng)

        history = []

        for it in range(max_iter):
            # Assignment
            labels = assign_clusters(X, C, w=w)

            # New centroids; keep old ones for empty clusters
            C_new, counts = update_centroids(X, labels, K)
            empty = counts == 0
            C_new[empty] = C[empty]

            obj = compute_objective(X, labels, C_new)
            history.append(obj)

            if verbose:
                print(f"  Run {run+1}, iter {it+1}: obj={obj:.4f}")

            # Check for empty-cluster reassignment
            if empty.any():
                # Re-assign empty centroid to a random point
                for k in np.where(empty)[0]:
                    C_new[k] = X[rng.integers(0, len(X))]

            # Convergence check (after first iteration)
            if it > 0 and abs(history[-2] - obj) / (abs(history[-2]) + 1e-12) < tol:
                break

            C = C_new

        if obj < best_obj:
            best_obj = obj
            best_labels = labels.copy()
            best_centroids = C_new.copy()
            best_history = history

    return best_labels, best_centroids, best_obj, best_history


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def cluster_purity(labels, true_labels):
    """
    Compute cluster purity: fraction of points that are in the majority
    class of their assigned cluster.
    """
    K = labels.max() + 1
    total = len(labels)
    correct = 0
    for k in range(K):
        mask = labels == k
        if mask.sum() == 0:
            continue
        counts = np.bincount(true_labels[mask])
        correct += counts.max()
    return correct / total


def cluster_entropy(labels, true_labels):
    """
    Mean entropy of label distribution inside each cluster.
    """
    K = labels.max() + 1
    entropies = []
    for k in range(K):
        mask = labels == k
        if mask.sum() == 0:
            continue
        counts = np.bincount(true_labels[mask])
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        entropies.append(-np.sum(probs * np.log(probs + 1e-12)))
    return float(np.mean(entropies))


def normalized_mutual_information(labels, true_labels):
    """
    Compute normalized mutual information (NMI) between cluster labels and true labels.
    NMI = 2 * MI(labels, true_labels) / (H(labels) + H(true_labels))
    """
    N = len(labels)
    K = labels.max() + 1
    C_classes = true_labels.max() + 1

    # Joint distribution
    joint = np.zeros((K, C_classes))
    for k, c in zip(labels, true_labels):
        joint[k, c] += 1
    joint /= N

    # Marginals
    p_k = joint.sum(axis=1, keepdims=True)   # (K, 1)
    p_c = joint.sum(axis=0, keepdims=True)   # (1, C)

    # MI
    mask = joint > 0
    mi = np.sum(joint[mask] * np.log(joint[mask] / (p_k * p_c + 1e-300)[mask]))

    # Entropies
    h_k = -np.sum(p_k[p_k > 0] * np.log(p_k[p_k > 0]))
    h_c = -np.sum(p_c[p_c > 0] * np.log(p_c[p_c > 0]))

    if h_k + h_c < 1e-12:
        return 0.0
    return float(2.0 * mi / (h_k + h_c))


def silhouette_score_fast(X, labels, sample_size=2000, random_state=42):
    """
    Approximate silhouette score using a random subsample.

    s(i) = (b(i) - a(i)) / max(a(i), b(i))
    a(i) = mean intra-cluster distance
    b(i) = mean distance to nearest other cluster
    """
    rng = np.random.default_rng(random_state)
    N = len(X)
    idx = rng.choice(N, size=min(sample_size, N), replace=False)
    Xs = X[idx]
    ls = labels[idx]

    K = labels.max() + 1
    scores = []
    for i in range(len(Xs)):
        k = ls[i]
        intra_mask = ls == k
        inter_masks = [ls == j for j in range(K) if j != k]

        if intra_mask.sum() > 1:
            a = np.mean(np.sqrt(np.sum((Xs[intra_mask] - Xs[i]) ** 2, axis=1)))
        else:
            a = 0.0

        b_vals = []
        for m in inter_masks:
            if m.sum() > 0:
                b_vals.append(np.mean(np.sqrt(np.sum((Xs[m] - Xs[i]) ** 2, axis=1))))
        b = min(b_vals) if b_vals else 0.0

        denom = max(a, b)
        s = (b - a) / denom if denom > 0 else 0.0
        scores.append(s)

    return float(np.mean(scores))


def train_test_cluster_evaluation(X_train, X_test, y_train, y_test,
                                  K, n_init=3, init='kmeans++',
                                  w=None, random_state=42):
    """
    Fit K-means on train set, evaluate on both train and test.

    Returns dict with train/test objectives, purity, NMI, entropy.
    """
    labels_tr, centroids, obj_tr, hist = kmeans(
        X_train, K, n_init=n_init, init=init, w=w, random_state=random_state
    )
    labels_te = assign_clusters(X_test, centroids, w=w)
    obj_te = compute_objective(X_test, labels_te, centroids)

    return {
        'train_obj': obj_tr,
        'test_obj': obj_te,
        'train_purity': cluster_purity(labels_tr, y_train),
        'test_purity': cluster_purity(labels_te, y_test),
        'train_nmi': normalized_mutual_information(labels_tr, y_train),
        'test_nmi': normalized_mutual_information(labels_te, y_test),
        'train_entropy': cluster_entropy(labels_tr, y_train),
        'test_entropy': cluster_entropy(labels_te, y_test),
        'centroids': centroids,
        'train_labels': labels_tr,
        'test_labels': labels_te,
        'history': hist,
    }
