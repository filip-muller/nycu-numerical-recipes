"""
Build a single Jupyter notebook from the individual task scripts.
"""
import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    return nbf.v4.new_markdown_cell(text)

def code(src):
    return nbf.v4.new_code_cell(src)

# ─── Title ───────────────────────────────────────────────────────────────────
cells.append(md("""# Challenge 1: K-Means Clustering for Images (MNIST)
**Numerical Recipes for Machine Learning**

This notebook contains all implementations and experiments for Challenge 1.
Each section corresponds to one assignment task.
"""))

# ─── Setup ────────────────────────────────────────────────────────────────────
cells.append(md("## 0. Setup and Imports"))
cells.append(code("""\
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.dpi'] = 100
import os, time, warnings
warnings.filterwarnings('ignore')
from sklearn.datasets import fetch_openml

os.makedirs('plots', exist_ok=True)
print("Imports OK")
"""))

# ─── Core routines ─────────────────────────────────────────────────────────────
cells.append(md("## 1. Core K-Means Routines"))
with open('kmeans_core.py') as f:
    cells.append(code(f.read()))

# ─── Load MNIST ────────────────────────────────────────────────────────────────
cells.append(md("## 2. Load MNIST Dataset"))
cells.append(code("""\
print("Loading MNIST (may take a moment)...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X_all = mnist.data.astype(np.float64) / 255.0  # Normalise to [0,1]
y_all = mnist.target.astype(int)

X = X_all[:60000]   # 60 000 training images
y = y_all[:60000]

print(f"Dataset shape: {X.shape}, labels: {np.unique(y)}")
"""))

# ─── Task 1 ───────────────────────────────────────────────────────────────────
cells.append(md("""## 3. Task 1 — Basic K-Means on MNIST

**Objectives:**
- Flatten 28×28 images to 784-dim vectors, normalize to [0,1]
- Run K-means for K ∈ {5, 10, 15, 20, 30}
- Compare random init vs K-means++ init
- Track objective per iteration
- Visualize centroids and nearest images
"""))

with open('task1_kmeans.py') as f:
    src = f.read()
# Remove the redundant data loading / imports at the top of the script
# and the PLOTS constant (already set)
# Keep as-is — cell can re-use globals defined above
src = src.replace("import numpy as np\n", "")
src = src.replace("import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\nimport matplotlib.gridspec as gridspec\n", "")
src = src.replace("from sklearn.datasets import fetch_openml\n", "")
src = src.replace("import os, time\n", "import time\n")
src = src.replace("from kmeans_core import (\n    kmeans, assign_clusters, compute_objective,\n    cluster_purity, normalized_mutual_information,\n    silhouette_score_fast, cluster_entropy\n)\n", "")
src = src.replace("PLOTS = 'plots'\nos.makedirs(PLOTS, exist_ok=True)\n", "")
# Remove the data loading section (already done)
src = src.replace("""\
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
""", "# (Data already loaded above)\nPLOTS = 'plots'\n")

cells.append(code(src))

# ─── Task 1a ──────────────────────────────────────────────────────────────────
cells.append(md("""## 4. Task 1a — How to Choose K?

Methods: Elbow curve, Silhouette score, Cluster purity (external), Stability.
"""))

with open('task1a_choose_k.py') as f:
    src = f.read()
for old, new in [
    ("import numpy as np\n", ""),
    ("import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n", ""),
    ("from sklearn.datasets import fetch_openml\n", ""),
    ("import os\n", ""),
    ("from kmeans_core import (\n    kmeans, assign_clusters, compute_objective,\n    cluster_purity, normalized_mutual_information,\n    silhouette_score_fast\n)\n", ""),
    ("PLOTS = 'plots'\nos.makedirs(PLOTS, exist_ok=True)\n\n# ============================================================\n# Load MNIST (use a subset for silhouette speed)\n# ============================================================\nprint(\"Loading MNIST...\")\nmnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')\nX_all = mnist.data.astype(np.float64) / 255.0\ny_all = mnist.target.astype(int)\n\nX = X_all[:60000]\ny = y_all[:60000]\n\n", "# (Data already loaded above)\nPLOTS = 'plots'\n"),
]:
    src = src.replace(old, new)

cells.append(code(src))

# ─── Task 2 ───────────────────────────────────────────────────────────────────
cells.append(md("""## 5. Task 2 — K-Means in the DFT Domain with Feature Weighting

Represents each image using 2D DFT log-magnitude features.
Implements weighted distance with several frequency weight masks.
"""))

with open('task2_dft.py') as f:
    src = f.read()
for old, new in [
    ("import numpy as np\n", ""),
    ("import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n", ""),
    ("from sklearn.datasets import fetch_openml\n", ""),
    ("import os\n", ""),
    ("from kmeans_core import (\n    kmeans, assign_clusters, compute_objective,\n    compute_weighted_distance_matrix,\n    cluster_purity, normalized_mutual_information\n)\n", ""),
    ("PLOTS = 'plots'\nos.makedirs(PLOTS, exist_ok=True)\n\n# ============================================================\n# 1. Load MNIST\n# ============================================================\nprint(\"Loading MNIST...\")\nmnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')\nX_all = mnist.data.astype(np.float64) / 255.0\ny_all = mnist.target.astype(int)\n\nX = X_all[:60000]\ny = y_all[:60000]\n\n", "# (Data already loaded above)\nPLOTS = 'plots'\n"),
]:
    src = src.replace(old, new)

cells.append(code(src))

# ─── Task 3 ───────────────────────────────────────────────────────────────────
cells.append(md("""## 6. Task 3 — Validating Clusters with Train/Test Splits

Fits centroids on training data, evaluates on test data.
Uses purity, NMI, entropy, cluster size imbalance, and objectives.
"""))

with open('task3_validation.py') as f:
    src = f.read()
for old, new in [
    ("import numpy as np\n", ""),
    ("import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n", ""),
    ("from sklearn.datasets import fetch_openml\n", ""),
    ("import os\n", ""),
    ("from kmeans_core import (\n    kmeans, assign_clusters, compute_objective,\n    cluster_purity, normalized_mutual_information,\n    cluster_entropy, train_test_cluster_evaluation\n)\n", ""),
    ("PLOTS = 'plots'\nos.makedirs(PLOTS, exist_ok=True)\n\n# ============================================================\n# 1. Load MNIST + DFT features\n# ============================================================\nprint(\"Loading MNIST...\")\nmnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')\nX_all = mnist.data.astype(np.float64) / 255.0\ny_all = mnist.target.astype(int)\n\nX = X_all[:60000]\ny = y_all[:60000]\n\n", "# (Data already loaded above)\nPLOTS = 'plots'\n"),
]:
    src = src.replace(old, new)

cells.append(code(src))

# ─── Task 4 ───────────────────────────────────────────────────────────────────
cells.append(md("""## 7. Task 4 — Hierarchical K-Means

Two-level hierarchical K-means: K1=5 coarse clusters, K2=4 sub-clusters each.
Compares with flat K-means at 20 total clusters.
"""))

with open('task4_hierarchical.py') as f:
    src = f.read()
for old, new in [
    ("import numpy as np\n", ""),
    ("import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n", ""),
    ("from sklearn.datasets import fetch_openml\n", ""),
    ("import os, time\n", "import time\n"),
    ("from kmeans_core import (\n    kmeans, assign_clusters, compute_objective,\n    cluster_purity, normalized_mutual_information, cluster_entropy\n)\n", ""),
    ("PLOTS = 'plots'\nos.makedirs(PLOTS, exist_ok=True)\n\n# ============================================================\n# 1. Load MNIST\n# ============================================================\nprint(\"Loading MNIST...\")\nmnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')\nX_all = mnist.data.astype(np.float64) / 255.0\ny_all = mnist.target.astype(int)\n\nX = X_all[:60000]\ny = y_all[:60000]\n\n", "# (Data already loaded above)\nPLOTS = 'plots'\n"),
]:
    src = src.replace(old, new)

cells.append(code(src))

# ─── Task 5 ───────────────────────────────────────────────────────────────────
cells.append(md("""## 8. Task 5 — K-Medoids with Edit Distance

Converts images to run-length encoded binary sequences.
Uses Levenshtein edit distance + K-medoids (no centroid averaging).
"""))

with open('task5_edit_distance.py') as f:
    src = f.read()
for old, new in [
    ("import numpy as np\n", ""),
    ("import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n", ""),
    ("from sklearn.datasets import fetch_openml\n", ""),
    ("import os, time\n", "import time\n"),
    ("PLOTS = 'plots'\nos.makedirs(PLOTS, exist_ok=True)\n\n# ============================================================\n# 1. Load MNIST (small subset for edit distance)\n# ============================================================\nprint(\"Loading MNIST...\")\nmnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')\nX_all = mnist.data.astype(np.float64) / 255.0\ny_all = mnist.target.astype(int)\n\n# Use only 500 images for the edit distance experiment (O(N^2) distance)\nN_EDIT = 500\nX_small = X_all[:N_EDIT]\ny_small = y_all[:N_EDIT]\n\nprint(f\"  Using {N_EDIT} images for edit distance clustering.\")\n", "# (Data already loaded above)\nPLOTS = 'plots'\nN_EDIT = 500\nX_small = X_all[:N_EDIT]\ny_small = y_all[:N_EDIT]\nprint(f\"  Using {N_EDIT} images for edit distance clustering.\")\n"),
    ("from kmeans_core import cluster_purity, normalized_mutual_information\n", ""),
    ("from kmeans_core import kmeans as kmeans_eucl\n", "kmeans_eucl = kmeans\n"),
]:
    src = src.replace(old, new)

cells.append(code(src))

# ─── Done ─────────────────────────────────────────────────────────────────────
cells.append(md("""## Summary

All tasks completed. Plots saved to `plots/` directory.

| Task | Description |
|------|-------------|
| 1    | Basic K-means on MNIST, K ∈ {5,10,15,20,30}, random vs K-means++ |
| 1a   | K-selection: elbow, silhouette, purity, stability |
| 2    | DFT-domain K-means with hand-designed frequency weights |
| 2a   | Gaussian bandwidth grid search for weight optimisation |
| 3    | Train/test validation pipeline with 5 metrics |
| 3a   | Train/test ratio comparison (50/50, 70/30, 80/20, 90/10) |
| 4    | Hierarchical K-means (2-level, K1=5, K2=4) |
| 5    | K-medoids with Levenshtein edit distance on RLE sequences |
"""))

nb.cells = cells
nbf.write(nb, 'challenge1_kmeans.ipynb')
print("Notebook written: challenge1_kmeans.ipynb")
