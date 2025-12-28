"""
Module 4 — direct Behaviour Subspace (HBS)
==========================================

This module constructs a low‑dimensional subspace capturing direct and
composed behaviour by performing PCA on the pooled activations from
Module 1.  It produces a set of principal components (``direct_subspace_components.npy``),
their explained variance ratios (``direct_subspace_explained_variance.npy``)
and the mean vector of the danger zone (direct + composed) activations
(``direct_subspace_mean.npy``).  Several plots are also generated
for analysis.

Downstream code uses the helper ``load_direct_space()`` defined at
the bottom of this file to load the mean and PCA basis without
repeating path logic.

Note: Visualization code remains intact but is not exercised by the
main pipeline unless this file is executed directly.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

PARENT_PATH = "/home/tahad/HAVOC/HAVOC"
# ============================================================
# PATHS
# ============================================================
ACT_DIR = f"{PARENT_PATH}/output/activations"
OUT_DIR = f"{PARENT_PATH}/output/analysis/module4"
SUBSPACE_DIR = f"{PARENT_PATH}/output/subspace"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(SUBSPACE_DIR, exist_ok=True)

# ============================================================
# CONFIGURATION
# ============================================================
LAYER = 20                      # Best layer selected earlier
TOP_K = 10                      # PCA components to keep
PLOT_SAMPLE_LIMIT = 2000        # Limit for t-SNE/UMAP to avoid slow runtime

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ============================================================
# direct SPACE CONSTRUCTION
# ============================================================
def run_direct_space_construction(layer: int = LAYER, top_k: int = TOP_K) -> None:
    """Compute the direct behaviour subspace using PCA on danger activations.

    Loads benign (B), direct (H) and composed (J) activations for the given
    layer, stacks ``H`` and ``J`` to form the danger set, performs PCA to
    obtain ``top_k`` principal components and saves the mean and components
    to ``SUBSPACE_DIR``.  A variety of diagnostic plots are written to
    ``OUT_DIR``.
    """
    print(f"Loading layer {layer} activations…")
    B = np.load(f"{ACT_DIR}/B_a_layer{layer}.npy")
    H = np.load(f"{ACT_DIR}/H_a_layer{layer}.npy")
    J = np.load(f"{ACT_DIR}/J_a_layer{layer}.npy")
    print(f"Benign:  {B.shape}")
    print(f"direct: {H.shape}")
    print(f"composed: {J.shape}")
    print("\nConstructing direct Behaviour Subspace (HBS)…")
    danger = np.vstack([H, J])
    pca = PCA(n_components=top_k)
    danger_pca = pca.fit(danger)
    # Save components and mean
    np.save(f"{SUBSPACE_DIR}/direct_subspace_components.npy", danger_pca.components_)
    np.save(f"{SUBSPACE_DIR}/direct_subspace_mean.npy", danger_pca.mean_)
    np.save(f"{SUBSPACE_DIR}/direct_subspace_explained_variance.npy", danger_pca.explained_variance_ratio_)
    print("Saved HBS components to:", SUBSPACE_DIR)
    # Scree plot
    plt.figure(figsize=(7,5))
    plt.plot(range(1, top_k+1), danger_pca.explained_variance_ratio_[:top_k], marker="o", linewidth=2)
    plt.title("Explained Variance (direct Behaviour Subspace)")
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{OUT_DIR}/hbs_scree_plot.png", dpi=300)
    plt.close()
    # PCA cluster plots (2D & 3D)
    labels = np.array([0]*len(B) + [1]*len(H) + [2]*len(J))
    X = np.vstack([B, H, J])
    # 2D
    pca2 = PCA(n_components=2).fit_transform(X)
    plt.figure(figsize=(7,6))
    plt.scatter(pca2[labels==0,0], pca2[labels==0,1], alpha=0.5, label="Benign")
    plt.scatter(pca2[labels==1,0], pca2[labels==1,1], alpha=0.5, label="direct")
    plt.scatter(pca2[labels==2,0], pca2[labels==2,1], alpha=0.5, label="composed")
    plt.title(f"PCA Cluster Plot — Layer {layer}")
    plt.legend()
    plt.savefig(f"{OUT_DIR}/pca_cluster_layer{layer}.png", dpi=300)
    plt.close()
    # 3D
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    pca3 = PCA(n_components=3).fit_transform(X)
    fig = plt.figure(figsize=(8,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pca3[labels==0,0], pca3[labels==0,1], pca3[labels==0,2], alpha=0.4, label="Benign")
    ax.scatter(pca3[labels==1,0], pca3[labels==1,1], pca3[labels==1,2], alpha=0.4, label="direct")
    ax.scatter(pca3[labels==2,0], pca3[labels==2,1], pca3[labels==2,2], alpha=0.4, label="composed")
    ax.set_title(f"3D PCA Cluster — Layer {layer}")
    ax.legend()
    plt.savefig(f"{OUT_DIR}/pca3d_cluster_layer{layer}.png", dpi=300)
    plt.close()
    # UMAP plot
    limit = min(PLOT_SAMPLE_LIMIT, len(X))
    umap_model = UMAP(n_components=2, n_neighbors=25, min_dist=0.1, metric="cosine")
    u = umap_model.fit_transform(X[:limit])
    plt.figure(figsize=(7,6))
    plt.scatter(u[labels[:limit]==0,0], u[labels[:limit]==0,1], alpha=0.5, label="Benign")
    plt.scatter(u[labels[:limit]==1,0], u[labels[:limit]==1,1], alpha=0.5, label="direct")
    plt.scatter(u[labels[:limit]==2,0], u[labels[:limit]==2,1], alpha=0.5, label="composed")
    plt.legend()
    plt.title(f"UMAP Cluster Plot — Layer {layer}")
    plt.savefig(f"{OUT_DIR}/umap_cluster_layer{layer}.png", dpi=300)
    plt.close()
    # t‑SNE plot
    tsne_model = TSNE(n_components=2, metric="cosine", perplexity=30, learning_rate=200)
    ts = tsne_model.fit_transform(X[:limit])
    plt.figure(figsize=(7,6))
    plt.scatter(ts[labels[:limit]==0,0], ts[labels[:limit]==0,1], alpha=0.5, label="Benign")
    plt.scatter(ts[labels[:limit]==1,0], ts[labels[:limit]==1,1], alpha=0.5, label="direct")
    plt.scatter(ts[labels[:limit]==2,0], ts[labels[:limit]==2,1], alpha=0.5, label="composed")
    plt.legend()
    plt.title(f"t-SNE Cluster Plot — Layer {layer}")
    plt.savefig(f"{OUT_DIR}/tsne_cluster_layer{layer}.png", dpi=300)
    plt.close()
    # Centroid similarity heatmap
    centroids = {
        "Benign": B.mean(axis=0),
        "direct": H.mean(axis=0),
        "composed": J.mean(axis=0)
    }
    names = ["Benign", "direct", "composed"]
    mat = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            mat[i,j] = cosine(centroids[names[i]], centroids[names[j]])
    plt.figure(figsize=(6,5))
    sns.heatmap(mat, annot=True, xticklabels=names, yticklabels=names, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title(f"Centroid Cosine Similarity — Layer {layer}")
    plt.savefig(f"{OUT_DIR}/centroid_similarity_heatmap.png", dpi=300)
    plt.close()
    print("\nModule 4 complete!")
    print(f"All visualizations saved to: {OUT_DIR}")
    print(f"direct Behaviour Subspace saved to: {SUBSPACE_DIR}")

# ============================================================
#  HELPER: LOAD direct SPACE FOR DOWNSTREAM MODULES
# ============================================================
def load_direct_space(subspace_dir: str = SUBSPACE_DIR):
    """Load the direct behaviour subspace (mean and components).

    This convenience function reads the PCA basis and mean vector
    computed by ``run_direct_space_construction`` and returns them as
    a tuple ``(mu_HJ, W)``.  ``mu_HJ`` corresponds to the mean of the
    danger zone (direct + composed) activations and ``W`` contains
    the principal components row‑wise.

    Args:
        subspace_dir: Directory containing the saved subspace files.

    Returns:
        ``(mu_HJ, W)`` where ``mu_HJ`` is a 1‑D numpy array and
        ``W`` is a 2‑D array of shape (k, hidden_dim).
    """
    mean = np.load(os.path.join(subspace_dir, "direct_subspace_mean.npy"))
    comps = np.load(os.path.join(subspace_dir, "direct_subspace_components.npy"))
    return mean, comps

if __name__ == "__main__":
    run_direct_space_construction(LAYER, TOP_K)


#CUDA_VISIBLE_DEVICES=3 nohup python module4_direct_representaiton_space.py > /home/tahad/HAVOC/HAVOC/logs/module4_direct_representaiton_space.log  2>&1 &