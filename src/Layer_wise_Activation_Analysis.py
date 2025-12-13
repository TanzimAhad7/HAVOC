import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D PCA plot

# ========================================
# CONFIG
# ========================================
data_dir   = "/home/tahad/HAVOC/HAVOC/output/activations"
output_dir = "/home/tahad/HAVOC/HAVOC/output/analysis"
layers     = list(range(18, 31))   # LLaMA3 mid-late layers
topN       = 10

alpha = 1.0     # weight for inter-class separation
beta  = 0.25    # weight for intra-class compactness (negative variance)

os.makedirs(output_dir, exist_ok=True)

def cos(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# ========================================
# STORAGE
# ========================================
S_inter = []            # directional separation
S_intra = []            # compactness (negative variance)
S_total = []            # HAVOC final score
JBShield_scores = []    # 1 - cos(B, H)

centroids      = {}
all_variances  = []     # [varB, varH, varJ] per layer
cos_BH_list    = []
cos_BJ_list    = []
cos_HJ_list    = []


# ========================================
# LAYER-BY-LAYER METRICS
# ========================================
for layer in layers:

    B = np.load(f"{data_dir}/B_a_layer{layer}.npy")
    H = np.load(f"{data_dir}/H_a_layer{layer}.npy")
    J = np.load(f"{data_dir}/J_a_layer{layer}.npy")

    # ---- Compute centroids ----
    cB, cH, cJ = B.mean(axis=0), H.mean(axis=0), J.mean(axis=0)
    centroids[layer] = (cB, cH, cJ)

    # ---- Cosine similarities (directional class separation) ----
    bh = cos(cB, cH)
    bj = cos(cB, cJ)
    hj = cos(cH, cJ)

    cos_BH_list.append(bh)
    cos_BJ_list.append(bj)
    cos_HJ_list.append(hj)

    # === INTER-CLASS SEPARATION SCORE (MATCHES PAPER) ===
    S_inter_layer = (1 - bh + 1 - bj + 1 - hj) / 3.0
    S_inter.append(S_inter_layer)

    # === JBShield Score (their metric: 1 - cos(B, H)) ===
    JBShield_scores.append(1 - bh)

    # ---- Intra-class variance (compactness) ----
    varB = np.var(B, axis=0).mean()
    varH = np.var(H, axis=0).mean()
    varJ = np.var(J, axis=0).mean()

    all_variances.append([varB, varH, varJ])

    # === HAVOC INTRA-CLASS SCORE ===
    # negative mean variance (lower variance = better)
    S_intra_layer = - (varB + varH + varJ) / 3.0
    S_intra.append(S_intra_layer)

    # === FINAL HAVOC SCORE ===
    S_total_layer = alpha * S_inter_layer + beta * S_intra_layer
    S_total.append(S_total_layer)


S_inter       = np.array(S_inter)
S_intra       = np.array(S_intra)
S_total       = np.array(S_total)
JBShield_scores = np.array(JBShield_scores)
all_variances = np.array(all_variances)  # shape: (num_layers, 3)

# ========================================
# BEST LAYERS (HAVOC & JBShield)
# ========================================
best_layer_havoc    = layers[int(np.argmax(S_total))]
best_layer_inter    = layers[int(np.argmax(S_inter))]
best_layer_jbshield = layers[int(np.argmax(JBShield_scores))]

print("Best layer (HAVOC total score):", best_layer_havoc)
print("Best layer (Inter-class only):", best_layer_inter)
print("Best layer (JBShield score):", best_layer_jbshield)


# ========================================
# PLOT 1: HAVOC TOTAL SCORE (USENIX LINE PLOT)
# ========================================
plt.figure(figsize=(9, 5))
plt.plot(layers, S_total, marker="o", linewidth=2, label="HAVOC Score")
plt.axvline(best_layer_havoc, color="red", linestyle="--",
            label=f"Best Layer = {best_layer_havoc}")
plt.title("HAVOC Total Layer Score")
plt.xlabel("Layer")
plt.ylabel("Score")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(f"{output_dir}/havoc_total_score.png", dpi=300)
plt.close()


# ========================================
# PLOT 2: INTER-CLASS & JBShield SCORES
# ========================================
plt.figure(figsize=(9, 5))
plt.plot(layers, S_inter, marker="o", label="Inter-Class Score")
plt.axvline(best_layer_inter, color="green", linestyle="--",
            label=f"Peak Inter-Class = {best_layer_inter}")
plt.title("Directional Inter-Class Separation")
plt.xlabel("Layer")
plt.ylabel("Inter-Class Score")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(f"{output_dir}/inter_class_score.png", dpi=300)
plt.close()

plt.figure(figsize=(9, 5))
plt.plot(layers, JBShield_scores, marker="o", label="JBShield (1 - cos(B,H))")
plt.axvline(best_layer_jbshield, color="purple", linestyle="--",
            label=f"JBShield Best Layer = {best_layer_jbshield}")
plt.title("JBShield Layer Score")
plt.xlabel("Layer")
plt.ylabel("Score")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(f"{output_dir}/jbshield_score.png", dpi=300)
plt.close()


# ========================================
# HEATMAP: COSINE SIMILARITIES (B-H, B-J, H-J)
# ========================================
cos_matrix = np.vstack([cos_BH_list, cos_BJ_list, cos_HJ_list])  # shape: (3, num_layers)

plt.figure(figsize=(10, 3))
sns.heatmap(
    cos_matrix,
    annot=False,
    cmap="coolwarm",
    vmin=-1.0,
    vmax=1.0,
    xticklabels=layers,
    yticklabels=["B-H", "B-J", "H-J"]
)
plt.title("Cosine Similarities Across Layers")
plt.xlabel("Layer")
plt.ylabel("Class Pair")
plt.tight_layout()
plt.savefig(f"{output_dir}/cosine_heatmap.png", dpi=300)
plt.close()


# ========================================
# LaTeX TABLE: LAYER SCORES
# ========================================
table_path = os.path.join(output_dir, "layer_scores_table.tex")

with open(table_path, "w") as f:
    f.write("% Auto-generated by HAVOC analysis script\n")
    f.write("\\begin{table*}[t]\n")
    f.write("\\centering\n")
    f.write("\\caption{Layer-wise separation and compactness scores for LLaMA-3 8B (layers 18--30).}\n")
    f.write("\\label{tab:havoc-layer-scores}\n")
    f.write("\\begin{tabular}{ccccc}\n")
    f.write("\\toprule\n")
    f.write("Layer & $S_{\\text{inter}}$ & $S_{\\text{intra}}$ & $S_{\\text{HAVOC}}$ & $S_{\\text{JBShield}}$\\\\\n")
    f.write("\\midrule\n")
    for idx, layer in enumerate(layers):
        f.write(
            f"{layer} & "
            f"{S_inter[idx]:.3f} & "
            f"{S_intra[idx]:.3e} & "
            f"{S_total[idx]:.3f} & "
            f"{JBShield_scores[idx]:.3f}\\\\\n"
        )
    f.write("\\bottomrule\n")
    f.write("\\end{tabular}\n")
    f.write("\\end{table*}\n")

print("LaTeX table written to:", table_path)


# ========================================
# PCA / UMAP / t-SNE CLUSTERS FOR ALL LAYERS
# ========================================
print("\n=== Generating PCA, UMAP, t-SNE plots for ALL layers ===")

for layer in layers:

    B = np.load(f"{data_dir}/B_a_layer{layer}.npy")
    H = np.load(f"{data_dir}/H_a_layer{layer}.npy")
    J = np.load(f"{data_dir}/J_a_layer{layer}.npy")

    X = np.vstack([B, H, J])
    labels = np.array([0]*len(B) + [1]*len(H) + [2]*len(J))

    # ------------ PCA 2D ------------
    pca_2d = PCA(n_components=2).fit_transform(X)
    plt.figure(figsize=(7, 6))
    plt.scatter(pca_2d[labels==0,0], pca_2d[labels==0,1], alpha=0.5, label="Benign")
    plt.scatter(pca_2d[labels==1,0], pca_2d[labels==1,1], alpha=0.5, label="Harmful")
    plt.scatter(pca_2d[labels==2,0], pca_2d[labels==2,1], alpha=0.5, label="Jailbreak")
    plt.legend()
    plt.title(f"PCA Cluster Plot — Layer {layer}")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pca_cluster_layer{layer}.png", dpi=300)
    plt.close()

    # ------------ UMAP 2D ------------
    um = UMAP(n_components=2, n_neighbors=25, min_dist=0.1, metric="cosine")
    u = um.fit_transform(X)
    plt.figure(figsize=(7, 6))
    plt.scatter(u[labels==0,0], u[labels==0,1], alpha=0.5, label="Benign")
    plt.scatter(u[labels==1,0], u[labels==1,1], alpha=0.5, label="Harmful")
    plt.scatter(u[labels==2,0], u[labels==2,1], alpha=0.5, label="Jailbreak")
    plt.legend()
    plt.title(f"UMAP Cluster Plot — Layer {layer}")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/umap_cluster_layer{layer}.png", dpi=300)
    plt.close()

    # ------------ t-SNE 2D ------------
    ts = TSNE(
        n_components=2,
        metric="cosine",
        perplexity=30,
        learning_rate=200
    ).fit_transform(X)

    plt.figure(figsize=(7, 6))
    plt.scatter(ts[labels==0,0], ts[labels==0,1], alpha=0.5, label="Benign")
    plt.scatter(ts[labels==1,0], ts[labels==1,1], alpha=0.5, label="Harmful")
    plt.scatter(ts[labels==2,0], ts[labels==2,1], alpha=0.5, label="Jailbreak")
    plt.legend()
    plt.title(f"t-SNE Cluster Plot — Layer {layer}")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/tsne_cluster_layer{layer}.png", dpi=300)
    plt.close()


# ========================================
# 3D PCA CLUSTER FOR BEST HAVOC LAYER
# ========================================
print("\n=== Generating 3D PCA cluster for best HAVOC layer ===")
best_layer = best_layer_havoc

B = np.load(f"{data_dir}/B_a_layer{best_layer}.npy")
H = np.load(f"{data_dir}/H_a_layer{best_layer}.npy")
J = np.load(f"{data_dir}/J_a_layer{best_layer}.npy")

X = np.vstack([B, H, J])
labels = np.array([0]*len(B) + [1]*len(H) + [2]*len(J))

pca_3d = PCA(n_components=3).fit_transform(X)

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(
    pca_3d[labels==0,0], pca_3d[labels==0,1], pca_3d[labels==0,2],
    alpha=0.5, label="Benign"
)
ax.scatter(
    pca_3d[labels==1,0], pca_3d[labels==1,1], pca_3d[labels==1,2],
    alpha=0.5, label="Harmful"
)
ax.scatter(
    pca_3d[labels==2,0], pca_3d[labels==2,1], pca_3d[labels==2,2],
    alpha=0.5, label="Jailbreak"
)

ax.set_title(f"3D PCA Cluster — Best HAVOC Layer {best_layer}")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.legend()
plt.tight_layout()
plt.savefig(f"{output_dir}/pca_3d_best_layer{best_layer}.png", dpi=300)
plt.close()


# ========================================
# LAYER IMPORTANCE RANKING ALGORITHM
# ========================================
print("\n=== Computing layer importance ranking (HAVOC-based) ===")

S_total_arr = np.array(S_total)
rank_indices = np.argsort(-S_total_arr)  # descending order

ranking_path = os.path.join(output_dir, "layer_rankings.txt")
with open(ranking_path, "w") as f:
    f.write("HAVOC Layer Importance Ranking (higher S_total = more important)\n")
    f.write("---------------------------------------------------------------\n\n")
    for rank, idx in enumerate(rank_indices, start=1):
        layer = layers[idx]
        f.write(
            f"Rank {rank:2d}: Layer {layer:2d}  |  "
            f"S_total = {S_total_arr[idx]:.4f}, "
            f"S_inter = {S_inter[idx]:.4f}, "
            f"S_intra = {S_intra[idx]:.4e}, "
            f"S_JB = {JBShield_scores[idx]:.4f}\n"
        )

print("Layer rankings written to:", ranking_path)


# ========================================
# PRINT SUMMARY
# ========================================
print("\n=========================================")
print(" HAVOC LAYER SELECTION SUMMARY")
print("=========================================")
print(f"→ Best layer (HAVOC Total Score): {best_layer_havoc}")
print(f"→ Best layer (Inter-Class Only):  {best_layer_inter}")
print(f"→ Best layer (JBShield Score):    {best_layer_jbshield}")
print("\nAll visualizations + LaTeX table + ranking saved to:")
print(output_dir)
print("=========================================\n")

