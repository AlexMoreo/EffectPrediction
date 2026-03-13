import os
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import tight_layout
from sklearn.decomposition import PCA
from itertools import combinations
from joblib import Parallel, delayed
from sklearn.cross_decomposition import CCA
from sklearn.manifold import MDS, TSNE
from adjustText import adjust_text
import json


def _first_cca_for_pair(i, j, X_blocks):
    Xi = X_blocks[i]
    Xj = X_blocks[j]

    n_comp = min(Xi.shape[1], Xj.shape[1])
    cca = CCA(n_components=1)
    Ui, Uj = cca.fit_transform(Xi, Xj)

    corr = np.corrcoef(Ui[:, 0], Uj[:, 0])[0, 1]
    return i, j, float(np.abs(corr))


def sanitize_block(X, group_name, tol=1e-8):
    X = np.asarray(X, dtype=np.float64)
    keep = np.std(X, axis=0) > tol
    X = X[:, keep]
    if X.shape[1] == 0:
        raise ValueError(f"Empty group {group_name} after cancelling null var columns")
    return X

def sanitize_blocks(X_blocks, group_names, groups):
    filtered_X_blocks = []
    filtered_group_names = []
    filtered_groups = []
    for X_block, group_name, group_idx in zip(X_blocks, group_names, groups):
        try:
            X_block = sanitize_block(X_block, group_name)
            filtered_X_blocks.append(X_block)
            filtered_group_names.append(group_name)
            filtered_groups.append(group_idx)
        except ValueError as e:
            print(e)
    return filtered_X_blocks, filtered_group_names, filtered_groups

def block_cca_heatmap_parallel(
    X,
    groups,
    group_names=None,
    n_jobs=-1,
    ax=None,
    title="Block-wise canonical correlation",
    annotate=True,
    pca_reduce=None
):
    X = np.asarray(X)
    if group_names is None:
        group_names = [f"G{i}" for i in range(len(groups))]

    X_blocks = [X[:, cols] for cols in groups]
    # revove colums with very low or null variance
    X_blocks, group_names, groups = sanitize_blocks(X_blocks, group_names, groups)
    n_groups = len(groups)

    if pca_reduce is not None:
        X_blocks = [PCA(n_components=min(pca_reduce, Xg.shape[1])).fit_transform(Xg) for Xg in X_blocks]


    pairs = list(combinations(range(n_groups), 2))

    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_first_cca_for_pair)(i, j, X_blocks)
        for i, j in pairs
    )

    corr_matrix = np.eye(n_groups, dtype=float)
    for i, j, corr in results:
        corr_matrix[i, j] = corr
        corr_matrix[j, i] = corr

    if ax is None:
        scale = max(n_groups // 9, 1)
        fig, ax = plt.subplots(figsize=(6*scale, 5*scale))

    im = ax.imshow(corr_matrix, vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(n_groups))
    ax.set_yticks(range(n_groups))
    ax.set_xticklabels(group_names, rotation=45, ha="right")
    ax.set_yticklabels(group_names)
    ax.set_title(title)

    if annotate:
        for i in range(n_groups):
            for j in range(n_groups):
                ax.text(j, i, f"{corr_matrix[i, j]:.2f}", ha="center", va="center")

    plt.tight_layout()
    return corr_matrix, group_names, ax

def get_group_indexes(feat_names, exclude_blocks=None, level=2):
    # use level=1 for macro groups
    # use level=2 for sub groups
    assert level in [1,2], 'wrong level, use 1 or 2'
    exclude_blocks = exclude_blocks or []
    # feature names have the following syntax: <GROUP>--<SUBGROUP>--name
    group_of_feat = np.asarray(['--'.join(f.split('--')[:level]) for f in feat_names])
    groups_unique = np.unique([g for g in group_of_feat if g not in exclude_blocks])
    all_idx = np.arange(len(feat_names))
    groups_idx = [all_idx[group_of_feat==g] for g in groups_unique]
    return groups_unique, groups_idx


def plot_block_embedding_colored(coords, subgroup_names, title="Block embedding", outpath=None):
    coords = np.asarray(coords)

    groups = []
    subgroups = []

    for name in subgroup_names:
        if "--" in name:
            group, subgroup = name.split("--", 1)
        else:
            group, subgroup = "Unknown", name
        groups.append(group)
        subgroups.append(subgroup)

    unique_groups = sorted(set(groups))
    cmap = plt.get_cmap("tab10")
    color_map = {g: cmap(i % 10) for i, g in enumerate(unique_groups)}

    plt.figure(figsize=(10, 8))

    for group in unique_groups:
        idx = [i for i, g in enumerate(groups) if g == group]
        plt.scatter(
            coords[idx, 0],
            coords[idx, 1],
            s=80,
            color=color_map[group],
            label=group,
            edgecolor="white", linewidth=0.7
        )

    texts = []

    for i, label in enumerate(subgroups):
        t = plt.text(
            coords[i, 0],
            coords[i, 1],
            label,
            fontsize=9
        )
        texts.append(t)

    # adjust_text(
    #     texts,
    #     arrowprops=dict(arrowstyle="-", color="gray", lw=0.5)
    # )

    adjust_text(
        texts,
        expand_points=(1.2, 1.4),
        expand_text=(1.2, 1.4),
        force_text=(0.2, 0.5),
        arrowprops=dict(arrowstyle="-", color="gray", lw=0.5)
    )

    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.legend(title="Group")
    plt.grid(alpha=0.3)
    if outpath is None:
        plt.tight_layout()
        plt.show()
    else:
        plt.savefig(outpath, bbox_inches="tight")

def plot_block_embedding_selected(
    coords,
    subgroup_names,
    selected_dict=None,
    title="Block embedding",
    figsize=(10, 8), outpath=None
):
    coords = np.asarray(coords)

    # Parsear GROUP y SUBGROUP
    groups = []
    subgroups = []
    for name in subgroup_names:
        if "--" in name:
            g, sg = name.split("--", 1)
        else:
            g, sg = "Unknown", name
        groups.append(g)
        subgroups.append(sg)

    unique_groups = sorted(set(groups))
    cmap = plt.get_cmap("tab20")
    color_map = {g: cmap(i % 20) for i, g in enumerate(unique_groups)}

    plt.figure(figsize=figsize)

    for g in unique_groups:
        idx = [i for i, gg in enumerate(groups) if gg == g]
        plt.scatter(
            coords[idx, 0],
            coords[idx, 1],
            s=90,
            color=color_map[g],
            label=g,
            edgecolor="white",
            linewidth=0.8,
            zorder=2
        )

    if selected_dict:
        importances = np.array(list(selected_dict.values()), dtype=float)
        imp_min = importances.min()
        imp_max = importances.max()

        def scale_linewidth(val, lw_min=1.5, lw_max=5.0):
            if imp_max == imp_min:
                return 3.0
            return lw_min + (val - imp_min) / (imp_max - imp_min) * (lw_max - lw_min)

        def scale_size(val, s_min=180, s_max=320):
            if imp_max == imp_min:
                return 240
            return s_min + (val - imp_min) / (imp_max - imp_min) * (s_max - s_min)

        for i, sel_name in enumerate(subgroup_names):
            if sel_name in selected_dict:
                imp = selected_dict[sel_name]
                plt.scatter(
                    coords[i, 0],
                    coords[i, 1],
                    s=scale_size(imp),
                    facecolors="none",
                    edgecolors="black",
                    linewidths=scale_linewidth(imp)**1.2,
                    zorder=4
                )

    # etiquetas
    texts = []
    for i, label in enumerate(subgroups):
        sel_name = subgroup_names[i]

        # opcional: añadir importancia a seleccionados
        if selected_dict is not None and sel_name in selected_dict:
            text_label = f"{label} ({selected_dict[sel_name]:.2f})"
            fontweight = "bold"
        else:
            text_label = label
            fontweight = "normal"

        t = plt.text(
            coords[i, 0],
            coords[i, 1],
            text_label,
            fontsize=9,
            fontweight=fontweight,
            zorder=5
        )
        texts.append(t)

    adjust_text(
        texts,
        arrowprops=dict(arrowstyle="-", color="gray", lw=0.5)
    )

    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.legend(title="Group", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.grid(alpha=0.3)
    if outpath is None:
        plt.tight_layout()
        plt.show()
    else:
        plt.savefig(outpath, bbox_inches="tight")

def represent_blocks_MDS(corr_matrix, subgroup_names, selected_dict=None, savepath=None):
    dist = 1 - corr_matrix

    mds = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=0
    )

    coords = mds.fit_transform(dist)

    plot_block_embedding_selected(
        coords,
        subgroup_names,
        selected_dict=selected_dict,
        title="MDS of block redundancy",
        outpath=savepath
    )


def represent_blocks_UMAP(corr_matrix, subgroup_names, selected_dict=None, random_state=0, savepath=None):
    import umap

    corr_matrix = 0.5 * (corr_matrix + corr_matrix.T)
    corr_matrix = np.clip(corr_matrix, 0.0, 1.0)

    # convertir similitud -> distancia
    dist_matrix = 1.0 - corr_matrix

    # UMAP sobre distancias precomputadas
    reducer = umap.UMAP(
        n_components=2,
        metric="precomputed",
        n_neighbors=min(10, len(subgroup_names) - 1),
        min_dist=0.1,
        random_state=random_state
    )

    coords = reducer.fit_transform(dist_matrix)

    plot_block_embedding_selected(
        coords,
        subgroup_names,
        selected_dict=selected_dict,
        title="UMAP of block redundancy",
        outpath=savepath
    )


def represent_blocks_tSNE(corr_matrix, subgroup_names, selected_dict=None, random_state=0, savepath=None):
    corr_matrix = np.asarray(corr_matrix, dtype=float)

    # por seguridad
    corr_matrix = 0.5 * (corr_matrix + corr_matrix.T)
    corr_matrix = np.clip(corr_matrix, 0.0, 1.0)

    # similitud -> distancia
    dist_matrix = 1.0 - corr_matrix
    np.fill_diagonal(dist_matrix, 0.0)

    n_blocks = len(subgroup_names)

    # la perplexity debe ser menor que el número de muestras
    perplexity = min(10, n_blocks - 1)

    tsne = TSNE(
        n_components=2,
        metric="precomputed",
        perplexity=perplexity,
        init="random",
        random_state=random_state
    )

    coords = tsne.fit_transform(dist_matrix)

    plot_block_embedding_selected(
        coords,
        subgroup_names,
        selected_dict=selected_dict,
        title="t-SNE of block redundancy",
        outpath=savepath
    )

def hierarchical_clustering(corr_matrix, subgroup_names, savepath=None):
    dist = 1 - corr_matrix
    from scipy.cluster.hierarchy import linkage, dendrogram

    Z = linkage(dist, method="average")

    plt.figure(figsize=(10, 5))
    dendrogram(Z, labels=subgroup_names)
    plt.title("Feature block redundancy dendrogram")
    if savepath is not None:
        plt.savefig(savepath, bbox_inches="tight")
    else:
        plt.show()


if __name__ == '__main__':
    from data import load_dataset
    path = '../datasets/toxicity_dataset'
    data = load_dataset(path, n_classes=5, filter_abandoned_activity=False)

    out_path = Path('../colineal_analysis/')
    os.makedirs(out_path, exist_ok=True)
    os.makedirs(out_path/'pickles', exist_ok=True)

    X = data.X

    group_names, groups = get_group_indexes(data.covariate_names, level=1)
    subgroup_names, subgroups = get_group_indexes(data.covariate_names, level=2)

    # first level (macro groups)
    n_components = 10
    out_name = f'colineal_group_pca{n_components}'
    pickle_path = out_path / 'pickles' / (out_name+'.pkl')
    if not os.path.exists(pickle_path):
        corr_matrix, group_names, ax = block_cca_heatmap_parallel(X, groups, group_names=group_names, pca_reduce=n_components)
        pickle.dump({'corr_matrix': corr_matrix, 'group_names': group_names}, open(pickle_path, 'wb'))
        plt.savefig(out_path / (out_name + '.pdf'), bbox_inches="tight")

    # second level (sub groups)
    n_components = 5
    out_name = f'colineal_subgroup_pca{n_components}'
    pickle_path = out_path / 'pickles' / (out_name + '.pkl')
    if not os.path.exists(pickle_path):
        corr_matrix, subgroup_names, ax = block_cca_heatmap_parallel(X, subgroups, group_names=subgroup_names, pca_reduce=n_components)
        pickle.dump({'corr_matrix': corr_matrix, 'group_names': subgroup_names}, open(pickle_path, 'wb'))
        plt.savefig(out_path / (out_name+'.pdf'), bbox_inches="tight")
    dict_output = pickle.load(open(pickle_path, 'rb'))

    # within sub group
    n_components = 5
    for group_name in group_names:
        out_name = f'colineal_{group_name}_pca{n_components}'
        pickle_path = out_path / 'pickles' / (out_name + '.pkl')
        if not os.path.exists(pickle_path):
            print(f'Generating CCA for {group_name}')
            subgr_names = []
            subgr_idx = []
            for subgroup_name, subgroup_idx in zip(subgroup_names, subgroups):
                gr_name, subgr_name = subgroup_name.split('--')
                if gr_name != group_name: continue
                subgr_names.append(subgr_name)
                subgr_idx.append(subgroup_idx)
            corr_matrix, subgr_names, ax = block_cca_heatmap_parallel(X, subgr_idx, group_names=subgr_names, pca_reduce=n_components)
            pickle.dump({'corr_matrix': corr_matrix, 'group_names': subgr_names}, open(pickle_path, 'wb'))
            plt.savefig(out_path / (out_name +'.pdf'), bbox_inches="tight")

    for dataset in ['activity', 'toxicity', 'diversity']:
        selected_features_path = f'../results/feat_importance/samplesize500/{dataset}/5_classes/feat_importance.json'
        with open(selected_features_path, 'r') as f:
            selected_dict = json.load(f)
            selected_dict = {feat: round(val*100, 1) for feat, val in selected_dict.items()}
        represent_blocks_MDS(corr_matrix = dict_output['corr_matrix'], subgroup_names = dict_output['group_names'], selected_dict=selected_dict, savepath=out_path/f'MDS_{dataset}.pdf')
        represent_blocks_UMAP(corr_matrix=dict_output['corr_matrix'], subgroup_names=dict_output['group_names'], selected_dict=selected_dict, savepath=out_path/f'UMAP_{dataset}.pdf')
        represent_blocks_tSNE(corr_matrix=dict_output['corr_matrix'], subgroup_names=dict_output['group_names'], selected_dict=selected_dict, savepath=out_path/f'tSNE_{dataset}.pdf')
    hierarchical_clustering(corr_matrix=dict_output['corr_matrix'], subgroup_names=dict_output['group_names'], savepath=out_path/f'Hierarchical_Clustering.pdf')

