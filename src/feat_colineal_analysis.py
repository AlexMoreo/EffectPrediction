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
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list, fcluster, optimal_leaf_ordering
from scipy.spatial.distance import squareform
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe


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

def block_cca_heatmap_parallel_(
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

def block_cca_heatmap_parallel(
    X,
    groups,
    group_names=None,
    n_jobs=-1,
    ax=None,
    title="Block-wise canonical correlation",
    annotate=True,
    pca_reduce=None,
    reorder_for_plot=True,
    use_optimal_leaf_ordering=True,
    draw_cluster_boundaries=False,
    n_clusters=4
):
    """
    Compute a block-wise CCA correlation matrix and optionally reorder it
    only for visualization purposes.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input feature matrix.

    groups : list of lists
        Each element contains the column indices corresponding to one block.

    group_names : list of str, optional
        Names of the feature blocks.

    n_jobs : int
        Number of parallel jobs.

    ax : matplotlib axis, optional
        Axis where the heatmap will be drawn.

    title : str
        Plot title.

    annotate : bool
        Whether to annotate each cell with its correlation value.

    pca_reduce : int or None
        If not None, reduce each block with PCA before running CCA.

    reorder_for_plot : bool
        If True, reorder rows/columns only for plotting, keeping the original
        correlation matrix unchanged in the returned values.

    use_optimal_leaf_ordering : bool
        If True, refine hierarchical ordering for cleaner visual grouping.

    draw_cluster_boundaries : bool
        If True, draw visual boundaries between clusters on the reordered plot.

    n_clusters : int
        Number of clusters used to draw boundaries if
        draw_cluster_boundaries=True.

    Returns
    -------
    corr_matrix : np.ndarray
        Original (non-reordered) block correlation matrix.

    group_names : list of str
        Original block names after sanitization.

    ax : matplotlib axis
        Axis containing the plot.
    """
    X = np.asarray(X)

    if group_names is None:
        group_names = [f"G{i}" for i in range(len(groups))]

    # Build block matrices
    X_blocks = [X[:, cols] for cols in groups]

    # Remove columns with near-zero variance and update metadata
    X_blocks, group_names, groups = sanitize_blocks(X_blocks, group_names, groups)
    n_groups = len(groups)

    # Optional PCA reduction per block
    if pca_reduce is not None:
        X_blocks = [
            PCA(n_components=min(pca_reduce, Xg.shape[1], Xg.shape[0] - 1)).fit_transform(Xg)
            for Xg in X_blocks
        ]

    # Compute pairwise first canonical correlations
    pairs = list(combinations(range(n_groups), 2))

    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_first_cca_for_pair)(i, j, X_blocks)
        for i, j in pairs
    )

    # Build the original correlation matrix
    corr_matrix = np.eye(n_groups, dtype=float)
    for i, j, corr in results:
        corr_matrix[i, j] = corr
        corr_matrix[j, i] = corr

    # ------------------------------------------------------------------
    # Prepare reordered copies ONLY for plotting
    # ------------------------------------------------------------------
    corr_plot = corr_matrix.copy()
    group_names_plot = list(group_names)
    boundaries = []

    if reorder_for_plot and n_groups > 1:
        # Convert similarity to distance
        dist = 1.0 - corr_plot
        np.fill_diagonal(dist, 0.0)

        # Hierarchical clustering requires condensed distance format
        dist_condensed = squareform(dist, checks=False)

        # Compute linkage for ordering
        Z = linkage(dist_condensed, method="average")

        # Optional refinement for a cleaner visual arrangement
        if use_optimal_leaf_ordering:
            Z = optimal_leaf_ordering(Z, dist_condensed)

        # Get leaf order
        order = leaves_list(Z)

        # Reorder only the plotting copies
        corr_plot = corr_plot[order][:, order]
        group_names_plot = [group_names_plot[i] for i in order]

        # Compute cluster boundaries if requested
        if draw_cluster_boundaries:
            cluster_labels = fcluster(Z, t=n_clusters, criterion="maxclust")
            cluster_labels_plot = cluster_labels[order]
            boundaries = np.where(np.diff(cluster_labels_plot) != 0)[0] + 1

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    if ax is None:
        scale = max(n_groups // 9, 1)
        fig, ax = plt.subplots(figsize=(6 * scale, 5 * scale))

    im = ax.imshow(corr_plot, vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(n_groups))
    ax.set_yticks(range(n_groups))
    ax.set_xticklabels(group_names_plot, rotation=45, ha="right")
    ax.set_yticklabels(group_names_plot)
    ax.set_title(title)

    # Annotate matrix values
    if annotate:
        for i in range(n_groups):
            for j in range(n_groups):
                ax.text(j, i, f"{corr_plot[i, j]:.2f}", ha="center", va="center")

    # Draw visual boundaries between clusters, if requested
    if len(boundaries)>0:
        for b in boundaries:
            ax.axhline(b - 0.5, color="white", lw=2)
            ax.axvline(b - 0.5, color="white", lw=2)

    plt.tight_layout()

    # Return the ORIGINAL matrix and ORIGINAL names (post-sanitization),
    # not the reordered plotting copies
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





def plot_block_embedding_selected_AB(
    coords,
    subgroup_names,
    selected_dict_A=None,
    selected_dict_B=None,
    title="Block embedding",
    figsize=(10, 8),
    outpath=None
):
    coords = np.asarray(coords)

    selected_dict_A = {} if selected_dict_A is None else dict(selected_dict_A)
    selected_dict_B = {} if selected_dict_B is None else dict(selected_dict_B)

    # prase GROUP and SUBGROUP
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

    fig, ax = plt.subplots(figsize=figsize)

    # color points by GROUP
    for g in unique_groups:
        idx = [i for i, gg in enumerate(groups) if gg == g]
        ax.scatter(
            coords[idx, 0],
            coords[idx, 1],
            s=90,
            color=color_map[g],
            label=g,
            edgecolor="white",
            linewidth=0.8,
            zorder=2
        )


    def make_scalers(selected_dict, lw_min=1.5, lw_max=5.0, s_min=180, s_max=320):
        if not selected_dict:
            return (
                lambda val: 0.0,
                lambda val: 0.0
            )

        importances = np.array(list(selected_dict.values()), dtype=float)
        imp_min = importances.min()
        imp_max = importances.max()

        def scale_linewidth(val):
            if imp_max == imp_min:
                return 3.0
            return lw_min + (val - imp_min) / (imp_max - imp_min) * (lw_max - lw_min)

        def scale_size(val):
            if imp_max == imp_min:
                return 240.0
            return s_min + (val - imp_min) / (imp_max - imp_min) * (s_max - s_min)

        return scale_linewidth, scale_size

    scale_lw_A, scale_size_A = make_scalers(selected_dict_A)
    scale_lw_B, scale_size_B = make_scalers(selected_dict_B)

    # Draw selection A/B
    for i, sel_name in enumerate(subgroup_names):
        x, y = coords[i, 0], coords[i, 1]
        in_A = sel_name in selected_dict_A
        in_B = sel_name in selected_dict_B

        # Only A -> black
        if in_A and not in_B:
            impA = float(selected_dict_A[sel_name])
            ax.scatter(
                x, y,
                s=scale_size_A(impA),
                facecolors="none",
                edgecolors="black",
                linewidths=scale_lw_A(impA) ** 1.1,
                zorder=4
            )

        # Only B -> red
        elif in_B and not in_A:
            impB = float(selected_dict_B[sel_name])
            ax.scatter(
                x, y,
                s=scale_size_B(impB),
                facecolors="none",
                edgecolors="red",
                linewidths=scale_lw_B(impB) ** 1.1,
                zorder=4
            )

        # Both -> double ring
        elif in_A and in_B:
            impA = float(selected_dict_A[sel_name])
            impB = float(selected_dict_B[sel_name])

            max_ring = max(scale_size_A(impA), scale_size_B(impB))
            min_ring = min(scale_size_A(impA), scale_size_B(impB))
            ax.scatter(
                x, y,
                # s=max_ring + 70,
                s=max_ring + min_ring,
                facecolors="none",
                edgecolors="red",
                linewidths=scale_lw_B(impB) ** 1.1,
                zorder=4
            )

            ax.scatter(
                x, y,
                s=min_ring,
                facecolors="none",
                edgecolors="black",
                linewidths=scale_lw_A(impA) ** 1.1,
                zorder=5
            )

    # labels
    texts = []
    for i, label in enumerate(subgroups):
        sel_name = subgroup_names[i]
        in_A = sel_name in selected_dict_A
        in_B = sel_name in selected_dict_B

        # if in_A and in_B:
        #     text_label = f"{label} (A:{selected_dict_A[sel_name]:.2f}, B:{selected_dict_B[sel_name]:.2f})"
        #     fontweight = "bold"
        # elif in_A:
        #     text_label = f"{label} (A:{selected_dict_A[sel_name]:.2f})"
        #     fontweight = "bold"
        # elif in_B:
        #     text_label = f"{label} (B:{selected_dict_B[sel_name]:.2f})"
        #     fontweight = "bold"
        if in_A or in_B:
            text_label = f"{label}"
            fontweight = "bold"
        else:
            text_label = label
            fontweight = "normal"

        t = ax.text(
            coords[i, 0],
            coords[i, 1],
            text_label,
            fontsize=9,
            fontweight=fontweight,
            color="black",
            zorder=6,
            path_effects=[
                pe.withStroke(linewidth=2, foreground="white", alpha=0.7),
            ]
        )
        texts.append(t)

    adjust_text(
        texts,
        ax=ax,
        arrowprops=dict(arrowstyle="-", color="gray", lw=0.5)
    )

    # Leyenda de grupos
    handles_groups, labels_groups = ax.get_legend_handles_labels()

    # Leyenda adicional para A/B

    if selected_dict_B is not None and len(selected_dict_B)>0:
        selection_handles = [
            Line2D([0], [0], marker="o", color="black", markerfacecolor="none",
                   markersize=10, linewidth=0, markeredgewidth=2, label="Selected in A"),
            Line2D([0], [0], marker="o", color="red", markerfacecolor="none",
                   markersize=10, linewidth=0, markeredgewidth=2, label="Selected in B"),
        ]
    else:
        selection_handles = []
        # selection_handles = [
        #     Line2D([0], [0], marker="o", color="black", markerfacecolor="none",
        #            markersize=10, linewidth=0, markeredgewidth=2, label="Selected"),
        #
        # ]

    legend1 = ax.legend(
        handles_groups,
        labels_groups,
        title="Group",
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )


    if len(selection_handles)>0:
        ax.add_artist(legend1)
        ax.legend(
            handles=selection_handles,
            title="Selection",
            bbox_to_anchor=(1.02, 0.55),
            loc="upper left"
        )

    ax.set_title(title)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.grid(alpha=0.3)

    if outpath is None:
        plt.tight_layout()
        plt.show()
    else:
        plt.savefig(outpath, bbox_inches="tight", pad_inches=0.6, dpi=300)
        plt.close(fig)

def represent_blocks_MDS(
    corr_matrix,
    subgroup_names,
    selected_dict_A=None,
    selected_dict_B=None,
    random_state=0,
    savepath=None
):
    dist = 1 - corr_matrix

    mds = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=0
    )

    coords = mds.fit_transform(dist)

    plot_block_embedding_selected_AB(
        coords,
        subgroup_names,
        selected_dict_A=selected_dict_A,
        selected_dict_B=selected_dict_B,
        title="MDS of block redundancy",
        outpath=savepath
    )


def represent_blocks_UMAP(
    corr_matrix,
    subgroup_names,
    selected_dict_A=None,
    selected_dict_B=None,
    random_state=0,
    savepath=None
):
    import umap

    corr_matrix = np.asarray(corr_matrix, dtype=float)
    corr_matrix = 0.5 * (corr_matrix + corr_matrix.T)
    corr_matrix = np.clip(corr_matrix, 0.0, 1.0)

    dist_matrix = 1.0 - corr_matrix
    np.fill_diagonal(dist_matrix, 0.0)

    reducer = umap.UMAP(
        n_components=2,
        metric="precomputed",
        n_neighbors=min(10, len(subgroup_names) - 1),
        min_dist=0.1,
        random_state=random_state
    )

    coords = reducer.fit_transform(dist_matrix)

    plot_block_embedding_selected_AB(
        coords,
        subgroup_names,
        selected_dict_A=selected_dict_A,
        selected_dict_B=selected_dict_B,
        title="UMAP of block redundancy",
        outpath=savepath
    )


def represent_blocks_tSNE(    corr_matrix,
    subgroup_names,
    selected_dict_A=None,
    selected_dict_B=None,
    random_state=0,
    savepath=None
):
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

    plot_block_embedding_selected_AB(
        coords,
        subgroup_names,
        selected_dict_A=selected_dict_A,
        selected_dict_B=selected_dict_B,
        title="t-SNE of block redundancy",
        outpath=savepath
    )

def hierarchical_clustering(corr_matrix, subgroup_names, savepath=None):
    dist = 1 - corr_matrix
    np.fill_diagonal(dist, 0.0)

    Z = linkage(squareform(dist, checks=False), method="average")

    plt.figure(figsize=(10, 5))
    dendrogram(Z, labels=subgroup_names)
    plt.title("Feature block redundancy dendrogram")
    if savepath is not None:
        plt.savefig(savepath, bbox_inches="tight")
    else:
        plt.show()


def hierarchical_clustering_selected(
    corr_matrix,
    subgroup_names,
    selected_dict_A=None,
    selected_dict_B=None,
    savepath=None,
    figsize=(12, 12)
):
    corr_matrix = np.asarray(corr_matrix, dtype=float)
    corr_matrix = 0.5 * (corr_matrix + corr_matrix.T)
    corr_matrix = np.clip(corr_matrix, 0.0, 1.0)

    selected_dict_A = {} if selected_dict_A is None else dict(selected_dict_A)
    selected_dict_B = {} if selected_dict_B is None else dict(selected_dict_B)

    # parsear GROUP y SUBGROUP
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

    # distancia y linkage
    dist = 1.0 - corr_matrix
    np.fill_diagonal(dist, 0.0)
    Z = linkage(squareform(dist, checks=False), method="average")

    fig, ax = plt.subplots(figsize=figsize)

    # usar subgroups como labels para que no salga GROUP--SUBGROUP entero
    ddata = dendrogram(
        Z,
        labels=subgroups,
        orientation="right",
        leaf_font_size=10,
        color_threshold=0,
        above_threshold_color="gray",
        ax=ax
    )

    ax.set_title("Feature block redundancy dendrogram")
    ax.set_xlabel("Distance (1 - canonical correlation)")
    ax.set_ylabel("")

    # orden real de las hojas en el plot
    leaf_order = ddata["leaves"]

    # colorear etiquetas por GROUP
    yticklabels = ax.get_yticklabels()
    for plot_pos, tick in enumerate(yticklabels):
        original_idx = leaf_order[plot_pos]
        tick.set_color(color_map[groups[original_idx]])

        # bold si está en alguna selección
        full_name = subgroup_names[original_idx]
        if full_name in selected_dict_A or full_name in selected_dict_B:
            tick.set_fontweight("bold")

    # coordenadas y de las hojas en scipy dendrogram horizontal:
    # suelen ser 5, 15, 25, ...
    y_positions = np.arange(len(leaf_order)) * 10 + 5

    # añadir marcadores A/B a la izquierda del texto
    x_min, x_max = ax.get_xlim()
    x_span = x_max - x_min

    x_A = x_min - 0.06 * x_span
    x_B = x_min - 0.03 * x_span

    for plot_pos, original_idx in enumerate(leaf_order):
        y = y_positions[plot_pos]
        full_name = subgroup_names[original_idx]

        if full_name in selected_dict_A:
            ax.scatter(
                x_A, y,
                s=45,
                color="black",
                zorder=5,
                clip_on=False
            )

        if full_name in selected_dict_B:
            ax.scatter(
                x_B, y,
                s=45,
                color="red",
                zorder=5,
                clip_on=False
            )

    # expandir límites para que se vean los marcadores
    ax.set_xlim(x_min - 0.10 * x_span, x_max)

    # leyenda grupos
    group_handles = [
        Line2D([0], [0], color=color_map[g], lw=3, label=g)
        for g in unique_groups
    ]

    sel_handles = [
        Line2D([0], [0], marker="o", color="black", linestyle="None", markersize=7, label="Selected in A"),
        Line2D([0], [0], marker="o", color="red", linestyle="None", markersize=7, label="Selected in B"),
    ]

    legend1 = ax.legend(
        handles=group_handles,
        title="Group",
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )
    ax.add_artist(legend1)

    ax.legend(
        handles=sel_handles,
        title="Selection",
        bbox_to_anchor=(1.02, 0.45),
        loc="upper left"
    )

    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, bbox_inches="tight", dpi=300)
        plt.close(fig)
    else:
        plt.show()


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def annotated_clustermap(
        corr_matrix,
        subgroup_names,
        selected_dict_A=None,
        selected_dict_B=None,
        figsize=(12, 12),
        cmap="viridis",
        n_clusters=None,
        savepath=None,
        cbar_pos=(1., 0.125, 0.02, 0.2)  # placement of colorbar (x, y, width, height) in figure fraction
):
    """
    Annotated clustermap with colorbar and legends placed to the right.
    - If both selected_dict_A and selected_dict_B are provided, two selection columns
      ("A" and "B") and two legend entries are shown.
    - If only one of them is provided, a single column "Selected" is shown and
      the legend uses label "Selected".
    """

    corr_matrix = np.asarray(corr_matrix, dtype=float)
    corr_matrix = 0.5 * (corr_matrix + corr_matrix.T)
    corr_matrix = np.clip(corr_matrix, 0.0, 1.0)

    # normalize inputs: keep as empty dicts if None
    selected_dict_A = {} if selected_dict_A is None else dict(selected_dict_A)
    selected_dict_B = {} if selected_dict_B is None else dict(selected_dict_B)

    # Parse GROUP and SUBGROUP from names "GROUP--SUBGROUP"
    groups = []
    subgroups = []
    for name in subgroup_names:
        if "--" in name:
            g, sg = name.split("--", 1)
        else:
            g, sg = "Unknown", name
        groups.append(g)
        subgroups.append(sg)

    # Build DataFrame for seaborn heatmap
    corr_df = pd.DataFrame(corr_matrix, index=subgroups, columns=subgroups)

    # Color palette for group annotation
    unique_groups = sorted(set(groups))
    group_palette = sns.color_palette("tab20", n_colors=max(2, len(unique_groups)))
    group_color_map = {g: group_palette[i % len(group_palette)] for i, g in enumerate(unique_groups)}
    group_colors = pd.Series(groups, index=subgroups).map(group_color_map)

    # Determine which selection columns to show
    show_A = bool(selected_dict_A)
    show_B = bool(selected_dict_B)

    # If exactly one selection dict is provided, show a single column called "Selected"
    annotation_columns = {"Group": group_colors}

    if show_A and show_B:
        # both supplied -> keep separate A and B columns
        sel_A_colors = pd.Series(
            ["black" if full_name in selected_dict_A else "white" for full_name in subgroup_names],
            index=subgroups
        )
        sel_B_colors = pd.Series(
            ["red" if full_name in selected_dict_B else "white" for full_name in subgroup_names],
            index=subgroups
        )
        annotation_columns["A"] = sel_A_colors
        annotation_columns["B"] = sel_B_colors

    elif show_A and not show_B:
        # only A -> single "Selected" column (black)
        sel_colors = pd.Series(
            ["black" if full_name in selected_dict_A else "white" for full_name in subgroup_names],
            index=subgroups
        )
        annotation_columns["Selected"] = sel_colors

    elif show_B and not show_A:
        # only B -> single "Selected" column (red)
        sel_colors = pd.Series(
            ["red" if full_name in selected_dict_B else "white" for full_name in subgroup_names],
            index=subgroups
        )
        annotation_columns["Selected"] = sel_colors

    # DataFrame used as row_colors / col_colors
    row_colors = pd.DataFrame(annotation_columns, index=subgroups)
    col_colors = row_colors.copy()

    # Compute clustering for boundaries (optional visual aid)
    dist = 1.0 - corr_matrix
    np.fill_diagonal(dist, 0.0)
    dist_condensed = squareform(dist, checks=False)
    Z = linkage(dist_condensed, method="average")
    cluster_labels = None
    if n_clusters is not None and n_clusters > 1:
        cluster_labels = fcluster(Z, t=n_clusters, criterion="maxclust")

    # Create clustermap with colorbar positioned to the right (cbar_pos)
    g = sns.clustermap(
        corr_df,
        row_cluster=True,
        col_cluster=True,
        row_colors=row_colors,
        col_colors=col_colors,
        cmap=cmap,
        vmin=0,
        vmax=1,
        linewidths=0,
        figsize=figsize,
        cbar_kws={"label": "Canonical correlation"},
        cbar_pos=cbar_pos
    )

    g.fig.suptitle("Annotated clustermap of block redundancy", y=1.02)

    # Force tick label readability
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90, fontsize=8)
    plt.setp(g.ax_heatmap.get_yticklabels(), fontsize=8)

    # Draw cluster boundaries using the visual order of the clustermap
    row_order = g.dendrogram_row.reordered_ind
    if cluster_labels is not None:
        row_order = g.dendrogram_row.reordered_ind
        cluster_labels_ordered = cluster_labels[row_order]
        boundaries = np.where(np.diff(cluster_labels_ordered) != 0)[0] + 1

        for b in boundaries:
            g.ax_heatmap.hlines(b, *g.ax_heatmap.get_xlim(), colors="white", linewidth=2)
            g.ax_heatmap.vlines(b, *g.ax_heatmap.get_ylim(), colors="white", linewidth=2)

    # Build legend handles
    group_handles = [Line2D([0], [0], color=group_color_map[g], lw=4, label=g) for g in unique_groups]

    # Build selection legend depending on which columns were shown
    selection_handles = []
    if show_A and show_B:
        selection_handles.append(Line2D([0], [0], marker="s", color="black", linestyle="None",
                                        markersize=8, label="Selected in A"))
        selection_handles.append(Line2D([0], [0], marker="s", color="red", linestyle="None",
                                        markersize=8, label="Selected in B"))
    elif show_A and not show_B:
        selection_handles.append(Line2D([0], [0], marker="s", color="black", linestyle="None",
                                        markersize=8, label="Selected"))
    elif show_B and not show_A:
        selection_handles.append(Line2D([0], [0], marker="s", color="red", linestyle="None",
                                        markersize=8, label="Selected"))

    # Place legends to the right: first the group legend, then the selection legend
    legend1 = g.ax_col_dendrogram.legend(
        handles=group_handles,
        title="Group",
        bbox_to_anchor=(1.15, -0.3),
        loc="upper left"
    )
    g.ax_col_dendrogram.add_artist(legend1)

    if selection_handles:
        g.ax_col_dendrogram.legend(
            handles=selection_handles,
            title="Selection",
            bbox_to_anchor=(1.15, -2),
            loc="upper left"
        )

    # Save or show
    if savepath is not None:

        plt.savefig(savepath, bbox_inches="tight", pad_inches=0.6, dpi=300)
        plt.close(g.fig)
    else:
        plt.show()

    return g, cluster_labels




def annotated_clustermap_with_boundaries(
    corr_matrix,
    subgroup_names,
    selected_dict_A=None,
    selected_dict_B=None,
    figsize=(12, 12),
    cmap="viridis",
    n_clusters=7,
    savepath=None
):
    """
    Create an annotated clustermap of block redundancy.

    The heatmap displays canonical correlations between feature blocks.
    Rows/columns are reordered via hierarchical clustering.

    Additional annotations indicate:
        - feature block group
        - blocks selected under discretization A
        - blocks selected under discretization B

    Cluster boundaries are drawn on the heatmap to highlight groups of
    highly redundant feature blocks.

    Parameters
    ----------
    corr_matrix : array-like (n_blocks x n_blocks)
        Matrix of canonical correlations between feature blocks.

    subgroup_names : list of str
        Names of feature blocks in the format "GROUP--SUBGROUP".

    selected_dict_A : dict (optional)
        Dictionary mapping block name -> importance for selection A.

    selected_dict_B : dict (optional)
        Dictionary mapping block name -> importance for selection B.

    figsize : tuple
        Size of the clustermap figure.

    cmap : str
        Colormap used for the heatmap.

    n_clusters : int
        Number of clusters used to draw visual boundaries.

    savepath : str or None
        If provided, the figure is saved to this path.
    """

    corr_matrix = np.asarray(corr_matrix, dtype=float)

    # Ensure symmetry and valid correlation range
    corr_matrix = 0.5 * (corr_matrix + corr_matrix.T)
    corr_matrix = np.clip(corr_matrix, 0.0, 1.0)

    selected_dict_A = {} if selected_dict_A is None else dict(selected_dict_A)
    selected_dict_B = {} if selected_dict_B is None else dict(selected_dict_B)

    # ------------------------------------------------------------------
    # Parse GROUP and SUBGROUP names
    # ------------------------------------------------------------------

    groups = []
    subgroups = []

    for name in subgroup_names:
        if "--" in name:
            g, sg = name.split("--", 1)
        else:
            g, sg = "Unknown", name

        groups.append(g)
        subgroups.append(sg)

    # Build DataFrame used by seaborn
    corr_df = pd.DataFrame(corr_matrix, index=subgroups, columns=subgroups)

    # ------------------------------------------------------------------
    # Build annotation color bars
    # ------------------------------------------------------------------

    unique_groups = sorted(set(groups))
    group_palette = sns.color_palette("tab20", n_colors=len(unique_groups))
    group_color_map = {g: group_palette[i] for i, g in enumerate(unique_groups)}

    # Map group → color
    group_colors = pd.Series(groups, index=subgroups).map(group_color_map)

    # Selection A annotation (black if selected)
    sel_A_colors = pd.Series(
        ["black" if name in selected_dict_A else "white" for name in subgroup_names],
        index=subgroups
    )

    # Selection B annotation (red if selected)
    sel_B_colors = pd.Series(
        ["red" if name in selected_dict_B else "white" for name in subgroup_names],
        index=subgroups
    )

    # Combine annotation bars
    row_colors = pd.DataFrame({
        "Group": group_colors,
        "A": sel_A_colors,
        "B": sel_B_colors
    }, index=subgroups)

    col_colors = row_colors.copy()

    # ------------------------------------------------------------------
    # Perform hierarchical clustering
    # ------------------------------------------------------------------

    # Convert similarity to distance
    dist = 1.0 - corr_matrix
    np.fill_diagonal(dist, 0.0)

    # Convert square matrix to condensed distance format
    dist_condensed = squareform(dist, checks=False)

    # Build linkage tree
    Z = linkage(dist_condensed, method="average")

    # Assign cluster labels for visual boundaries
    cluster_labels = fcluster(Z, t=n_clusters, criterion="maxclust")



    # ------------------------------------------------------------------
    # Create clustermap
    # ------------------------------------------------------------------

    g = sns.clustermap(
        corr_df,
        row_cluster=True,
        col_cluster=True,
        row_colors=row_colors,
        col_colors=col_colors,
        cmap=cmap,
        vmin=0,
        vmax=1,
        linewidths=0,
        figsize=figsize,
        cbar_kws={"label": "Canonical correlation"}
    )

    g.fig.suptitle("Annotated clustermap of block redundancy", y=1.02)

    g.ax_heatmap.set_xticks(np.arange(len(subgroups)) + 0.5)
    g.ax_heatmap.set_xticklabels(subgroups, rotation=90)

    g.ax_heatmap.set_yticks(np.arange(len(subgroups)) + 0.5)
    g.ax_heatmap.set_yticklabels(subgroups)

    # ------------------------------------------------------------------
    # Determine visual order of rows and columns after clustering
    # ------------------------------------------------------------------

    row_order = g.dendrogram_row.reordered_ind
    col_order = g.dendrogram_col.reordered_ind

    row_clusters = cluster_labels[row_order]
    col_clusters = cluster_labels[col_order]

    # Identify cluster boundaries
    row_boundaries = np.where(np.diff(row_clusters) != 0)[0] + 1
    col_boundaries = np.where(np.diff(col_clusters) != 0)[0] + 1

    # ------------------------------------------------------------------
    # Draw cluster boundary lines on the heatmap
    # ------------------------------------------------------------------

    for b in row_boundaries:
        g.ax_heatmap.hlines(
            b,
            *g.ax_heatmap.get_xlim(),
            colors="white",
            linewidth=2
        )

    for b in col_boundaries:
        g.ax_heatmap.vlines(
            b,
            *g.ax_heatmap.get_ylim(),
            colors="white",
            linewidth=2
        )

    # ------------------------------------------------------------------
    # Improve label readability
    # ------------------------------------------------------------------

    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90, fontsize=8)
    plt.setp(g.ax_heatmap.get_yticklabels(), fontsize=8)

    # ------------------------------------------------------------------
    # Build legend explaining annotations
    # ------------------------------------------------------------------

    group_handles = [
        Line2D([0], [0], color=group_color_map[g], lw=4, label=g)
        for g in unique_groups
    ]

    selection_handles = [
        Line2D([0], [0], marker="s", color="black", linestyle="None",
               markersize=8, label="Selected in A"),
        Line2D([0], [0], marker="s", color="red", linestyle="None",
               markersize=8, label="Selected in B")
    ]

    legend1 = g.ax_col_dendrogram.legend(
        handles=group_handles,
        title="Group",
        bbox_to_anchor=(1.05, 1),
        loc="upper left"
    )

    g.ax_col_dendrogram.add_artist(legend1)

    g.ax_col_dendrogram.legend(
        handles=selection_handles,
        title="Selection",
        bbox_to_anchor=(1.05, 0.6),
        loc="upper left"
    )

    # ------------------------------------------------------------------
    # Save or display figure
    # ------------------------------------------------------------------

    if savepath is not None:
        plt.savefig(savepath, bbox_inches="tight", dpi=300)
        plt.close(g.fig)
    else:
        plt.show()

    return g, cluster_labels


def summarize_selection_robustness(
    corr_matrix,
    subgroup_names,
    selected_dict_A,
    selected_dict_B,
    dataset_name
):
    """
    Summarize robustness of two block selections.

    Parameters
    ----------
    corr_matrix : array-like of shape (n_blocks, n_blocks)
        Pairwise redundancy/similarity matrix between blocks
        (e.g., canonical correlations).

    subgroup_names : list of str
        Block names in the same order as corr_matrix.

    selected_dict_A : dict
        Mapping {block_name: importance} for selection A.

    selected_dict_B : dict
        Mapping {block_name: importance} for selection B.

    Returns
    -------
    results : dict
        Dictionary with summary statistics.

    text_summary : str
        A ready-to-use paragraph summarizing the results.
    """
    corr_matrix = np.asarray(corr_matrix, dtype=float)

    A = dict(selected_dict_A)
    B = dict(selected_dict_B)

    A_set = set(A.keys())
    B_set = set(B.keys())

    shared = sorted(A_set & B_set)
    A_only = sorted(A_set - B_set)
    B_only = sorted(B_set - A_set)
    disagreements = A_only + B_only
    union = sorted(A_set | B_set)

    name_to_idx = {name: i for i, name in enumerate(subgroup_names)}

    def safe_mean(values):
        values = [v for v in values if not np.isnan(v)]
        return float(np.mean(values)) if values else np.nan

    def safe_median(values):
        values = [v for v in values if not np.isnan(v)]
        return float(np.median(values)) if values else np.nan

    def safe_std(values):
        values = [v for v in values if not np.isnan(v)]
        return float(np.std(values)) if values else np.nan

    def safe_ratio(num, den):
        return float(num) / float(den) if den > 0 else np.nan

    # Importance summaries
    shared_importance_A = [A[name] for name in shared if name in A]
    shared_importance_B = [B[name] for name in shared if name in B]

    A_only_importance = [A[name] for name in A_only if name in A]
    B_only_importance = [B[name] for name in B_only if name in B]
    disagree_importance = A_only_importance + B_only_importance

    # Cross-selection redundancy for disagreements
    A_idx = [name_to_idx[name] for name in A_set if name in name_to_idx]
    B_idx = [name_to_idx[name] for name in B_set if name in name_to_idx]

    A_only_maxcorr = {}
    for name in A_only:
        i = name_to_idx[name]
        A_only_maxcorr[name] = float(np.max(corr_matrix[i, B_idx])) if B_idx else np.nan

    B_only_maxcorr = {}
    for name in B_only:
        i = name_to_idx[name]
        B_only_maxcorr[name] = float(np.max(corr_matrix[i, A_idx])) if A_idx else np.nan

    disagreement_maxcorr = list(A_only_maxcorr.values()) + list(B_only_maxcorr.values())

    # Weighted version using importance
    # Each disagreement is weighted by its own selection importance
    weighted_vals = []
    for name in A_only:
        if not np.isnan(A_only_maxcorr[name]):
            weighted_vals.append((A[name], A_only_maxcorr[name]))
    for name in B_only:
        if not np.isnan(B_only_maxcorr[name]):
            weighted_vals.append((B[name], B_only_maxcorr[name]))

    if weighted_vals:
        weights = np.array([w for w, _ in weighted_vals], dtype=float)
        vals = np.array([v for _, v in weighted_vals], dtype=float)
        weighted_mean_maxcorr = float(np.average(vals, weights=weights))
    else:
        weighted_mean_maxcorr = np.nan

    # Overlap metrics
    jaccard = safe_ratio(len(shared), len(union))
    overlap_A = safe_ratio(len(shared), len(A_set))
    overlap_B = safe_ratio(len(shared), len(B_set))

    results = {
        "n_A": len(A_set),
        "n_B": len(B_set),
        "n_shared": len(shared),
        "n_A_only": len(A_only),
        "n_B_only": len(B_only),
        "n_disagreements": len(disagreements),
        "jaccard": jaccard,
        "overlap_fraction_A": overlap_A,
        "overlap_fraction_B": overlap_B,

        "shared_blocks": shared,
        "A_only_blocks": A_only,
        "B_only_blocks": B_only,

        "shared_importance_mean_A": safe_mean(shared_importance_A),
        "shared_importance_mean_B": safe_mean(shared_importance_B),
        "shared_importance_median_A": safe_median(shared_importance_A),
        "shared_importance_median_B": safe_median(shared_importance_B),

        "A_only_importance_mean": safe_mean(A_only_importance),
        "B_only_importance_mean": safe_mean(B_only_importance),
        "disagreement_importance_mean": safe_mean(disagree_importance),
        "disagreement_importance_median": safe_median(disagree_importance),
        "disagreement_importance_std": safe_std(disagree_importance),

        "A_only_maxcorr": A_only_maxcorr,
        "B_only_maxcorr": B_only_maxcorr,
        "disagreement_maxcorr_mean": safe_mean(disagreement_maxcorr),
        "disagreement_maxcorr_median": safe_median(disagreement_maxcorr),
        "disagreement_maxcorr_std": safe_std(disagreement_maxcorr),
        "weighted_disagreement_maxcorr_mean": weighted_mean_maxcorr,
    }

    text_summary = (
        f"For dataset {dataset_name}, selection A contains {results['n_A']} blocks and selection B contains {results['n_B']} blocks. "
        f"They share {results['n_shared']} blocks exactly "
        f"(Jaccard overlap = {results['jaccard']:.2f}). "
        f"The blocks selected under both criteria tend to have higher importance "
        f"(mean importance: A = {results['shared_importance_mean_A']:.3f}, "
        f"B = {results['shared_importance_mean_B']:.3f}) "
        f"than discrepant blocks (mean importance = {results['disagreement_importance_mean']:.3f}). "
        f"For blocks selected under only one discretization criterion, the mean maximum canonical correlation "
        f"with the blocks selected under the alternative criterion is "
        f"{results['disagreement_maxcorr_mean']:.3f} "
        f"(median = {results['disagreement_maxcorr_median']:.3f}; "
        f"importance-weighted mean = {results['weighted_disagreement_maxcorr_mean']:.3f}), "
        f"indicating that disagreements mainly involve highly redundant feature blocks."
    )

    return results, text_summary

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, optimal_leaf_ordering
from scipy.spatial.distance import squareform


def annotated_clustermap_optimal(
    corr_matrix,
    subgroup_names,
    selected_dict_A=None,
    selected_dict_B=None,
    figsize=(12, 12),
    cmap="viridis",
    savepath=None
):
    corr_matrix = np.asarray(corr_matrix, dtype=float)
    corr_matrix = 0.5 * (corr_matrix + corr_matrix.T)
    corr_matrix = np.clip(corr_matrix, 0.0, 1.0)

    selected_dict_A = {} if selected_dict_A is None else dict(selected_dict_A)
    selected_dict_B = {} if selected_dict_B is None else dict(selected_dict_B)

    groups = []
    subgroups = []
    for name in subgroup_names:
        if "--" in name:
            g, sg = name.split("--", 1)
        else:
            g, sg = "Unknown", name
        groups.append(g)
        subgroups.append(sg)

    corr_df = pd.DataFrame(corr_matrix, index=subgroups, columns=subgroups)

    unique_groups = sorted(set(groups))
    group_palette = sns.color_palette("tab20", n_colors=len(unique_groups))
    group_color_map = {g: group_palette[i] for i, g in enumerate(unique_groups)}

    group_colors = pd.Series(groups, index=subgroups).map(group_color_map)
    sel_A_colors = pd.Series(
        ["black" if full_name in selected_dict_A else "white" for full_name in subgroup_names],
        index=subgroups
    )
    sel_B_colors = pd.Series(
        ["red" if full_name in selected_dict_B else "white" for full_name in subgroup_names],
        index=subgroups
    )

    row_colors = pd.DataFrame({
        "Group": group_colors,
        "A": sel_A_colors,
        "B": sel_B_colors
    }, index=subgroups)
    col_colors = row_colors.copy()

    # Distancia para clustering
    dist = 1.0 - corr_matrix
    np.fill_diagonal(dist, 0.0)

    # linkage normal sobre vector condensado
    dist_condensed = squareform(dist, checks=False)
    Z = linkage(dist_condensed, method="average")

    # optimal leaf ordering
    Z_opt = optimal_leaf_ordering(Z, dist_condensed)

    g = sns.clustermap(
        corr_df,
        row_linkage=Z_opt,
        col_linkage=Z_opt,
        row_colors=row_colors,
        col_colors=col_colors,
        cmap=cmap,
        vmin=0,
        vmax=1,
        linewidths=0,
        figsize=figsize,
        cbar_kws={"label": "Canonical correlation"}
    )

    g.fig.suptitle("Annotated clustermap of block redundancy", y=1.02)

    # Forzar todas las etiquetas
    g.ax_heatmap.set_xticks(np.arange(len(subgroups)) + 0.5)
    g.ax_heatmap.set_xticklabels(g.data2d.columns, rotation=90, fontsize=8)

    g.ax_heatmap.set_yticks(np.arange(len(subgroups)) + 0.5)
    g.ax_heatmap.set_yticklabels(g.data2d.index, fontsize=8)

    if savepath is not None:
        plt.savefig(savepath, bbox_inches="tight", dpi=300)
        plt.close(g.fig)
    else:
        plt.show()

    return g

def annotated_clustermap_optimal_with_boundaries(
    corr_matrix,
    subgroup_names,
    selected_dict_A=None,
    selected_dict_B=None,
    figsize=(14, 14),
    cmap="viridis",
    method="average",
    n_clusters=10,
    savepath=None
):
    corr_matrix = np.asarray(corr_matrix, dtype=float)
    corr_matrix = 0.5 * (corr_matrix + corr_matrix.T)
    corr_matrix = np.clip(corr_matrix, 0.0, 1.0)

    selected_dict_A = {} if selected_dict_A is None else dict(selected_dict_A)
    selected_dict_B = {} if selected_dict_B is None else dict(selected_dict_B)

    groups = []
    subgroups = []
    for name in subgroup_names:
        if "--" in name:
            g, sg = name.split("--", 1)
        else:
            g, sg = "Unknown", name
        groups.append(g)
        subgroups.append(sg)

    corr_df = pd.DataFrame(corr_matrix, index=subgroups, columns=subgroups)

    unique_groups = sorted(set(groups))
    group_palette = sns.color_palette("tab20", n_colors=len(unique_groups))
    group_color_map = {g: group_palette[i] for i, g in enumerate(unique_groups)}

    group_colors = pd.Series(groups, index=subgroups).map(group_color_map)
    sel_A_colors = pd.Series(
        ["black" if full_name in selected_dict_A else "white" for full_name in subgroup_names],
        index=subgroups
    )
    sel_B_colors = pd.Series(
        ["red" if full_name in selected_dict_B else "white" for full_name in subgroup_names],
        index=subgroups
    )

    row_colors = pd.DataFrame({
        "Group": group_colors,
        "A": sel_A_colors,
        "B": sel_B_colors
    }, index=subgroups)
    col_colors = row_colors.copy()

    # Distancia para clustering
    dist = 1.0 - corr_matrix
    np.fill_diagonal(dist, 0.0)
    dist_condensed = squareform(dist, checks=False)

    Z = linkage(dist_condensed, method=method)
    Z_opt = optimal_leaf_ordering(Z, dist_condensed)

    # etiquetas de cluster para cada bloque original
    cluster_labels = fcluster(Z_opt, t=n_clusters, criterion="maxclust")

    g = sns.clustermap(
        corr_df,
        row_linkage=Z_opt,
        col_linkage=Z_opt,
        row_colors=row_colors,
        col_colors=col_colors,
        cmap=cmap,
        vmin=0,
        vmax=1,
        linewidths=0,
        figsize=figsize,
        cbar_kws={"label": "Canonical correlation"}
    )

    g.fig.suptitle("Annotated clustermap of block redundancy", y=1.02)

    # Forzar todas las etiquetas
    ordered_rows = g.dendrogram_row.reordered_ind
    ordered_cols = g.dendrogram_col.reordered_ind

    g.ax_heatmap.set_xticks(np.arange(len(ordered_cols)) + 0.5)
    g.ax_heatmap.set_xticklabels(
        [subgroups[i] for i in ordered_cols],
        rotation=90,
        fontsize=8
    )

    g.ax_heatmap.set_yticks(np.arange(len(ordered_rows)) + 0.5)
    g.ax_heatmap.set_yticklabels(
        [subgroups[i] for i in ordered_rows],
        fontsize=8
    )

    # Reordenar las etiquetas de cluster al orden visual
    row_clusters_ordered = cluster_labels[ordered_rows]
    col_clusters_ordered = cluster_labels[ordered_cols]

    # Encontrar fronteras: posiciones donde cambia el cluster
    row_boundaries = np.where(np.diff(row_clusters_ordered) != 0)[0] + 1
    col_boundaries = np.where(np.diff(col_clusters_ordered) != 0)[0] + 1

    # Dibujar líneas sobre el heatmap
    for b in row_boundaries:
        g.ax_heatmap.hlines(
            b, *g.ax_heatmap.get_xlim(),
            colors="white", linewidth=2.0
        )

    for b in col_boundaries:
        g.ax_heatmap.vlines(
            b, *g.ax_heatmap.get_ylim(),
            colors="white", linewidth=2.0
        )

    if savepath is not None:
        plt.savefig(savepath, bbox_inches="tight", dpi=300)
        plt.close(g.fig)
    else:
        plt.show()

    return g, cluster_labels

def load_selected_features_json(selected_features_path, subgroups_names):
    with open(selected_features_path, 'r') as f:
        selected_dict = json.load(f)
        selected_dict = {feat: round(val * 100, 1) for feat, val in selected_dict.items() if feat in subgroups_names}
        return selected_dict


if __name__ == '__main__':
    from data import load_dataset
    path = '../datasets/toxicity_dataset'
    data = load_dataset(path, n_classes=5, filter_abandoned_activity=False)

    out_path = Path('../colineal_analysis/')
    os.makedirs(out_path, exist_ok=True)
    os.makedirs(out_path / 'colineal', exist_ok=True)
    os.makedirs(out_path / 'pickles', exist_ok=True)

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
        plt.savefig(out_path / 'colineal' / (out_name + '.pdf'), bbox_inches="tight")

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
            corr_matrix, subgr_names, ax = block_cca_heatmap_parallel(X, subgr_idx, group_names=subgr_names,
                                                                      pca_reduce=n_components)
            pickle.dump({'corr_matrix': corr_matrix, 'group_names': subgr_names}, open(pickle_path, 'wb'))
            plt.savefig(out_path / 'colineal' / (out_name + '.pdf'), bbox_inches="tight")

    # second level (sub groups)
    n_components = 5
    out_name = f'colineal_subgroup_pca{n_components}'
    pickle_path = out_path / 'pickles' / (out_name + '.pkl')
    if not os.path.exists(pickle_path):
        corr_matrix, subgroup_names, ax = block_cca_heatmap_parallel(X, subgroups, group_names=subgroup_names, pca_reduce=n_components)
        pickle.dump({'corr_matrix': corr_matrix, 'group_names': subgroup_names}, open(pickle_path, 'wb'))
        plt.savefig(out_path / 'colineal' / (out_name+'.pdf'), bbox_inches="tight")
    dict_output = pickle.load(open(pickle_path, 'rb'))
    corr_matrix, subgroup_names = dict_output['corr_matrix'], dict_output['group_names']

    for dataset in ['activity']: #, 'toxicity', 'diversity']:
        selA = load_selected_features_json(f'../results/feat_importance/samplesize500/{dataset}/5_classes/feat_importance.json', subgroup_names)
        selB = load_selected_features_json(f'../results_new/feat_importance/samplesize500/{dataset}/5_classes/feat_importance.json', subgroup_names)

        # represent_blocks_MDS(corr_matrix, subgroup_names, selected_dict_A=selA, selected_dict_B=selB, savepath=out_path/f'MDS_{dataset}.pdf')
        represent_blocks_UMAP(corr_matrix, subgroup_names, selected_dict_A=selA, selected_dict_B=selB, savepath=out_path/f'UMAP_{dataset}_AB.pdf')
        represent_blocks_UMAP(corr_matrix, subgroup_names, selected_dict_A=selA, savepath=out_path / f'UMAP_{dataset}_main.pdf')
        # represent_blocks_tSNE(corr_matrix, subgroup_names, selected_dict_A=selA, selected_dict_B=selB, savepath=out_path/f'tSNE_{dataset}.pdf')
        # hierarchical_clustering_selected(corr_matrix, subgroup_names, selected_dict_A=selA, selected_dict_B=selB, savepath=out_path/f'hierarchical_clustering_{dataset}.pdf')
        # annotated_clustermap(corr_matrix, subgroup_names, selected_dict_A=selA, selected_dict_B=selB, savepath=out_path/f'clustermap_AB_{dataset}.png')
        # annotated_clustermap(corr_matrix, subgroup_names, selected_dict_A=selA, savepath=out_path / f'clustermap_main_{dataset}.png')
        # annotated_clustermap_with_boundaries(corr_matrix, subgroup_names, selected_dict_A=selA, selected_dict_B=selB, savepath=out_path/f'clustermap_bound_AB_{dataset}.png')
        annotated_clustermap_optimal(corr_matrix, subgroup_names, selected_dict_A=selA, selected_dict_B=selB, savepath=out_path / f'clustermap_optim_AB_{dataset}.png')
        # annotated_clustermap_optimal_with_boundaries(corr_matrix, subgroup_names, selected_dict_A=selA, selected_dict_B=selB, savepath=out_path / f'clustermap_bound_AB_{dataset}.png')
        results, text_summary = summarize_selection_robustness(corr_matrix, subgroup_names, selected_dict_A=selA, selected_dict_B=selB, dataset_name=dataset)
        print(text_summary)

    # hierarchical_clustering(corr_matrix, subgroup_names, savepath=out_path/f'Hierarchical_Clustering.pdf')

