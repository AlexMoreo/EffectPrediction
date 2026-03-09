import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from itertools import combinations
from joblib import Parallel, delayed
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler


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
    return corr_matrix, ax

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


if __name__ == '__main__':
    from data import load_dataset
    path = '../datasets/toxicity_dataset'
    data = load_dataset(path, n_classes=5, filter_abandoned_activity=False)

    X = data.X

    # first level (macro groups)
    group_names, groups = get_group_indexes(data.covariate_names, level=1)
    n_components = 10
    # corr_matrix, ax = block_cca_heatmap_parallel(X, groups, group_names=group_names, pca_reduce=n_components)
    # plt.savefig(f'../fig/colineal_group_pca{n_components}.pdf', bbox_inches="tight")

    # second level (sub groups)
    subgroup_names, subgroups = get_group_indexes(data.covariate_names, level=2)
    n_components = 5
    # corr_matrix, ax = block_cca_heatmap_parallel(X, subgroups, group_names=subgroup_names, pca_reduce=n_components)
    # plt.savefig(f'../fig/colineal_subgroup_pca{n_components}.pdf', bbox_inches="tight")

    # within sub group
    for group_name in group_names:
        print(f'Generating CCA for {group_name}')
        subgr_names = []
        subgr_idx = []
        for subgroup_name, subgroup_idx in zip(subgroup_names, subgroups):
            gr_name, subgr_name = subgroup_name.split('--')
            if gr_name != group_name: continue
            subgr_names.append(subgr_name)
            subgr_idx.append(subgroup_idx)
        n_components = 5
        corr_matrix, ax = block_cca_heatmap_parallel(X, subgr_idx, group_names=subgr_names, pca_reduce=n_components)
        plt.savefig(f'../fig/colineal_{group_name}_pca{n_components}.pdf', bbox_inches="tight")
