import os
import pickle
from os.path import join
from itertools import product

import pandas as pd
import quapy as qp
import numpy as np
from quapy.data import LabelledCollection
from quapy.method.aggregative import PCC
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from data import load_dataset
from methods import methods
import warnings

from utils import mmd_pairwise_rbf_blocks, mmd_rbf
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import quapy.functional as F

warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
warnings.filterwarnings("ignore", category=ConvergenceWarning, module='sklearn')


def main(data, n_classes):
    n_subreddits = len(data.subreddit_names)
    classes = np.arange(n_classes)

    X = data.X
    data_by_subreddit = []
    for subreddit_idx, subreddit_name in enumerate(data.subreddit_names):
        subreddit_sel = data.subreddits[subreddit_idx]
        Xsub = X[subreddit_sel]
        ysub = data.y[subreddit_sel]
        if np.prod(F.prevalence_from_labels(ysub, classes=classes))!=0:
            data_by_subreddit.append(LabelledCollection(Xsub, ysub, classes=classes))

    period = -1  # only global test
    print(f'\trunning {period=}')

    ae_matrix = np.zeros(shape=(n_subreddits, n_subreddits), dtype=float)
    rae_matrix = np.zeros(shape=(n_subreddits, n_subreddits), dtype=float)
    mmd_matrix   = np.zeros(shape=(n_subreddits, n_subreddits), dtype=float)

    for i, train in enumerate(data_by_subreddit):
        pcc = PCC(LogisticRegression())
        print(train.prevalence())
        pcc.fit(train)
        for j, test in enumerate(data_by_subreddit):
            if i==j: continue
            pred_prev = pcc.quantify(test.X)
            true_prev = test.prevalence()
            mae = qp.error.ae(true_prev, pred_prev)
            qp.environ['SAMPLE_SIZE']=len(test)
            mrae = qp.error.rae(true_prev, pred_prev)
            ae_matrix[i,j] = mae
            rae_matrix[i, j] = mrae
            if i < j:
                mmd = mmd_rbf(train.X, test.X)
                mmd_matrix[i, j] = mmd
                mmd_matrix[j, i] = mmd

    off_diagonal_mask = ~np.eye(n_subreddits, dtype=bool)
    ae_errs = ae_matrix[off_diagonal_mask].flatten()
    rae_errs = rae_matrix[off_diagonal_mask].flatten()
    mmds = mmd_matrix[off_diagonal_mask].flatten()

    plt.clf()

    plt.scatter(mmds, ae_errs, color='blue', alpha=0.7, label='ae')
    plt.scatter(mmds, rae_errs, color='red', alpha=0.7, label='rae')
    correlation, p_value = pearsonr(mmds, ae_errs)
    plt.title(f'{dataset_name} {n_classes} classes, p={correlation:.5f} (p-val {p_value:.5f})')

    plt.show()

if __name__ == '__main__':
    results_dir = '../results_corr'
    dataset_dir = '../datasets'

    n_classes_list = [3]
    dataset_names = ['diversity', 'toxicity'] # 'activity',
    for dataset_name, n_classes in product(dataset_names, n_classes_list):
        print(f'running {dataset_name=} {n_classes}')
        data = load_dataset(join(dataset_dir, f'{dataset_name}_dataset'), n_classes=n_classes, filter_out_multiple_subreddits=True)
        os.makedirs(join(results_dir, f'{n_classes}_classes', dataset_name), exist_ok=True)
        main(data, n_classes)