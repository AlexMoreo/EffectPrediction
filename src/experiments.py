import os
import pickle
from os.path import join
from itertools import product

import pandas as pd
import quapy as qp
import numpy as np
from quapy.data import LabelledCollection
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from data import load_dataset
from methods import methods
import warnings

from utils import mmd_pairwise_rbf_blocks

warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
warnings.filterwarnings("ignore", category=ConvergenceWarning, module='sklearn')


def experiment(data, n_classes, method, policy):
    n_periods = data.y_periods.shape[0]
    n_subreddits = len(data.subreddit_names)
    classes = np.arange(n_classes)

    results = []

    X = data.X
    X_by_subreddit = []
    for subreddit_idx, subreddit_name in enumerate(data.subreddit_names):
        subreddit_sel = data.subreddits[subreddit_idx]
        Xsub = X[subreddit_sel]
        X_by_subreddit.append(Xsub)
    blocks_idx = list(data.prefix_idx.values())
    policy.feed(X_by_subreddit, blocks_idx=blocks_idx, gammas=1.)

    # gets a vector [-1, 0, ..., 6] where -1 is the global experiment, and the rest
    # are the indexes of the periods (currently 7)
    periods = np.arange(n_periods+1)-1
    for period in periods:
        if period==-1: continue
        period = int(period)
        print(f'\trunning {period=}')

        y = data.y if period == -1 else data.y_periods[period]

        data_by_subreddit = []
        for subreddit_idx, subreddit_name in enumerate(data.subreddit_names):
            Xsub = X_by_subreddit[subreddit_idx]
            subreddit_sel = data.subreddits[subreddit_idx]
            ysub = y[subreddit_sel]
            data_by_subreddit.append(LabelledCollection(Xsub, ysub, classes=classes))

        # blocks_idx = list(data.prefix_idx.values())
        # policy.feed(data_by_subreddit, blocks_idx=blocks_idx, gammas=1.)

        def job(i):
            i=int(i)
            test_data = data_by_subreddit[i]
            train_data_list = [data for j, data in enumerate(data_by_subreddit) if j != i]
            selection = policy.get_selection(test_index=i)
            method.fit(train_data_list, select=selection)
            predicted_prevalence = method.quantify(test_data.X)
            true_prevalence = test_data.prevalence()

            ae = qp.error.ae(true_prevalence, predicted_prevalence)
            rae = qp.error.rae(true_prevalence, predicted_prevalence, eps=1. / (2. * len(test_data)))

            return {'period': period, 'subreddit_idx_test': i, 'ae': ae, 'rae': rae}

        results_period = qp.util.parallel(job, np.arange(n_subreddits), n_jobs=-1, asarray=False, backend='loky')
        results.extend(results_period)

    results = pd.DataFrame(results)

    return results


def main(data, n_classes):
    base_classifier = LogisticRegression()
    for method_name, method, policy in methods(base_classifier, data.prefix_idx):
        print(f'running {method_name}')
        result_path = join(results_dir, f'{n_classes}_classes', dataset_name, method_name+'.pkl')

        # skip experiment already computed
        if os.path.exists(result_path):
            print(f'experiment {result_path} already exist; skipping')
            continue

        # compute the results for this experiment
        results = experiment(data, n_classes, method, policy)
        with open(result_path, 'wb') as foo:
            pickle.dump(results, foo, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    results_dir = '../results_global'
    dataset_dir = '../datasets'

    n_classes_list = [3, 5]
    dataset_names = ['diversity', 'toxicity', 'activity']
    for dataset_name, n_classes in product(dataset_names, n_classes_list):
        print(f'running {dataset_name=} {n_classes}')
        data = load_dataset(join(dataset_dir, f'{dataset_name}_dataset'), n_classes=n_classes, filter_out_multiple_subreddits=True)
        os.makedirs(join(results_dir, f'{n_classes}_classes', dataset_name), exist_ok=True)
        main(data, n_classes)