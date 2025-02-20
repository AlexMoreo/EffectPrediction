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
import warnings
from quapy.functional import prevalence_from_labels

from methods import methods, PCCrecalib, new_methods
from utils import mmd_pairwise_rbf_blocks

warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
warnings.filterwarnings("ignore", category=ConvergenceWarning, module='sklearn')


def dump(results, result_path):
    with open(result_path, 'wb') as foo:
        pickle.dump(results, foo, pickle.HIGHEST_PROTOCOL)


def experiment(data, n_classes, target, method, policy):
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
    policy.feed(X_by_subreddit, blocks_idx=blocks_idx, gammas=1., conf_val=0.01)

    # gets a vector [-1, 0, ..., 6] where -1 is the global experiment, and the rest
    # are the indexes of the periods (currently 7)
    if target=='global':
        periods = [-1]
    elif target=='periods':
        periods = np.arange(n_periods)
    else:
        raise ValueError(f'{target=} not understood')

    for period in periods:
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

        def job(i):  # one of the jobs in a LOO evaluation
            i=int(i)
            test_data = data_by_subreddit[i]
            train_data_list = [data for j, data in enumerate(data_by_subreddit) if j != i]
            selection = policy.get_selection(test_index=i)
            method.fit(train_data_list, select=selection)
            predicted_prevalence = method.quantify(test_data.X)
            true_prevalence = test_data.prevalence()

            ae = qp.error.ae(true_prevalence, predicted_prevalence)
            rae = qp.error.rae(true_prevalence, predicted_prevalence, eps=1. / (2. * len(test_data)))
            nmd = qp.error.normalized_match_distance(true_prevalence, predicted_prevalence)

            return {'period': period, 'subreddit_idx_test': i, 'ae': ae, 'rae': rae, 'nmd': nmd}

        results_period = qp.util.parallel(job, np.arange(n_subreddits), n_jobs=-1, asarray=False, backend='loky')
        results.extend(results_period)

    results = pd.DataFrame(results)

    return results


def new_experiment(data, n_classes, target, method):
    assert target == 'global', 'not implemented'
    period = -1
    y = data.y if period == -1 else data.y_periods[period]

    n_subreddits = len(data.subreddit_names)
    classes = np.arange(n_classes)

    X = data.X
    n = X.shape[0]

    results = []
    for test_idx in range(n_subreddits):
        # if test_idx not in [1,3,6,11]:
        #     continue
        print(f'testing for {test_idx}')
        test_sel = data.subreddits[test_idx]
        test_sel_size = test_sel.sum()

        #select all training instances, i.e., all instances which either
        # (i) are not test instances, or
        # (ii) has a label for any other subreddit (even if this is a test instance)
        train_sel = ~test_sel
        for train_idx in range(n_subreddits):
            if train_idx == test_idx:
                continue
            train_sel = np.logical_or(train_sel, data.subreddits[train_idx])

        common_labels = np.logical_and(train_sel, test_sel)
        only_train = np.logical_and(train_sel, ~common_labels)
        only_test  = np.logical_and(test_sel, ~common_labels)

        Xsrc = X[only_train]
        ysrc = y[only_train]

        Xtgt = X[common_labels]
        ytgt = y[common_labels]

        Xtgt_rest = X[only_test]
        ytgt_rest = y[only_test]

        print(f'{test_idx=}\ttrain-only={only_train.sum()}'
              f'\tcommon={common_labels.sum()}\t({LabelledCollection(Xtgt, ytgt, classes).counts()})'
              f'\ttest-only={only_test.sum()}\t({LabelledCollection(Xtgt_rest, ytgt_rest, classes).counts()})'
              f'\t')

        method.fit(Xsrc, ysrc, Xtgt, ytgt)
        predicted_prevalence = method.join_quantify(Xtgt, ytgt, Xtgt_rest)

        true_prevalence = prevalence_from_labels(y[test_sel], classes=classes)

        ae = qp.error.ae(true_prevalence, predicted_prevalence)
        rae = qp.error.rae(true_prevalence, predicted_prevalence, eps=1. / (2. * test_sel_size))
        nmd = qp.error.normalized_match_distance(true_prevalence, predicted_prevalence)

        result = {'period': period, 'subreddit_idx_test': test_idx, 'ae': ae, 'rae': rae, 'nmd': nmd}
        results.append(result)

    results = pd.DataFrame(results)

    return results


def main(data, n_classes, target, dataset_name, results_dir):
    base_classifier = LogisticRegression()
    for method_name, method, policy in methods(base_classifier, data.prefix_idx):
        print(f'running {method_name}')
        result_path = join(results_dir, f'{n_classes}_classes', dataset_name, method_name+'.pkl')

        # skip experiment already computed
        if os.path.exists(result_path):
            print(f'experiment {result_path} already exist; skipping')
            continue

        # compute the results for this experiment
        results = experiment(data, n_classes, target, method, policy)
        dump(results, result_path)


    # new experiments after the last meeting
    # for method_name, method in new_methods(base_classifier):
    #     print(f'running new method {method_name}')
    #     result_path = join(results_dir, f'{n_classes}_classes', dataset_name, method_name+'.pkl')
    #
    #     skip experiment already computed
        # if not os.path.exists(result_path):
        #     results = new_experiment(data, n_classes, target, method)
        #     dump(results, result_path)


if __name__ == '__main__':
    dataset_dir = '../datasets'

    targets = ['global'] #, 'periods']
    n_classes_list = [5]
    dataset_names = ['diversity', 'toxicity', 'activity']
    for dataset_name, n_classes, target in product(dataset_names, n_classes_list, targets):
        results_dir = f'../results_{target}'
        print(f'running {dataset_name=} {n_classes}')
        data = load_dataset(join(dataset_dir, f'{dataset_name}_dataset'), n_classes=n_classes, filter_out_multiple_subreddits=False)
        os.makedirs(join(results_dir, f'{n_classes}_classes', dataset_name), exist_ok=True)
        main(data, n_classes, target, dataset_name, results_dir)