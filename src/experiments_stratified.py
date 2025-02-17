import os
import pickle
from os.path import join
from itertools import product

import pandas as pd
import quapy as qp
import numpy as np
from quapy.data import LabelledCollection
from quapy.method.aggregative import PCC, PACC, CC
from quapy.method.base import BaseQuantifier
from quapy.method.confidence import WithConfidenceABC, ConfidenceIntervals, AggregativeBootstrap
from quapy.method.non_aggregative import MaximumLikelihoodPrevalenceEstimation
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from data import load_dataset
import warnings
from classification import BlockEnsembleClassifier
from quapy.functional import prevalence_from_labels, strprev

from utils import mmd_pairwise_rbf_blocks

warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
warnings.filterwarnings("ignore", category=ConvergenceWarning, module='sklearn')


def dump(results, result_path):
    with open(result_path, 'wb') as foo:
        pickle.dump(results, foo, pickle.HIGHEST_PROTOCOL)


def experiment(data, n_classes, method, method_name):
    n_subreddits = len(data.subreddit_names)
    classes = np.arange(n_classes)

    X = data.X
    y = data.y

    n = X.shape[0]
    np.random.seed(0)
    random_order = np.random.permutation(n)
    test_split_point = 2*n//3
    test_idx = random_order[test_split_point:]

    random_order = random_order[:test_split_point]
    init_train_size = 500
    train_idx = random_order[:init_train_size]
    remainder = random_order[init_train_size:]

    batch_size = 100

    Xte = X[test_idx]
    yte = y[test_idx]
    test = LabelledCollection(Xte, yte, classes=classes)

    Xtr = X[train_idx]
    ytr = y[train_idx]
    train = LabelledCollection(Xtr, ytr, classes=classes)
    # test = test.sampling(len(test), *test.prevalence()[::-1])
    true_prevalence = test.prevalence()
    print(strprev(true_prevalence))

    results_nmd = []
    if not isinstance(method, WithConfidenceABC):
        while(len(remainder)>batch_size):
            warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
            warnings.filterwarnings("ignore", category=ConvergenceWarning, module='sklearn')

            method.fit(train)
            predicted_prevalence = method.quantify(test.X)
            nmd = qp.error.normalized_match_distance(true_prevalence, predicted_prevalence)
            results_nmd.append({'method': method_name, '#train': len(train), 'nmd': nmd})

            print(f'[no-CI] train_size={len(train)} {nmd=:.5f}')

            posteriors = method.classifier.predict_proba(X[remainder])
            conf = posteriors.max(axis=1)
            order = np.argsort(conf)
            remainder = remainder[order]

            next_batch_idx = remainder[:batch_size]
            remainder = remainder[batch_size:]
            new_Xtr = X[next_batch_idx]
            new_ytr = y[next_batch_idx]
            next_batch = LabelledCollection(new_Xtr, new_ytr, classes=classes)

            train = train + next_batch
    else:
        while (len(remainder) > batch_size):
            warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
            warnings.filterwarnings("ignore", category=ConvergenceWarning, module='sklearn')

            method.fit(train, val_split=5)
            predicted_prevalence = method.quantify(test.X)
            nmd = qp.error.normalized_match_distance(true_prevalence, predicted_prevalence)
            results_nmd.append({'method': method_name, '#train': len(train), 'nmd': nmd})

            print(f'[with-CI] train_size={len(train)} {nmd=:.5f}')

            np.random.shuffle(remainder)
            n_remainder = len(remainder)
            n_remaining_batches = n_remainder // batch_size
            most_uncertain = None
            rest = None
            worst_uncertainty = None
            for i in range(n_remaining_batches):
                batch_idx = remainder[i * batch_size:(i + 1) * batch_size]
                rest_idx = np.concatenate([remainder[:i * batch_size], remainder[(i + 1) * batch_size:]])
                Xbatch = X[batch_idx]
                prev, conf_intervals = method.quantify_conf(Xbatch)
                uncertainty = conf_intervals.simplex_portion()
                if worst_uncertainty is None or uncertainty > worst_uncertainty:
                    worst_uncertainty = uncertainty
                    most_uncertain = batch_idx
                    rest = rest_idx
                    # print(f'\tworst uncertainty found={worst_uncertainty:.4f}')

            remainder = rest
            new_Xtr = X[most_uncertain]
            new_ytr = y[most_uncertain]
            next_batch = LabelledCollection(new_Xtr, new_ytr, classes=classes)
            train = train + next_batch

    results_nmd = pd.DataFrame(results_nmd)
    return results_nmd


def methods(base_classifier, block_ids):
    # yield 'MLPE', MaximumLikelihoodPrevalenceEstimation()
    # yield 'CC', CC(base_classifier)
    yield 'PCC', PCC(base_classifier)
    yield 'PCC-CI', AggregativeBootstrap(PCC(base_classifier), n_train_samples=50, n_test_samples=50, confidence_level=0.95)
    # yield 'bPCC', PCC(BlockEnsembleClassifier(base_classifier, blocks_ids=block_ids))
    # yield 'PACC', PACC(base_classifier)


def main(data, n_classes, dataset_name):
    base_classifier = LogisticRegression()
    all_results = []
    for method_name, method in methods(base_classifier, data.prefix_idx):
        print(f'running {method_name}')
        results = experiment(data, n_classes, method, method_name)
        all_results.append(results)

    df = pd.concat(all_results)
    df['dataset'] = dataset_name
    df['n_classes'] = n_classes
    df.to_csv(f'../results_active_learning/{dataset_name}_{n_classes}_classes.csv')


if __name__ == '__main__':
    dataset_dir = '../datasets'
    n_classes_list = [3,5]
    dataset_names = ['diversity', 'toxicity', 'activity']
    for dataset_name, n_classes in product(dataset_names, n_classes_list):
        print(f'running {dataset_name=} {n_classes}')
        data = load_dataset(join(dataset_dir, f'{dataset_name}_dataset'), n_classes=n_classes, filter_out_multiple_subreddits=False)
        main(data, n_classes, dataset_name)

# CC [0.00318037 0.01162731 0.02090877 0.04247854 0.03984796 0.05510725 0.05329431 0.05905753 0.06550041 0.17857143]
# PCC [0.00280356 0.00440107 0.00244653 0.01039527 0.00457421 0.00200411 0.00193409 0.00514357 0.00458079 0.13097832]
# PACC [0.19307214 0.07399234 0.15285762 0.14943895 0.1104644  0.11927672  0.13393173 0.16204134 0.09514025 0.28571429]