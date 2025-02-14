import os
import pickle
from os.path import join
from itertools import product

import pandas as pd
import quapy as qp
import numpy as np
from quapy.data import LabelledCollection
from quapy.method.aggregative import PCC, PACC, CC
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from data import load_dataset
import warnings
from quapy.functional import prevalence_from_labels

from utils import mmd_pairwise_rbf_blocks

warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
warnings.filterwarnings("ignore", category=ConvergenceWarning, module='sklearn')


def dump(results, result_path):
    with open(result_path, 'wb') as foo:
        pickle.dump(results, foo, pickle.HIGHEST_PROTOCOL)


def experiment(data, n_classes, method):
    n_subreddits = len(data.subreddit_names)
    classes = np.arange(n_classes)

    results = []

    X = data.X
    y = data.y

    n = X.shape[0]
    np.random.seed(0)
    random_order = np.random.permutation(n)
    n_batches = 10
    batch_size = n//n_batches

    def job(batch_id):
        split_point = (batch_id+1)*batch_size
        train_idx, test_idx = random_order[:split_point], random_order[split_point:]
        Xtr = X[train_idx]
        ytr = y[train_idx]
        Xte = X[test_idx]
        yte = y[test_idx]
        train = LabelledCollection(Xtr, ytr, classes=classes)
        test  = LabelledCollection(Xte, yte, classes=classes)
        method.fit(train)
        predicted_prevalence = method.quantify(test.X)
        true_prevalence = test.prevalence()

        # ae = qp.error.ae(true_prevalence, predicted_prevalence)
        nmd = qp.error.normalized_match_distance(true_prevalence, predicted_prevalence)
        return nmd

    results_nmd = qp.util.parallel(job, np.arange(n_batches), n_jobs=-1, asarray=True, backend='loky')

    return results_nmd


def methods(base_classifier):
    yield 'CC', CC(base_classifier)
    yield 'PCC', PCC(base_classifier)
    yield 'PACC', PACC(base_classifier)


def main(data, n_classes, dataset_name):
    base_classifier = LogisticRegression()
    for method_name, method in methods(base_classifier):
        print(f'running {method_name}')
        results = experiment(data, n_classes, method)
        print(results)

if __name__ == '__main__':
    dataset_dir = '../datasets'
    n_classes_list = [5]
    dataset_names = ['diversity', 'toxicity', 'activity']
    for dataset_name, n_classes in product(dataset_names, n_classes_list):
        print(f'running {dataset_name=} {n_classes}')
        data = load_dataset(join(dataset_dir, f'{dataset_name}_dataset'), n_classes=n_classes, filter_out_multiple_subreddits=False)
        main(data, n_classes, dataset_name)

# CC [0.00318037 0.01162731 0.02090877 0.04247854 0.03984796 0.05510725 0.05329431 0.05905753 0.06550041 0.17857143]
# PCC [0.00280356 0.00440107 0.00244653 0.01039527 0.00457421 0.00200411 0.00193409 0.00514357 0.00458079 0.13097832]
# PACC [0.19307214 0.07399234 0.15285762 0.14943895 0.1104644  0.11927672  0.13393173 0.16204134 0.09514025 0.28571429]