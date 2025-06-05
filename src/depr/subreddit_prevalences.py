import os
import pickle
from os.path import join
from itertools import product

import pandas as pd
import quapy as qp
import numpy as np
from quapy.data import LabelledCollection
import quapy.functional as F
from quapy.method.aggregative import PACC
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from data import load_dataset, merge_data
import warnings
from classification import BlockEnsembleClassifier
from quapy.functional import prevalence_from_labels, strprev, uniform_prevalence

from utils import mmd_pairwise_rbf_blocks


warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
warnings.filterwarnings("ignore", category=ConvergenceWarning, module='sklearn')


def show_subreddit_stats(data, n_classes):
    n_subreddits = len(data.subreddit_names)
    classes = np.arange(n_classes)

    X = data.X
    y = data.y

    accs = []
    for subreddit_idx in range(n_subreddits):
        name = data.subreddit_names[subreddit_idx]
        in_test = data.subreddits[subreddit_idx]

        train = LabelledCollection(X[~in_test], y[~in_test], classes=classes)
        test  = LabelledCollection(X[in_test], y[in_test], classes=classes)

        lr = LogisticRegression()
        lr.fit(*train.Xy)
        y_pred = lr.predict(test.X)
        y_true = test.y
        acc = (y_true==y_pred).mean()
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_micro = f1_score(y_true, y_pred, average='micro')

        pacc = PACC().fit(train)
        estim_prev = pacc.quantify(test.X)
        nmd = qp.error.nmd(test.prevalence(), estim_prev)
        nmd_mlpe = qp.error.nmd(test.prevalence(), train.prevalence())


        print(f'subreddit={name} has {len(test)} instances;\tacc={acc:.4f} Mf1={f1_macro:.4f} mf1={f1_micro:.4f}\tPACC {nmd=:.4f} MLPE {nmd_mlpe:.4f}\tprevalence={F.strprev(test.prevalence())}')



if __name__ == '__main__':
    n_classes_list = [3]
    dataset_names = ['activity'] #, 'toxicity'] # 'diversity',

    # features_blocks = [None, np.str_('ACTIVITY'), np.str_('DEMOGRAPHIC'), np.str_('EMBEDDINGS'), np.str_('EMOTIONS'), np.str_('LIWC'), np.str_('SENTIMENT'), np.str_('SOC'), np.str_('WRITING')]

    for dataset_name, n_classes in product(dataset_names, n_classes_list):
        # old_path = join('../datasets/old_features', f'{dataset_name}_dataset')
        new_path = join('../datasets/new_features', f'{dataset_name}_dataset')

        print(f'running {dataset_name=} {n_classes}')
        # data_old = load_dataset(old_path, n_classes=n_classes)
        data_new = load_dataset(new_path, n_classes=n_classes)
        # data = merge_data(data_old, data_new)
        data = data_new
        results = show_subreddit_stats(data, n_classes)

