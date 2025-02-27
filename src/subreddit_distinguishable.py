import os
import pickle
from os.path import join
from itertools import product

import pandas as pd
import quapy as qp
import numpy as np
from quapy.data import LabelledCollection
from quapy.evaluation import evaluation_report
from quapy.method.aggregative import PCC, PACC, CC, KDEyML, KDEyCS, KDEyHD, EMQ, ACC
from quapy.method.base import BaseQuantifier
from quapy.method.confidence import WithConfidenceABC, ConfidenceIntervals, AggregativeBootstrap
from quapy.method.non_aggregative import MaximumLikelihoodPrevalenceEstimation
from quapy.model_selection import GridSearchQ
from quapy.protocol import UPP
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
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


def experiment_domain_discernibility(data, n_classes):
    n_subreddits = len(data.subreddit_names)
    classes = np.arange(n_classes)

    X = data.X
    y = data.y

    print(f'X.shape[1] = {X.shape[1]}')
    if X.shape[1] > 100:
        X = PCA(n_components=100).fit_transform(X)

    qp.environ['SAMPLE_SIZE'] = 500
    # alldata = LabelledCollection(X, y, classes=classes)
    # train, test = alldata.split_stratified()
    # pacc = PACC().fit(train)
    # mae = qp.evaluation.evaluate(pacc, protocol=UPP(test, repeats=100), error_metric='mae', verbose=False)
    # print(f'\tPPS w\o subreddits MAE={mae:.5f}')

    accs = []
    for subreddit_idx in range(n_subreddits):
        in_test = data.subreddits[subreddit_idx]
        print(f'{data.subreddit_names[subreddit_idx]} has {sum(in_test)} instances')

        # domain separation
        domain = LabelledCollection(X, labels=(in_test*1), classes=[0,1])
        tr, te = domain.split_stratified()

        domain_classifier = LogisticRegression()
        domain_classifier.fit(*tr.Xy)
        is_in_test = domain_classifier.predict(te.X)
        acc = (is_in_test==te.y).mean()
        f1 = f1_score(te.y, is_in_test)
        accs.append(acc)

        # quantification
        train = LabelledCollection(X[~in_test], y[~in_test], classes=classes)
        test  = LabelledCollection(X[in_test], y[in_test], classes=classes)
        pacc = PACC().fit(train)
        mae = qp.evaluation.evaluate(pacc, protocol=UPP(test, repeats=100), error_metric='mae', verbose=False)
        print(f'\tdomain detection acc={acc:.4f}\tf1={f1:.4f}\t MAE={mae:.5f}')

    print(f'\nMean acc={np.mean(accs):.5f}')


def methods(base_classifier, block_ids):
    # yield 'MLPE', MaximumLikelihoodPrevalenceEstimation()
    # yield 'CC', CC(base_classifier)
    # yield 'PCC', PCC(base_classifier)
    # yield 'PCC-CI', AggregativeBootstrap(PCC(base_classifier), n_train_samples=50, n_test_samples=50, confidence_level=0.95)
    # yield 'bPCC', PCC(BlockEnsembleClassifier(base_classifier, blocks_ids=block_ids))
    # yield 'ACC', ACC(base_classifier)
    yield 'PACC', PACC(base_classifier)
    # yield 'EMQ', EMQ(base_classifier)
    # yield 'EMQ-cal', EMQ(CalibratedClassifierCV(base_classifier))
    # yield 'KDEy-ML', KDEyML(base_classifier)
    # yield 'KDEy-CS', KDEyCS(base_classifier)
    # yield 'KDEy-HD', KDEyHD(base_classifier)




if __name__ == '__main__':
    n_classes_list = [3]
    dataset_names = ['activity'] #, 'toxicity'] # 'diversity',

    features_blocks = [None, np.str_('ACTIVITY'), np.str_('DEMOGRAPHIC'), np.str_('EMBEDDINGS'), np.str_('EMOTIONS'), np.str_('LIWC'), np.str_('SENTIMENT'), np.str_('SOC'), np.str_('WRITING')]

    for dataset_name, n_classes, feat_block in product(dataset_names, n_classes_list, features_blocks):
        old_path = join('../datasets/old_features', f'{dataset_name}_dataset')
        new_path = join('../datasets/new_features', f'{dataset_name}_dataset')

        print(f'running {dataset_name=} {n_classes}')
        # data_old = load_dataset(old_path, n_classes=n_classes)
        data_new = load_dataset(new_path, n_classes=n_classes, features_blocks=feat_block)
        # data = merge_data(data_old, data_new)
        data = data_new
        base_classifier = LogisticRegression()
        for method_name, method in methods(base_classifier, data.prefix_idx):
            results = experiment_domain_discernibility(data, n_classes)

