import os
import pickle
from os.path import join
from itertools import product
import pandas as pd
import quapy as qp
import numpy as np
import sklearn.preprocessing
from quapy.data import LabelledCollection
from quapy.method.aggregative import PCC, PACC, CC, KDEyML, KDEyCS, KDEyHD, EMQ, ACC
from quapy.method.base import BaseQuantifier
from quapy.method.confidence import WithConfidenceABC, ConfidenceIntervals, AggregativeBootstrap
from quapy.method.non_aggregative import MaximumLikelihoodPrevalenceEstimation as MLPE
from quapy.model_selection import GridSearchQ
from quapy.protocol import UPP
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from typing_extensions import deprecated

from data import load_dataset, merge_data
from quapy.error import ae, nmd
from quapy.evaluation import evaluation_report
import warnings
from classification import BlockEnsembleClassifier
from quapy.functional import prevalence_from_labels, strprev, uniform_prevalence, softmax

from utils import mmd_pairwise_rbf_blocks

from sklearn.metrics.pairwise import pairwise_kernels


def prepare_dataset(dataset_name, n_classes, features, data_dir='../datasets'):
    old_path = f'{data_dir}/old_features/{dataset_name}_dataset'
    new_path = f'{data_dir}/new_features/{dataset_name}_dataset'
    if features=='old':
        data = load_dataset(old_path, n_classes=n_classes)
        data.prefix_idx = None
    elif features=='new':
        data = load_dataset(new_path, n_classes=n_classes)
    elif features=='both':
        data_old = load_dataset(old_path, n_classes=n_classes)
        data_new = load_dataset(new_path, n_classes=n_classes)
        data = merge_data(data_old, data_new)
        data.prefix_idx = None
    else:
        raise ValueError(f'feature type {features} not understood')
    return data


def project_onto_pv(X, Xpv):
    # M = pairwise_kernels(normalize(X, norm='l2'), normalize(Xpv, norm='l2'), metric='rbf')
    M = pairwise_kernels(X, Xpv, metric='linear')
    M = normalize(M, norm='l2')
    # M = softmax(M)
    return M


def train_and_test(train, test):
    q = PACC()
    q.fit(train)
    qp.environ['SAMPLE_SIZE']=500
    protocol = UPP(test, repeats=100, random_state=0)
    mae = qp.evaluation.evaluate(q, protocol=protocol, error_metric='mae')
    print(f'{mae=:.4f}')


if __name__ == '__main__':

    n_classes = 3
    dataset_name = 'activity'
    features_choice = 'new'
    data = prepare_dataset(dataset_name, n_classes, features_choice)

    subreddit_target = 1
    in_test = data.subreddits[subreddit_target]
    in_train = np.vstack([data.subreddits[:subreddit_target], data.subreddits[subreddit_target+1:]]).sum(axis=0).astype(bool)
    commons = in_train & in_test
    in_test = in_test & ~commons
    in_train = in_train & ~commons

    print(f'in_train = {in_train.sum()}')
    print(f'commons = {commons.sum()}')
    print(f'in_test = {in_test.sum()}')

    Xtr=data.X[in_train]
    ytr=data.y[in_train]

    Xpv = data.X[commons]
    ypv = data.y[commons]

    Xte = data.X[in_test]
    yte = data.y[in_test]

    Ptr = project_onto_pv(Xtr, Xpv)
    Pte = project_onto_pv(Xte, Xpv)

    pca = PCA(n_components=500)
    Ptr = pca.fit_transform(Ptr)
    Pte = pca.transform(Pte)

    Xtrain = LabelledCollection(Xtr, ytr)
    Xtest  = LabelledCollection(Xte, yte, Xtrain.classes_)

    Ptrain = LabelledCollection(Ptr, ytr)
    Ptest = LabelledCollection(Pte, yte, Ptrain.classes_)

    # train_and_test(Xtrain, Xtest)
    train_and_test(Ptrain, Ptest)

