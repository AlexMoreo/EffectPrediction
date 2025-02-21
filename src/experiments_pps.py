import os
import pickle
from os.path import join
from itertools import product
import pandas as pd
import quapy as qp
import numpy as np
from quapy.data import LabelledCollection
from quapy.method.aggregative import PCC, PACC, CC, KDEyML, KDEyCS, KDEyHD, EMQ, ACC
from quapy.method.base import BaseQuantifier
from quapy.method.confidence import WithConfidenceABC, ConfidenceIntervals, AggregativeBootstrap
from quapy.method.non_aggregative import MaximumLikelihoodPrevalenceEstimation as MLPE
from quapy.model_selection import GridSearchQ
from quapy.protocol import UPP
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from data import load_dataset, merge_data
from quapy.error import ae, nmd
from quapy.evaluation import evaluation_report
import warnings
from classification import BlockEnsembleClassifier
from quapy.functional import prevalence_from_labels, strprev, uniform_prevalence

from utils import mmd_pairwise_rbf_blocks


def experiment_pps_by_subreddit(
        dataset_name,
        n_classes,
        features,
        min_subreddit_instances=1000,
        sample_size=500,
        result_dir='../results',
        dataset_dir='../datasets'):

    config = f'min{min_subreddit_instances}_size{sample_size}'  # experiment macro setup
    result_dir=join(result_dir, config, dataset_name, f'{n_classes}_classes')
    os.makedirs(result_dir, exist_ok=True)

    qp.environ['SAMPLE_SIZE'] = sample_size

    # load data
    data = prepare_dataset(dataset_name, n_classes, features, data_dir=dataset_dir)

    n_subreddits = len(data.subreddit_names)
    classes = np.arange(n_classes)

    X = data.X
    y = data.y
    X = PCA(n_components=100).fit_transform(X)

    all_reports=[]
    for method_name, method, param_grid in methods():
        report_path = join(result_dir, f'{method_name}__{features}.csv')
        print(report_path)
        if os.path.exists(report_path):
            method_report = qp.util.load_report(report_path)
        else:
            subreddit_reports = []
            for subreddit_idx in range(n_subreddits):
                n_instances = sum(data.subreddits[subreddit_idx])
                if n_instances < min_subreddit_instances:
                    continue

                in_test = data.subreddits[subreddit_idx]
                train = LabelledCollection(X[~in_test], y[~in_test], classes=classes)
                test  = LabelledCollection(X[in_test], y[in_test], classes=classes)

                # model training
                if len(param_grid)==0:
                    method.fit(train)
                else:
                    devel, validation = train.split_stratified()
                    model_selection = GridSearchQ(
                        model=method,
                        param_grid=param_grid,
                        protocol=UPP(validation, repeats=250),
                        n_jobs=-1,
                        refit=True,
                        verbose=False
                    ).fit(devel)
                    method = model_selection.best_model()

                # model test
                test_protocol = UPP(test, repeats=1000)

                report = evaluation_report(method, protocol=test_protocol, error_metrics=[nmd, ae], verbose=True)
                report['method'] = method_name
                report['subreddit'] = data.subreddit_names[subreddit_idx]
                report['features'] = features
                report['dataset'] = dataset_name
                report['n_classes'] = n_classes
                subreddit_reports.append(report)

            method_report = pd.concat(subreddit_reports)
            method_report.to_csv(report_path)

        all_reports.append(method_report)

    return all_reports


def experiment_pps_random_split(
        dataset_name,
        n_classes,
        features,
        min_subreddit_instances=1000,
        sample_size=500,
        result_dir='../results/random_split',
        dataset_dir='../datasets'):

    config = f'min{min_subreddit_instances}_size{sample_size}'  # experiment macro setup
    result_dir=join(result_dir, config, dataset_name, f'{n_classes}_classes')
    os.makedirs(result_dir, exist_ok=True)

    qp.environ['SAMPLE_SIZE'] = sample_size

    # load data
    data = prepare_dataset(dataset_name, n_classes, features, data_dir=dataset_dir)

    classes = np.arange(n_classes)

    X = data.X
    y = data.y
    X = PCA(n_components=100).fit_transform(X)

    all_data = LabelledCollection(X, y, classes=classes)
    training_pool, test = all_data.split_stratified(train_prop=8000, random_state=0)
    n_batches = 16
    batch_size = len(training_pool)//n_batches
    np.random.seed(0)
    random_order = np.random.permutation(len(training_pool))

    all_reports=[]
    for method_name, method, param_grid in methods():
        report_path = join(result_dir, f'{method_name}__{features}.csv')
        print(report_path)
        if os.path.exists(report_path):
            method_report = qp.util.load_report(report_path)
        else:
            trainsize_reports = []
            for batch in range(n_batches):
                tr_selection = random_order[:(batch+1)*batch_size]
                train = training_pool.sampling_from_index(tr_selection)

                # model training
                if len(param_grid)==0:
                    method.fit(train)
                else:
                    devel, validation = train.split_stratified()
                    model_selection = GridSearchQ(
                        model=method,
                        param_grid=param_grid,
                        protocol=UPP(validation, repeats=250),
                        n_jobs=-1,
                        refit=True,
                        verbose=False
                    ).fit(devel)
                    method = model_selection.best_model()

                # model test
                test_protocol = UPP(test, repeats=1000)

                report = evaluation_report(method, protocol=test_protocol, error_metrics=[nmd, ae], verbose=False)
                report['method'] = method_name
                report['tr_size'] = len(train)
                report['features'] = features
                report['dataset'] = dataset_name
                report['n_classes'] = n_classes
                trainsize_reports.append(report)
                print(f'\ttrain-size={len(train)} got mae={report.mean(numeric_only=True)["ae"]:.3f}')

            method_report = pd.concat(trainsize_reports)
            method_report.to_csv(report_path)

        all_reports.append(method_report)

    return all_reports




def methods():
    params_lr = {'classifier__C': np.logspace(-4, 4, 9), 'classifier__class_weight': ['balanced', None]}
    params_kde = {**params_lr, 'bandwidth': np.linspace(0.005, 0.15, 20)}

    yield 'MLPE', MLPE(), {}
    yield 'CC', CC(), params_lr
    yield 'PCC', PCC(), params_lr
    # yield 'bPCC', PCC(BlockEnsembleClassifier(base_classifier, blocks_ids=block_ids)), params_lr
    yield 'ACC', ACC(), params_lr
    yield 'PACC', PACC(), params_lr
    yield 'EMQ', EMQ(), params_lr
    yield 'KDEy-ML', KDEyML(), params_kde
    # yield 'KDEy-CS', KDEyCS(), params_kde
    # yield 'KDEy-HD', KDEyHD(), params_kde


def show_results_by_subreddit(reports:pd.DataFrame):
    print('NORMALIZED MATCH DISTANCE')
    print(reports.pivot_table(
        index=['n_classes', 'dataset', 'subreddit'],
        columns=['method', 'features'],
        values='nmd'
    ))
    print()
    print('ABSOLUTE ERROR')
    print(reports.pivot_table(
        index=['n_classes', 'dataset', 'subreddit'],
        columns=['method', 'features'],
        values='ae'
    ))

def show_results_random_split(reports:pd.DataFrame):
    print('NORMALIZED MATCH DISTANCE')
    print(reports.pivot_table(
        index=['n_classes', 'dataset', 'tr_size'],
        columns=['method', 'features'],
        values='nmd'
    ))
    print()
    print('ABSOLUTE ERROR')
    print(reports.pivot_table(
        index=['n_classes', 'dataset', 'tr_size'],
        columns=['method', 'features'],
        values='ae'
    ))

def prepare_dataset(dataset_name, n_classes, features, data_dir='../datasets'):
    old_path = f'{data_dir}/old_features/{dataset_name}_dataset'
    new_path = f'{data_dir}/new_features/{dataset_name}_dataset'
    if features=='old':
        data = load_dataset(old_path, n_classes=n_classes)
    elif features=='new':
        data = load_dataset(new_path, n_classes=n_classes)
    elif features=='both':
        data_old = load_dataset(old_path, n_classes=n_classes)
        data_new = load_dataset(new_path, n_classes=n_classes)
        data = merge_data(data_old, data_new)
    else:
        raise ValueError(f'feature type {features} not understood')
    return data


if __name__ == '__main__':
    n_classes_list = [3, 5]
    dataset_names = ['activity', 'toxicity', 'diversity']
    features_choice = ['old', 'new', 'both']
    all_reports = []
    for dataset_name, n_classes, features in product(dataset_names, n_classes_list, features_choice):
        reports = experiment_pps_random_split(dataset_name, n_classes, features)
        all_reports.extend(reports)

    show_results_random_split(pd.concat(all_reports))

