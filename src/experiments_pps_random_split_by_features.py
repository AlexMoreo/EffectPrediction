import os
import argparse
import pickle
from os.path import join
from itertools import product
import pandas as pd
import quapy as qp
import numpy as np
from quapy.data import LabelledCollection
from quapy.method.aggregative import PCC, PACC, CC, KDEyML, KDEyCS, KDEyHD, EMQ, ACC
from quapy.method.non_aggregative import MaximumLikelihoodPrevalenceEstimation as MLPE
from quapy.model_selection import GridSearchQ
from quapy.protocol import UPP
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

import data
from data import load_dataset, FEATURE_PREFIXES
from quapy.error import ae, nmd
from quapy.evaluation import evaluation_report



def experiment_pps_random_split(
        dataset_name,
        n_classes,
        sample_size=500,
        features='all',
        result_dir='../results/random_split_merged',
        dataset_dir='../datasets/merged_features'):

    config = f'samplesize{sample_size}'  # experiment macro setup
    result_dir=join(result_dir, config, dataset_name, f'{n_classes}_classes')
    os.makedirs(result_dir, exist_ok=True)

    qp.environ['SAMPLE_SIZE'] = sample_size

    # load data
    data = load_dataset(f'{dataset_dir}/{dataset_name}_dataset', n_classes=n_classes, features_blocks=features)
    if isinstance(features, str):
        feature_names = features
    else: # list of feature names
        feature_names = '--'.join(features)

    classes = np.arange(n_classes)

    X = data.X
    y = data.y
    # X = PCA(n_components=100).fit_transform(X)

    all_data = LabelledCollection(X, y, classes=classes)
    training_pool, test = all_data.split_stratified(train_prop=8000, random_state=0)
    n_batches = 16
    batch_size = len(training_pool)//n_batches
    np.random.seed(0)
    random_order = np.random.permutation(len(training_pool))

    all_reports=[]
    for method_name, method, param_grid in select(methods(data.prefix_idx)):
        report_path = join(result_dir, f'{method_name}__{feature_names}.csv')
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


def select(methods):
    for method_name, method, params in methods:
        if args.method=='all' or method_name in args.method:
            yield method_name, method, params


def methods(block_ids=None):
    params_lr = {'classifier__C': np.logspace(-4, 4, 9), 'classifier__class_weight': ['balanced', None]}
    params_kde = {**params_lr, 'bandwidth': np.linspace(0.005, 0.15, 20)}

    yield 'MLPE', MLPE(), {}
    yield 'CC', CC(), params_lr
    # yield 'PCC', PCC(), params_lr
    # yield 'bPCC', PCC(BlockEnsembleClassifier(base_classifier, blocks_ids=block_ids)), params_lr
    # yield 'ACC', ACC(), params_lr
    yield 'PACC', PACC(), params_lr
    yield 'EMQ', EMQ(), params_lr
    # if block_ids is not None:
    #     yield 'EMQ-b', EMQ(BlockEnsembleClassifier(LogisticRegression(), blocks_ids=block_ids, kfcv=5)), {}
    yield 'KDEy-ML', KDEyML(), params_kde
    # yield 'KDEy-CS', KDEyCS(), params_kde
    # yield 'KDEy-HD', KDEyHD(), params_kde


def show_results_random_split(reports:pd.DataFrame):
    pd.set_option('display.max_rows', 1000)
    pd.set_option('display.max_columns', 1000)
    pd.set_option('display.width', 1000)
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

def prepare_dataset(dataset_name, n_classes, data_dir):
    path = f'{data_dir}/{dataset_name}_dataset'
    data = load_dataset(path, n_classes=n_classes)
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Launch feature test for effect prediction.")
    parser.add_argument('--method', type=str, default='all', help='Name of the method to use')

    args = parser.parse_args()

    n_classes_list = [3, 5]
    dataset_names = ['activity', 'toxicity', 'diversity']
    feature_blocks = ['all'] + FEATURE_PREFIXES
    all_reports = []
    for dataset_name, n_classes, features in product(dataset_names, n_classes_list, feature_blocks):
        reports = experiment_pps_random_split(dataset_name, n_classes, features=features)
        all_reports.extend(reports)

    show_results_random_split(pd.concat(all_reports))

