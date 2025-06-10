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
from commons import SAMPLE_SIZE, get_full_path
# from classification import BlockEnsembleClassifier
from data import load_dataset, FEATURE_GROUP_PREFIXES
from quapy.error import ae, nmd
from quapy.evaluation import evaluation_report


def load_data(dataset_dir, dataset_name, n_classes, features, n_batches, seed=0):
    data = load_dataset(f'{dataset_dir}/{dataset_name}_dataset', n_classes=n_classes, features_blocks=features)
    classes = np.arange(n_classes)

    X = data.X
    y = data.y
    # X = PCA(n_components=100).fit_transform(X)

    all_data = LabelledCollection(X, y, classes=classes)
    training_pool, test = all_data.split_stratified(train_prop=8000, random_state=0)

    batch_size = len(training_pool) // n_batches
    np.random.seed(seed)
    random_order = np.random.permutation(len(training_pool))

    return training_pool, test, batch_size, random_order, classes, data.prefix_idx


def experiment_pps_random_split_all_methods(
        dataset_name,
        n_classes,
        sample_size=SAMPLE_SIZE,
        features='all',
        result_dir='../results/random_split_features',
        dataset_dir='../datasets'):

    all_reports = []
    for method_name in select_methods():
        if method_name == 'MLPE' and features != 'all':
            continue
        report, _ = experiment_label_shift(
            dataset_name,
            n_classes,
            method_name,
            sample_size,
            features,
            result_dir=result_dir,
            dataset_dir=dataset_dir
        )
        all_reports.append(report)
    return all_reports


def experiment_label_shift(
        dataset_name,
        n_classes,
        method_name,
        sample_size=SAMPLE_SIZE,
        features='all',
        features_short_id=None,
        result_dir='../results/random_split_features',
        dataset_dir='../datasets',
        n_runs=5
):

    qp.environ['SAMPLE_SIZE'] = sample_size
    result_dir = get_full_path(result_dir, dataset_name, n_classes, sample_size)

    if isinstance(features, str):
        feature_names = features
    else: # list of feature names
        feature_names = '::'.join(features)

    # the path of the resulting report is generated automatically, by also taking into account
    # the feature blocks (features) that concurred in the experiment, by generating a concatenation
    # of their names (e.g., <feat_block_1>::<feat_block_2>, see feature_names). However, these may be too many
    # as to generate a valid filename. In these cases, it is better to explicitly provide a
    # short name (by setting features_short_id to a valid string)
    if features_short_id is None:
        features_short_id = feature_names

    # load data
    n_batches=16

    report_path = join(result_dir, f'{method_name}__{features_short_id}.csv')
    print(report_path)
    if os.path.exists(report_path):
        method_report = qp.util.load_report(report_path)
    else:
        trainsize_reports = []
        for run in range(n_runs):
            print(f'Running {dataset_name} {method_name} {run=}')
            training_pool, test, batch_size, random_order, classes, blocks_ids \
                = load_data(dataset_dir, dataset_name, n_classes, features, n_batches, seed=run)

            method, param_grid = new_method(method_name, blocks_ids)

            for batch in range(n_batches):
                tr_selection = random_order[:(batch+1)*batch_size]
                train = training_pool.sampling_from_index(tr_selection)

                # model training
                if len(param_grid)==0:
                    method.fit(train)
                else:
                    devel, validation = train.split_stratified(random_state=0)
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
                report['features'] = feature_names
                report['dataset'] = dataset_name
                report['n_classes'] = n_classes
                report['run'] = run
                trainsize_reports.append(report)
                print(f'\ttrain-size={len(train)} got nmd={report.mean(numeric_only=True)["nmd"]:.3f}')

        method_report = pd.concat(trainsize_reports)
        method_report.to_csv(report_path)

    return method_report, report_path


def select_methods():
    if args.method == 'all':
        return ['MLPE', 'CC', 'PACC', 'EMQ'] #, 'KDEy-ML']
    else:
        return [args.method]


def new_method(method_name, blocks_ids=None):
    params_lr = {'classifier__C': np.logspace(-4, 4, 9), 'classifier__class_weight': ['balanced', None]}
    params_kde = {**params_lr, 'bandwidth': np.linspace(0.005, 0.15, 20)}

    factory = {
        'MLPE': (MLPE(), {}),
        'CC': (CC(), params_lr),
        'PACC': (PACC(), params_lr),
        'EMQ': (EMQ(), params_lr),
        # 'o-SLD': ()
        #'EMQ-b': (EMQ(BlockEnsembleClassifier(LogisticRegression(), blocks_ids=blocks_ids, kfcv=5)), {}),
        #'KDEy-ML': (KDEyML(), params_kde)
    }
    if method_name not in factory:
        raise ValueError(f'unknown method; valid ones are {factory.keys()}')

    return factory[method_name]


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
    parser.add_argument('--feats', type=str, default='all',
                        help='Feature blocks to use: all=all blocks concatenated, full= all + 1st-level + 2nd level')

    args = parser.parse_args()

    if args.feats == 'all':
        feature_blocks = ['all']
    elif args.feats == 'full':
        feature_blocks = ['all'] + FEATURE_GROUP_PREFIXES + data.FEATURE_SUBGROUP_PREFIXES
    else:
        raise ValueError('unrecognized --feats, valid args are "all" and "full"')

    n_classes_list = [5]
    dataset_names = ['activity', 'toxicity', 'diversity']

    all_reports = []
    for dataset_name, n_classes, features in product(dataset_names, n_classes_list, feature_blocks):
        reports = experiment_pps_random_split_all_methods(dataset_name, n_classes, features=features)
        all_reports.extend(reports)

    show_results_random_split(pd.concat(all_reports))

