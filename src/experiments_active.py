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
from quapy.method.non_aggregative import MaximumLikelihoodPrevalenceEstimation
from quapy.model_selection import GridSearchQ
from quapy.protocol import UPP
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from data import load_dataset, merge_data
import warnings
from classification import BlockEnsembleClassifier
from quapy.functional import prevalence_from_labels, strprev, uniform_prevalence

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

    from_subreddit = data.subreddits[1]


    n = X.shape[0]
    np.random.seed(0)
    # random_order = np.random.permutation(n)

    # test_split_point = 2*n//3
    idx = np.arange(len(y))
    test_idx = idx[from_subreddit]
    remainder = idx[~from_subreddit]
    # test_idx = random_order[test_split_point:]

    Xte = X[test_idx]
    yte = y[test_idx]
    test = LabelledCollection(Xte, yte, classes=classes)
    # test = test.sampling(len(test), *test.prevalence()[::-1])

    # random_order = random_order[:test_split_point]
    # pool = LabelledCollection(X[random_order], y[random_order], classes=classes)
    # pool = pool.sampling(5000, *uniform_prevalence(n_classes))
    # random_order = np.random.permutation(len(pool))

    # X = pool.X
    # y = pool.y

    init_train_size = 500
    np.random.shuffle(remainder)
    train_idx = remainder[:init_train_size]
    remainder = remainder[init_train_size:]

    batch_size = 100

    Xtr = X[train_idx]
    ytr = y[train_idx]
    train = LabelledCollection(Xtr, ytr, classes=classes)
    # test = test.sampling(len(test), *test.prevalence()[::-1])
    true_prevalence = test.prevalence()
    print(strprev(true_prevalence))

    results_nmd = []

    while(len(remainder)>batch_size):

        if not isinstance(method, WithConfidenceABC):
            method.fit(train)
            predicted_prevalence = method.quantify(test.X)
            nmd = qp.error.normalized_match_distance(true_prevalence, predicted_prevalence)
            results_nmd.append({'method': method_name, '#train': len(train), 'nmd': nmd})

            print(f'{method_name} train_size={len(train)} {nmd=:.5f}')

            if hasattr(method, 'classifier'):
                posteriors = method.classifier.predict_proba(X[remainder])
                conf = posteriors.max(axis=1)
                order = np.argsort(conf)
                remainder = remainder[order]

            next_batch_idx = remainder[:batch_size]
            remainder = remainder[batch_size:]
            new_Xtr = X[next_batch_idx]
            new_ytr = y[next_batch_idx]
            next_batch = LabelledCollection(new_Xtr, new_ytr, classes=classes)
        else:
            method.fit(train, val_split=5)
            predicted_prevalence = method.quantify(test.X)
            nmd = qp.error.normalized_match_distance(true_prevalence, predicted_prevalence)
            results_nmd.append({'method': method_name, '#train': len(train), 'nmd': nmd})

            print(f'{method_name} train_size={len(train)} {nmd=:.5f}')

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


# def experiment_natural_prev_subreddit(data, n_classes, method, method_name):
#     n_subreddits = len(data.subreddit_names)
#     classes = np.arange(n_classes)
#
#     aes, nmds = [], []
#     for subreddit in range(n_subreddits):
#         sel = data.subreddits[subreddit]
#         X = data.X
#         y = data.y
#
#         X = PCA(n_components=100).fit_transform(X)
#
#         train = LabelledCollection(X[~sel], y[~sel])
#         test = LabelledCollection(X[sel], y[sel])
#
#         method.fit(train)
#         estim_prev = method.quantify(test.X)
#         true_prev = test.prevalence()
#
#         aes.append(qp.error.ae(true_prev, estim_prev))
#         nmds.append(qp.error.nmd(true_prev, estim_prev))
#
#     print(f'{method_name=} got MAE={np.mean(aes):.4f} NMD={np.mean(nmds):.4f}')
#
#     return None
#     results_nmd = pd.DataFrame(results_nmd)
#     return results_nmd

def experiment_pps(data, n_classes, method, method_name):
    n_subreddits = len(data.subreddit_names)
    classes = np.arange(n_classes)

    X = data.X
    y = data.y

    # X = PCA(n_components=100).fit_transform(X)

    # data = LabelledCollection(X, y)
    # train, test = data.split_stratified(random_state=0)
    sub_idx = 11
    in_test=data.subreddits[sub_idx]
    print(f'in {data.subreddit_names[sub_idx]} with {sum(in_test)} instances')
    train = LabelledCollection(X[~in_test], y[~in_test], classes=classes)
    test = LabelledCollection(X[in_test], y[in_test], classes=classes)

    # if method_name!='PACC': return
    # domain_classifier = LogisticRegression()
    # domain_classifier.fit(X, y=in_test)
    # posteriors = domain_classifier.predict_proba(X[~in_test])
    # weights = (posteriors[:, 0] + 1e-7) / (posteriors[:, 1] + 1e-7)
    # # weights = (posteriors[:, 1] + 1e-7) / (posteriors[:, 1] + 1e-7)
    #
    # cls = LogisticRegression()
    # cls.fit(*train.Xy, sample_weight=weights)
    # pacc = PACC(classifier=cls)
    # pacc.fit(train, fit_classifier=False, val_split=train)
    #
    # qp.environ['SAMPLE_SIZE']=1000
    # report = qp.evaluation.evaluation_report(pacc, protocol=UPP(test, repeats=500), error_metrics=[qp.error.nmd, 'ae'])
    # means = report.mean(numeric_only=True)

    qp.environ['SAMPLE_SIZE'] = 1000
    if method_name not in ['MLPE']:
        dev, val = train.split_stratified(random_state=0)
        if "KDE" in method_name:
            param_grid = {'classifier__C': np.logspace(-4, 4, 9), 'classifier__class_weight':['balanced', None], 'bandwidth':np.linspace(0.005, 0.15, 20)}
        else:
            param_grid = {'classifier__C': np.logspace(-4, 4, 9), 'classifier__class_weight': ['balanced', None]}

        modsel = GridSearchQ(
            model=method,
            param_grid=param_grid,
            protocol=UPP(val, repeats=100),
            n_jobs=-1,
            refit=False,
            verbose=True
        ).fit(dev)
        method = modsel.best_model()
        print(modsel.best_params_)
    else:
        print('no model selection')
        method.fit(train)

    report = qp.evaluation.evaluation_report(method, protocol=UPP(test, repeats=100), error_metrics=[qp.error.nmd, 'ae'], verbose=True)
    means = report.mean(numeric_only=True)
    print(means)

    print(f'{method_name=} got MAE={means["ae"]:.4f} NMD={means["nmd"]:.4f}')

    return means
    # results_nmd = pd.DataFrame(results_nmd)
    # return results_nmd


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


def show_plot(df, dataset, n_classes):
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set(style="whitegrid")

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="#train", y="nmd", hue="method", marker="o", palette="tab10")

    plt.xlabel("Training Samples")
    plt.ylabel("NMD Error")
    plt.title(f'{dataset} {n_classes} classes')

    plt.legend(title="Method")
    plt.savefig(f'../fig/{dataset}_{n_classes}_classes.png')
    # plt.show()


def main(data, n_classes, dataset_name):
    base_classifier = LogisticRegression(max_iter=3000)
    all_results = []
    for method_name, method in methods(base_classifier, data.prefix_idx):
        # print(f'running {method_name}')
        results = experiment_pps(data, n_classes, method, method_name)
        # experiment_natural_prev_subreddit(data, n_classes, method, method_name)
        # all_results.append(results)

    # df = pd.concat(all_results)
    # df['dataset'] = dataset_name
    # df['n_classes'] = n_classes
    # df.to_csv(f'../results_active_learning/{dataset_name}_{n_classes}_classes.csv')
    # show_plot(df, dataset_name, n_classes)




if __name__ == '__main__':
    n_classes_list = [3]
    dataset_names = ['activity', 'toxicity'] # 'diversity',
    for dataset_name, n_classes in product(dataset_names, n_classes_list):
        print(f'running {dataset_name=} {n_classes}')
        data_old = load_dataset(join('../datasets/old_features', f'{dataset_name}_dataset'), n_classes=n_classes, filter_out_multiple_subreddits=False, filter_abandoned_activity=False)
        # data_new = load_dataset(join('../datasets/new_features', f'{dataset_name}_dataset'), n_classes=n_classes, filter_out_multiple_subreddits=False, filter_abandoned_activity=False)
        # data = merge_data(data_old, data_new)
        data = data_old
        main(data, n_classes, dataset_name)

