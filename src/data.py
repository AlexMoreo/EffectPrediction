import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import StandardScaler
from sklearn.utils import Bunch
from collections import OrderedDict


THRESHOLD_3_CLASSES={
    'activity': [0.4],
    'diversity': [0.3],
    'toxicity': [0.4],
}

THRESHOLD_5_CLASSES={
    'activity': [0.2, 0.55],
    'diversity': [0.2, 0.55],
    'toxicity': [0.2, 0.55],
}

SUBREDDIT_NAMES = [
        'CCJ2',
        'ChapoTrapHouse',
        'ConsumeProduct',
        'DarkHumorAndMemes',
        'DarkJokeCentral',
        'DebateAltRight',
        'GenderCritical',
        'HateCrimeHoaxes',
        'OandAExclusiveForum',
        'ShitNeoconsSay',
        'TheNewRight',
        'The_Donald',
        'Wojak',
        'imgoingtohellforthis2',
        'soyboys'
    ]


def load_dataset(path, n_classes, filter_out_multiple_subreddits=False, filter_abandoned_activity=True):

    # asserts
    if n_classes not in [3, 5]:
        raise ValueError(f'unexpected {n_classes=}; valid values are 3, 5')

    if not os.path.exists(path):
        raise ValueError(f'file {path} not found!')

    # load dataframe
    df = pd.read_json(path)

    if filter_out_multiple_subreddits:
        df = df[df[SUBREDDIT_NAMES].sum(axis=1) == 1]

    if 'activity' in path and filter_abandoned_activity:
        df = df[df['label'] != -1]

    if 'index' in df.columns:
        # unused, and only present in some datasets
        df.pop('index')

    # prepare the class-specific thresholds for binning the scores into labels
    thresholds = THRESHOLD_3_CLASSES if n_classes==3 else THRESHOLD_5_CLASSES
    threshold_values = None
    for data_name in thresholds.keys():
        if data_name in path:
            threshold_values = thresholds[data_name]
    assert threshold_values is not None, \
        f'unknown threshold for dataset in path {path}'

    # make threshold_values symmetric and bounded by inf; e.g., [0.2, 0.55] -> [-inf, -0.55, -0.2, 0.2, 0.55, inf]
    threshold_values = sorted(threshold_values)
    threshold_values = [-np.inf] + [-t for t in threshold_values[::-1]] + threshold_values + [np.inf]
    new_labels = np.arange(len(threshold_values)-1)

    # parsing information
    authors = df.pop('author')  # unused
    label_scores = {
        'global': df.pop('label').values
    }
    for period in range(7):
        label_scores[f'label_{period+1}'] = df.pop(f'label_{period+1}').values

    label_classes = {}
    for label, scores in label_scores.items():
        label_classes[label] = pd.cut(label_scores[label], bins=threshold_values, labels=new_labels, right=False).to_numpy()

    n_covariates = 623
    n_subreddits = len(SUBREDDIT_NAMES)
    assert len(df.columns) == n_covariates+n_subreddits, 'unexpected number of columns'
    covariates = df.iloc[:,:n_covariates].values
    covariate_names = df.columns.values[:n_covariates]

    assert all(df.columns.values[-n_subreddits:] == SUBREDDIT_NAMES), 'unexpected subreddit names'
    subreddits = df.values[:, -n_subreddits:].astype(bool).T

    # get the feature blocks ids and prefixes
    column_prefixes = [n.split('_')[0] for n in covariate_names]
    prefix_idx = OrderedDict()
    for prefix in sorted(np.unique(column_prefixes)):
        prefix_idx[prefix] = np.char.startswith(column_prefixes, prefix)

    # standardize all covariates; doing this once and for all is not flawed
    # since, while quantifying a set, we assume to have access to the covariates
    # of the test set, and since standardization is unsupervised.
    zscorer = StandardScaler()
    covariates = zscorer.fit_transform(covariates)

    data = Bunch(
        X=covariates,
        y=label_classes['global'],
        scores=label_scores['global'],
        y_periods=np.vstack([label_classes[f'label_{period+1}'] for period in range(7)]),
        scores_periods=np.vstack([label_scores[f'label_{period + 1}'] for period in range(7)]),
        covariate_names=covariate_names,
        subreddit_names=SUBREDDIT_NAMES,
        subreddits=subreddits,
        prefix_idx=prefix_idx,
    )

    return data



if __name__ == '__main__':
    path = '../datasets/diversity_dataset'
    data = load_dataset(path, n_classes=5)
    print(data)