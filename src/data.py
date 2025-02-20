import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import StandardScaler
from sklearn.utils import Bunch
from collections import OrderedDict


THRESHOLD_2_CLASSES={
    # abandoned users have -1 score, this binarizes the dataset into abandoned or not abandoned
    'activity': [-0.99],
}

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
    if n_classes not in [2, 3, 5]:
        raise ValueError(f'unexpected {n_classes=}; valid values are 3, 5')

    if n_classes == 2:
        assert 'activity' in path, 'binarization only works for the "activity" dataset'
        assert not filter_abandoned_activity, 'binarization and filtering abandoned are not compatible'

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
    thresholds = {
        2: THRESHOLD_2_CLASSES,
        3: THRESHOLD_3_CLASSES,
        5: THRESHOLD_5_CLASSES
    }.get(n_classes, None)
    assert thresholds is not None, 'unknown thresholds'

    threshold_values = None
    for data_name in thresholds.keys():
        if data_name in path:
            threshold_values = thresholds[data_name]
    assert threshold_values is not None, \
        f'unknown threshold for dataset in path {path}'

    # make threshold_values symmetric and bounded by inf; e.g., [0.2, 0.55] -> [-inf, -0.55, -0.2, 0.2, 0.55, inf]
    if n_classes in [3,5]:
        threshold_values = sorted(threshold_values)
        threshold_values = [-np.inf] + [-t for t in threshold_values[::-1]] + threshold_values + [np.inf]
    elif n_classes==2:
        threshold_values = [-np.inf] + threshold_values + [np.inf]
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

    n_new_covariates = 623
    n_old_covariates = 143
    n_subreddits = len(SUBREDDIT_NAMES)
    assert len(df.columns) in [n_new_covariates+n_subreddits, n_old_covariates+n_subreddits], \
        'unexpected number of columns'

    assert all(df.columns.values[-n_subreddits:] == SUBREDDIT_NAMES), 'unexpected subreddit names'
    subreddits = df.values[:, -n_subreddits:].astype(bool).T
    covariates = df.iloc[:, :-n_subreddits].values
    covariate_names = df.columns.values[:-n_subreddits]

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
        authors=authors,
        scores=label_scores['global'],
        y_periods=np.vstack([label_classes[f'label_{period+1}'] for period in range(7)]),
        scores_periods=np.vstack([label_scores[f'label_{period + 1}'] for period in range(7)]),
        covariate_names=covariate_names,
        subreddit_names=SUBREDDIT_NAMES,
        subreddits=subreddits,
        prefix_idx=prefix_idx,
    )

    return data


def merge_data(data_1, data_2):
    def _sort_bunch_by_author(data):
        order = np.argsort(data.authors)
        return Bunch(
            X=data.X[order],
            y=data.y[order],
            authors=data.authors[order],
            scores=data.scores[order],
            y_periods=data.y_periods.T[order].T,
            scores_periods=data.scores_periods.T[order].T,
            covariate_names=data.covariate_names,
            subreddit_names=data.subreddit_names,
            subreddits=data.subreddits.T[order].T,
            prefix_idx=data.prefix_idx
        )

    assert sorted(data_1.authors) == sorted(data_2.authors), 'different authors'
    data_1 = _sort_bunch_by_author(data_1)
    data_2 = _sort_bunch_by_author(data_2)
    return Bunch(
        X=np.hstack([data_1.X, data_2.X]),
        y=data_1.y,
        authors=data_1.authors,
        scores=data_1.scores,
        y_periods=data_1.y_periods,
        scores_periods=data_1.scores_periods,
        covariate_names=data_1.covariate_names,
        subreddit_names=data_1.subreddit_names,
        subreddits=data_1.subreddits,
        prefix_idx=data_1.prefix_idx
    )


if __name__ == '__main__':
    print('EN DATA!')
    path = '../datasets/old_features/activity_dataset'
    data_old = load_dataset(path, n_classes=5, filter_abandoned_activity=False)
    path = '../datasets/new_features/activity_dataset'
    data_new = load_dataset(path, n_classes=5, filter_abandoned_activity=False)
    merge_data(data_old, data_new)
