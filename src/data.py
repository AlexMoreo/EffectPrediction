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
    'activity': [-0.4, 0.4],
    'diversity': [-0.3, 0.3],
    'toxicity': [-0.4, 0.4],
}

THRESHOLD_5_CLASSES={
    'activity': [-0.55, -0.2, 0.2, 0.55],
    'diversity': [-0.55, -0.2, 0.2, 0.55],
    'toxicity': [-0.55, -0.2, 0.2, 0.55],
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

FEATURE_PREFIXES = [
    'ACTIVITY',
    'DEMOGRAPHIC',
    'EMBEDDINGS',
    'EMOTIONS',
    'LIWC',
    'RELATIONAL',
    'SENTIMENT',
    'SOC_PSY',
    'TOXICITY',
    'WRITING_STYLE'
]

def extract_prefixes(features):
    import re
    prefixes = set()
    for feat in features:
        match = re.match(r'^([A-Z_]+)_', feat)
        if match:
            prefixes.add(match.group(1))
    return sorted(list(prefixes))


def load_dataset(path, n_classes, filter_out_multiple_subreddits=False, filter_abandoned_activity=False,
                 features_blocks='all'):

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

    # make threshold_values bounded by inf
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

    n_covariates = 765
    n_subreddits = len(SUBREDDIT_NAMES)
    assert len(df.columns) == n_covariates+n_subreddits, 'unexpected number of columns'

    assert all(df.columns.values[-n_subreddits:] == SUBREDDIT_NAMES), 'unexpected subreddit names'
    subreddits = df.values[:, -n_subreddits:].astype(bool).T
    covariates = df.iloc[:, :-n_subreddits].values
    covariate_names = df.columns.values[:-n_subreddits].astype(str)

    # get the feature blocks ids and prefixes
    column_prefixes = extract_prefixes(covariate_names)
    assert column_prefixes == FEATURE_PREFIXES, 'unexpected feature prefixes'
    prefix_idx = OrderedDict()
    for prefix in sorted(column_prefixes):
        prefix_idx[prefix] = np.char.startswith(covariate_names, prefix)

    # filter by feature_blocks
    covariate_blocks = []
    if features_blocks != 'all':
        print(f'selecting {features_blocks} from {list(prefix_idx.keys())}')
        if isinstance(features_blocks, str):
            features_blocks=[features_blocks]
        for feature_block in features_blocks:
            covariate_blocks.append(covariates[:,prefix_idx[feature_block]])
        covariates = np.hstack(covariate_blocks)

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


if __name__ == '__main__':
    path = '../datasets/merged_features/activity_dataset'
    data = load_dataset(path, n_classes=5, filter_abandoned_activity=False, features_blocks='all')
    print(data.X.shape)
