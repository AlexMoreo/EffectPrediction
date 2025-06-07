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

FEATURE_GROUP_PREFIXES = [
    'ACTIVITY', 'EMBEDDINGS', 'EMOTIONS', 'LIWC', 'RELATIONAL', 'SENTIMENT', 'SOC_PSY', 'TOXICITY', 'WRITING_STYLE'
]

FEATURE_SUBGROUP_PREFIXES_depr = ['ACTIVITY--COMMENTS',
 'ACTIVITY--MENTIONS_LINKS',
 'ACTIVITY--SUBMISSIONS',
 'ACTIVITY--TIME',
 'ACTIVITY--TRENDS',
 'EMBEDDINGS--HIDDEN',
 'EMOTIONS--EIL',
 'EMOTIONS--EMOSCORES',
 'EMOTIONS--VAD',
 'LIWC--AFFECT',
 'LIWC--BIO',
 'LIWC--COGNITION',
 'LIWC--CONVERSATION',
 'LIWC--CULTURE',
 'LIWC--DRIVES',
 'LIWC--LIFESTYLE',
 'LIWC--LINGUISTIC',
 'LIWC--MOTIVATION',
 'LIWC--PUNCTUATION',
 'LIWC--SOCIAL',
 'LIWC--SPATIAL',
 'LIWC--SUMMARY',
 'LIWC--TEMPORAL',
 'RELATIONAL--AUTHORS',
 'RELATIONAL--ENGAGEMENT',
 'RELATIONAL--SUBR',
 'RELATIONAL--THREADS',
 'SENTIMENT--CPD',
 'SENTIMENT--NEG',
 'SENTIMENT--NEG-EMOJI',
 'SENTIMENT--NEU',
 'SENTIMENT--NEU-EMOJI',
 'SENTIMENT--POS',
 'SENTIMENT--POS-EMOJI',
 'SENTIMENT--SENT-EMOJI',
 'SOC_PSY--AUTHORITY',
 'SOC_PSY--DEMOGRAPHIC',
 'SOC_PSY--FAIRNESS',
 'SOC_PSY--GENERAL',
 'SOC_PSY--HARM',
 'SOC_PSY--INGROUP',
 'SOC_PSY--OCEAN',
 'SOC_PSY--PURITY',
 'TOXICITY--ID-ATTACK',
 'TOXICITY--INSULT',
 'TOXICITY--OBSCENE',
 'TOXICITY--SEVERE-TOXICITY',
 'TOXICITY--THREAT',
 'TOXICITY--TOXICITY',
 'WRITING_STYLE--ADPOSITIONS',
 'WRITING_STYLE--ADV',
 'WRITING_STYLE--ADVERBS',
 'WRITING_STYLE--ARTICLES',
 'WRITING_STYLE--AUX',
 'WRITING_STYLE--CONG',
 'WRITING_STYLE--DET',
 'WRITING_STYLE--FLESCH-KINKAID',
 'WRITING_STYLE--INTJ',
 'WRITING_STYLE--IRONY',
 'WRITING_STYLE--NER',
 'WRITING_STYLE--NOUNS',
 'WRITING_STYLE--NOVELTY',
 'WRITING_STYLE--PRON',
 'WRITING_STYLE--PRONOUNS',
 'WRITING_STYLE--PROPN',
 'WRITING_STYLE--SCONJ',
 'WRITING_STYLE--SMOG',
 'WRITING_STYLE--SPELL_ERRORS',
 'WRITING_STYLE--STOPWORDS',
 'WRITING_STYLE--SYM',
 'WRITING_STYLE--VERB',
 'WRITING_STYLE--VERBS']

FEATURE_SUBGROUP_PREFIXES = ['ACTIVITY--COMMENTS',
     'ACTIVITY--MENTIONS_LINKS',
     'ACTIVITY--SUBMISSIONS',
     'ACTIVITY--TIME',
     'ACTIVITY--TRENDS',
     'EMOTIONS--EIL',
     'EMOTIONS--EMOSCORES',
     'EMOTIONS--VAD',
     'LIWC--AFFECT',
     'LIWC--BIO',
     'LIWC--COGNITION',
     'LIWC--CONVERSATION',
     'LIWC--CULTURE',
     'LIWC--DRIVES',
     'LIWC--LIFESTYLE',
     'LIWC--LINGUISTIC',
     'LIWC--MOTIVATION',
     'LIWC--PUNCTUATION',
     'LIWC--SOCIAL',
     'LIWC--SPATIAL',
     'LIWC--SUMMARY',
     'LIWC--TEMPORAL',
     'RELATIONAL--AUTHORS',
     'RELATIONAL--ENGAGEMENT',
     'RELATIONAL--SUBR',
     'RELATIONAL--THREADS',
     'SENTIMENT--CPD',
     'SENTIMENT--NEG',
     'SENTIMENT--NEG-EMOJI',
     'SENTIMENT--NEU',
     'SENTIMENT--NEU-EMOJI',
     'SENTIMENT--POS',
     'SENTIMENT--POS-EMOJI',
     'SENTIMENT--SENT-EMOJI',
     'SOC_PSY--AUTHORITY',
     'SOC_PSY--DEMOGRAPHIC',
     'SOC_PSY--FAIRNESS',
     'SOC_PSY--GENERAL',
     'SOC_PSY--HARM',
     'SOC_PSY--INGROUP',
     'SOC_PSY--OCEAN',
     'SOC_PSY--PURITY',
     'TOXICITY--ID-ATTACK',
     'TOXICITY--INSULT',
     'TOXICITY--OBSCENE',
     'TOXICITY--SEVERE-TOXICITY',
     'TOXICITY--THREAT',
     'TOXICITY--TOXICITY',
     'WRITING_STYLE--ADPOSITIONS',
     'WRITING_STYLE--ADV',
     'WRITING_STYLE--ARTICLES',
     'WRITING_STYLE--AUX',
     'WRITING_STYLE--CONG',
     'WRITING_STYLE--DET',
     'WRITING_STYLE--FLESCH-KINKAID',
     'WRITING_STYLE--INTJ',
     'WRITING_STYLE--IRONY',
     'WRITING_STYLE--NER',
     'WRITING_STYLE--NOUNS',
     'WRITING_STYLE--NOVELTY',
     'WRITING_STYLE--PRON',
     'WRITING_STYLE--PROPN',
     'WRITING_STYLE--SCONJ',
     'WRITING_STYLE--SMOG',
     'WRITING_STYLE--SPELL_ERRORS',
     'WRITING_STYLE--STOPWORDS',
     'WRITING_STYLE--SYM',
     'WRITING_STYLE--VERB']

def extract_prefixes(features, covariate_names, level=0):
    # FEATGROUP__FEATSUBGROUP__FEATID
    # level=0  -> list of distinct FEATGROUP
    # level=1  -> list of distinct FEATGROUP--FEATSUBGROUP
    # level=2  -> list of distinct FEATGROUP--FEATSUBGROUP--FEATID
    assert level in [0,1,2], 'unexpected level; use 0 for group, 1 for subgroup, 2 for featID'
    parts = set()
    for feat in features:
        split_parts = feat.split('--')
        assert len(split_parts) == 3, f'unexpected covariate name "{feat}"'
        part_route = '--'.join(split_parts[:level+1])
        if level<2:
            # bugfix: the prefixes in level==1 are not unique, e.g., 'WRITING_STYLE--PRON', 'WRITING_STYLE--PRONOUNS'
            part_route+='--'  # this forces the prefix to be compared entirely
        parts.add(part_route)

    prefixes = sorted(parts)

    prefix_idx = OrderedDict()
    for prefix in sorted(prefixes):
        prefix_key = prefix
        if prefix_key.endswith('--'):
            # bugfix: removes the additional '--' if any
            prefix_key = prefix_key[:-2]
        prefix_idx[prefix_key] = np.char.startswith(covariate_names, prefix)

    return prefix_idx



def load_dataset(path,
                 n_classes,
                 filter_out_multiple_subreddits=False,
                 filter_abandoned_activity=False,
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

    n_covariates = 753
    n_subreddits = len(SUBREDDIT_NAMES)
    assert len(df.columns) == n_covariates+n_subreddits, 'unexpected number of columns'

    assert all(df.columns.values[-n_subreddits:] == SUBREDDIT_NAMES), 'unexpected subreddit names'
    subreddits = df.values[:, -n_subreddits:].astype(bool).T
    covariates = df.iloc[:, :-n_subreddits].values
    df.columns = df.columns.str.replace(r"^EMBEDDINGS_", "EMBEDDINGS--HIDDEN--DIM", regex=True)  # this group did not follow the convention <father>--<soon>--<featID>
    covariate_names = df.columns.values[:-n_subreddits].astype(str)

    # get the feature blocks ids and prefixes
    if features_blocks == 'all':
        level = 0
    elif isinstance(features_blocks, str):
        level = features_blocks.count('--')
    else: # is a list of feature blocks
        levels = list(set([feat_block.count('--') for feat_block in features_blocks]))
        assert len(levels)==1, 'mix of hierarchies from blocks not implemented'
        level = levels[0]

    prefix_idx = extract_prefixes(covariate_names, covariate_names, level=level)

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
    path = '../datasets/activity_dataset'
    data = load_dataset(path, n_classes=5, filter_abandoned_activity=False)
    print(data.X.shape)
