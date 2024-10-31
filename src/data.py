import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler



def load_dataset(path, with_periods=False, return_period=0):
    df = pd.read_json(path)

    subbreddits_names = [
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

    # print(df.columns)


    idx = list(range(624))
    covariate_idx = np.asarray(idx[:3]+idx[4:])
    covariates = df.values[:, covariate_idx]
    covariate_names = df.columns.values[covariate_idx]

    if with_periods:
        subbreddits_start_idx = -len(subbreddits_names)
        assert list(df.columns[subbreddits_start_idx:]) == subbreddits_names, 'the subbreddit names do not coincide'
        subbreddits = df.values[:, subbreddits_start_idx:]

        label_scores = df.values[:, 624:subbreddits_start_idx]

        assert return_period<7, 'period out of range'
        label_scores = label_scores[:, return_period]

        if 'activity_dataset' in path:
            bin_values = [-np.inf, -0.4, 0.4, np.inf]
        elif 'toxicity_dataset' in path:
            bin_values = [-np.inf, -0.3, 0.3, np.inf]
        elif 'diversity_dataset' in path:
            bin_values = [-np.inf, -0.3, 0.3, np.inf]
        else:
            raise ValueError('bin values unknown for this dataset')

        new_labels = [0,1,2]
        labels = pd.cut(label_scores, bins=bin_values, labels=new_labels, right=False).to_numpy()

    else:
        labels = df.values[:,624].astype(int)

        assert list(df.columns[625:])==subbreddits_names, 'the subbreddit names do not coincide'
        subbreddits = df.values[:, 625:]

    return covariate_names, covariates, labels, subbreddits_names, subbreddits
