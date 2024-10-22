import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler



def load_dataset(path):
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

    print(df.columns)

    covariates = np.hstack((df.values[:, :3], df.values[:, 4:624]))
    labels = df.values[:,624].astype(int)
    subbreddits = df.values[:, 625:]

    scaler = StandardScaler()
    covariates = scaler.fit_transform(covariates)

    return covariates, labels, subbreddits_names, subbreddits
