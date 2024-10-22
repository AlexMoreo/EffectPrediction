import numpy as np
import quapy as qp
from quapy.data import LabelledCollection
import quapy.functional as F
from quapy.method.aggregative import PACC, EMQ, CC, KDEyML
from quapy.protocol import UPP
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from data import load_dataset

path = '../datasets/toxicity_dataset'

covariates, labels, subreddits_names, subreddits = qp.util.pickled_resource(path + '.pkl', load_dataset, path)

# n_subreddits = subreddits.shape[1]
# for subbreddit_id in range(n_subreddits):
#     sel = subreddits[:,subbreddit_id].astype(bool)
#     X, y = covariates[sel], labels[sel]
#     collection = LabelledCollection(X, y)
#
#     subreddit_name = subreddits_names[subbreddit_id]
#
#     print(f'{subreddit_name} has {len(collection)} instances, prevalence={F.strprev(collection.prevalence())}' )

# sel = np.logical_or(labels==0, labels==2)
# covariates = covariates[sel]
# labels = labels[sel]

data = LabelledCollection(covariates, labels)
train, test = data.split_stratified(0.5, random_state=0)

print(f"data length = {len(data)}")

qp.environ["SAMPLE_SIZE"] = 1000


# quant = CC(LogisticRegression())
# quant = EMQ(LogisticRegression())
# quant = KDEyML(LogisticRegression())
quant = PACC(LogisticRegression())


train, val = train.split_stratified(train_prop=0.6, random_state=0)

quant = qp.model_selection.GridSearchQ(
    model= quant,
    param_grid={'classifier__C': np.logspace(-4,4,7), 'classifier__class_weight': ['balanced', None]}, # 'bandwidth': np.linspace(0.01, 0.2, 20)},
    protocol=UPP(val),
    refit=True,
    n_jobs=-1,
    verbose=True
).fit(train)

# quant.fit(train)

upp = UPP(test, repeats=1000)

report = qp.evaluation.evaluation_report(quant, protocol=upp, error_metrics=['mae', 'mrae'])

import pandas as pd
pd.set_option('display.max_columns', 100)

print(report)

print(report.mean(numeric_only=True))

