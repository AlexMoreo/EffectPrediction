from typing import OrderedDict
from sklearn.model_selection import cross_val_predict
from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn import clone
import quapy.functional as F
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.tools import add_constant


class BlockEnsembleClassifier(BaseEstimator):
    """
    An ensemble that trains a 1st tier of feature-block-specific classifiers, that
    takes a final decision based on a meta-classifier that is trained on the outputs
    of the 1st tier.

    :param base_estimator: the estimators to use in the 1st tier and meta classifier
    :param blocks_ids: a dictionary in which the keys are the feature-block prefix
        and the values are the column indices of the corresponding features
    """

    def __init__(self, base_estimator: BaseEstimator, blocks_ids: OrderedDict, kfcv=0):
        self.base_estimator = base_estimator
        self.blocks_ids = blocks_ids
        self.kfcv = kfcv

    def separate_blocks(self, X):
        blocks = {prefix: X[:,index] for prefix, index in self.blocks_ids.items()}
        return blocks

    def first_tier_predict_proba(self, blocks):
        Ps = []
        for prefix in self.blocks_ids.keys():
            X = blocks[prefix]
            Ps.append(self.learners[prefix].predict_proba(X))
        P = np.hstack(Ps)
        return P

    def fit(self, X, y):
        blocks = self.separate_blocks(X)
        train_counts = F.counts_from_labels(y, classes=np.unique(y))
        if self.kfcv==0 or any(train_counts < self.kfcv):
            self.learners = {
                prefix: clone(self.base_estimator).fit(block, y) for prefix, block in blocks.items()
            }
            P = self.first_tier_predict_proba(blocks)
        else:
            self.learners = {}
            Ps = []
            for prefix, block in blocks.items():
                learner_block = clone(self.base_estimator)
                block_P = cross_val_predict(learner_block, block, y, cv=self.kfcv, n_jobs=self.kfcv, method='predict_proba')
                Ps.append(block_P)
                learner_block.fit(block, y)
                self.learners[prefix] = learner_block
            P = np.hstack(Ps)

        self.meta = clone(self.base_estimator)
        self.meta.fit(P, y)
        self.classes_ = self.meta.classes_
        return self

    def predict(self, X):
        blocks = self.separate_blocks(X)
        P = self.first_tier_predict_proba(blocks)
        return self.meta.predict(P)

    def predict_proba(self, X):
        blocks = self.separate_blocks(X)
        P = self.first_tier_predict_proba(blocks)
        return self.meta.predict_proba(P)


class OrderedLogisticRegression(BaseEstimator):
    """
    Wrapper of statsmodels' ordered LR
    """

    def __init__(self):
        self.olr = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        self.constant_cols_idx = np.isclose(X.var(axis=0), 0)
        if (self.constant_cols_idx).any():  # if there are constant columns
            X = X[:, ~self.constant_cols_idx]  # remove constants

        self.olr = OrderedModel(y, X, distr='logit')
        self.res_log = self.olr.fit(method='lbfgs', disp=False)
        return self

    def predict(self, X):
        print('in predict')
        posteriors = self.predict_proba(X)
        return np.argmax(posteriors, axis=1)

    def predict_proba(self, X):
        print('in predict_proba')
        X = np.asarray(X)
        if (self.constant_cols_idx).any():  # if there are constant columns
            X = X[:, ~self.constant_cols_idx]  # remove constants
        P = self.res_log.model.predict(self.res_log.params, X)
        return P


if __name__ == '__main__':
    import quapy as qp
    data = qp.datasets.fetch_UCIMulticlassDataset(qp.datasets.UCI_MULTICLASS_DATASETS[2])
    train, test = data.train_test
    olr = OrderedLogisticRegression()
    olr.fit(*train.Xy)
    pred_prob = olr.predict_proba(test.X)
    print(pred_prob)


