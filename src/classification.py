from typing import OrderedDict
from sklearn.model_selection import cross_val_predict
from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn import clone
import quapy.functional as F


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


if __name__ == '__main__':
    from data import load_dataset
    from sklearn.model_selection import cross_val_score, cross_val_predict
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
    warnings.filterwarnings("ignore", category=ConvergenceWarning, module='sklearn')

    
    for path in [
        '../datasets/datasets_periods/activity_dataset',
        '../datasets/datasets_periods/toxicity_dataset',
        '../datasets/datasets_periods/diversity_dataset'
    ]:

        lr_scores, block_scores = [], []
        for period in range(7):
            print('testing for period', period)
            cov_names, covariates, labels, subreddits_names, subreddits = load_dataset(path, with_periods=True, return_period=period)

            cls = LogisticRegression()
            lr_score = cross_val_score(cls, covariates, labels, n_jobs=-1, scoring='accuracy')
            lr_scores.append(np.mean(lr_score))

            cls = BlockEnsembleClassifier(cls, cov_names)
            block_score = cross_val_score(cls, covariates, labels, n_jobs=-1, scoring='accuracy')
            block_scores.append(np.mean(block_score))

        print(path)
        print(f'LR score = {np.mean(lr_scores):.4f}')
        print(f'Block score = {np.mean(block_scores):.4f}')

