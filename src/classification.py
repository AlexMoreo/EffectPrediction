from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn import clone


class BlockEnsembleClassifier(BaseEstimator):

    def __init__(self, base_estimator, column_names):
        self.base_estimator = base_estimator
        self.column_names = column_names

    def separate_blocks_idx(self, column_names):
        column_prefixes = [n.split('_')[0] for n in column_names]
        self.column_prefixes = sorted(np.unique(column_prefixes))
        self.prefix_idx = {prefix: np.char.startswith(column_prefixes, prefix) for prefix in self.column_prefixes}
        return self.prefix_idx

    def separate_blocks(self, X):
        blocks = {prefix: X[:,index] for prefix, index in self.prefix_idx.items()}
        return blocks

    def first_tier_predict_proba(self, blocks):
        Ps = []
        for prefix in self.column_prefixes:
            X = blocks[prefix]
            Ps.append(self.learners[prefix].predict_proba(X))
        P = np.hstack(Ps)
        return P

    def fit(self, X, y):
        self.separate_blocks_idx(self.column_names)
        blocks = self.separate_blocks(X)
        self.learners = {
            prefix: clone(self.base_estimator).fit(block, y) for prefix, block in blocks.items()
        }
        P = self.first_tier_predict_proba(blocks)
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
    from sklearn.model_selection import cross_val_score
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

    
    for path in [
        './datasets/datasets_periods/activity_dataset',
        './datasets/datasets_periods/toxicity_dataset',
        './datasets/datasets_periods/diversity_dataset'
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

