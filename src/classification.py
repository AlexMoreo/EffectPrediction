from itertools import product
from typing import OrderedDict

from quapy.data import LabelledCollection
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_predict, GridSearchCV
from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn import clone
import quapy.functional as F

# from data import load_dataset, merge_data


# from statsmodels.miscmodels.ordinal_model import OrderedModel
# from statsmodels.tools import add_constant


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


# class OrderedLogisticRegression(BaseEstimator):
#     """
#     Wrapper of statsmodels' ordered LR
#     """
#
#     def __init__(self):
#         self.olr = None
#
#     def fit(self, X, y):
#         X = np.asarray(X)
#         y = np.asarray(y)
#
#         self.constant_cols_idx = np.isclose(X.var(axis=0), 0)
#         if (self.constant_cols_idx).any():  # if there are constant columns
#             X = X[:, ~self.constant_cols_idx]  # remove constants
#
#         self.olr = OrderedModel(y, X, distr='logit')
#         self.res_log = self.olr.fit(method='lbfgs', disp=False)
#         return self
#
#     def predict(self, X):
#         print('in predict')
#         posteriors = self.predict_proba(X)
#         return np.argmax(posteriors, axis=1)
#
#     def predict_proba(self, X):
#         print('in predict_proba')
#         X = np.asarray(X)
#         if (self.constant_cols_idx).any():  # if there are constant columns
#             X = X[:, ~self.constant_cols_idx]  # remove constants
#         P = self.res_log.model.predict(self.res_log.params, X)
#         return P
#
#
if __name__ == '__main__':
    from os.path import join
    from quapy.method.aggregative import CC, PACC, EMQ, PCC
    from sklearn.ensemble import RandomForestClassifier
    import quapy as qp
    import warnings
    from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, f_classif

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        targets = ['global']  # , 'periods']
        n_classes_list = [3,5]
        # dataset_names = ['diversity', 'toxicity', 'activity']
        dataset_names = ['diversity']
        for dataset_name, n_classes, target in product(dataset_names, n_classes_list, targets):
            print(f'running {dataset_name=} {n_classes}')
            data_old = load_dataset(join('../datasets/old_features', f'{dataset_name}_dataset'), n_classes=n_classes,
                                filter_out_multiple_subreddits=False,
                                filter_abandoned_activity=False)
            data_new = load_dataset(join('../datasets/new_features', f'{dataset_name}_dataset'), n_classes=n_classes,
                                filter_out_multiple_subreddits=False,
                                filter_abandoned_activity=False)
            # data = data_old
            data = merge_data(data_old, data_new)
            print(data.X.shape)

            y = data.y
            X = data.X
            # y = (data.scores > 0).astype(int)
            # print(np.logical_and(y != 1, y != 3))
            # sel = np.logical_and(y!=1, y!=3)
            # sel = y!=1
            # sel = np.logical_and(sel, y!=2)
            # sel = np.logical_and(sel, y != 3)
            # X=X[sel]
            # y=y[sel]
            # y = np.asarray([{0:0, 4:1}[label] for label in y])  # <-- relabeling removing difficult in-between classes
            X = PCA(n_components=100).fit_transform(X)
            # X = SelectKBest(k=100, score_func=f_classif).fit_transform(X, y)
            lc = LabelledCollection(X, y)
            train, test = lc.split_stratified()
            cls = LogisticRegression()
            # optim = GridSearchCV(
            #     estimator=LogisticRegression(),
            #     param_grid={'C':np.logspace(-3,3,7), 'class_weight':['balanced', None]},
            #     n_jobs=-1,
            #     refit=False,
            # ).fit(X,y)
            # print(optim.best_params_)
            # cls = LogisticRegression(**optim.best_params_)
            # cls = LogisticRegression(C=1)
            # cls = BlockEnsembleClassifier(LogisticRegression(), blocks_ids=data.prefix_idx)
            # cls = RandomForestClassifier()
            pacc = PACC(cls, n_jobs=-1)
            pacc.fit(train)
            print(pacc.Pte_cond_estim_)
            # print(f'rank={np.linalg.matrix_rank(pacc.Pte_cond_estim_)}')
            # p_hat = pacc.quantify(test.X)
            # ae = qp.error.ae(test.prevalence(), p_hat)
            # print(f'train prev {qp.functional.strprev(train.prevalence())}')
            # print(f'true prev {qp.functional.strprev(test.prevalence())}')
            # print(f'estim prev {qp.functional.strprev(p_hat)}')
            # print(f'pacc {ae=:.4f}')
            #
            # cc = CC().fit(train)
            # p_hat = cc.quantify(test.X)
            # ae = qp.error.ae(test.prevalence(), p_hat)
            # print(f'CC estim prev {qp.functional.strprev(p_hat)}')
            # print(f'cc {ae=:.4f}')
            #
            # pcc = PCC(cc.classifier)
            # p_hat = pcc.quantify(test.X)
            # ae = qp.error.ae(test.prevalence(), p_hat)
            # print(f'PCC estim prev {qp.functional.strprev(p_hat)}')
            # print(f'pcc {ae=:.4f}')

            # cls = BlockEnsembleClassifier(LogisticRegression(), blocks_ids=data.prefix_idx)
            # cls.fit(*train.Xy)
            # y_hat = cls.predict(test.X)
            # accuracy = (y_hat==test.y).mean()
            # print(f'classifier accuracy = {accuracy:.4f}')
            # if n_classes == 2:
            #     f1 = f1_score(test.y, y_hat)
            #     print(f'classifier f1 = {f1:.4f}')
            # else:
            #     f1_macro = f1_score(test.y, y_hat, average='macro')
            #     f1_micro = f1_score(test.y, y_hat, average='micro')
            #     print(f'classifier Macro f1 = {f1_macro:.4f}')
            #     print(f'classifier Micro f1 = {f1_micro:.4f}')


            cls = LogisticRegression()
            cls.fit(*train.Xy)
            y_hat = cls.predict(test.X)
            accuracy = (y_hat==test.y).mean()
            print(f'classifier accuracy = {accuracy:.4f}')
            if n_classes == 2:
                f1 = f1_score(test.y, y_hat)
                print(f'classifier f1 = {f1:.4f}')
            else:
                f1_macro = f1_score(test.y, y_hat, average='macro')
                f1_micro = f1_score(test.y, y_hat, average='micro')
                print(f'classifier Macro f1 = {f1_macro:.4f}')
                print(f'classifier Micro f1 = {f1_micro:.4f}')


            # qp.environ["SAMPLE_SIZE"]=200
            # mae = qp.evaluation.evaluate(pacc, qp.protocol.UPP(test, repeats=100), error_metric='mae')
            # print(f'PACC {mae=:.4f}')


            # mae = qp.evaluation.evaluate(cc, qp.protocol.UPP(test, repeats=100), error_metric='mae')
            # print(f'CC {mae=:.4f}')

