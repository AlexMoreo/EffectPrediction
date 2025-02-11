from abc import ABC, abstractmethod
from copy import deepcopy
from typing import List, Iterable
import numpy as np
import quapy.functional
from quapy.data import LabelledCollection
from quapy.method.base import BaseQuantifier
from quapy.method.non_aggregative import MaximumLikelihoodPrevalenceEstimation as MLPE
from quapy.method.aggregative import CC, PCC, PACC, EMQ, KDEyML
from quapy.method.meta import Ensemble
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from statsmodels.miscmodels.ordinal_model import OrderedModel
# from mord import LogisticIT
import warnings

from sklearn.linear_model import LogisticRegression

from classification import BlockEnsembleClassifier, OrderedLogisticRegression
from classifier_calibrators import IsotonicCalibration, LasCalCalibration, TransCalCalibrator, HeadToTailCalibrator, \
    CpcsCalibrator
from utils import mmd_pairwise_rbf_blocks, mmd_pairwise_rbf_blocks_pval

warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')


def join_subreddits(*args: Iterable['LabelledCollection']):
    """
    Returns a new :class:`LabelledCollection` as the union of the collections given in input.

    :param args: instances of :class:`LabelledCollection`
    :return: a :class:`LabelledCollection` representing the union of both collections
    """

    args = [lc for lc in args if lc is not None]
    assert len(args) > 0, 'empty list is not allowed for mix'

    instances = np.concatenate([lc.instances for lc in args])
    labels = np.concatenate([lc.labels for lc in args])

    classes = sorted(np.unique(labels))
    return LabelledCollection(instances, labels, classes=classes)


class SelectAndQuantify(BaseQuantifier):
    """
    A simple method that simply concatenates all the training sets into a unique
    training set, and then trains a surrogate quantifier on it
    """

    def __init__(self, base_quantifier):
        self.base_quantifier = base_quantifier

    def fit(self, data_list:List[LabelledCollection], select:List[bool]):
        n_classes = max(d.n_classes for d in data_list)
        selected = [data for sel, data in zip(select,data_list) if sel==True]
        training = join_subreddits(*selected)
        if training.n_classes < n_classes:
            print('the selection does not contain positive examples for all classes... backing up to a concatenation')
            training = join_subreddits(*data_list)
        self.base_quantifier.fit(training)
        return self

    def quantify(self, instances):
        return self.base_quantifier.quantify(instances)


class EnsembleQuantifier(BaseQuantifier):
    def __init__(self, base_quantifier):
        self.ensemble = []
        self.base_quantifier = base_quantifier

    def fit(self, data_list: List[LabelledCollection], select: List[bool]):
        sel_data = [data for sel, data in zip(select, data_list)]
        for data in sel_data:
            if np.prod(data.counts())>0: # discard datasets with empty classes
                q = deepcopy(self.base_quantifier)
                try:
                    q.fit(data)
                    self.ensemble.append(q)
                except Exception as e:
                    print(f'skipping one member due to: {e}')
        return self

    def quantify(self, instances):
        predictions = [q.quantify(instances) for q in self.ensemble]
        predictions = np.vstack(predictions)
        return predictions.mean(axis=0)


class SelectionPolicy(ABC):
    def feed(self, Xs:List[np.ndarray], **kwargs):
        self.Xs = Xs

    @abstractmethod
    def get_selection(self, test_index:int)->List[bool]:
        ...

class SelectAllPolicy(SelectionPolicy):
    def get_selection(self, test_index: int) -> List[bool]:
        n = len(self.Xs)
        return [True] * (n - 1)


class SelectMedianMMDPolicy(SelectionPolicy):
    def feed(self, Xs:List[np.ndarray], **kwargs):
        self.mmd = mmd_pairwise_rbf_blocks(Xs, **kwargs)

    def get_selection(self, test_index: int) -> List[bool]:
        mmds_wrt_test = self.mmd[test_index]
        # remove its own comparison, the index test_index
        mmds_wrt_test = [v for i,v in enumerate(mmds_wrt_test) if i!=test_index]
        # select datasets for which the mmd is below the median
        median_val = np.median(mmds_wrt_test)
        sel = [v < median_val for v in mmds_wrt_test]
        return sel


class SelectStatSimilarMMDPolicy(SelectionPolicy):
    def feed(self, Xs:List[np.ndarray], conf_val=0.01, **kwargs):
        self.conf_val = conf_val
        mmd, self.pvals = mmd_pairwise_rbf_blocks_pval(Xs, **kwargs)
        self.backup =SelectMedianMMDPolicy()
        self.backup.mmd = mmd

    def get_selection(self, test_index: int) -> List[bool]:
        pvals_test = self.pvals[test_index]
        # remove its own comparison, the index test_index
        pvals_test = [v for i,v in enumerate(pvals_test) if i!=test_index]
        # select datasets for which the pval is above the conf value
        sel = [pval > self.conf_val for pval in pvals_test]
        # if no index has been chosen, then select by median
        if not any(sel):
            print(f'no set selected for {test_index=}; backing up to median policy')
            sel = self.backup.get_selection(test_index)
        return sel


class FromKnownPart(ABC):
    def join_quantify(self, Xtgt, ytgt, X):
        n_known = len(ytgt)
        n_unknown = X.shape[0]
        n = n_known+n_unknown
        priors_knw = quapy.functional.prevalence_from_labels(ytgt, classes=self.base_classifier.classes_)
        priors_unk = self.predict_proba(X).mean(axis=0)
        priors = priors_knw*(n_known/n) + priors_unk*(n_unknown/n)
        return priors


class PCCrecalib(FromKnownPart):

    def __init__(self, base_classifier):
        self.base_classifier = base_classifier

    def fit(self, Xsrc, ysrc, Xtgt, ytgt):
        self.base_classifier.fit(Xsrc, ysrc)
        self.calibrated = CalibratedClassifierCV(estimator=FrozenEstimator(self.base_classifier))
        self.calibrated.fit(Xtgt, ytgt)
        # Ptgt = self.base_classifier.predict_proba(Xtgt)
        # self.calibrator = IsotonicCalibration().fit(Ptgt, ytgt)
        return self

    def predict_proba(self, X):
        # uncal = self.base_classifier.predict_proba(X)
        # calib = self.calibrator.calibrate(uncal)
        calib = self.calibrated.predict_proba(X)
        return calib



class PCCnaive(FromKnownPart):

    def __init__(self, base_classifier):
        self.base_classifier = base_classifier

    def fit(self, Xsrc, ysrc, Xtgt, ytgt):
        X = np.vstack([Xsrc, Xtgt])
        y = np.concatenate([ysrc, ytgt])
        self.base_classifier.fit(X, y)
        return self

    def predict_proba(self, X):
        calib = self.base_classifier.predict_proba(X)
        return calib


class PCClascal(FromKnownPart):

    def __init__(self, base_classifier):
        self.base_classifier = base_classifier

    def fit(self, Xsrc, ysrc, Xtgt, ytgt):
        self.calibrator = LasCalCalibration(prob2logits=True)
        self.base_classifier.fit(Xsrc, ysrc)
        self.Ptgt = self.base_classifier.predict_proba(Xtgt)
        self.ytgt = ytgt
        return self

    def predict_proba(self, X):
        P = self.base_classifier.predict_proba(X)
        Pcal = self.calibrator.calibrate(self.Ptgt, self.ytgt, P)
        return Pcal


class PCCTransCal(FromKnownPart):
    def __init__(self, base_classifier):
        self.base_classifier = base_classifier

    def fit(self, Xsrc, ysrc, Xtgt, ytgt):
        self.calibrator = TransCalCalibrator(prob2logits=True)
        self.base_classifier.fit(Xsrc, ysrc)
        self.Xsrc = Xsrc
        self.ysrc = ysrc
        self.Xtgt = Xtgt
        self.ytgt = ytgt
        self.Ptgt = self.base_classifier.predict_proba(Xtgt)
        return self

    def predict_proba(self, X):
        P = self.base_classifier.predict_proba(X)
        Pcal = self.calibrator.calibrate(self.Xsrc, self.ysrc, self.Xtgt, self.Ptgt, self.ytgt, X, P)
        return Pcal


class PCC_Cpcs(FromKnownPart):
    def __init__(self, base_classifier):
        self.base_classifier = base_classifier

    def fit(self, Xsrc, ysrc, Xtgt, ytgt):
        self.calibrator = CpcsCalibrator(prob2logits=True)
        self.base_classifier.fit(Xsrc, ysrc)
        self.Xsrc = Xsrc
        self.ysrc = ysrc
        self.Xtgt = Xtgt
        self.ytgt = ytgt
        self.Ptgt = self.base_classifier.predict_proba(Xtgt)
        return self

    def predict_proba(self, X):
        P = self.base_classifier.predict_proba(X)
        Pcal = self.calibrator.calibrate(self.Xsrc, self.ysrc, self.Xtgt, self.Ptgt, self.ytgt, X, P)
        return Pcal


class PCCHead2Tail(FromKnownPart):
    def __init__(self, base_classifier):
        self.base_classifier = base_classifier

    def fit(self, Xsrc, ysrc, Xtgt, ytgt):
        self.calibrator = HeadToTailCalibrator(prob2logits=True)
        self.base_classifier.fit(Xsrc, ysrc)
        self.Xsrc = Xsrc
        self.ysrc = ysrc
        self.Xtgt = Xtgt
        self.ytgt = ytgt
        self.Ptgt = self.base_classifier.predict_proba(Xtgt)
        return self

    def predict_proba(self, X):
        P = self.base_classifier.predict_proba(X)
        Pcal = self.calibrator.calibrate(self.Xsrc, self.ysrc, self.Xtgt, self.Ptgt, self.ytgt, X, P)
        return Pcal

class CalibrateExtrapolate:
    pass


def methods(base_classifier, prefix_idx):
    yield 'MLPE', SelectAndQuantify(MLPE()), SelectAllPolicy()
    yield 'CC', SelectAndQuantify(CC(deepcopy(base_classifier))), SelectAllPolicy()
    yield 'PCC', SelectAndQuantify(PCC(deepcopy(base_classifier))), SelectAllPolicy()
    # yield 'o-PCC', SelectAndQuantify(PCC(LogisticIT())), SelectAllPolicy()
    # yield 'PCC-sel', SelectAndQuantify(PCC(deepcopy(base_classifier))), SelectMedianMMDPolicy()
    # yield 'PCC-psel', SelectAndQuantify(PCC(deepcopy(base_classifier))), SelectStatSimilarMMDPolicy()
    # yield 'bPCC', SelectAndQuantify(PCC(BlockEnsembleClassifier(deepcopy(base_classifier), prefix_idx))), SelectAllPolicy()
    yield 'bPCC-cv', SelectAndQuantify(PCC(BlockEnsembleClassifier(deepcopy(base_classifier), prefix_idx, kfcv=5))), SelectAllPolicy()
    # yield 'PACC', SelectAndQuantify(PACC(deepcopy(base_classifier))), SelectAllPolicy()
    # yield 'EMQ', SelectAndQuantify(EMQ(deepcopy(base_classifier))), SelectAllPolicy()
    # yield 'KDEy', SelectAndQuantify(KDEyML(deepcopy(base_classifier))), SelectAllPolicy()
    # yield 'EPCC', EnsembleQuantifier(PCC(deepcopy(base_classifier))), SelectAllPolicy()
    # yield 'EbPCC-cv', EnsembleQuantifier(PCC(BlockEnsembleClassifier(deepcopy(base_classifier), prefix_idx, kfcv=5))), SelectAllPolicy()
    # yield 'EbPCC-cv-sel', EnsembleQuantifier(PCC(BlockEnsembleClassifier(deepcopy(base_classifier), prefix_idx, kfcv=5))), SelectMedianMMDPolicy()
    # yield 'bPCC-sel', SelectAndQuantify(PCC(BlockEnsembleClassifier(deepcopy(base_classifier), prefix_idx))), SelectMedianMMDPolicy()
    # yield 'bPCC-cv-sel', SelectAndQuantify(PCC(BlockEnsembleClassifier(deepcopy(base_classifier), prefix_idx, kfcv=5))), SelectMedianMMDPolicy()

    from quapy.method.meta import MedianEstimator

def new_methods(base_classifier):
    yield 'PCC-recalib', PCCrecalib(base_classifier=base_classifier)
    yield 'PCC-norcalib', PCCnaive(base_classifier=base_classifier)
    yield 'PCC-lascal',  PCClascal(base_classifier=base_classifier)
    yield 'PCC-transcal', PCCTransCal(base_classifier=base_classifier)
    yield 'PCC-cpcs', PCC_Cpcs(base_classifier=base_classifier)
    # yield 'PCC-head2tail', PCCHead2Tail(base_classifier=base_classifier)