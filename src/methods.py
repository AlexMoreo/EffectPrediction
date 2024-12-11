from abc import ABC, abstractmethod
from copy import deepcopy
from typing import List, Iterable
import numpy as np
from quapy.data import LabelledCollection
from quapy.method.base import BaseQuantifier
from quapy.method.non_aggregative import MaximumLikelihoodPrevalenceEstimation as MLPE
from quapy.method.aggregative import CC, PCC, PACC, EMQ, KDEyML
from quapy.method.meta import Ensemble
import warnings

from sklearn.linear_model import LogisticRegression

from classification import BlockEnsembleClassifier
from utils import mmd_pairwise_rbf_blocks

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

    classes = np.unique(labels).sort()
    return LabelledCollection(instances, labels, classes=classes)


class SelectAndQuantify(BaseQuantifier):
    """
    A simple method that simply concatenates all the training sets into a unique
    training set, and then trains a surrogate quantifier on it
    """

    def __init__(self, base_quantifier):
        self.base_quantifier = base_quantifier

    def fit(self, data_list:List[LabelledCollection], select:List[bool]):
        selected = [data for sel, data in zip(select,data_list) if sel==True]
        training = join_subreddits(*selected)
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
        # Xs = [data.X for data in subreddits_data]
        self.mmd = mmd_pairwise_rbf_blocks(Xs, **kwargs)

    def get_selection(self, test_index: int) -> List[bool]:
        mmds_wrt_test = self.mmd[test_index]
        # remove its own comparison, the index test_index
        mmds_wrt_test = [v for i,v in enumerate(mmds_wrt_test) if i!=test_index]
        # select datasets for which the mmd is below the median
        median_val = np.median(mmds_wrt_test)
        return [v < median_val for v in mmds_wrt_test]




def methods(base_classifier, prefix_idx):
    yield 'MLPE', SelectAndQuantify(MLPE()), SelectAllPolicy()
    yield 'CC', SelectAndQuantify(CC(deepcopy(base_classifier))), SelectAllPolicy()
    yield 'PCC', SelectAndQuantify(PCC(deepcopy(base_classifier))), SelectAllPolicy()
    # yield 'bPCC', SelectAndQuantify(PCC(BlockEnsembleClassifier(deepcopy(base_classifier), prefix_idx))), SelectAllPolicy()
    yield 'bPCC-cv', SelectAndQuantify(PCC(BlockEnsembleClassifier(deepcopy(base_classifier), prefix_idx, kfcv=5))), SelectAllPolicy()
    yield 'PACC', SelectAndQuantify(PACC(deepcopy(base_classifier))), SelectAllPolicy()
    yield 'EPCC', EnsembleQuantifier(PCC(deepcopy(base_classifier))), SelectAllPolicy()
    yield 'EbPCC-cv', EnsembleQuantifier(PCC(BlockEnsembleClassifier(deepcopy(base_classifier), prefix_idx, kfcv=5))), SelectAllPolicy()
    yield 'EbPCC-cv-sel', EnsembleQuantifier(PCC(BlockEnsembleClassifier(deepcopy(base_classifier), prefix_idx, kfcv=5))), SelectMedianMMDPolicy()
    # yield 'bPCC-sel', SelectAndQuantify(PCC(BlockEnsembleClassifier(deepcopy(base_classifier), prefix_idx))), SelectMedianMMDPolicy()
    # yield 'bPCC-cv-sel', SelectAndQuantify(PCC(BlockEnsembleClassifier(deepcopy(base_classifier), prefix_idx, kfcv=5))), SelectMedianMMDPolicy()

    from quapy.method.meta import MedianEstimator