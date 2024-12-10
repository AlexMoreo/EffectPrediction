import numpy as np
import quapy as qp
from quapy.data import LabelledCollection, Dataset
import quapy.functional as F
from quapy.method.aggregative import PACC, EMQ, CC, KDEyML, PCC
from quapy.method.non_aggregative import MaximumLikelihoodPrevalenceEstimation
from quapy.protocol import UPP
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from tqdm import tqdm
from classification import BlockEnsembleClassifier

from _depr.derp_data import load_dataset
import warnings

from utils import mmd_rbf_blocks

warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

# path = '../datasets/datasets_periods/activity_dataset'
# path = '../datasets/datasets_periods/toxicity_dataset'
path = '../datasets/depr_/datasets_periods/diversity_dataset'


def fit_and_test(quantifier, train, test, error_fn):
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
    quantifier.fit(train)
    pred_prev = quantifier.quantify(test.X)
    err = error_fn(test.prevalence(), pred_prev)
    return err, pred_prev


def experiment(i):
    sel = subreddits[:, i].astype(bool)
    classes = sorted(np.unique(labels))
    train = LabelledCollection(covariates[~sel], labels[~sel], classes=classes)
    test = LabelledCollection(covariates[sel], labels[sel], classes=classes)

    zscorer = StandardScaler()
    train.instances = zscorer.fit_transform(train.instances)
    test.instances = zscorer.transform(test.instances)

    qp.environ["SAMPLE_SIZE"] = len(test)

    # quant = EMQ(CalibratedClassifierCV(LogisticRegression()), n_jobs=-1)
    # quant = PCC(CalibratedClassifierCV(LogisticRegression(), n_jobs=-1))

    mlpe = MaximumLikelihoodPrevalenceEstimation()
    cc = CC(LogisticRegression())
    # pcc = PCC(LogisticRegression())
    # pcc = PCC(BlockEnsembleClassifier(LogisticRegression(), column_names=cov_names))
    # quant_pps = PACC(LogisticRegression())
    # quant_pps = PACC(BlockEnsembleClassifier(LogisticRegression(), column_names=cov_names))

    trains = []
    for j in range(n_subreddits):
        if j==i: continue
        sel = subreddits[:, j].astype(bool)
        covs_sel = covariates[sel]
        covs_sel = zscorer.transform(covs_sel)
        trains.append(LabelledCollection(covs_sel, labels[sel], classes=classes))

    # mmds = []
    # for train_j in trains:
    #     mmd = mmd_rbf_blocks(train_j.X, test.X, blocks_idx, gammas=1.)
    #     mmds.append(mmd)

    # choose only the best
    # min_mmd_idx = np.argmin(mmds)
    # train_sel = trains[min_mmd_idx]

    # choose all below the median mmd
    # median_mmd = np.median(mmds)
    # sel_trains = [train_j for mmd_j, train_j in zip(mmds,trains) if mmd_j < median_mmd]
    # train_sel = LabelledCollection.join(*sel_trains)

    # pcc_mmd = PCC(BlockEnsembleClassifier(LogisticRegression(), column_names=cov_names))

    err_noshift, _ = fit_and_test(mlpe, train, test, error_fn=qp.error.ae)
    err_cc=-1
    # err_cc, _ = fit_and_test(cc, train, test, error_fn=qp.error.ae)
    # err_pcc, pcc_prev_hat = fit_and_test(pcc, train, test, error_fn=qp.error.ae)
    err_pcc=-1
    # err_pcc_sel, pcc_prev_hat_sel = fit_and_test(pcc_mmd, train_sel, test, error_fn=qp.error.ae)
    err_pcc_sel=-1
    # err_quant, quant_prev_hat = fit_and_test(quant_pps, train, test, error_fn=qp.error.ae)

    # print(f"test-prevalence=\t{F.strprev(test.prevalence())}")
    # print(f"train-prevalence=\t{F.strprev(train.prevalence())}\t{err_noshift:.5f}")
    # print(f"PCC-prevalence=\t{F.strprev(pcc_prev_hat)}\t{err_pcc:.5f}")
    # print(f"Quant-prevalence=\t{F.strprev(quant_prev_hat)}\t{err_quant:.5f}")
    # print()

    return err_noshift, err_cc, err_pcc, err_pcc_sel


for period in range(7):
    cov_names, covariates, labels, subreddits_names, subreddits = load_dataset(path, with_periods=True, return_period=period)

    # column_prefixes, prefix_idx = separate_blocks_idx(cov_names)
    # blocks_idx = prefix_idx.values()

    n_subreddits = len(subreddits_names)

    results = qp.util.parallel(experiment, np.arange(n_subreddits), n_jobs=-1, asarray=False, backend='loky')

    mlpe_errors, cc_errors, pcc_errors, pcc_sel_errors = list(zip(*results))

    print('-'*80)
    print('ERRORS for period', period)
    print('-'*80)
    print(f'MLPE = {np.mean(mlpe_errors):.4f}+-{np.std(mlpe_errors):.4f}')
    print(f'CC = {np.mean(cc_errors):.4f}+-{np.std(cc_errors):.4f}')
    print(f'PCC = {np.mean(pcc_errors):.4f}+-{np.std(pcc_errors):.4f}')
    print(f'PCC-sel = {np.mean(pcc_sel_errors):.4f}+-{np.std(pcc_sel_errors):.4f}')
    # print(f'Quant = {np.mean(q_errors):.4f}+-{np.std(q_errors):.4f}')

"""
activity_dataset
--------------------------------------------------------------------------------
ERRORS for period 0
--------------------------------------------------------------------------------
MLPE = 0.0148+-0.0055
CC = 0.0067+-0.0047
PCC = 0.0089+-0.0053
PCC-sel = 0.0077+-0.0039
--------------------------------------------------------------------------------
ERRORS for period 1
--------------------------------------------------------------------------------
MLPE = 0.0160+-0.0050
CC = 0.0104+-0.0041
PCC = 0.0105+-0.0051
PCC-sel = 0.0076+-0.0058
--------------------------------------------------------------------------------
ERRORS for period 2
--------------------------------------------------------------------------------
MLPE = 0.0149+-0.0063
CC = 0.0079+-0.0034
PCC = 0.0091+-0.0061
PCC-sel = 0.0088+-0.0077
--------------------------------------------------------------------------------
ERRORS for period 3
--------------------------------------------------------------------------------
MLPE = 0.0128+-0.0049
CC = 0.0090+-0.0049
PCC = 0.0059+-0.0032
PCC-sel = 0.0064+-0.0065
--------------------------------------------------------------------------------
ERRORS for period 4
--------------------------------------------------------------------------------
MLPE = 0.0139+-0.0061
CC = 0.0078+-0.0055
PCC = 0.0094+-0.0050
PCC-sel = 0.0074+-0.0041
--------------------------------------------------------------------------------
ERRORS for period 5
--------------------------------------------------------------------------------
MLPE = 0.0138+-0.0048
CC = 0.0069+-0.0034
PCC = 0.0102+-0.0074
PCC-sel = 0.0078+-0.0067
--------------------------------------------------------------------------------
ERRORS for period 6
--------------------------------------------------------------------------------
MLPE = 0.0183+-0.0065
CC = 0.0071+-0.0058
PCC = 0.0124+-0.0056
PCC-sel = 0.0091+-0.0052


toxicity_dataset
--------------------------------------------------------------------------------
ERRORS for period 0
--------------------------------------------------------------------------------
MLPE = 0.0339+-0.0248
CC = 0.0805+-0.0418
PCC = 0.0316+-0.0188
PCC-sel = 0.0282+-0.0163
--------------------------------------------------------------------------------
ERRORS for period 1
--------------------------------------------------------------------------------
MLPE = 0.0391+-0.0213
CC = 0.1435+-0.0583
PCC = 0.0269+-0.0192
PCC-sel = 0.0269+-0.0200
--------------------------------------------------------------------------------
ERRORS for period 2
--------------------------------------------------------------------------------
MLPE = 0.0361+-0.0181
CC = 0.1727+-0.0578
PCC = 0.0330+-0.0205
PCC-sel = 0.0317+-0.0219
--------------------------------------------------------------------------------
ERRORS for period 3
--------------------------------------------------------------------------------
MLPE = 0.0393+-0.0224
CC = 0.1774+-0.0415
PCC = 0.0362+-0.0157
PCC-sel = 0.0281+-0.0157
--------------------------------------------------------------------------------
ERRORS for period 4
--------------------------------------------------------------------------------
MLPE = 0.0330+-0.0190
CC = 0.1854+-0.0454
PCC = 0.0288+-0.0139
PCC-sel = 0.0289+-0.0114
--------------------------------------------------------------------------------
ERRORS for period 5
--------------------------------------------------------------------------------
MLPE = 0.0384+-0.0136
CC = 0.1977+-0.0408
PCC = 0.0291+-0.0142
PCC-sel = 0.0260+-0.0144
--------------------------------------------------------------------------------
ERRORS for period 6
--------------------------------------------------------------------------------
MLPE = 0.0476+-0.0220
CC = 0.1840+-0.0299
PCC = 0.0284+-0.0120
PCC-sel = 0.0234+-0.0089


diversity_dataset
--------------------------------------------------------------------------------
ERRORS for period 0
--------------------------------------------------------------------------------
MLPE = 0.0312+-0.0147
CC = 0.1325+-0.0237
PCC = 0.0351+-0.0162
--------------------------------------------------------------------------------
ERRORS for period 1
--------------------------------------------------------------------------------
MLPE = 0.0400+-0.0215
CC = 0.1293+-0.0393
PCC = 0.0375+-0.0157
--------------------------------------------------------------------------------
ERRORS for period 2
--------------------------------------------------------------------------------
MLPE = 0.0434+-0.0273
CC = 0.1251+-0.0429
PCC = 0.0367+-0.0201
--------------------------------------------------------------------------------
ERRORS for period 3
--------------------------------------------------------------------------------
MLPE = 0.0451+-0.0260
CC = 0.1207+-0.0398
PCC = 0.0386+-0.0168
--------------------------------------------------------------------------------
ERRORS for period 4
--------------------------------------------------------------------------------
MLPE = 0.0358+-0.0203
CC = 0.1126+-0.0335
PCC = 0.0336+-0.0139
--------------------------------------------------------------------------------
ERRORS for period 5
--------------------------------------------------------------------------------
MLPE = 0.0363+-0.0189
CC = 0.1093+-0.0301
PCC = 0.0292+-0.0178
--------------------------------------------------------------------------------
ERRORS for period 6
--------------------------------------------------------------------------------
MLPE = 0.0434+-0.0193
CC = 0.1070+-0.0250
PCC = 0.0310+-0.0195

"""