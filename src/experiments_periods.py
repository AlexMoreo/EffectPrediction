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


from data import load_dataset
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

# path = '../datasets/datasets_periods/activity_dataset'
path = '../datasets/datasets_periods/toxicity_dataset'
# path = '../datasets/datasets_periods/diversity_dataset'


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
    pcc = PCC(BlockEnsembleClassifier(LogisticRegression(), column_names=cov_names))
    # quant_pps = PACC(LogisticRegression())
    # quant_pps = PACC(BlockEnsembleClassifier(LogisticRegression(), column_names=cov_names))

    # train, val = train.split_stratified(train_prop=0.6, random_state=0)
    # quant = qp.model_selection.GridSearchQ(
    #     model= quant,
    #     param_grid={'classifier__C': np.logspace(-4,4,7), 'classifier__class_weight': ['balanced', None]}, # 'bandwidth': np.linspace(0.01, 0.2, 20)},
    #     protocol=UPP(val),
    #     refit=True,
    #     n_jobs=-1,
    #     verbose=True
    # ).fit(train).best_model()

    err_noshift, _ = fit_and_test(mlpe, train, test, error_fn=qp.error.ae)
    err_cc, _ = fit_and_test(cc, train, test, error_fn=qp.error.ae)
    err_pcc, pcc_prev_hat = fit_and_test(pcc, train, test, error_fn=qp.error.ae)
    # err_quant, quant_prev_hat = fit_and_test(quant_pps, train, test, error_fn=qp.error.ae)

    # print(f"test-prevalence=\t{F.strprev(test.prevalence())}")
    # print(f"train-prevalence=\t{F.strprev(train.prevalence())}\t{err_noshift:.5f}")
    # print(f"PCC-prevalence=\t{F.strprev(pcc_prev_hat)}\t{err_pcc:.5f}")
    # print(f"Quant-prevalence=\t{F.strprev(quant_prev_hat)}\t{err_quant:.5f}")
    # print()

    return err_noshift, err_cc, err_pcc #, err_quant


for period in range(7):
    cov_names, covariates, labels, subreddits_names, subreddits = load_dataset(path, with_periods=True, return_period=period)

    # cov_names, covariates, labels, subreddits_names, subreddits = qp.util.pickled_resource(path + '.pkl', load_dataset, path)
    #
    n_subreddits = len(subreddits_names)
    # print(cov_names)

    results = qp.util.parallel(experiment, np.arange(n_subreddits), n_jobs=-1, asarray=False, backend='loky')

    # mlpe_errors, cc_errors, pcc_errors, q_errors = list(zip(*results))
    mlpe_errors, cc_errors, pcc_errors = list(zip(*results))

    print('-'*80)
    print('ERRORS for period', period)
    print('-'*80)
    print(f'MLPE = {np.mean(mlpe_errors):.4f}+-{np.std(mlpe_errors):.4f}')
    print(f'CC = {np.mean(cc_errors):.4f}+-{np.std(cc_errors):.4f}')
    print(f'PCC = {np.mean(pcc_errors):.4f}+-{np.std(pcc_errors):.4f}')
    # print(f'Quant = {np.mean(q_errors):.4f}+-{np.std(q_errors):.4f}')

"""
activity_dataset
--------------------------------------------------------------------------------
ERRORS for period 0
--------------------------------------------------------------------------------
MLPE = 0.0148+-0.0055
CC = 0.0067+-0.0047
PCC = 0.0089+-0.0053
--------------------------------------------------------------------------------
ERRORS for period 1
--------------------------------------------------------------------------------
MLPE = 0.0160+-0.0050
CC = 0.0104+-0.0041
PCC = 0.0105+-0.0051
--------------------------------------------------------------------------------
ERRORS for period 2
--------------------------------------------------------------------------------
MLPE = 0.0149+-0.0063
CC = 0.0079+-0.0034
PCC = 0.0091+-0.0061
--------------------------------------------------------------------------------
ERRORS for period 3
--------------------------------------------------------------------------------
MLPE = 0.0128+-0.0049
CC = 0.0090+-0.0049
PCC = 0.0059+-0.0032
--------------------------------------------------------------------------------
ERRORS for period 4
--------------------------------------------------------------------------------
MLPE = 0.0139+-0.0061
CC = 0.0078+-0.0055
PCC = 0.0094+-0.0050
--------------------------------------------------------------------------------
ERRORS for period 5
--------------------------------------------------------------------------------
MLPE = 0.0138+-0.0048
CC = 0.0069+-0.0034
PCC = 0.0102+-0.0074
--------------------------------------------------------------------------------
ERRORS for period 6
--------------------------------------------------------------------------------
MLPE = 0.0183+-0.0065
CC = 0.0071+-0.0058
PCC = 0.0124+-0.0056



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